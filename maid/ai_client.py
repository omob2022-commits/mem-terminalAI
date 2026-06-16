"""AI client — wraps the Anthropic Claude API for memory management decisions.

Sends a JSON memory snapshot and recent action history to Claude,
parses the structured JSON response into an ``Action``, and never
raises exceptions to the caller (returns ``NoAction`` on any error).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict
from typing import Any

import anthropic

from maid.models import (
    Action,
    ActionRecord,
    ActionSource,
    AdjustSwappiness,
    Confidence,
    DropCaches,
    KillProcess,
    MemSnapshot,
    NoAction,
    ProcessInfo,
    PSIPressure,
    ReniceProcess,
)

logger = logging.getLogger(__name__)

_MODEL: str = "claude-sonnet-4-6"
_MAX_RETRIES: int = 2
_BACKOFF_BASE_S: float = 1.0
_TIMEOUT_S: float = 15.0
_MAX_TOKENS: int = 1024

# ── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT: str = """\
You are a Linux memory management assistant embedded in the MAID daemon.

You receive a JSON object with two keys:
  • "snapshot" — current system memory state (RAM, swap, PSI pressure, top processes).
  • "recent_actions" — list of actions already taken recently (may be empty).

Your job is to decide the single best action to relieve memory pressure, or to
do nothing if the system is healthy.

Respond ONLY with a JSON object (no markdown, no commentary).  Valid shapes:

1. Kill a process:
   {"action": "kill_process", "pid": <int>, "name": "<str>", "reason": "<str>"}

2. Renice a process:
   {"action": "renice_process", "pid": <int>, "name": "<str>", "new_nice": <int>, "reason": "<str>"}

3. Drop page/dentry/inode caches:
   {"action": "drop_caches", "level": <1|2|3>, "reason": "<str>"}

4. Adjust vm.swappiness:
   {"action": "adjust_swappiness", "value": <0–200>, "reason": "<str>"}

5. Do nothing:
   {"action": "no_action", "reason": "<str>"}

Rules:
  • NEVER recommend killing protected processes (pid 1, systemd, compositors,
    any process with oom_score_adj == -1000).
  • Prefer less destructive actions (drop_caches, renice) over killing.
  • Only recommend kill_process when memory pressure is severe and the target
    is clearly the dominant consumer.
  • Keep "reason" concise but informative (one sentence).
"""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _process_to_dict(proc: ProcessInfo) -> dict[str, Any]:
    """Serialise a ``ProcessInfo`` to a plain dict."""
    return {
        "pid": proc.pid,
        "name": proc.name,
        "vm_rss_kb": proc.vm_rss_kb,
        "vm_swap_kb": proc.vm_swap_kb,
        "oom_score": proc.oom_score,
        "oom_score_adj": proc.oom_score_adj,
        "nice": proc.nice,
    }


def _psi_to_dict(psi: PSIPressure) -> dict[str, float]:
    """Serialise ``PSIPressure`` to a plain dict."""
    return {
        "some_avg10": psi.some_avg10,
        "some_avg60": psi.some_avg60,
        "full_avg10": psi.full_avg10,
        "full_avg60": psi.full_avg60,
    }


def snapshot_to_dict(snap: MemSnapshot) -> dict[str, Any]:
    """Convert a ``MemSnapshot`` to a JSON-serialisable dict.

    Timestamps are ISO-8601 strings; tuples become lists; ``None`` stays
    ``None`` (→ ``null``).
    """
    return {
        "timestamp": snap.timestamp.isoformat(),
        "mem_total_kb": snap.mem_total_kb,
        "mem_available_kb": snap.mem_available_kb,
        "mem_used_kb": snap.mem_used_kb,
        "swap_total_kb": snap.swap_total_kb,
        "swap_free_kb": snap.swap_free_kb,
        "swap_used_kb": snap.swap_used_kb,
        "cached_kb": snap.cached_kb,
        "buffers_kb": snap.buffers_kb,
        "ram_used_pct": round(snap.ram_used_pct, 2),
        "swap_used_pct": round(snap.swap_used_pct, 2),
        "processes": [_process_to_dict(p) for p in snap.processes],
        "psi": _psi_to_dict(snap.psi) if snap.psi is not None else None,
    }


def _action_record_to_dict(rec: ActionRecord) -> dict[str, Any]:
    """Serialise an ``ActionRecord`` for the AI context window."""
    # Use dataclasses.asdict for the inner Action, then flatten.
    action_dict = asdict(rec.action)
    # Tag with the action class name so Claude sees what was done.
    action_dict["action_type"] = type(rec.action).__name__
    return {
        "timestamp": rec.timestamp.isoformat(),
        "source": rec.source.value,
        "executed": rec.executed,
        "success": rec.success,
        "error": rec.error,
        "action": action_dict,
    }


# ── JSON extraction from response text ──────────────────────────────────────

_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n(.*?)\n\s*```",
    re.DOTALL,
)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from Claude's response text.

    Handles both raw JSON and JSON wrapped in markdown code fences.

    Raises ``ValueError`` if no valid JSON object can be extracted.
    """
    # 1. Try the raw text as-is.
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2. Try extracting from a markdown code block.
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 3. Last resort: find the first { … } substring.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON object found in response")


# ── Response → Action mapping ───────────────────────────────────────────────


def _parse_action(data: dict[str, Any]) -> Action:
    """Convert a validated dict from Claude into a concrete ``Action``.

    Raises ``ValueError`` on missing/invalid fields.
    """
    action_type: str = data.get("action", "")

    if action_type == "kill_process":
        return KillProcess(
            pid=int(data["pid"]),
            name=str(data["name"]),
            reason=str(data["reason"]),
        )

    if action_type == "renice_process":
        return ReniceProcess(
            pid=int(data["pid"]),
            name=str(data["name"]),
            new_nice=int(data["new_nice"]),
            reason=str(data["reason"]),
        )

    if action_type == "drop_caches":
        level = int(data["level"])
        if level not in (1, 2, 3):
            raise ValueError(f"Invalid drop_caches level: {level}")
        return DropCaches(level=level, reason=str(data["reason"]))

    if action_type == "adjust_swappiness":
        value = int(data["value"])
        if not 0 <= value <= 200:
            raise ValueError(f"Swappiness out of range: {value}")
        return AdjustSwappiness(value=value, reason=str(data["reason"]))

    if action_type == "no_action":
        return NoAction(reason=str(data.get("reason", "AI chose no action")))

    raise ValueError(f"Unknown action type: {action_type!r}")


# ── Client class ─────────────────────────────────────────────────────────────


class AIClient:
    """Synchronous wrapper around the Anthropic Claude API.

    Never raises exceptions to the caller.  Every public method returns a
    concrete ``Action``; errors are logged and yield ``NoAction``.
    """

    __slots__ = ("_client", "available")

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set — AI client disabled.  "
                "Only local heuristics will be used."
            )
            self._client: anthropic.Anthropic | None = None
            self.available: bool = False
            return

        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=_TIMEOUT_S,
        )
        self.available = True
        logger.info("AI client initialised (model=%s).", _MODEL)

    # ── Public API ───────────────────────────────────────────────────────

    def analyze(
        self,
        snapshot: MemSnapshot,
        recent_actions: list[ActionRecord],
    ) -> Action:
        """Send *snapshot* and *recent_actions* to Claude and return an ``Action``.

        On any error (network, parsing, timeout, …) returns ``NoAction``
        with a descriptive reason — never raises.
        """
        if not self.available or self._client is None:
            return NoAction(reason="AI client not available")

        payload = {
            "snapshot": snapshot_to_dict(snapshot),
            "recent_actions": [
                _action_record_to_dict(r) for r in recent_actions
            ],
        }
        user_content = json.dumps(payload, indent=2)

        return self._call_with_retries(user_content)

    # ── Internals ────────────────────────────────────────────────────────

    def _call_with_retries(self, user_content: str) -> Action:
        """Invoke the Claude API with retry + exponential backoff.

        Returns ``NoAction`` if all attempts fail.
        """
        assert self._client is not None  # guaranteed by caller

        last_error: str = "unknown error"
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._single_call(user_content)

            except anthropic.APITimeoutError:
                last_error = "API request timed out"
                logger.warning(
                    "Claude API timeout (attempt %d/%d).",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                )

            except anthropic.APIStatusError as exc:
                last_error = f"API status {exc.status_code}: {exc.message}"
                logger.warning(
                    "Claude API error %d (attempt %d/%d): %s",
                    exc.status_code,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    exc.message,
                )

            except anthropic.APIConnectionError as exc:
                last_error = f"Connection error: {exc}"
                logger.warning(
                    "Claude connection error (attempt %d/%d): %s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    exc,
                )

            except Exception as exc:  # noqa: BLE001
                # Unexpected error — don't retry.
                logger.exception("Unexpected error calling Claude API.")
                return NoAction(
                    reason=f"AI error: {exc}",
                    confidence=Confidence.LOW,
                )

            # Exponential backoff before next attempt.
            if attempt < _MAX_RETRIES:
                delay = _BACKOFF_BASE_S * (2**attempt)
                logger.debug("Retrying in %.1fs …", delay)
                time.sleep(delay)

        return NoAction(
            reason=f"AI unavailable after {_MAX_RETRIES + 1} attempts: {last_error}",
            confidence=Confidence.LOW,
        )

    def _single_call(self, user_content: str) -> Action:
        """Make one API call and parse the response into an ``Action``.

        Raises ``anthropic.*Error`` on transport/API failures.
        Returns ``NoAction`` only on malformed *response content*.
        """
        assert self._client is not None

        message = self._client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        # Extract text from the response.
        text_parts = [
            block.text
            for block in message.content
            if hasattr(block, "text")
        ]
        if not text_parts:
            logger.warning("Claude returned no text content.")
            return NoAction(
                reason="AI response malformed",
                confidence=Confidence.LOW,
            )

        raw_text = "\n".join(text_parts)
        logger.debug("Raw Claude response: %s", raw_text[:500])

        try:
            data = _extract_json(raw_text)
        except ValueError:
            logger.warning(
                "Could not extract JSON from Claude response: %s",
                raw_text[:300],
            )
            return NoAction(
                reason="AI response malformed",
                confidence=Confidence.LOW,
            )

        try:
            return _parse_action(data)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse action from AI response: %s", exc)
            return NoAction(
                reason="AI response malformed",
                confidence=Confidence.LOW,
            )
