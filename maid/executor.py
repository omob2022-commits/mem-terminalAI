"""Executor — applies MAID actions to the running system.

Handles KillProcess, ReniceProcess, DropCaches, AdjustSwappiness, and
NoAction.  Every action is logged as a JSON line to ``config.log_file``.
Dry-run mode logs intent without touching the system.  Exceptions are
caught internally; the caller always receives an ``ActionRecord``.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from maid.models import (
    Action,
    ActionRecord,
    ActionSource,
    AdjustSwappiness,
    Config,
    DropCaches,
    KillProcess,
    NoAction,
    ReniceProcess,
)

logger = logging.getLogger(__name__)

# ── Default confirmation callback ────────────────────────────────────────────

_SIGKILL_GRACE_SECONDS: float = 3.0
_NICE_MIN: int = 0
_NICE_MAX: int = 19


def default_confirm(prompt: str) -> bool:
    """Print *prompt* to stdout and wait for ``y``/``n`` on stdin.

    Returns ``True`` when the user types a string starting with ``y``
    (case-insensitive), ``False`` otherwise.  Any I/O error is treated
    as rejection.
    """
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
        return answer.startswith("y")
    except (EOFError, KeyboardInterrupt):
        return False


# ── Helpers ──────────────────────────────────────────────────────────────────


def _action_type_name(action: Action) -> str:
    """Return the simple class name of an action, e.g. ``'KillProcess'``."""
    return type(action).__name__


def _action_details(action: Action) -> dict[str, object]:
    """Serialise action fields to a plain dict for JSON logging."""

    def _convert(obj: object) -> object:
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    raw = asdict(action)  # type: ignore[arg-type]
    return {k: _convert(v) for k, v in raw.items()}


def _write_log_line(path: Path, data: dict[str, object]) -> None:
    """Append a single JSON object as a line to *path*.

    Creates parent directories if they do not exist.  Errors are logged
    but never propagated.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, default=str) + "\n")
    except OSError as exc:
        logger.error("Failed to write audit log to %s: %s", path, exc)


# ── Executor class ───────────────────────────────────────────────────────────


class Executor:
    """Apply :pydata:`Action` objects to the running Linux system.

    Parameters
    ----------
    config:
        Runtime configuration (``dry_run``, ``auto_confirm``, ``log_file``).
    confirm_callback:
        Optional callable that receives a human-readable prompt and
        returns ``True`` to proceed.  When ``None``, :func:`default_confirm`
        is used.
    """

    __slots__ = ("_config", "_confirm")

    def __init__(
        self,
        config: Config,
        confirm_callback: Callable[[str], bool] | None = None,
    ) -> None:
        self._config = config
        self._confirm: Callable[[str], bool] = confirm_callback or default_confirm

    # ── public entry point ───────────────────────────────────────────────

    def execute(self, action: Action, source: ActionSource) -> ActionRecord:
        """Execute *action* and return an audit record.

        In dry-run mode the action is logged but not applied.  On any
        failure an ``ActionRecord`` with ``success=False`` is returned —
        this method **never** raises.
        """
        now = datetime.now(tz=timezone.utc)
        action_name = _action_type_name(action)

        # -- dry-run shortcut --
        if self._config.dry_run:
            logger.info("[DRY-RUN] Would execute %s: %s", action_name, action)
            record = ActionRecord(
                timestamp=now,
                action=action,
                source=source,
                executed=False,
                success=True,
            )
            self._log_record(record)
            return record

        # -- dispatch --
        try:
            success, error = self._dispatch(action)
        except Exception as exc:  # noqa: BLE001 — intentional catch-all
            logger.exception("Unexpected error executing %s", action_name)
            success, error = False, str(exc)

        record = ActionRecord(
            timestamp=now,
            action=action,
            source=source,
            executed=True,
            success=success,
            error=error,
        )
        self._log_record(record)
        return record

    # ── internal dispatch ────────────────────────────────────────────────

    def _dispatch(self, action: Action) -> tuple[bool, str | None]:
        """Route *action* to the appropriate handler.

        Returns ``(success, error_message_or_None)``.
        """
        if isinstance(action, KillProcess):
            return self._kill(action)
        if isinstance(action, ReniceProcess):
            return self._renice(action)
        if isinstance(action, DropCaches):
            return self._drop_caches(action)
        if isinstance(action, AdjustSwappiness):
            return self._adjust_swappiness(action)
        if isinstance(action, NoAction):
            return self._no_action(action)
        # Unknown action type — should never happen given the Union, but
        # be defensive.
        logger.warning("Unknown action type: %s", type(action).__name__)
        return False, f"Unknown action type: {type(action).__name__}"

    # ── action handlers ──────────────────────────────────────────────────

    def _kill(self, action: KillProcess) -> tuple[bool, str | None]:
        """Send SIGTERM to *pid*; escalate to SIGKILL after grace period."""
        pid = action.pid

        # Confirmation gate
        if not self._config.auto_confirm:
            prompt = (
                f"Kill process {action.name!r} (PID {pid})? "
                f"Reason: {action.reason}"
            )
            if not self._confirm(prompt):
                logger.info("User declined to kill PID %d (%s)", pid, action.name)
                return False, "User declined"

        # SIGTERM
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to PID %d (%s)", pid, action.name)
        except ProcessLookupError:
            logger.info("PID %d already gone before SIGTERM", pid)
            return True, None
        except PermissionError as exc:
            logger.warning("Permission denied sending SIGTERM to PID %d: %s", pid, exc)
            return False, f"PermissionError: {exc}"
        except OSError as exc:
            logger.warning("OSError sending SIGTERM to PID %d: %s", pid, exc)
            return False, f"OSError: {exc}"

        # Grace period — check whether the process exited.
        time.sleep(_SIGKILL_GRACE_SECONDS)

        try:
            # Signal 0 checks existence without sending a real signal.
            os.kill(pid, 0)
        except ProcessLookupError:
            # Process exited cleanly after SIGTERM.
            logger.info("PID %d exited after SIGTERM", pid)
            return True, None
        except PermissionError:
            # Can't check — assume SIGTERM was enough.
            logger.info(
                "Cannot verify PID %d status (permission); assuming SIGTERM worked",
                pid,
            )
            return True, None

        # Still alive — escalate.
        try:
            os.kill(pid, signal.SIGKILL)
            logger.warning("Sent SIGKILL to PID %d (%s)", pid, action.name)
            return True, None
        except ProcessLookupError:
            logger.info("PID %d exited between check and SIGKILL", pid)
            return True, None
        except PermissionError as exc:
            logger.warning(
                "Permission denied sending SIGKILL to PID %d: %s", pid, exc
            )
            return False, f"SIGTERM sent but SIGKILL failed: PermissionError: {exc}"
        except OSError as exc:
            logger.warning("OSError sending SIGKILL to PID %d: %s", pid, exc)
            return False, f"SIGTERM sent but SIGKILL failed: OSError: {exc}"

    def _renice(self, action: ReniceProcess) -> tuple[bool, str | None]:
        """Change the nice value of *pid*, clamping to [0, 19]."""
        nice = max(_NICE_MIN, min(_NICE_MAX, action.new_nice))
        if nice != action.new_nice:
            logger.info(
                "Clamped requested nice %d → %d for PID %d",
                action.new_nice,
                nice,
                action.pid,
            )

        try:
            os.setpriority(os.PRIO_PROCESS, action.pid, nice)
            logger.info(
                "Reniced PID %d (%s) to %d", action.pid, action.name, nice
            )
            return True, None
        except ProcessLookupError:
            logger.info("PID %d no longer exists", action.pid)
            return False, f"Process {action.pid} no longer exists"
        except PermissionError as exc:
            logger.warning(
                "Permission denied renicing PID %d: %s", action.pid, exc
            )
            return False, f"PermissionError: {exc}"
        except OSError as exc:
            logger.warning("OSError renicing PID %d: %s", action.pid, exc)
            return False, f"OSError: {exc}"

    def _drop_caches(self, action: DropCaches) -> tuple[bool, str | None]:
        """Write cache-drop level to ``/proc/sys/vm/drop_caches``.

        Requires root.  Uses ``sudo tee`` via subprocess so that the
        daemon itself does not need to run as root.
        """
        level = action.level
        if level not in (1, 2, 3):
            return False, f"Invalid drop_caches level: {level}"

        try:
            result = subprocess.run(
                ["sudo", "tee", "/proc/sys/vm/drop_caches"],
                input=str(level),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode != 0:
                err = result.stderr.strip() or f"tee exited with {result.returncode}"
                logger.warning("drop_caches failed: %s", err)
                return False, err
            logger.info("Dropped caches (level %d)", level)
            return True, None
        except PermissionError as exc:
            logger.warning("Permission denied dropping caches: %s", exc)
            return False, f"PermissionError: {exc}"
        except FileNotFoundError as exc:
            logger.warning("sudo/tee not found: %s", exc)
            return False, f"FileNotFoundError: {exc}"
        except subprocess.TimeoutExpired:
            logger.warning("drop_caches command timed out")
            return False, "Command timed out (sudo may be waiting for a password)"

    def _adjust_swappiness(self, action: AdjustSwappiness) -> tuple[bool, str | None]:
        """Set ``vm.swappiness`` via ``sysctl``.  Requires root."""
        value = action.value
        if not (0 <= value <= 200):
            return False, f"Swappiness value out of range: {value}"

        try:
            result = subprocess.run(
                ["sudo", "sysctl", "-w", f"vm.swappiness={value}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode != 0:
                err = result.stderr.strip() or f"sysctl exited with {result.returncode}"
                logger.warning("adjust_swappiness failed: %s", err)
                return False, err
            logger.info("Set vm.swappiness = %d", value)
            return True, None
        except PermissionError as exc:
            logger.warning("Permission denied adjusting swappiness: %s", exc)
            return False, f"PermissionError: {exc}"
        except FileNotFoundError as exc:
            logger.warning("sudo/sysctl not found: %s", exc)
            return False, f"FileNotFoundError: {exc}"
        except subprocess.TimeoutExpired:
            logger.warning("sysctl command timed out")
            return False, "Command timed out (sudo may be waiting for a password)"

    def _no_action(self, action: NoAction) -> tuple[bool, str | None]:
        """Log that no action was taken.  Always succeeds."""
        logger.debug("NoAction: %s", action.reason)
        return True, None

    # ── audit logging ────────────────────────────────────────────────────

    def _log_record(self, record: ActionRecord) -> None:
        """Persist *record* as a JSON line to ``config.log_file``."""
        data: dict[str, object] = {
            "timestamp": record.timestamp.isoformat(),
            "action_type": _action_type_name(record.action),
            "details": _action_details(record.action),
            "source": record.source.value,
            "executed": record.executed,
            "success": record.success,
            "error": record.error,
        }
        _write_log_line(self._config.log_file, data)
