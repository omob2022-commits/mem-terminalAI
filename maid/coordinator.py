"""Coordinator — two-layer decision engine for MAID.

Layer 1: fast local heuristics (always runs).
Layer 2: AI fallback (runs when heuristics are uncertain *and*
         rate-limit / sustained-pressure conditions are met).

Thread safety: ``action_history`` and ``mode`` are guarded by
``_lock`` so the TUI can read them from the main thread while the
collector thread writes.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

from maid.models import (
    Action,
    ActionRecord,
    ActionSource,
    AdjustSwappiness,
    Confidence,
    Config,
    DaemonMode,
    DropCaches,
    KillProcess,
    MemSnapshot,
    NoAction,
    ReniceProcess,
)

logger = logging.getLogger(__name__)

# ── Deduplication constants ──────────────────────────────────────────────────

_RENICE_DEDUP_S: float = 120.0
"""Seconds to suppress a duplicate renice on the same PID."""

_EMERGENCY_RAM_PCT: float = 95.0
"""RAM% above which actions bypass cooldown."""

_EMERGENCY_SWAP_PCT: float = 90.0
"""Swap% above which actions bypass cooldown."""

_AI_FALLBACK_LOW_PCT: float = 75.0
"""Lower RAM threshold that, when sustained, triggers the AI fallback."""

_AI_FALLBACK_HIGH_PCT: float = 90.0
"""Upper RAM threshold; above this the heuristics should handle it."""


# ── Protocols for sibling modules ────────────────────────────────────────────


@runtime_checkable
class Executor(Protocol):
    """Interface fulfilled by the executor module."""

    def execute(self, action: Action, *, dry_run: bool) -> ActionRecord:
        """Execute *action* and return a completed record.

        When *dry_run* is ``True`` the executor must **not** perform any
        side-effects but still return a record with ``executed=False``.
        """
        ...


@runtime_checkable
class AIClient(Protocol):
    """Interface fulfilled by the AI client module."""

    def analyse(self, snapshot: MemSnapshot) -> Action:
        """Ask the AI model to decide on an action for *snapshot*."""
        ...


@dataclass(frozen=True, slots=True)
class HeuristicResult:
    """Return type of ``heuristics.evaluate``."""

    action: Action
    confidence: Confidence


@runtime_checkable
class HeuristicsModule(Protocol):
    """Interface for the heuristics layer (module-level function)."""

    def evaluate(self, snapshot: MemSnapshot) -> HeuristicResult:
        ...


# ── Coordinator ──────────────────────────────────────────────────────────────


class Coordinator:
    """Orchestrates heuristics → AI fallback pipeline.

    Parameters
    ----------
    config:
        Immutable runtime configuration.
    executor:
        Responsible for actually performing actions (or dry-running them).
    ai_client:
        Optional AI backend. ``None`` disables Layer 2 entirely.
    heuristics:
        The heuristics module (must expose an ``evaluate`` function).
    """

    def __init__(
        self,
        config: Config,
        executor: Executor,
        ai_client: AIClient | None,
        heuristics: HeuristicsModule,
    ) -> None:
        self._config = config
        self._executor = executor
        self._ai_client = ai_client
        self._heuristics = heuristics

        # ── Public state (guarded by _lock) ──────────────────────────────
        self._lock = threading.Lock()
        self._action_history: list[ActionRecord] = []
        self._mode: DaemonMode = (
            DaemonMode.DRY_RUN if config.dry_run else DaemonMode.MONITORING
        )

        # ── Private timing state ─────────────────────────────────────────
        self._cooldown_until: datetime | None = None
        self._last_ai_call: datetime | None = None
        self._sustained_high_since: datetime | None = None

        # ── Deduplication maps ───────────────────────────────────────────
        self._recent_reniced: dict[int, datetime] = {}
        self._killed_pids: set[int] = set()

    # ── Thread-safe property access ──────────────────────────────────────

    @property
    def action_history(self) -> list[ActionRecord]:
        """Return a *copy* of the action history (thread-safe)."""
        with self._lock:
            return list(self._action_history)

    @property
    def mode(self) -> DaemonMode:
        with self._lock:
            return self._mode

    @mode.setter
    def mode(self, value: DaemonMode) -> None:
        with self._lock:
            self._mode = value

    # ── Public entry points ──────────────────────────────────────────────

    def process_snapshot(self, snapshot: MemSnapshot) -> ActionRecord | None:
        """Main poll-cycle entry point.

        1. Run heuristic evaluation.
        2. If HIGH confidence and non-trivial → execute (respecting cooldown).
        3. Otherwise, attempt AI fallback if conditions are met.

        Returns the :class:`ActionRecord` produced, or ``None`` when
        nothing was done.
        """
        self._update_sustained_tracker(snapshot)
        self._expire_renice_dedup()

        result = self._run_heuristics(snapshot)
        if result is None:
            return None

        # Layer 1: high-confidence, actionable recommendation.
        if result.confidence is Confidence.HIGH and not isinstance(
            result.action, NoAction
        ):
            return self._maybe_execute(
                result.action,
                source=ActionSource.LOCAL,
                snapshot=snapshot,
            )

        # Layer 2: AI fallback (only when enabled).
        if self._should_call_ai(result, snapshot):
            return self._call_ai(snapshot)

        logger.debug(
            "No action taken — heuristic confidence=%s, AI fallback not triggered",
            result.confidence.value,
        )
        return None

    def force_ai_analysis(self, snapshot: MemSnapshot) -> ActionRecord | None:
        """User-triggered AI analysis (keybind).

        Bypasses sustained-pressure and cooldown checks but still
        respects the AI rate limit.
        """
        if self._ai_client is None:
            logger.warning("AI analysis requested but no AI client configured")
            return None

        now = self._now()
        if self._last_ai_call is not None:
            elapsed = (now - self._last_ai_call).total_seconds()
            if elapsed < self._config.ai_rate_limit_s:
                logger.info(
                    "AI rate-limited — %.1fs until next call allowed",
                    self._config.ai_rate_limit_s - elapsed,
                )
                return None

        return self._call_ai(snapshot)

    # ── Layer 1: heuristics ──────────────────────────────────────────────

    def _run_heuristics(self, snapshot: MemSnapshot) -> HeuristicResult | None:
        """Run the heuristic evaluation, catching unexpected errors."""
        try:
            return self._heuristics.evaluate(snapshot)
        except Exception:
            logger.exception("Heuristic evaluation failed")
            return None

    # ── Layer 2: AI fallback logic ───────────────────────────────────────

    def _should_call_ai(
        self,
        heuristic_result: HeuristicResult,
        snapshot: MemSnapshot,
    ) -> bool:
        """Decide whether to invoke the AI backend."""
        if self._ai_client is None:
            return False

        # Respect AI rate limit.
        now = self._now()
        if self._last_ai_call is not None:
            elapsed = (now - self._last_ai_call).total_seconds()
            if elapsed < self._config.ai_rate_limit_s:
                return False

        # Condition 1: heuristics returned LOW confidence.
        if heuristic_result.confidence is Confidence.LOW:
            return True

        # Condition 2: the previous action failed.
        last = self._last_record()
        if last is not None and not last.success:
            return True

        # Condition 3: RAM in 75-90% band sustained > threshold.
        if self._is_sustained_high(snapshot, now):
            return True

        return False

    def _call_ai(self, snapshot: MemSnapshot) -> ActionRecord | None:
        """Invoke the AI client and execute the returned action."""
        if self._ai_client is None:
            return None

        self.mode = DaemonMode.AI_THINKING
        self._last_ai_call = self._now()

        try:
            action = self._ai_client.analyse(snapshot)
        except Exception:
            logger.exception("AI analysis failed")
            self.mode = DaemonMode.MONITORING
            return None

        record = self._maybe_execute(
            action,
            source=ActionSource.AI,
            snapshot=snapshot,
        )

        # Restore mode (execute path may have set COOLDOWN).
        if self.mode is DaemonMode.AI_THINKING:
            self.mode = DaemonMode.MONITORING

        return record

    # ── Execution gate (cooldown + dedup) ────────────────────────────────

    def _maybe_execute(
        self,
        action: Action,
        *,
        source: ActionSource,
        snapshot: MemSnapshot,
    ) -> ActionRecord | None:
        """Apply cooldown, dedup, and then delegate to the executor."""
        now = self._now()

        if isinstance(action, NoAction):
            logger.debug("Explicit NoAction: %s", action.reason)
            return None

        # Deduplication checks.
        if isinstance(action, ReniceProcess):
            last_renice = self._recent_reniced.get(action.pid)
            if last_renice is not None:
                age = (now - last_renice).total_seconds()
                if age < _RENICE_DEDUP_S:
                    logger.info(
                        "Skipping renice for PID %d — reniced %.0fs ago",
                        action.pid,
                        age,
                    )
                    return None

        if isinstance(action, KillProcess):
            if action.pid in self._killed_pids:
                logger.info(
                    "Skipping kill for PID %d — already killed this session",
                    action.pid,
                )
                return None

        # Cooldown check (emergency actions bypass).
        if not self._is_emergency(snapshot) and self._is_in_cooldown(now):
            logger.info(
                "Action suppressed by cooldown (expires %s)",
                self._cooldown_until,
            )
            return None

        # Execute.
        record = self._executor.execute(action, dry_run=self._config.dry_run)
        self._record_action(record, now)

        # Post-execution bookkeeping.
        if record.success and record.executed:
            self._start_cooldown(now)

            if isinstance(action, ReniceProcess):
                self._recent_reniced[action.pid] = now

            if isinstance(action, KillProcess):
                self._killed_pids.add(action.pid)

        return record

    # ── Cooldown helpers ─────────────────────────────────────────────────

    def _is_in_cooldown(self, now: datetime) -> bool:
        if self._cooldown_until is None:
            return False
        if now >= self._cooldown_until:
            self._cooldown_until = None
            if self.mode is DaemonMode.COOLDOWN:
                self.mode = DaemonMode.MONITORING
            return False
        return True

    def _start_cooldown(self, now: datetime) -> None:
        self._cooldown_until = now + timedelta(seconds=self._config.cooldown_s)
        self.mode = DaemonMode.COOLDOWN
        logger.info(
            "Cooldown started — next action allowed after %s",
            self._cooldown_until,
        )

    @staticmethod
    def _is_emergency(snapshot: MemSnapshot) -> bool:
        """True when the system is in a critical state that bypasses cooldown."""
        return (
            snapshot.ram_used_pct > _EMERGENCY_RAM_PCT
            or snapshot.swap_used_pct > _EMERGENCY_SWAP_PCT
        )

    # ── Sustained high-memory tracker ────────────────────────────────────

    def _update_sustained_tracker(self, snapshot: MemSnapshot) -> None:
        """Update the timestamp tracking when RAM first exceeded the
        AI-fallback threshold (75 %)."""
        in_band = _AI_FALLBACK_LOW_PCT <= snapshot.ram_used_pct <= _AI_FALLBACK_HIGH_PCT
        if in_band:
            if self._sustained_high_since is None:
                self._sustained_high_since = snapshot.timestamp
                logger.debug(
                    "RAM entered AI-fallback band (%.1f%%) — tracking started",
                    snapshot.ram_used_pct,
                )
        else:
            if self._sustained_high_since is not None:
                logger.debug(
                    "RAM left AI-fallback band (%.1f%%) — tracker reset",
                    snapshot.ram_used_pct,
                )
            self._sustained_high_since = None

    def _is_sustained_high(self, snapshot: MemSnapshot, now: datetime) -> bool:
        """Return True if RAM has been in the 75-90 % band longer than
        ``ai_sustained_threshold_s``."""
        if self._sustained_high_since is None:
            return False
        if not (_AI_FALLBACK_LOW_PCT <= snapshot.ram_used_pct <= _AI_FALLBACK_HIGH_PCT):
            return False
        elapsed = (now - self._sustained_high_since).total_seconds()
        return elapsed >= self._config.ai_sustained_threshold_s

    # ── Dedup expiry ─────────────────────────────────────────────────────

    def _expire_renice_dedup(self) -> None:
        """Evict stale entries from the renice deduplication map."""
        now = self._now()
        stale = [
            pid
            for pid, ts in self._recent_reniced.items()
            if (now - ts).total_seconds() >= _RENICE_DEDUP_S
        ]
        for pid in stale:
            del self._recent_reniced[pid]

    # ── History helpers ──────────────────────────────────────────────────

    def _record_action(self, record: ActionRecord, now: datetime) -> None:
        """Append *record* to the history, trimming to max size."""
        with self._lock:
            self._action_history.append(record)
            overflow = len(self._action_history) - self._config.max_action_history
            if overflow > 0:
                del self._action_history[:overflow]

    def _last_record(self) -> ActionRecord | None:
        with self._lock:
            if self._action_history:
                return self._action_history[-1]
        return None

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _now() -> datetime:
        """Return the current UTC time (extracted for testability)."""
        return datetime.now(tz=timezone.utc)
