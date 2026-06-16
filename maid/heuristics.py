"""Stateless heuristic rules for memory pressure evaluation.

Every function in this module is *pure* — no I/O, no side effects.
The main entry point is :func:`evaluate`, which inspects a
:class:`~maid.models.MemSnapshot` and returns a recommended action
together with a confidence level.

Rule priority (highest → lowest):
    1. RAM > 90 %        → DropCaches(level=1) + KillProcess
    2. RAM > 80 % ∧ Swap > 70 % → ReniceProcess (top consumer, +10)
    3. Swap > 90 %       → KillProcess (single worst swap hog)
    4. RAM 75 – 90 %     → NoAction / LOW  (hint: consider AI fallback)
    5. RAM < 70 %        → NoAction / HIGH (all good)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from maid.models import (
    Action,
    Confidence,
    DropCaches,
    KillProcess,
    NoAction,
    ReniceProcess,
    is_protected,
)

if TYPE_CHECKING:
    from maid.models import MemSnapshot, ProcessInfo

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def pick_killable(processes: tuple[ProcessInfo, ...]) -> ProcessInfo | None:
    """Return the best kill candidate from *processes*, or ``None``.

    Selection strategy:
        1. Filter out protected processes.
        2. Sort the remaining ones by ``oom_score`` descending (the kernel's
           own ranking of who to kill first) so we align with OOM-killer
           behaviour.
        3. Return the top candidate.

    This function is pure and never mutates its input.
    """
    candidates = [p for p in processes if not is_protected(p)]
    if not candidates:
        return None
    # Highest oom_score → most "killable" according to the kernel.
    candidates.sort(key=lambda p: p.oom_score, reverse=True)
    return candidates[0]


def pick_reniceable(processes: tuple[ProcessInfo, ...]) -> ProcessInfo | None:
    """Return the top RSS non-protected process that isn't already at nice 19.

    Selection strategy:
        1. Filter out protected processes and those already at max nice (19).
        2. Sort by ``vm_rss_kb`` descending — we want to slow down the
           biggest memory consumer.
        3. Return the top candidate.

    This function is pure and never mutates its input.
    """
    candidates = [
        p for p in processes if not is_protected(p) and p.nice < 19
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.vm_rss_kb, reverse=True)
    return candidates[0]


# ── Main evaluation ─────────────────────────────────────────────────────────


def evaluate(snapshot: MemSnapshot) -> tuple[Action | None, Confidence]:
    """Evaluate *snapshot* and return ``(action, confidence)``.

    Rules are checked in strict priority order; the first matching rule
    wins.  If no rule matches clearly the function returns
    ``(None, Confidence.LOW)`` to signal ambiguity to the coordinator.

    Parameters
    ----------
    snapshot:
        A frozen, point-in-time view of system memory and processes.

    Returns
    -------
    tuple[Action | None, Confidence]
        The recommended action (or ``None``) and how confident the
        engine is about the recommendation.
    """
    ram_pct = snapshot.ram_used_pct
    swap_pct = snapshot.swap_used_pct

    # ── Rule 1: RAM > 90 % ──────────────────────────────────────────────
    if ram_pct > 90.0:
        return _rule_critical_ram(snapshot)

    # ── Rule 2: RAM > 80 % AND Swap > 70 % ──────────────────────────────
    if ram_pct > 80.0 and swap_pct > 70.0:
        return _rule_high_ram_high_swap(snapshot)

    # ── Rule 3: Swap > 90 % ─────────────────────────────────────────────
    if swap_pct > 90.0:
        return _rule_critical_swap(snapshot)

    # ── Rule 4: RAM 75 – 90 % (sustained pressure) ─────────────────────
    if 75.0 <= ram_pct <= 90.0:
        return (
            NoAction(
                reason=(
                    f"RAM at {ram_pct:.1f}% — sustained pressure zone; "
                    "consider AI-driven analysis"
                ),
            ),
            Confidence.LOW,
        )

    # ── Rule 5: RAM < 70 % — all clear ─────────────────────────────────
    if ram_pct < 70.0:
        return (
            NoAction(
                reason=f"RAM at {ram_pct:.1f}% — system healthy",
                confidence=Confidence.HIGH,
            ),
            Confidence.HIGH,
        )

    # Fallback: RAM is between 70 % and 75 % — not clearly stressed.
    return (None, Confidence.LOW)


# ── Private rule implementations ────────────────────────────────────────────


def _rule_critical_ram(
    snapshot: MemSnapshot,
) -> tuple[Action, Confidence]:
    """RAM > 90 %: drop page caches **and** kill the lowest-priority
    non-essential process.

    If no killable process exists we still return DropCaches (something
    is better than nothing).
    """
    target = pick_killable(snapshot.processes)
    if target is not None:
        action: Action = KillProcess(
            pid=target.pid,
            name=target.name,
            reason=(
                f"RAM at {snapshot.ram_used_pct:.1f}% (critical); "
                f"killing '{target.name}' (pid {target.pid}, "
                f"RSS {target.vm_rss_kb} kB, oom_score {target.oom_score})"
            ),
            confidence=Confidence.HIGH,
        )
        logger.info(
            "Rule 1 triggered — critical RAM (%.1f%%): recommending kill "
            "of '%s' (pid %d)",
            snapshot.ram_used_pct,
            target.name,
            target.pid,
        )
        return (action, Confidence.HIGH)

    # No killable target — fall back to DropCaches only.
    action = DropCaches(
        level=1,
        reason=(
            f"RAM at {snapshot.ram_used_pct:.1f}% (critical); "
            "dropping page caches (no killable process found)"
        ),
        confidence=Confidence.HIGH,
    )
    logger.info(
        "Rule 1 triggered — critical RAM (%.1f%%) but no killable process; "
        "recommending cache drop",
        snapshot.ram_used_pct,
    )
    return (action, Confidence.HIGH)


def _rule_high_ram_high_swap(
    snapshot: MemSnapshot,
) -> tuple[Action, Confidence]:
    """RAM > 80 % AND Swap > 70 %: renice the top memory consumer (+10)."""
    target = pick_reniceable(snapshot.processes)
    if target is not None:
        new_nice = min(target.nice + 10, 19)
        action: Action = ReniceProcess(
            pid=target.pid,
            name=target.name,
            new_nice=new_nice,
            reason=(
                f"RAM {snapshot.ram_used_pct:.1f}% / Swap "
                f"{snapshot.swap_used_pct:.1f}%; renicing '{target.name}' "
                f"(pid {target.pid}) from nice {target.nice} → {new_nice}"
            ),
            confidence=Confidence.HIGH,
        )
        logger.info(
            "Rule 2 triggered — high RAM+swap: renicing '%s' (pid %d) "
            "to nice %d",
            target.name,
            target.pid,
            new_nice,
        )
        return (action, Confidence.HIGH)

    # Every process is either protected or already at nice 19.
    return (None, Confidence.LOW)


def _rule_critical_swap(
    snapshot: MemSnapshot,
) -> tuple[Action, Confidence]:
    """Swap > 90 %: kill the single worst swap-memory hog."""
    # Sort non-protected processes by swap usage descending to find the
    # worst offender.
    candidates = [p for p in snapshot.processes if not is_protected(p)]
    if not candidates:
        return (None, Confidence.LOW)

    candidates.sort(key=lambda p: p.vm_swap_kb, reverse=True)
    target = candidates[0]

    action: Action = KillProcess(
        pid=target.pid,
        name=target.name,
        reason=(
            f"Swap at {snapshot.swap_used_pct:.1f}% (critical); "
            f"killing top swap consumer '{target.name}' "
            f"(pid {target.pid}, VmSwap {target.vm_swap_kb} kB)"
        ),
        confidence=Confidence.HIGH,
    )
    logger.info(
        "Rule 3 triggered — critical swap (%.1f%%): killing '%s' (pid %d, "
        "VmSwap %d kB)",
        snapshot.swap_used_pct,
        target.name,
        target.pid,
        target.vm_swap_kb,
    )
    return (action, Confidence.HIGH)


# ── Action description ──────────────────────────────────────────────────────


def describe_action(action: Action) -> str:
    """Return a human-readable, one-line description of *action*.

    Parameters
    ----------
    action:
        Any variant of :data:`~maid.models.Action`.

    Returns
    -------
    str
        A short sentence suitable for logging or TUI display.

    Examples
    --------
    >>> from maid.models import NoAction, Confidence
    >>> describe_action(NoAction(reason="all good"))
    'No action needed: all good'
    """
    if isinstance(action, KillProcess):
        return (
            f"Kill process '{action.name}' (pid {action.pid}): "
            f"{action.reason}"
        )
    if isinstance(action, ReniceProcess):
        return (
            f"Renice process '{action.name}' (pid {action.pid}) "
            f"to nice {action.new_nice}: {action.reason}"
        )
    if isinstance(action, DropCaches):
        return f"Drop caches (level {action.level}): {action.reason}"
    if isinstance(action, NoAction):
        return f"No action needed: {action.reason}"
    # AdjustSwappiness or any future variant
    return f"Action: {action!r}"
