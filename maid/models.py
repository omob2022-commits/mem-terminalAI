"""Shared data models for MAID — Memory AI Daemon.

All typed dataclasses, enums, and type aliases used across modules.
No mutable global state; everything is passed explicitly.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union


# ── Enums ────────────────────────────────────────────────────────────────────


class Confidence(enum.Enum):
    """How certain the heuristic engine is about its recommendation."""

    HIGH = "high"
    LOW = "low"


class ActionSource(enum.Enum):
    """Where a decision originated."""

    LOCAL = "LOCAL"
    AI = "AI"
    USER = "USER"


class DaemonMode(enum.Enum):
    """Current operational mode shown in the TUI status bar."""

    MONITORING = "monitoring"
    COOLDOWN = "cooldown"
    AI_THINKING = "ai_thinking"
    PAUSED = "paused"
    DRY_RUN = "dry_run"


# ── Process info ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    """Snapshot of a single process from /proc/*/status."""

    pid: int
    name: str
    vm_rss_kb: int  # VmRSS in kB
    vm_swap_kb: int  # VmSwap in kB
    oom_score: int  # /proc/pid/oom_score
    oom_score_adj: int  # /proc/pid/oom_score_adj
    nice: int  # current nice value


# ── Memory snapshot ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PSIPressure:
    """Memory pressure stall information from /proc/pressure/memory."""

    some_avg10: float
    some_avg60: float
    full_avg10: float
    full_avg60: float


@dataclass(frozen=True, slots=True)
class MemSnapshot:
    """Complete system memory state at a point in time."""

    timestamp: datetime

    # From /proc/meminfo (all in kB)
    mem_total_kb: int
    mem_available_kb: int
    swap_total_kb: int
    swap_free_kb: int
    cached_kb: int
    buffers_kb: int

    # Derived percentages (0.0–100.0)
    ram_used_pct: float
    swap_used_pct: float

    # Top processes sorted by RSS descending
    processes: tuple[ProcessInfo, ...]

    # PSI pressure (may be None if /proc/pressure/memory is unreadable)
    psi: PSIPressure | None = None

    @property
    def mem_used_kb(self) -> int:
        return self.mem_total_kb - self.mem_available_kb

    @property
    def swap_used_kb(self) -> int:
        return self.swap_total_kb - self.swap_free_kb


# ── Actions ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class KillProcess:
    """Request to kill a process."""

    pid: int
    name: str
    reason: str
    confidence: Confidence = Confidence.HIGH


@dataclass(frozen=True, slots=True)
class ReniceProcess:
    """Request to change a process's nice value."""

    pid: int
    name: str
    new_nice: int
    reason: str
    confidence: Confidence = Confidence.HIGH


@dataclass(frozen=True, slots=True)
class DropCaches:
    """Request to drop page/dentry/inode caches.

    level: 1 = page cache, 2 = dentries+inodes, 3 = both.
    """

    level: int  # 1, 2, or 3
    reason: str
    confidence: Confidence = Confidence.HIGH


@dataclass(frozen=True, slots=True)
class AdjustSwappiness:
    """Request to change vm.swappiness via sysctl."""

    value: int  # 0–200
    reason: str
    confidence: Confidence = Confidence.HIGH


@dataclass(frozen=True, slots=True)
class NoAction:
    """Explicit decision to do nothing."""

    reason: str
    confidence: Confidence = Confidence.HIGH


# Union of all possible actions the decision engine can emit.
Action = Union[KillProcess, ReniceProcess, DropCaches, AdjustSwappiness, NoAction]


# ── Action history ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """Persisted record of an executed (or dry-run) action."""

    timestamp: datetime
    action: Action
    source: ActionSource
    executed: bool  # False when in dry-run mode
    success: bool
    error: str | None = None


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Config:
    """Runtime configuration — parsed from CLI args, never mutated after init."""

    dry_run: bool = False
    auto_confirm: bool = False
    poll_interval_s: float = 3.0
    ram_threshold_pct: float = 90.0
    cooldown_s: float = 60.0
    ai_sustained_threshold_s: float = 30.0
    ai_rate_limit_s: float = 30.0
    max_action_history: int = 20
    log_dir: Path = field(
        default_factory=lambda: Path.home() / ".local" / "share" / "maid"
    )
    log_file: Path = field(
        default_factory=lambda: Path.home()
        / ".local"
        / "share"
        / "maid"
        / "actions.log"
    )


# ── Protected processes ──────────────────────────────────────────────────────

# Basenames of processes that must never be killed or reniced.
PROTECTED_NAMES: frozenset[str] = frozenset(
    {
        "init",
        "systemd",
        "Xorg",
        "Xwayland",
        "gnome-shell",
        "kwin_wayland",
        "kwin_x11",
        "plasmashell",
        "sway",
        "weston",
        "mutter",
        "maid",        # this daemon itself
        "memgpt",
    }
)


def is_protected(proc: ProcessInfo) -> bool:
    """Return True if the process must never be acted on."""
    if proc.pid == 1:
        return True
    if proc.oom_score_adj == -1000:
        return True
    if proc.name in PROTECTED_NAMES:
        return True
    # Kernel threads have VmRSS == 0 and names in square brackets
    if proc.vm_rss_kb == 0:
        return True
    return False
