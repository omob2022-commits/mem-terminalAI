"""Collector — polls /proc to build a MemSnapshot.

Reads /proc/meminfo, /proc/*/status, /proc/*/oom_score{,_adj}, and
/proc/pressure/memory to produce a frozen :class:`MemSnapshot` on
every call to :func:`collect`.

Pure-Python, no external dependencies, no subprocess calls.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from maid.models import MemSnapshot, ProcessInfo, PSIPressure

logger = logging.getLogger(__name__)

# ── /proc paths ──────────────────────────────────────────────────────────────

_PROC = Path("/proc")
_MEMINFO = _PROC / "meminfo"
_PRESSURE_MEMORY = _PROC / "pressure" / "memory"

# Fields we extract from /proc/meminfo (all values are in kB).
_MEMINFO_FIELDS: frozenset[str] = frozenset(
    {
        "MemTotal",
        "MemAvailable",
        "SwapTotal",
        "SwapFree",
        "Cached",
        "Buffers",
    }
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_meminfo(proc_root: Path = _PROC) -> dict[str, int]:
    """Parse /proc/meminfo and return requested fields in kB.

    Returns a dict like ``{"MemTotal": 16384000, ...}``.  Missing fields
    are logged at WARNING and default to ``0``.
    """
    meminfo_path = proc_root / "meminfo"
    result: dict[str, int] = {}
    try:
        text = meminfo_path.read_text()
    except OSError:
        logger.error("Cannot read %s — returning zeroes", meminfo_path)
        return {name: 0 for name in _MEMINFO_FIELDS}

    for line in text.splitlines():
        # Typical line: "MemTotal:       16384000 kB"
        parts = line.split(":")
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        if key not in _MEMINFO_FIELDS:
            continue
        # Value is "   16384000 kB" — strip units and whitespace.
        value_str = parts[1].strip().split()[0]
        try:
            result[key] = int(value_str)
        except ValueError:
            logger.warning("Non-integer value for %s: %r", key, value_str)
            result[key] = 0

    # Ensure every requested field has a value.
    for name in _MEMINFO_FIELDS:
        if name not in result:
            logger.warning("Field %s missing from %s", name, meminfo_path)
            result.setdefault(name, 0)

    return result


def _read_int(path: Path, default: int = 0) -> int:
    """Read a single integer from a /proc file, returning *default* on error."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return default


def _parse_process(pid_dir: Path) -> ProcessInfo | None:
    """Build a :class:`ProcessInfo` from ``/proc/<pid>/``.

    Returns ``None`` when the process cannot be read (permission denied,
    vanished mid-read, or is a kernel thread with no RSS).
    """
    pid_str = pid_dir.name
    try:
        pid = int(pid_str)
    except ValueError:
        return None  # not a numeric directory

    # ── /proc/<pid>/status ───────────────────────────────────────────────
    status_path = pid_dir / "status"
    try:
        status_text = status_path.read_text()
    except OSError:
        return None  # permission denied or process vanished

    name: str = ""
    vm_rss_kb: int = 0
    vm_swap_kb: int = 0
    found_rss = False

    for line in status_text.splitlines():
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("VmRSS:"):
            try:
                vm_rss_kb = int(line.split(":")[1].strip().split()[0])
                found_rss = True
            except (ValueError, IndexError):
                pass
        elif line.startswith("VmSwap:"):
            try:
                vm_swap_kb = int(line.split(":")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    # Skip kernel threads: they have no VmRSS line or VmRSS == 0.
    if not found_rss or vm_rss_kb == 0:
        return None

    # ── OOM scores ───────────────────────────────────────────────────────
    oom_score = _read_int(pid_dir / "oom_score")
    oom_score_adj = _read_int(pid_dir / "oom_score_adj")

    # ── nice value ───────────────────────────────────────────────────────
    try:
        nice = os.getpriority(os.PRIO_PROCESS, pid)
    except OSError:
        nice = 0

    return ProcessInfo(
        pid=pid,
        name=name,
        vm_rss_kb=vm_rss_kb,
        vm_swap_kb=vm_swap_kb,
        oom_score=oom_score,
        oom_score_adj=oom_score_adj,
        nice=nice,
    )


def _parse_psi(proc_root: Path = _PROC) -> PSIPressure | None:
    """Parse /proc/pressure/memory and return a :class:`PSIPressure`.

    Returns ``None`` when the file is unreadable (older kernels < 4.20,
    or CONFIG_PSI disabled).

    Expected format (two relevant lines)::

        some avg10=0.00 avg60=0.00 avg300=0.00 total=0
        full avg10=0.00 avg60=0.00 avg300=0.00 total=0
    """
    pressure_path = proc_root / "pressure" / "memory"
    try:
        text = pressure_path.read_text()
    except OSError:
        logger.debug("PSI not available at %s", pressure_path)
        return None

    some_avg10 = 0.0
    some_avg60 = 0.0
    full_avg10 = 0.0
    full_avg60 = 0.0

    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        kind = parts[0]  # "some" or "full"
        if kind not in ("some", "full"):
            continue

        # Parse key=value pairs in the rest of the line.
        kv: dict[str, str] = {}
        for token in parts[1:]:
            if "=" in token:
                k, v = token.split("=", 1)
                kv[k] = v

        try:
            if kind == "some":
                some_avg10 = float(kv.get("avg10", "0"))
                some_avg60 = float(kv.get("avg60", "0"))
            elif kind == "full":
                full_avg10 = float(kv.get("avg10", "0"))
                full_avg60 = float(kv.get("avg60", "0"))
        except ValueError as exc:
            logger.warning("Malformed PSI line: %r (%s)", line, exc)

    return PSIPressure(
        some_avg10=some_avg10,
        some_avg60=some_avg60,
        full_avg10=full_avg10,
        full_avg60=full_avg60,
    )


def _collect_processes(proc_root: Path = _PROC) -> list[ProcessInfo]:
    """Iterate over /proc/<pid>/ directories and collect process info.

    Silently skips processes we cannot read (permission denied, vanished
    between listing and reading, or kernel threads).
    """
    processes: list[ProcessInfo] = []
    try:
        entries = proc_root.iterdir()
    except OSError:
        logger.error("Cannot list %s", proc_root)
        return processes

    for entry in entries:
        if not entry.name.isdigit():
            continue
        info = _parse_process(entry)
        if info is not None:
            processes.append(info)

    return processes


# ── Public API ───────────────────────────────────────────────────────────────


def collect(top_n: int = 10, *, proc_root: Path = _PROC) -> MemSnapshot:
    """Collect a full memory snapshot from /proc.

    Parameters
    ----------
    top_n:
        Number of top-RSS processes to include in the snapshot.
    proc_root:
        Root of the proc filesystem.  Override for testing.

    Returns
    -------
    MemSnapshot
        Frozen dataclass with system memory info, top processes, and
        optional PSI data.
    """
    now = datetime.now(tz=timezone.utc)

    # ── System-wide memory info ──────────────────────────────────────────
    mi = _parse_meminfo(proc_root)
    mem_total = mi["MemTotal"]
    mem_available = mi["MemAvailable"]
    swap_total = mi["SwapTotal"]
    swap_free = mi["SwapFree"]
    cached = mi["Cached"]
    buffers = mi["Buffers"]

    # Derived percentages — guard against zero-division.
    if mem_total > 0:
        ram_used_pct = round(
            (mem_total - mem_available) / mem_total * 100.0, 2
        )
    else:
        ram_used_pct = 0.0

    if swap_total > 0:
        swap_used_pct = round(
            (swap_total - swap_free) / swap_total * 100.0, 2
        )
    else:
        swap_used_pct = 0.0

    # ── Per-process info ─────────────────────────────────────────────────
    all_procs = _collect_processes(proc_root)
    # Sort descending by RSS, then take top N.
    all_procs.sort(key=lambda p: p.vm_rss_kb, reverse=True)
    top_procs = tuple(all_procs[:top_n])

    # ── PSI pressure ─────────────────────────────────────────────────────
    psi = _parse_psi(proc_root)

    return MemSnapshot(
        timestamp=now,
        mem_total_kb=mem_total,
        mem_available_kb=mem_available,
        swap_total_kb=swap_total,
        swap_free_kb=swap_free,
        cached_kb=cached,
        buffers_kb=buffers,
        ram_used_pct=ram_used_pct,
        swap_used_pct=swap_used_pct,
        processes=top_procs,
        psi=psi,
    )
