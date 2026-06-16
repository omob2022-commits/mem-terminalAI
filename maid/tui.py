"""Terminal UI for MAID — Memory AI Daemon.

Uses the ``rich`` library to render a live-updating dashboard with:
- System header (hostname, uptime, clock)
- Memory / swap bars with colour-coded thresholds
- Top-10 process table by RSS
- Recent action log from the coordinator
- Status bar with keybind hints

Keyboard handling uses raw-mode stdin so the TUI never blocks
the coordinator or collector threads.
"""

from __future__ import annotations

import logging
import os
import select
import signal
import socket
import sys
import termios
import threading
import time
import tty
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Protocol, Sequence

from rich.box import ROUNDED
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from maid.models import (
    ActionRecord,
    ActionSource,
    DaemonMode,
    DropCaches,
    KillProcess,
    MemSnapshot,
    NoAction,
    ProcessInfo,
    ReniceProcess,
    AdjustSwappiness,
)

if TYPE_CHECKING:
    from maid.models import Config

logger = logging.getLogger(__name__)


# ── Protocols for loose coupling ─────────────────────────────────────────────


class Coordinator(Protocol):
    """Minimal interface the TUI expects from the coordinator."""

    mode: DaemonMode
    action_history: Sequence[ActionRecord]
    latest_snapshot: MemSnapshot | None

    def force_ai_analysis(self) -> None: ...


# ── Helpers ──────────────────────────────────────────────────────────────────


def format_bytes(kb: int) -> str:
    """Convert a value in **kB** to a compact human-readable string.

    Examples
    --------
    >>> format_bytes(512)
    '512.0 KB'
    >>> format_bytes(1_572_864)
    '1.5 GB'
    """
    if kb < 0:
        return "0 KB"
    value = float(kb)
    for unit in ("KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    # Unreachable, but keeps mypy happy.
    return f"{value:.1f} TB"  # pragma: no cover


def _read_uptime() -> str:
    """Return a human-friendly uptime string from ``/proc/uptime``."""
    try:
        raw = open("/proc/uptime").read()  # noqa: SIM115
        secs = int(float(raw.split()[0]))
        days, remainder = divmod(secs, 86400)
        hours, remainder = divmod(remainder, 3600)
        mins, _ = divmod(remainder, 60)
        parts: list[str] = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        parts.append(f"{mins}m")
        return " ".join(parts)
    except (OSError, ValueError, IndexError):
        return "??"


def _pct_color(pct: float) -> str:
    """Return a rich colour name based on a usage percentage."""
    if pct < 60.0:
        return "green"
    if pct < 80.0:
        return "yellow"
    return "red"


def _build_bar(pct: float, width: int = 20) -> Text:
    """Create a coloured bar like ``[████████░░░░] 78.3%``."""
    clamped = max(0.0, min(100.0, pct))
    filled = int(round(clamped / 100.0 * width))
    empty = width - filled
    colour = _pct_color(clamped)
    bar = Text("[", style="bold")
    bar.append("█" * filled, style=colour)
    bar.append("░" * empty, style="dim")
    bar.append("] ", style="bold")
    bar.append(f"{clamped:5.1f}%", style=f"bold {colour}")
    return bar


def _action_description(record: ActionRecord) -> str:
    """Return a one-line description of an ``ActionRecord``."""
    act = record.action
    if isinstance(act, KillProcess):
        return f"Kill PID {act.pid} ({act.name}): {act.reason}"
    if isinstance(act, ReniceProcess):
        return f"Renice PID {act.pid} ({act.name}) → {act.new_nice}: {act.reason}"
    if isinstance(act, DropCaches):
        return f"Drop caches (level {act.level}): {act.reason}"
    if isinstance(act, AdjustSwappiness):
        return f"Swappiness → {act.value}: {act.reason}"
    if isinstance(act, NoAction):
        return f"No action: {act.reason}"
    return str(act)


# ── Keybind help text ────────────────────────────────────────────────────────

_HELP_TEXT = """\
[bold cyan]MAID — Keybinds[/]

  [bold]q[/]  Quit the TUI
  [bold]a[/]  Force AI analysis now
  [bold]d[/]  Toggle dry-run mode
  [bold]p[/]  Pause / resume monitoring
  [bold]?[/]  Show / hide this help

Press any key to dismiss."""


# ── TUI class ────────────────────────────────────────────────────────────────


class TUI:
    """Rich-based terminal dashboard for MAID.

    Parameters
    ----------
    config:
        Runtime configuration (``dry_run`` may be toggled via keybind).
    coordinator:
        The coordinator instance exposing ``mode``, ``action_history``,
        ``latest_snapshot``, and ``force_ai_analysis()``.
    collector_fn:
        Not used directly by the TUI but stored so the caller can wire
        up the collector lifecycle alongside the TUI.
    """

    __slots__ = (
        "_config",
        "_coordinator",
        "_collector_fn",
        "_console",
        "_stop_event",
        "_show_help",
        "_old_termios",
        "_sigwinch_received",
    )

    def __init__(
        self,
        config: Config,
        coordinator: Coordinator,
        collector_fn: Callable[..., object],
    ) -> None:
        self._config = config
        self._coordinator = coordinator
        self._collector_fn = collector_fn
        self._console = Console()
        self._stop_event = threading.Event()
        self._show_help = False
        self._old_termios: list[object] | None = None
        self._sigwinch_received = threading.Event()

    # ── public API ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Blocking main loop — call from the main thread.

        Sets the terminal to raw mode, starts ``rich.live.Live``, and
        refreshes every ``config.poll_interval_s`` seconds (default 3 s).
        Exits cleanly when **q** is pressed or ``_stop_event`` is set.
        """
        self._install_sigwinch()
        self._enter_raw_mode()
        try:
            with Live(
                self._build_layout(),
                console=self._console,
                screen=True,
                refresh_per_second=1,
                transient=True,
            ) as live:
                while not self._stop_event.is_set():
                    self._process_keys()

                    if self._sigwinch_received.is_set():
                        self._sigwinch_received.clear()
                        self._console.size  # forces recalculation

                    live.update(self._build_layout())
                    # Sleep in small steps so key-presses feel responsive.
                    self._stop_event.wait(timeout=0.25)
        except Exception:
            logger.exception("TUI crashed — exiting gracefully")
        finally:
            self._restore_terminal()

    def stop(self) -> None:
        """Signal the TUI to exit its run-loop."""
        self._stop_event.set()

    # ── terminal raw-mode management ─────────────────────────────────────

    def _enter_raw_mode(self) -> None:
        """Switch stdin to raw mode so we can read single key-presses."""
        try:
            fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(fd)
            tty.setraw(fd)
        except (termios.error, OSError, ValueError):
            logger.warning("Could not set terminal to raw mode")

    def _restore_terminal(self) -> None:
        """Restore original terminal settings."""
        if self._old_termios is not None:
            try:
                fd = sys.stdin.fileno()
                termios.tcsetattr(fd, termios.TCSADRAIN, self._old_termios)
            except (termios.error, OSError, ValueError):
                pass  # best-effort

    def _install_sigwinch(self) -> None:
        """Handle SIGWINCH (terminal resize) without crashing."""
        try:
            signal.signal(signal.SIGWINCH, self._on_sigwinch)
        except (OSError, ValueError):
            pass  # running in a context without signal support

    def _on_sigwinch(self, _signum: int, _frame: object) -> None:
        self._sigwinch_received.set()

    # ── keyboard handling ────────────────────────────────────────────────

    def _process_keys(self) -> None:
        """Read any pending keystrokes from stdin (non-blocking)."""
        try:
            fd = sys.stdin.fileno()
        except ValueError:
            return

        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            try:
                ch = os.read(fd, 1).decode("utf-8", errors="ignore")
            except OSError:
                break
            self._handle_key(ch)

    def _handle_key(self, ch: str) -> None:
        """Dispatch a single key-press."""
        if self._show_help:
            # Any key dismisses the help overlay.
            self._show_help = False
            return

        ch_lower = ch.lower()

        if ch_lower == "q":
            self._stop_event.set()

        elif ch_lower == "a":
            threading.Thread(
                target=self._safe_force_ai,
                daemon=True,
                name="tui-force-ai",
            ).start()

        elif ch_lower == "d":
            self._config.dry_run = not self._config.dry_run
            logger.info("Dry-run toggled to %s via TUI", self._config.dry_run)

        elif ch_lower == "p":
            if self._coordinator.mode is DaemonMode.PAUSED:
                self._coordinator.mode = DaemonMode.MONITORING
            else:
                self._coordinator.mode = DaemonMode.PAUSED
            logger.info("Mode set to %s via TUI", self._coordinator.mode.value)

        elif ch == "?":
            self._show_help = True

    def _safe_force_ai(self) -> None:
        """Call ``coordinator.force_ai_analysis()`` in a background thread."""
        try:
            self._coordinator.force_ai_analysis()
        except Exception:
            logger.exception("force_ai_analysis() failed")

    # ── layout building ──────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        """Compose the full-screen layout.

        Returns a ``rich.layout.Layout`` tree:
        ::

            ┌──────── header ────────┐
            │  hostname │ uptime │ ⏰  │
            ├──────── body ──────────┤
            │ memory │  processes    │
            ├──────── log ───────────┤
            │ recent actions         │
            ├──────── status ────────┤
            │ mode + keybind hints   │
            └────────────────────────┘
        """
        root = Layout(name="root")

        root.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=3),
            Layout(name="log", size=9),
            Layout(name="status", size=3),
        )

        root["body"].split_row(
            Layout(name="memory", ratio=2, minimum_size=36),
            Layout(name="processes", ratio=3, minimum_size=50),
        )

        root["header"].update(self._render_header())
        root["memory"].update(self._render_memory())
        root["processes"].update(self._render_processes())
        root["log"].update(self._render_action_log())
        root["status"].update(self._render_status_bar())

        if self._show_help:
            return self._wrap_with_help(root)

        return root

    def _wrap_with_help(self, base: Layout) -> Layout:
        """Overlay a help panel on top of the base layout."""
        overlay = Layout(name="overlay")
        overlay.split_column(
            Layout(name="spacer_top", size=4),
            Layout(name="help_center", size=14),
            Layout(name="spacer_bottom"),
        )
        overlay["help_center"].split_row(
            Layout(name="pad_left"),
            Layout(
                Panel(
                    _HELP_TEXT,
                    title="[bold]Help[/]",
                    box=ROUNDED,
                    style="on #1c1c2e",
                    border_style="bright_cyan",
                ),
                name="help_panel",
                ratio=2,
            ),
            Layout(name="pad_right"),
        )
        # Rich doesn't natively support z-index overlays, so we simply
        # replace the layout with the help view.
        return overlay

    # ── individual panel renderers ───────────────────────────────────────

    def _render_header(self) -> Panel:
        """Hostname · uptime · current time."""
        hostname = socket.gethostname()
        uptime = _read_uptime()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header = Text()
        header.append("  🖥  ", style="bold cyan")
        header.append(hostname, style="bold white")
        header.append("    ⏱  up ", style="dim")
        header.append(uptime, style="bold white")
        header.append("    🕒  ", style="dim")
        header.append(now, style="bold white")

        return Panel(
            header,
            box=ROUNDED,
            style="on #1a1a2e",
            border_style="bright_blue",
        )

    def _render_memory(self) -> Panel:
        """RAM bar, swap bar, and PSI pressure line."""
        snap = self._coordinator.latest_snapshot

        if snap is None:
            return Panel(
                Text("Waiting for first snapshot…", style="dim italic"),
                title="[bold]Memory[/]",
                box=ROUNDED,
                border_style="bright_green",
            )

        lines: list[Text | str] = []

        # ── RAM bar ──
        ram_bar = Text("RAM:  ", style="bold")
        ram_bar.append_text(_build_bar(snap.ram_used_pct))
        ram_bar.append(
            f"  ({format_bytes(snap.mem_used_kb)} / {format_bytes(snap.mem_total_kb)})",
            style="dim",
        )
        lines.append(ram_bar)
        lines.append("")

        # ── Swap bar ──
        swap_bar = Text("Swap: ", style="bold")
        swap_bar.append_text(_build_bar(snap.swap_used_pct))
        swap_bar.append(
            f"  ({format_bytes(snap.swap_used_kb)} / {format_bytes(snap.swap_total_kb)})",
            style="dim",
        )
        lines.append(swap_bar)
        lines.append("")

        # ── PSI ──
        if snap.psi is not None:
            psi_line = Text("PSI:  ", style="bold")
            psi_line.append(
                f"some_avg10={snap.psi.some_avg10:5.2f}  "
                f"some_avg60={snap.psi.some_avg60:5.2f}  "
                f"full_avg10={snap.psi.full_avg10:5.2f}  "
                f"full_avg60={snap.psi.full_avg60:5.2f}",
                style="cyan",
            )
            lines.append(psi_line)
        else:
            lines.append(Text("PSI:  unavailable", style="dim italic"))

        return Panel(
            Group(*lines),
            title="[bold]Memory[/]",
            box=ROUNDED,
            border_style="bright_green",
        )

    def _render_processes(self) -> Panel:
        """Top-10 processes by RSS in a ``rich.table.Table``."""
        table = Table(
            title="Top Processes by RSS",
            box=ROUNDED,
            show_lines=False,
            header_style="bold bright_white on #2a2a4a",
            title_style="bold",
            expand=True,
            pad_edge=True,
        )
        table.add_column("PID", justify="right", style="cyan", width=8)
        table.add_column("Name", style="white", ratio=2)
        table.add_column("RSS", justify="right", style="green", width=10)
        table.add_column("Swap", justify="right", style="yellow", width=10)
        table.add_column("Nice", justify="right", width=5)
        table.add_column("OOM", justify="right", width=6)

        snap = self._coordinator.latest_snapshot
        if snap is None:
            table.add_row("—", "waiting…", "—", "—", "—", "—")
            return Panel(
                table,
                box=ROUNDED,
                border_style="bright_magenta",
            )

        top_procs = snap.processes[:10]
        if not top_procs:
            table.add_row("—", "no processes", "—", "—", "—", "—")
        else:
            # Determine a threshold for "high memory" colouring.
            max_rss = top_procs[0].vm_rss_kb if top_procs else 1
            for proc in top_procs:
                row_style = ""
                if max_rss > 0 and proc.vm_rss_kb / max_rss > 0.8:
                    row_style = "bold red"
                elif proc.oom_score >= 800:
                    row_style = "bold red"

                table.add_row(
                    str(proc.pid),
                    proc.name,
                    format_bytes(proc.vm_rss_kb),
                    format_bytes(proc.vm_swap_kb),
                    str(proc.nice),
                    str(proc.oom_score),
                    style=row_style,
                )

        return Panel(
            table,
            box=ROUNDED,
            border_style="bright_magenta",
        )

    def _render_action_log(self) -> Panel:
        """Last 5 ``ActionRecord``s from the coordinator."""
        history = self._coordinator.action_history
        recent = list(history)[-5:] if history else []

        if not recent:
            content = Text("  No actions recorded yet.", style="dim italic")
            return Panel(
                content,
                title="[bold]Action Log[/]",
                box=ROUNDED,
                border_style="bright_yellow",
            )

        lines: list[Text] = []
        for record in reversed(recent):  # newest first
            line = Text()
            ts = record.timestamp.strftime("%H:%M:%S")
            line.append(f" {ts} ", style="dim")

            # Source tag
            tag_styles = {
                ActionSource.LOCAL: ("bold white on blue", "[LOCAL]"),
                ActionSource.AI: ("bold white on magenta", "  [AI] "),
                ActionSource.USER: ("bold white on green", "[USER] "),
            }
            tag_style, tag_text = tag_styles.get(
                record.source, ("bold", f"[{record.source.value}]")
            )
            line.append(tag_text, style=tag_style)
            line.append(" ", style="")

            # Description
            desc = _action_description(record)
            if not record.success:
                desc += " ✗"
                line.append(desc, style="red")
            elif not record.executed:
                desc += " (dry-run)"
                line.append(desc, style="italic yellow")
            else:
                line.append(desc, style="white")

            lines.append(line)

        return Panel(
            Group(*lines),
            title="[bold]Action Log[/]",
            box=ROUNDED,
            border_style="bright_yellow",
        )

    def _render_status_bar(self) -> Panel:
        """Current mode + keybind hints."""
        mode = self._coordinator.mode
        mode_styles: dict[DaemonMode, tuple[str, str]] = {
            DaemonMode.MONITORING: ("bold green", "● MONITORING"),
            DaemonMode.COOLDOWN: ("bold yellow", "◉ COOLDOWN"),
            DaemonMode.AI_THINKING: ("bold magenta", "⟳ AI THINKING"),
            DaemonMode.PAUSED: ("bold red", "⏸ PAUSED"),
            DaemonMode.DRY_RUN: ("bold cyan", "⚑ DRY-RUN"),
        }
        style, label = mode_styles.get(mode, ("bold", mode.value.upper()))

        status = Text()
        status.append(f"  {label}", style=style)

        if self._config.dry_run:
            status.append("  [dry-run ON]", style="bold cyan")

        # Keybind hints
        status.append("    ", style="")
        hints = [
            ("q", "quit"),
            ("a", "AI"),
            ("d", "dry-run"),
            ("p", "pause"),
            ("?", "help"),
        ]
        for key, desc in hints:
            status.append(f" {key}", style="bold bright_white")
            status.append(f"={desc}", style="dim")

        return Panel(
            status,
            box=ROUNDED,
            border_style="bright_blue",
        )
