"""Entry point for the MAID daemon.

Parses CLI arguments, wires up all components, and runs the main loop
with a background collector thread and coordinator thread while the TUI
occupies the main thread.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from maid.models import Config, MemSnapshot

if TYPE_CHECKING:
    from maid.ai_client import AIClient
    from maid.collector import Collector
    from maid.coordinator import Coordinator
    from maid.executor import Executor
    from maid.tui import TUI

logger = logging.getLogger(__name__)

# ── Shared mutable container for the latest snapshot ────────────────────────

_snapshot_lock = threading.Lock()
_latest_snapshot: MemSnapshot | None = None


def _store_snapshot(snap: MemSnapshot) -> None:
    """Thread-safe update of the latest snapshot."""
    global _latest_snapshot
    with _snapshot_lock:
        _latest_snapshot = snap


def _load_snapshot() -> MemSnapshot | None:
    """Thread-safe read of the latest snapshot."""
    with _snapshot_lock:
        return _latest_snapshot


# ── CLI argument parsing ────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Construct the ArgumentParser for the ``maid`` CLI."""
    parser = argparse.ArgumentParser(
        prog="maid",
        description="MAID — Memory AI Daemon: terminal-based Linux memory management.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Enable dry-run mode (no system modifications).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        default=False,
        help="Auto-confirm dangerous actions (no y/n prompts).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        metavar="N",
        help="Poll interval in seconds (default: 3).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        metavar="N",
        help="RAM threshold percentage for heuristic actions (default: 90).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        default=False,
        help="Disable AI fallback entirely.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override log directory (default: ~/.local/share/maid/).",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a ``Config`` from parsed CLI arguments."""
    cfg = Config(
        dry_run=args.dry_run,
        auto_confirm=args.auto,
        poll_interval_s=args.interval,
        ram_threshold_pct=args.threshold,
    )
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
        cfg.log_file = args.log_dir / "actions.log"
    return cfg


# ── Logging setup ───────────────────────────────────────────────────────────


def setup_logging(log_dir: Path) -> None:
    """Configure root logger to write to *stderr* and a rotating file.

    The file is ``maid.log`` inside *log_dir*.
    """
    log_file = log_dir / "maid.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Formatter shared by both handlers
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # stderr handler — INFO and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)

    # File handler — DEBUG and above
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


# ── Background threads ─────────────────────────────────────────────────────


def _collector_loop(
    config: Config,
    stop_event: threading.Event,
) -> None:
    """Continuously collect memory snapshots until *stop_event* is set.

    Each snapshot is stored in the module-level ``_latest_snapshot`` via
    :func:`_store_snapshot` so the coordinator thread can consume it.
    """
    logger.info("Collector thread started (interval=%.1fs)", config.poll_interval_s)
    while not stop_event.is_set():
        try:
            snap = collector.collect()
            _store_snapshot(snap)
        except Exception:
            logger.exception("Error in collector loop")
        stop_event.wait(timeout=config.poll_interval_s)
    logger.info("Collector thread exiting")


def _coordinator_loop(
    coordinator: Coordinator,
    config: Config,
    stop_event: threading.Event,
) -> None:
    """Continuously process the latest snapshot until *stop_event* is set.

    Sleeps for half the poll interval between iterations so the coordinator
    reacts faster than the collector produces.
    """
    logger.info("Coordinator thread started")
    process_interval = max(config.poll_interval_s / 2.0, 0.5)
    while not stop_event.is_set():
        snap = _load_snapshot()
        if snap is not None:
            try:
                coordinator.process_snapshot(snap)
            except Exception:
                logger.exception("Error in coordinator loop")
        stop_event.wait(timeout=process_interval)
    logger.info("Coordinator thread exiting")


# ── Signal handling ─────────────────────────────────────────────────────────


def _install_signal_handlers(stop_event: threading.Event) -> None:
    """Register handlers for SIGTERM and SIGINT that trigger a clean shutdown."""

    def _handler(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating shutdown", sig_name)
        stop_event.set()
        # Flush all log handlers so nothing is lost
        for handler in logging.getLogger().handlers:
            handler.flush()
        print("\nMAID shutting down...")  # noqa: T201
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


# ── Main entry point ───────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, wire components, and run the daemon.

    Parameters
    ----------
    argv:
        Explicit argument list (defaults to ``sys.argv[1:]``).
        Useful for testing.
    """
    # ── 1. Parse args & build config ────────────────────────────────────
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    # ── 2. Ensure log directory exists ──────────────────────────────────
    config.log_dir.mkdir(parents=True, exist_ok=True)

    # ── 3. Set up logging ───────────────────────────────────────────────
    setup_logging(config.log_dir)
    logger.info("MAID starting (dry_run=%s, auto=%s)", config.dry_run, config.auto_confirm)

    # ── 4. Create components ────────────────────────────────────────────
    # Late imports to avoid circular dependencies and keep startup fast.
    import maid.collector as collector  # noqa: PLC0415
    from maid.coordinator import Coordinator  # noqa: PLC0415
    from maid.executor import Executor  # noqa: PLC0415
    from maid.tui import TUI  # noqa: PLC0415

    ai_client: AIClient | None = None
    if not args.no_ai:
        from maid.ai_client import AIClient  # noqa: PLC0415

        try:
            ai_client = AIClient(config)
            logger.info("AI client initialised")
        except Exception:
            logger.exception("Failed to initialise AI client — continuing without AI")
            ai_client = None
    else:
        logger.info("AI fallback disabled via --no-ai")

    executor = Executor(config)
    coordinator = Coordinator(config=config, executor=executor, ai_client=ai_client)

    # ── 5. Shared stop event ────────────────────────────────────────────
    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    # ── 6. Start background threads ─────────────────────────────────────
    collector_thread = threading.Thread(
        target=_collector_loop,
        args=(config, stop_event),
        name="maid-collector",
        daemon=True,
    )
    coordinator_thread = threading.Thread(
        target=_coordinator_loop,
        args=(coordinator, config, stop_event),
        name="maid-coordinator",
        daemon=True,
    )
    collector_thread.start()
    coordinator_thread.start()

    # ── 7. Run TUI on the main thread ───────────────────────────────────
    tui = TUI(config=config, coordinator=coordinator, snapshot_fn=_load_snapshot)
    try:
        tui.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in TUI — shutting down")
    except Exception:
        logger.exception("Unhandled exception in TUI")
        raise
    finally:
        # Ensure background threads wind down
        stop_event.set()
        logger.info("Waiting for background threads to finish...")
        collector_thread.join(timeout=5.0)
        coordinator_thread.join(timeout=5.0)
        # Flush all log handlers
        for handler in logging.getLogger().handlers:
            handler.flush()
        print("MAID shutting down...")  # noqa: T201


if __name__ == "__main__":
    main()
