# MAID - Memory AI Daemon

MAID is a real-time memory monitoring and enforcement tool that runs in the terminal. It uses local heuristics for fast common-case decisions, and falls back to the Claude API (`claude-sonnet-4-6`) for complex or ambiguous situations.

## Installation

1. Clone or copy the project files to a local directory.
2. Make sure you are using Python 3.11+.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Setup

MAID uses Anthropic's Claude API for complex memory management decisions. You must set the `ANTHROPIC_API_KEY` environment variable before running the daemon if you want AI fallback to work.

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

*Note: If the key is not set, MAID will still run using local heuristics but AI features will be disabled.*

## Usage

Start the daemon by running `main.py`:

```bash
python3 main.py
```

### Example Run Commands

- Run in dry-run mode to see what would happen without making any changes:
  ```bash
  python3 main.py --dry-run
  ```

- Run aggressively by changing the RAM threshold and auto-confirming kills:
  ```bash
  python3 main.py --threshold 85 --auto
  ```

- Run purely locally without AI fallback:
  ```bash
  python3 main.py --no-ai
  ```

### Command-line Flags

| Flag | Description | Default |
|---|---|---|
| `--dry-run` | Log actions instead of executing them (no system modifications). | `False` |
| `--auto` | Auto-confirm dangerous actions like KillProcess without y/n prompt. | `False` |
| `--interval N` | Polling interval in seconds (how often to sample memory state). | `3.0` |
| `--threshold N` | RAM usage threshold percentage before taking heuristic action. | `90.0` |
| `--no-ai` | Disable AI fallback entirely. | `False` |
| `--log-dir PATH` | Directory to save `actions.log`. | `~/.local/share/maid` |

## Keybinds

While the TUI is running, you can use the following keys:
- `q`: Quit the daemon.
- `a`: Force an immediate AI analysis.
- `d`: Toggle dry-run mode on/off.
- `p`: Pause/resume enforcement.
- `?`: Show the help overlay.

## Project Structure

- `models.py` - Shared data classes and configurations.
- `collector.py` - Reads system memory states.
- `heuristics.py` - Fast local rule evaluation.
- `ai_client.py` - Claude API integration.
- `coordinator.py` - Orchestrates heuristics and AI fallback.
- `executor.py` - Executes system actions (kill, renice, drop_caches).
- `tui.py` - Terminal user interface built with `rich`.
- `main.py` - Entry point and thread management.
