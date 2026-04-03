"""Terminal UI for PSST using rich.

States:
  IDLE         — waiting for hotkey
  RECORDING    — mic is active, show elapsed time
  TRANSCRIBING — Whisper is running
  PROCESSING   — LLM cleanup is running
  DONE         — text copied to clipboard
  ERROR        — something failed
  CANCELLED    — recording was cancelled by user

The UI never blocks; all display calls are safe to call from the main thread.
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum, auto
from typing import Deque, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

from psst import __version__


class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()
    PROCESSING = auto()
    DONE = auto()
    ERROR = auto()
    CANCELLED = auto()


# Color/label for each state
_STATE_STYLE: dict[State, tuple[str, str]] = {
    State.IDLE:         ("dim",          "Idle — hold hotkey to record"),
    State.RECORDING:    ("bold red",     "Recording..."),
    State.TRANSCRIBING: ("bold yellow",  "Transcribing..."),
    State.PROCESSING:   ("bold cyan",    "Cleaning up..."),
    State.DONE:         ("bold green",   "Copied to clipboard"),
    State.ERROR:        ("bold red",     "Error"),
    State.CANCELLED:    ("bold yellow",  "Recording Cancelled"),
}


class UI:
    def __init__(self, history_size: int = 10) -> None:
        self.console = Console()
        self._state = State.IDLE
        self._history: Deque[Tuple[str, str]] = deque(maxlen=history_size)
        self._recording_start: float = 0.0

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def print_banner(
        self,
        hotkey: str,
        model: str,
        device: str,
        cleanup: bool,
        cancel_hotkey: str = "escape",
    ) -> None:
        lines = [
            f"[bold cyan]PSST[/] v{__version__} — Push, Speak, Send Text",
            "",
            f"  Hotkey  : [bold]{hotkey}[/]",
            f"  Cancel  : [bold]{cancel_hotkey}[/]",
            f"  Model   : [bold]{model}[/]",
            f"  Device  : [bold]{device}[/]",
            f"  Cleanup : [bold]{'on' if cleanup else 'off'}[/]",
            "",
            "[dim]Hold the hotkey to record. Release to transcribe. Ctrl+C to quit.[/]",
        ]
        self.console.print(
            Panel(
                "\n".join(lines),
                title="[bold blue]PSST[/]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 2),
            )
        )

    # ------------------------------------------------------------------
    # Admin warning
    # ------------------------------------------------------------------

    def print_admin_warning(self) -> None:
        """Print a prominent warning about running without admin privileges."""
        self.console.print(
            "[bold yellow][!] Running without Admin privileges. "
            "Global hotkeys are disabled. "
            "This terminal window must be in focus to start recording.[/]"
        )

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def set_state(self, state: State, detail: str = "") -> None:
        self._state = state
        style, label = _STATE_STYLE[state]

        if state == State.RECORDING:
            self._recording_start = time.monotonic()
            self.console.print(f"[{style}][REC] {label}[/]")

        elif state == State.DONE:
            elapsed = ""
            if detail:
                elapsed = f"  ({len(detail)} chars)"
            self.console.print(f"[{style}][OK] {label}{elapsed}[/]")

        elif state == State.ERROR:
            msg = f" — {detail}" if detail else ""
            self.console.print(f"[{style}][!!] {label}{msg}[/]")

        elif state == State.CANCELLED:
            self.console.print(f"[{style}][CANCEL] {label}[/]")

        elif state == State.IDLE:
            pass  # Idle is silent after the banner

        else:
            self.console.print(f"[{style}]{label}[/]")

    def update_recording_time(self, elapsed: float) -> None:
        """Overwrite the current line with elapsed recording time."""
        # rich doesn't support true terminal carriage-return easily,
        # so we just let the Recording... message stand and skip noisy updates.
        pass

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def add_to_history(self, text: str, cleaned: bool = False) -> None:
        tag = "cleaned" if cleaned else "whisper"
        self._history.appendleft((text, tag))
        # Print a preview: first 120 chars of the transcription
        preview = text[:120].replace("\n", " ")
        if len(text) > 120:
            preview += "…"
        self.console.print(f'[dim]  "{preview}"[/]')

    def print_history(self) -> None:
        if not self._history:
            self.console.print("[dim]No history yet.[/]")
            return
        self.console.rule("[dim]Session history[/]")
        for i, (text, tag) in enumerate(self._history, 1):
            preview = text[:100].replace("\n", " ")
            if len(text) > 100:
                preview += "…"
            self.console.print(f"[dim]{i:2d}. [{tag}] {preview}[/]")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def print_error(self, msg: str) -> None:
        self.console.print(f"[bold red]Error:[/] {msg}")

    def print_info(self, msg: str) -> None:
        self.console.print(f"[dim]{msg}[/]")

    def print_quit(self) -> None:
        self.console.print("\n[dim]Goodbye.[/]")
