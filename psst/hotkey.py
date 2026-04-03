"""Global hotkey handling for PSST using the `keyboard` library.

Design:
- The keyboard listener thread ONLY puts events on the queue.
- Lock + pressed flag prevents spurious key-repeat events.
- Supports hold-to-record mode: press = start, release = stop.
- Combo hotkeys (e.g. ctrl+shift+space) are handled via keyboard.hook
  with manual state tracking, because keyboard.on_press_key() doesn't
  support combos reliably on all Windows configurations.
- Non-admin fallback: keyboard.hook() is wrapped in try/except; if it
  raises (PermissionError, OSError, ImportError) the app continues with
  a warning — keyboard may still work for the focused window.

Events placed on queue:
  ("press",)   — hotkey pressed (start recording)
  ("release",) — hotkey released (stop recording)
  ("cancel",)  — cancel key pressed (discard recording)
  ("quit",)    — user pressed Ctrl+C (handled separately in main loop)
"""

from __future__ import annotations

import queue
import threading
from typing import Optional, Set

import keyboard


class HotkeyListener:
    def __init__(
        self,
        combo: str,
        event_queue: queue.Queue,
        cancel_key: Optional[str] = None,
    ) -> None:
        self.combo = combo.lower().strip()
        self.event_queue = event_queue

        self._lock = threading.Lock()
        self._pressed = False
        self._cancel_pressed = False
        self._hook = None

        # Parse the combo into modifier keys + trigger key
        parts = [p.strip() for p in self.combo.split("+")]
        self._modifiers: Set[str] = set(parts[:-1])
        self._trigger: str = parts[-1]

        # Optional cancel key (single key, no combo)
        self._cancel_key: Optional[str] = cancel_key.lower().strip() if cancel_key else None

        # Normalize modifier names keyboard library may use
        self._modifier_map = {
            "ctrl":  {"ctrl", "left ctrl", "right ctrl"},
            "shift": {"shift", "left shift", "right shift"},
            "alt":   {"alt", "left alt", "right alt"},
            "win":   {"windows", "left windows", "right windows"},
        }

    def _modifiers_held(self) -> bool:
        """Return True if all required modifier keys are currently held."""
        for mod in self._modifiers:
            canonical = self._modifier_map.get(mod, {mod})
            if not any(keyboard.is_pressed(k) for k in canonical):
                return False
        return True

    def _on_key_event(self, event: keyboard.KeyboardEvent) -> None:
        """Keyboard hook callback — runs in keyboard listener thread.
        Must be minimal: only queue.put() allowed here.
        """
        key_name = event.name.lower() if event.name else ""

        # ------------------------------------------------------------------
        # Cancel key (checked first — single key, no modifiers required)
        # ------------------------------------------------------------------
        if self._cancel_key and key_name == self._cancel_key:
            if event.event_type == keyboard.KEY_DOWN:
                with self._lock:
                    if self._cancel_pressed:
                        return  # key-repeat
                    self._cancel_pressed = True
                self.event_queue.put(("cancel",))
            elif event.event_type == keyboard.KEY_UP:
                with self._lock:
                    self._cancel_pressed = False
            return

        # ------------------------------------------------------------------
        # Main recording hotkey
        # ------------------------------------------------------------------
        if key_name != self._trigger:
            return

        if event.event_type == keyboard.KEY_DOWN:
            if not self._modifiers_held():
                return
            with self._lock:
                if self._pressed:
                    return  # Key repeat — ignore
                self._pressed = True
            self.event_queue.put(("press",))

        elif event.event_type == keyboard.KEY_UP:
            with self._lock:
                if not self._pressed:
                    return
                self._pressed = False
            self.event_queue.put(("release",))

    def start(self) -> bool:
        """Register the keyboard hook. Returns True on success, False on
        permission error (non-admin fallback — caller should warn the user)."""
        try:
            self._hook = keyboard.hook(self._on_key_event)
            return True
        except (PermissionError, OSError, ImportError):
            return False

    def stop(self) -> None:
        """Unregister the keyboard hook."""
        if self._hook is not None:
            try:
                keyboard.unhook(self._hook)
            except Exception:
                pass
            self._hook = None
        with self._lock:
            self._pressed = False
            self._cancel_pressed = False
