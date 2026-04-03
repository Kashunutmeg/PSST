"""System tray icon for PSST using pystray + Pillow.

Architecture:
  - TrayIcon runs pystray's blocking icon.run() in a daemon thread.
  - "Quit" menu item puts ("quit",) on the shared event queue so the main
    loop exits cleanly — same path as Ctrl+C.
  - update_status() refreshes the menu label in real-time.
  - If pystray or Pillow are not installed, start() is a no-op (graceful
    degradation — the app works fine without a tray icon).
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

try:
    import pystray
    from PIL import Image, ImageDraw, ImageFont
    _TRAY_AVAILABLE = True
except ImportError:
    _TRAY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Icon generation
# ---------------------------------------------------------------------------

def _make_icon_image() -> "Image.Image":
    """Generate a simple 64×64 PSST tray icon using Pillow."""
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dark blue filled circle
    margin = 2
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=(30, 80, 160, 255),
    )

    # White "P" — try to load a large font, fall back gracefully
    try:
        # load_default(size=N) requires Pillow >= 10.1.0
        font = ImageFont.load_default(size=36)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((16, 10), "P", fill=(255, 255, 255, 255), font=font)

    return img


# ---------------------------------------------------------------------------
# TrayIcon class
# ---------------------------------------------------------------------------

class TrayIcon:
    def __init__(
        self,
        event_queue: queue.Queue,
        initial_status: str = "Idle",
    ) -> None:
        self._queue = event_queue
        self._status = initial_status
        self._icon: Optional["pystray.Icon"] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_menu(self) -> "pystray.Menu":
        return pystray.Menu(
            pystray.MenuItem(
                f"Status: {self._status}",
                None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit PSST", self._on_quit),
        )

    def _on_quit(self, icon: "pystray.Icon", item: object) -> None:
        self._queue.put(("quit",))
        try:
            icon.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the tray icon in a background daemon thread."""
        if not _TRAY_AVAILABLE:
            return

        try:
            icon_image = _make_icon_image()
            self._icon = pystray.Icon(
                name="PSST",
                icon=icon_image,
                title="PSST — Push, Speak, Send Text",
                menu=self._build_menu(),
            )
            self._thread = threading.Thread(
                target=self._icon.run,
                daemon=True,
                name="psst-tray",
            )
            self._thread.start()
        except Exception:
            # Tray is best-effort — never crash the main app
            self._icon = None

    def update_status(self, status: str) -> None:
        """Update the status label shown in the tray context menu."""
        self._status = status
        if self._icon is not None:
            try:
                self._icon.menu = self._build_menu()
                self._icon.update_menu()
            except Exception:
                pass

    def stop(self) -> None:
        """Stop the tray icon."""
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass
            self._icon = None
