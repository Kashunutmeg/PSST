"""Clipboard output for PSST.

Primary:  pyperclip (cross-platform, handles encoding correctly)
Fallback: ctypes Windows API (in case pyperclip fails / no xclip on Linux)

The user's dictation must NEVER be lost.  If both methods fail, we at
minimum print the text to stdout so the user can copy it manually.
"""

from __future__ import annotations

import sys
from typing import Optional


def _copy_ctypes(text: str) -> bool:
    """Windows-only clipboard write via ctypes. Returns True on success."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        CF_UNICODETEXT = 13
        GMEM_MOVEABLE = 0x0002

        user32 = ctypes.windll.user32      # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        encoded = text.encode("utf-16-le") + b"\x00\x00"

        if not user32.OpenClipboard(None):
            return False
        try:
            user32.EmptyClipboard()
            h_mem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
            if not h_mem:
                return False
            ptr = kernel32.GlobalLock(h_mem)
            if not ptr:
                kernel32.GlobalFree(h_mem)
                return False
            ctypes.memmove(ptr, encoded, len(encoded))
            kernel32.GlobalUnlock(h_mem)
            user32.SetClipboardData(CF_UNICODETEXT, h_mem)
            return True
        finally:
            user32.CloseClipboard()
    except Exception:
        return False


def copy_to_clipboard(text: str, fallback: bool = True) -> bool:
    """Copy text to clipboard. Returns True if any method succeeded."""
    # Primary: pyperclip
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        pass

    # Fallback: ctypes Windows API
    if fallback and sys.platform == "win32":
        if _copy_ctypes(text):
            return True

    # Last resort: print so user doesn't lose the text
    print(f"\n[PSST] Clipboard unavailable. Transcription:\n{text}\n")
    return False
