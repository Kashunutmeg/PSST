"""Audio cues for PSST using stdlib only (wave + struct) + winsound.

Generates short WAV tones programmatically and plays them asynchronously
so they never block the main thread or the recording pipeline.

Tones:
  start  — 440 Hz  (A4, "listening")
  stop   — 330 Hz  (E4, "done recording") — overridden by pssst.wav if present
  done   — 660 Hz  (E5, "copied to clipboard")
  error  — 220 Hz  (A3, "something went wrong")
  cancel — 180 Hz  (low short tone, "recording cancelled")
"""

from __future__ import annotations

import math
import os
import struct
import wave
from pathlib import Path
from typing import Dict

import winsound

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOUNDS_DIR = Path(__file__).parent.parent / "sounds"

TONE_SPEC: Dict[str, tuple[float, float]] = {
    # name -> (frequency_hz, duration_seconds)
    "start":  (440.0, 0.12),
    "stop":   (330.0, 0.10),
    "done":   (660.0, 0.14),
    "error":  (220.0, 0.20),
    "cancel": (180.0, 0.08),
}

SAMPLE_RATE = 44100
AMPLITUDE = 0.4  # 0..1, keep it gentle


# ---------------------------------------------------------------------------
# WAV generation
# ---------------------------------------------------------------------------

def _generate_wav(path: Path, freq: float, duration: float) -> None:
    """Write a sine-wave WAV file using only stdlib (wave + struct)."""
    n_samples = int(SAMPLE_RATE * duration)
    # Apply a simple envelope (fade-in 10 ms, fade-out 20 ms) to avoid clicks
    fade_in  = int(SAMPLE_RATE * 0.010)
    fade_out = int(SAMPLE_RATE * 0.020)

    samples: list[int] = []
    for i in range(n_samples):
        t = i / SAMPLE_RATE
        sample = math.sin(2.0 * math.pi * freq * t)

        # Envelope
        if i < fade_in:
            sample *= i / fade_in
        elif i >= n_samples - fade_out:
            sample *= (n_samples - i) / fade_out

        samples.append(int(sample * AMPLITUDE * 32767))

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


def _ensure_sounds() -> Dict[str, Path]:
    """Generate any missing WAV files and return {name: path} mapping."""
    paths: Dict[str, Path] = {}
    for name, (freq, dur) in TONE_SPEC.items():
        p = SOUNDS_DIR / f"{name}.wav"
        if not p.exists():
            _generate_wav(p, freq, dur)
        paths[name] = p
    return paths


# Cache paths after first generation
_sound_paths: Dict[str, Path] = {}


def _get_paths() -> Dict[str, Path]:
    global _sound_paths
    if not _sound_paths:
        _sound_paths = _ensure_sounds()
    return _sound_paths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def play(name: str) -> None:
    """Play a named audio cue asynchronously. Silently ignores errors."""
    try:
        paths = _get_paths()
        if name not in paths:
            return
        winsound.PlaySound(
            str(paths[name]),
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
        )
    except Exception:
        pass  # Audio cues are best-effort; never crash the main loop


def play_start() -> None:
    play("start")

def play_stop() -> None:
    """Play stop sound. Uses custom pssst.wav from CWD if present."""
    try:
        custom = Path.cwd() / "pssst.wav"
        if custom.exists():
            winsound.PlaySound(
                str(custom),
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
            )
            return
    except Exception:
        pass
    play("stop")

def play_done() -> None:
    play("done")

def play_error() -> None:
    play("error")

def play_cancel() -> None:
    play("cancel")
