"""PSST main event loop.

Threading model:
  Thread 1 (main)  : event loop, transcription, LLM, clipboard, UI
  Thread 2 (keyboard): only puts ("press",) / ("release",) on queue
  Thread 3 (PortAudio): only appends audio chunks via callback

Invariant: keyboard and audio threads NEVER do heavy work.
"""

from __future__ import annotations

import logging
import queue
import sys
import time
from typing import Optional

import numpy as np

from psst import __version__
from psst.config import get_config, Config
from psst.recorder import Recorder
from psst.hotkey import HotkeyListener
from psst.output import copy_to_clipboard
from psst.ui import UI, State

# Optional modules — imported lazily so Phase 1 works without them
_transcriber = None
_cleanup = None


def _get_transcriber(cfg: Config):
    """Lazy-import transcriber; returns None if not available."""
    global _transcriber
    if _transcriber is not None:
        return _transcriber
    try:
        from psst.transcriber import Transcriber
        _transcriber = Transcriber(
            model_name=cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
            language=cfg.language,
            vad_filter=cfg.vad_filter,
        )
        return _transcriber
    except ImportError:
        return None


def _get_cleanup(cfg: Config):
    """Lazy-import cleanup backend; returns None if not available or disabled."""
    global _cleanup
    if not cfg.cleanup_enabled:
        return None
    if _cleanup is not None:
        return _cleanup
    try:
        from psst.cleanup import get_backend
        _cleanup = get_backend(cfg)
        return _cleanup
    except ImportError:
        return None


def _transcribe(audio: np.ndarray, cfg: Config, ui: UI) -> Optional[str]:
    """Run transcription. Returns text or None on failure."""
    transcriber = _get_transcriber(cfg)
    if transcriber is None:
        # Phase 1 placeholder: return a stub so the clipboard/UI flow works
        ui.print_info("[Phase 1 placeholder] faster-whisper not installed yet.")
        return "[transcription placeholder — install faster-whisper]"
    try:
        return transcriber.transcribe(audio)
    except Exception as exc:
        ui.print_error(f"Transcription failed: {exc}")
        return None


def _cleanup_text(text: str, cfg: Config, ui: UI) -> str:
    """Run optional LLM cleanup. Always returns text (falls back to raw on error)."""
    if not cfg.cleanup_enabled:
        return text
    backend = _get_cleanup(cfg)
    if backend is None:
        return text
    ui.set_state(State.PROCESSING)
    try:
        return backend.clean(text) or text
    except Exception as exc:
        ui.print_error(f"LLM cleanup failed: {exc}")
        return text


def _setup_logging(cfg: Config) -> None:
    if cfg.log_to_file:
        logging.basicConfig(
            filename=cfg.log_file,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )
    else:
        logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run() -> None:
    cfg = get_config()
    _setup_logging(cfg)

    ui = UI(history_size=cfg.history_size)
    event_queue: queue.Queue = queue.Queue()

    # Audio cues (Windows only, best-effort) — import once, close over reference
    _audio_cues = None
    if cfg.audio_cues and sys.platform == "win32":
        try:
            from psst import audio_cues as _audio_cues
        except Exception:
            pass

    def _cue(name: str) -> None:
        if _audio_cues is not None:
            try:
                _audio_cues.play(name)
            except Exception:
                pass

    recorder = Recorder(
        sample_rate=cfg.sample_rate,
        channels=cfg.channels,
        min_duration=cfg.min_recording_seconds,
        max_duration=cfg.max_recording_seconds,
    )

    listener = HotkeyListener(combo=cfg.hotkey, event_queue=event_queue)

    # Determine effective device for banner
    device_label = cfg.device
    if device_label == "auto":
        try:
            import ctranslate2
            supported = ctranslate2.get_supported_compute_types("cuda")
            device_label = "cuda (auto)" if supported else "cpu (auto)"
        except Exception:
            device_label = "cpu (auto)"

    ui.print_banner(
        hotkey=cfg.hotkey,
        model=cfg.model,
        device=device_label,
        cleanup=cfg.cleanup_enabled,
    )
    ui.set_state(State.IDLE)

    # Lazy-load transcriber in background so startup is instant
    # (model download / GPU init happens on first transcription)

    listener.start()

    transcribing = False  # True while we're processing audio (ignore hotkey)

    try:
        while True:
            try:
                event = event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            kind = event[0]

            # ----------------------------------------------------------
            # PRESS — start recording
            # ----------------------------------------------------------
            if kind == "press":
                if transcribing:
                    # Ignore hotkey while busy — never lose previous result
                    continue
                started = recorder.start()
                if started:
                    ui.set_state(State.RECORDING)
                    _cue("start")

            # ----------------------------------------------------------
            # RELEASE — stop recording, transcribe, copy
            # ----------------------------------------------------------
            elif kind == "release":
                audio = recorder.stop()
                _cue("stop")

                if audio is None:
                    ui.print_info("Recording too short — ignored.")
                    ui.set_state(State.IDLE)
                    continue

                transcribing = True
                ui.set_state(State.TRANSCRIBING)

                try:
                    text = _transcribe(audio, cfg, ui)

                    if not text:
                        ui.set_state(State.ERROR, "Empty transcription")
                        _cue("error")
                        continue

                    text = _cleanup_text(text, cfg, ui)

                    copied = copy_to_clipboard(text, fallback=cfg.clipboard_fallback)

                    if copied:
                        ui.set_state(State.DONE, text)
                        ui.add_to_history(text, cleaned=cfg.cleanup_enabled)
                        _cue("done")
                        logging.info("Transcribed: %s", text)
                    else:
                        ui.set_state(State.ERROR, "Clipboard write failed")
                        _cue("error")

                finally:
                    transcribing = False
                    ui.set_state(State.IDLE)

            # ----------------------------------------------------------
            # QUIT
            # ----------------------------------------------------------
            elif kind == "quit":
                break

    except KeyboardInterrupt:
        pass

    finally:
        listener.stop()
        if recorder.is_recording:
            recorder.stop()
        ui.print_quit()
