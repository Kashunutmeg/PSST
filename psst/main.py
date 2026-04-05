"""PSST main event loop.

Threading model:
  Thread 1 (main)  : event loop, transcription, LLM, clipboard, UI
  Thread 2 (keyboard): only puts ("press",) / ("release",) / ("cancel",) on queue
  Thread 3 (PortAudio): only appends audio chunks via callback
  Thread 4 (tray)  : pystray daemon thread — only puts ("quit",) on queue

Invariant: keyboard and audio threads NEVER do heavy work.
"""

from __future__ import annotations

import logging
import queue
import sys
from typing import Optional

import numpy as np

from psst import __version__
from psst.config import get_config, Config, is_admin
from psst.recorder import Recorder
from psst.hotkey import HotkeyListener
from psst.output import copy_to_clipboard
from psst.tray import TrayIcon
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


def _active_instruction(cfg: Config) -> Optional[str]:
    """Return the instruction string for the active prompt profile, or None."""
    profile = cfg.prompts.get(cfg.active_prompt, {})
    if isinstance(profile, dict):
        instruction = profile.get("instruction")
        return instruction if instruction else None
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
    instruction = _active_instruction(cfg)
    try:
        return backend.clean(text, instruction=instruction) or text
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

def _ensure_ico() -> "Optional[str]":
    """Ensure assets/icon.ico exists (converting from .png if needed).

    Returns the .ico path as a string, or None if no icon is available.
    """
    from pathlib import Path
    project_dir = Path(__file__).resolve().parent.parent
    ico_path = project_dir / "assets" / "icon.ico"
    png_path = project_dir / "assets" / "icon.png"

    if ico_path.is_file():
        return str(ico_path)

    if png_path.is_file():
        try:
            from PIL import Image
            img = Image.open(str(png_path))
            sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            img.save(str(ico_path), format="ICO", sizes=sizes)
            return str(ico_path)
        except Exception:
            pass

    return None


def _set_console_icon() -> None:
    """Set the console window icon on Windows. Best-effort, silent on failure."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        icon_file = _ensure_ico()
        if not icon_file:
            return

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x0010
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1

        hwnd = kernel32.GetConsoleWindow()
        if not hwnd:
            return

        h_icon = user32.LoadImageW(0, icon_file, IMAGE_ICON, 0, 0, LR_LOADFROMFILE)
        if h_icon:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, h_icon)
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, h_icon)
    except Exception:
        pass


def run() -> None:
    cfg = get_config()
    _setup_logging(cfg)
    _set_console_icon()

    ui = UI(history_size=cfg.history_size)
    event_queue: queue.Queue = queue.Queue()

    # Admin check — show warning if not elevated
    admin_mode = is_admin()
    admin_status = "admin" if admin_mode else "no-admin"

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

    listener = HotkeyListener(
        combo=cfg.hotkey,
        event_queue=event_queue,
        cancel_key=cfg.cancel_hotkey,
    )

    # System tray
    tray = TrayIcon(
        event_queue=event_queue,
        initial_status=f"Idle ({admin_status})",
    )

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
        cancel_hotkey=cfg.cancel_hotkey,
    )

    if not admin_mode:
        ui.print_admin_warning()

    # Eager cleanup model init — download + load before the user starts dictating
    if cfg.cleanup_enabled:
        ui.print_info("Loading cleanup model...")
        backend = _get_cleanup(cfg)
        if backend is not None and hasattr(backend, '_load'):
            try:
                backend._load()
                # Detect actual GPU usage from the loaded model
                gpu_layers = getattr(backend, 'n_gpu_layers', 0)
                llm = getattr(backend, '_llm', None)
                if llm is not None:
                    # Check if llama-cpp-python was built with GPU offload support.
                    # 0.3.x exposes this as a function (llama_supports_gpu_offload);
                    # older versions exposed module-level constants.
                    try:
                        import llama_cpp
                        if hasattr(llama_cpp, 'llama_supports_gpu_offload'):
                            has_cuda = bool(llama_cpp.llama_supports_gpu_offload())
                        else:
                            has_cuda = getattr(llama_cpp, 'LLAMA_SUPPORTS_GPU_OFFLOAD', False)
                            if not has_cuda:
                                has_cuda = hasattr(llama_cpp.llama_cpp, 'GGML_USE_CUDA') or \
                                           hasattr(llama_cpp.llama_cpp, 'GGML_USE_CUBLAS')
                    except Exception:
                        has_cuda = False

                    if gpu_layers != 0 and not has_cuda:
                        ui.console.print(
                            "[bold yellow][!] n_gpu_layers=-1 but llama-cpp-python was "
                            "built WITHOUT CUDA support -- running on CPU.[/]\n"
                            "[bold yellow]    To fix: pip install llama-cpp-python --force-reinstall "
                            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121[/]"
                        )
                        accel = "CPU (CUDA not available)"
                    elif gpu_layers != 0:
                        accel = "GPU"
                    else:
                        accel = "CPU"
                else:
                    accel = "GPU" if gpu_layers != 0 else "CPU"
                ui.print_info(f"Cleanup model ready ({accel}).")
            except Exception as exc:
                ui.print_info(f"Cleanup model failed to load: {exc}")
        elif backend is None:
            ui.print_info("Cleanup backend unavailable — falling back to Whisper output.")

    ui.set_state(State.IDLE)

    tray.start()

    hook_ok = listener.start()
    if not hook_ok and admin_mode:
        # Hook failed even though we appear to be admin — warn anyway
        ui.print_info("Warning: keyboard hook failed to register.")

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
                    tray.update_status("Recording...")
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
                    tray.update_status(f"Idle ({admin_status})")
                    continue

                transcribing = True
                ui.set_state(State.TRANSCRIBING)
                tray.update_status("Transcribing...")

                try:
                    text = _transcribe(audio, cfg, ui)

                    if not text:
                        ui.set_state(State.ERROR, "Empty transcription")
                        tray.update_status(f"Idle ({admin_status})")
                        _cue("error")
                        continue

                    raw_text = text
                    text = _cleanup_text(text, cfg, ui)
                    actually_cleaned = (text != raw_text)

                    copied = copy_to_clipboard(text, fallback=cfg.clipboard_fallback)

                    if copied:
                        ui.set_state(State.DONE, text)
                        ui.add_to_history(text, cleaned=actually_cleaned)
                        tray.update_status(f"Idle ({admin_status})")
                        _cue("done")
                        logging.info("Transcribed: %s", text)
                    else:
                        ui.set_state(State.ERROR, "Clipboard write failed")
                        tray.update_status(f"Idle ({admin_status})")
                        _cue("error")

                finally:
                    transcribing = False
                    ui.set_state(State.IDLE)

            # ----------------------------------------------------------
            # CANCEL — discard recording
            # ----------------------------------------------------------
            elif kind == "cancel":
                if recorder.is_recording:
                    recorder.cancel()
                    _cue("cancel")
                    ui.set_state(State.CANCELLED)
                    tray.update_status(f"Idle ({admin_status})")
                    ui.set_state(State.IDLE)
                transcribing = False

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
        tray.stop()
        ui.print_quit()
