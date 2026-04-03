"""Configuration loading for PSST.

Search order:
  1. Path given via --config CLI flag
  2. ./config.toml  (CWD)
  3. %APPDATA%/psst/config.toml  (Windows user config)
  4. ~/.psst/config.toml
  5. Built-in defaults
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError as exc:
        raise ImportError(
            "tomli is required on Python < 3.11. Install it with: pip install tomli"
        ) from exc

from psst import __version__


# ---------------------------------------------------------------------------
# Admin detection
# ---------------------------------------------------------------------------

def is_admin() -> bool:
    """Return True if running with elevated (admin) privileges on Windows."""
    if sys.platform != "win32":
        return True  # Non-Windows: concept doesn't apply
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Default prompt profiles
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS: Dict[str, Dict[str, str]] = {
    "default": {
        "name": "Default Dictation",
        "instruction": (
            "Clean up this dictated text. Fix grammar, punctuation, and remove "
            "filler words. Do not change the meaning. Return only the cleaned text."
        ),
    },
    "code": {
        "name": "Code Docstrings",
        "instruction": (
            "This is dictated documentation for code. Format it as a proper "
            "docstring. Fix grammar and make it technical and concise."
        ),
    },
    "actions": {
        "name": "Action Items",
        "instruction": (
            "Extract action items from this dictated text. Format as a numbered "
            "list of clear, actionable tasks."
        ),
    },
}


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Hotkey
    hotkey: str = "ctrl+shift+space"
    cancel_hotkey: str = "escape"

    # Audio recording
    sample_rate: int = 16000
    channels: int = 1
    min_recording_seconds: float = 0.3
    max_recording_seconds: float = 300.0

    # Whisper transcription
    model: str = "base"
    device: str = "auto"          # "auto" | "cuda" | "cpu"
    compute_type: str = "auto"    # "auto" | "float16" | "int8" | "float32"
    language: Optional[str] = None  # None = auto-detect
    vad_filter: bool = True

    # LLM cleanup
    cleanup_enabled: bool = True
    cleanup_backend: str = "llama_cpp"   # "llama_cpp" | "ollama"
    cleanup_model: str = "qwen3.5:4b"
    cleanup_timeout: int = 10
    ollama_url: str = "http://localhost:11434"
    llama_cpp_model_path: str = ""

    # Prompt profiles
    active_prompt: str = "default"
    prompts: Dict[str, Any] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_PROMPTS.items()}
    )

    # UI
    history_size: int = 10
    audio_cues: bool = True
    log_to_file: bool = False
    log_file: str = "psst.log"

    # Output
    clipboard_fallback: bool = True  # use ctypes fallback if pyperclip fails


# ---------------------------------------------------------------------------
# TOML loading helpers
# ---------------------------------------------------------------------------

def _find_config_file(override: Optional[str] = None) -> Optional[Path]:
    """Return the first config file that exists, or None."""
    candidates: List[Path] = []

    if override:
        candidates.append(Path(override))
    else:
        # 1. CWD
        candidates.append(Path.cwd() / "config.toml")
        # 2. %APPDATA%/psst/config.toml (Windows)
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "psst" / "config.toml")
        # 3. ~/.psst/config.toml
        candidates.append(Path.home() / ".psst" / "config.toml")

    for path in candidates:
        if path.is_file():
            return path
    return None


def _apply_section(cfg: Config, section: Dict[str, Any], prefix: str = "") -> None:
    """Recursively apply a TOML section dict onto a Config dataclass."""
    valid_fields = {f.name for f in fields(cfg)}
    for key, value in section.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            # Nested table — flatten with underscore prefix
            _apply_section(cfg, value, prefix=f"{full_key}_")
        elif full_key in valid_fields:
            current = getattr(cfg, full_key)
            # Basic type coercion
            if current is not None and not isinstance(value, type(current)):
                try:
                    value = type(current)(value)
                except (ValueError, TypeError):
                    pass
            setattr(cfg, full_key, value)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load config from TOML file merged with defaults."""
    cfg = Config()
    found = _find_config_file(config_path)
    if found:
        with open(found, "rb") as fh:
            data = tomllib.load(fh)
        # Extract [prompts.*] before _apply_section flattens nested dicts
        prompts_data = data.pop("prompts", None)
        _apply_section(cfg, data)
        if isinstance(prompts_data, dict):
            # Merge user-defined profiles with defaults
            cfg.prompts.update(prompts_data)
    return cfg


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="psst",
        description="PSST — Push, Speak, Send Text: local speech-to-text dictation",
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="Path to config.toml (overrides search order)",
    )
    parser.add_argument(
        "--model", metavar="NAME",
        help="Whisper model name (tiny/base/small/medium/large-v3)",
    )
    parser.add_argument(
        "--hotkey", metavar="COMBO",
        help='Hotkey combo, e.g. "ctrl+shift+space"',
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"],
        help="Compute device for Whisper",
    )
    parser.add_argument(
        "--cleanup", action="store_true", default=None,
        help="Enable LLM cleanup of transcription",
    )
    parser.add_argument(
        "--no-audio-cues", action="store_true", default=False,
        help="Disable audio feedback sounds",
    )
    parser.add_argument(
        "--language", metavar="LANG",
        help='Force language code, e.g. "en" (default: auto-detect)',
    )
    parser.add_argument(
        "--log", action="store_true", default=False,
        help="Write session log to psst.log",
    )
    parser.add_argument(
        "--prompt", metavar="PROFILE",
        help='Active prompt profile name, e.g. "default", "code", "actions"',
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser.parse_args()


def get_config() -> Config:
    """Parse CLI args, load config file, merge overrides. Returns final Config."""
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.model:
        cfg.model = args.model
    if args.hotkey:
        cfg.hotkey = args.hotkey
    if args.device:
        cfg.device = args.device
    if args.cleanup is True:
        cfg.cleanup_enabled = True
    if args.no_audio_cues:
        cfg.audio_cues = False
    if args.language:
        cfg.language = args.language
    if args.log:
        cfg.log_to_file = True
    if args.prompt:
        cfg.active_prompt = args.prompt

    return cfg
