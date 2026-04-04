"""LLM-based transcription cleanup for PSST.

Backends:
  OllamaBackend    — HTTP to localhost:11434/api/generate (uses urllib, not requests)
  LlamaCppBackend  — llama-cpp-python bindings (optional, not in default requirements)

Contract:
  - clean(text, instruction=None) always returns a string (raw text on any failure)
  - If instruction is None, the built-in SYSTEM_PROMPT is used
  - All failures are logged and swallowed — never lose the user's dictation

System prompt:
  Clean up the transcription: fix punctuation, remove filler words (um, uh,
  like), capitalise sentences. Do NOT change the meaning or add content.
  Return only the cleaned text with no preamble.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Optional

from rich.console import Console

from psst.config import Config


SYSTEM_PROMPT = (
    "You are a transcription cleanup assistant. "
    "The user will provide raw speech-to-text output. "
    "Your job: fix punctuation, capitalise sentences, remove filler words "
    "(um, uh, like, you know, sort of), and fix obvious repetitions. "
    "Do NOT change the meaning, add information, or summarise. "
    "Return ONLY the cleaned text — no preamble, no explanation."
)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CleanupBackend(ABC):
    @abstractmethod
    def clean(self, text: str, instruction: Optional[str] = None) -> str:
        """Return cleaned text. Must never raise — return raw text on failure.

        Args:
            text: Raw transcription text to clean.
            instruction: Optional system prompt override. If None, the default
                SYSTEM_PROMPT is used.
        """
        ...


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaBackend(CleanupBackend):
    def __init__(self, base_url: str, model: str, timeout: int = 10) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._generate_url = f"{self.base_url}/api/generate"

    def clean(self, text: str, instruction: Optional[str] = None) -> str:
        system = instruction if instruction is not None else SYSTEM_PROMPT
        payload = {
            "model": self.model,
            "system": system,
            "prompt": text,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._generate_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                cleaned = body.get("response", "").strip()
                return cleaned if cleaned else text
        except urllib.error.URLError as exc:
            logging.warning("Ollama unavailable: %s — returning raw text", exc)
            return text
        except Exception as exc:
            logging.warning("Ollama cleanup failed: %s — returning raw text", exc)
            return text


# ---------------------------------------------------------------------------
# llama-cpp-python backend
# ---------------------------------------------------------------------------

class LlamaCppBackend(CleanupBackend):
    def __init__(self, model_path: str, chat_format: str = "chatml", timeout: int = 60) -> None:
        self.model_path = model_path
        self.chat_format = chat_format
        self.timeout = timeout
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return self._llm
        try:
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "See https://github.com/abetlen/llama-cpp-python for GPU install."
            ) from exc
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            chat_format=self.chat_format,
            verbose=False,
        )
        return self._llm

    def clean(self, text: str, instruction: Optional[str] = None) -> str:
        system = instruction if instruction is not None else SYSTEM_PROMPT
        try:
            llm = self._load()
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                max_tokens=512,
            )
            cleaned = output["choices"][0]["message"]["content"].strip()
            return cleaned if cleaned else text
        except Exception as exc:
            logging.warning("LlamaCpp cleanup failed: %s — returning raw text", exc)
            return text


# ---------------------------------------------------------------------------
# Model auto-download
# ---------------------------------------------------------------------------

_console = Console()


def _ensure_model(cfg: Config) -> Optional[str]:
    """Return a path to the GGUF model file, downloading from HF if needed.

    Priority:
      1. cfg.llama_cpp_model_path (local override) — returned as-is
      2. cfg.llama_cpp_repo_id / cfg.llama_cpp_filename — downloaded via
         huggingface_hub and cached locally

    Returns None on any failure (caller should fall back gracefully).
    """
    # Local file override — skip download entirely
    if cfg.llama_cpp_model_path:
        return cfg.llama_cpp_model_path

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
    except ImportError:
        logging.warning(
            "huggingface_hub is not installed — cannot auto-download LLM model. "
            "Install it with: pip install huggingface-hub"
        )
        _console.print(
            "[bold yellow][PSST][/] huggingface-hub is not installed — "
            "cannot auto-download cleanup model. "
            "Install it with: pip install huggingface-hub"
        )
        return None

    try:
        _console.print(
            f"[bold cyan][PSST][/] Downloading cleanup model "
            f"[bold]{cfg.llama_cpp_filename}[/] from "
            f"[bold]{cfg.llama_cpp_repo_id}[/] (~2.7 GB, first run only)..."
        )
        path = hf_hub_download(
            repo_id=cfg.llama_cpp_repo_id,
            filename=cfg.llama_cpp_filename,
        )
        _console.print(
            f"[bold green][PSST][/] Model cached at: [dim]{path}[/]"
        )
        return path
    except Exception as exc:
        logging.warning("Failed to download LLM model: %s", exc)
        _console.print(
            f"[bold yellow][PSST][/] Could not download cleanup model: {exc}"
        )
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_backend(cfg: Config) -> Optional[CleanupBackend]:
    """Instantiate and return the configured cleanup backend, or None."""
    backend_name = cfg.cleanup_backend.lower()

    if backend_name == "ollama":
        return OllamaBackend(
            base_url=cfg.ollama_url,
            model=cfg.cleanup_model,
            timeout=cfg.cleanup_timeout,
        )

    if backend_name in ("llama_cpp", "llama-cpp", "llamacpp"):
        # Don't download a 2.7 GB model if the runtime isn't even installed
        if importlib.util.find_spec("llama_cpp") is None:
            logging.warning(
                "llama-cpp-python is not installed — cleanup disabled. "
                "See https://github.com/abetlen/llama-cpp-python for install instructions."
            )
            _console.print(
                "[bold yellow][PSST][/] llama-cpp-python is not installed — "
                "LLM cleanup disabled. Install it with:\n"
                "  pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )
            return None
        model_path = _ensure_model(cfg)
        if model_path is None:
            logging.warning("No LLM model available — cleanup disabled.")
            return None
        return LlamaCppBackend(
            model_path=model_path,
            chat_format=cfg.chat_format,
            timeout=cfg.cleanup_timeout,
        )

    logging.error("Unknown cleanup backend: %s", backend_name)
    return None
