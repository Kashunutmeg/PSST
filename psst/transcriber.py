"""Whisper transcription via faster-whisper + CTranslate2.

CUDA detection uses ctranslate2.get_supported_compute_types(), NOT torch.
This avoids pulling in the 2 GB PyTorch dependency.

Compute type selection:
  CUDA available  → float16
  CPU only        → int8

Model is auto-downloaded to HuggingFace cache on first use.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np


def _detect_device_and_compute(
    device_pref: str, compute_pref: str
) -> tuple[str, str]:
    """Return (device, compute_type) based on preferences and what's available."""
    import ctranslate2

    if device_pref == "cuda" or device_pref == "auto":
        try:
            cuda_types = ctranslate2.get_supported_compute_types("cuda")
            has_cuda = bool(cuda_types)
        except Exception:
            has_cuda = False
    else:
        has_cuda = False

    if device_pref == "cuda" and not has_cuda:
        logging.warning("CUDA requested but not available — falling back to CPU")

    device = "cuda" if has_cuda else "cpu"

    if compute_pref != "auto":
        return device, compute_pref

    # Auto-select compute type
    if device == "cuda":
        # float16 is fastest on modern NVIDIA GPUs
        return "cuda", "float16"
    else:
        return "cpu", "int8"


class Transcriber:
    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        language: Optional[str] = None,
        vad_filter: bool = True,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.vad_filter = vad_filter
        self._model = None

        self.device, self.compute_type = _detect_device_and_compute(device, compute_type)
        logging.info(
            "Transcriber: model=%s device=%s compute=%s",
            model_name, self.device, self.compute_type,
        )

    def _load_model(self):
        """Lazy-load the faster-whisper model (triggers download on first run)."""
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        return self._model

    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe a float32 numpy array. Returns text or None."""
        model = self._load_model()

        # faster-whisper expects float32 1-D array at 16 kHz
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim != 1:
            audio = audio[:, 0]

        # Normalise if needed (sounddevice returns -1..1 float32 already)
        peak = np.abs(audio).max()
        if peak > 1.0:
            audio = audio / peak

        segments, info = model.transcribe(
            audio,
            language=self.language,
            vad_filter=self.vad_filter,
        )

        logging.info(
            "Detected language: %s (%.2f confidence)",
            info.language, info.language_probability,
        )

        # Consume the generator and join segments
        parts = [seg.text for seg in segments]
        text = " ".join(parts).strip()
        return text if text else None
