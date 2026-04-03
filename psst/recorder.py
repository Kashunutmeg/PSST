"""Audio recording via sounddevice InputStream callback mode.

Design:
- Callback appends chunks to a list; NEVER does heavy work.
- threading.Lock guards start/stop to prevent races from key-repeat.
- indata.copy() is mandatory in the callback — sounddevice reuses its buffers.
- start() returns immediately.
- stop() joins the stream and returns a numpy float32 array, or None if the
  recording was too short / empty.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import numpy as np
import sounddevice as sd


class Recorder:
    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = "float32"

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        min_duration: float = 0.3,
        max_duration: float = 60.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.min_duration = min_duration
        self.max_duration = max_duration

        self._lock = threading.Lock()
        self._chunks: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._start_time: float = 0.0
        self._recording = False

    # ------------------------------------------------------------------
    # Internal callback — called by PortAudio thread
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        # indata.copy() is mandatory — sounddevice reuses this buffer
        self._chunks.append(indata.copy())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Begin recording. Returns True if started, False if already recording."""
        with self._lock:
            if self._recording:
                return False
            self._chunks = []
            self._start_time = time.monotonic()
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.DTYPE,
                callback=self._callback,
            )
            self._stream.start()
            self._recording = True
            return True

    def stop(self) -> Optional[np.ndarray]:
        """Stop recording. Returns float32 numpy array, or None if too short/empty."""
        with self._lock:
            if not self._recording:
                return None

            elapsed = time.monotonic() - self._start_time
            stream = self._stream

            self._stream = None
            self._recording = False

        # Stop/close outside the lock to avoid potential deadlock with callback
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        if elapsed < self.min_duration:
            return None

        if not self._chunks:
            return None

        audio = np.concatenate(self._chunks, axis=0)

        # Flatten to 1-D if mono
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Trim to max_duration
        max_samples = int(self.max_duration * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        return audio

    def cancel(self) -> None:
        """Stop recording and discard all captured audio chunks."""
        with self._lock:
            if not self._recording:
                return
            stream = self._stream
            self._stream = None
            self._recording = False
            self._chunks = []

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    @property
    def is_recording(self) -> bool:
        return self._recording

    def elapsed(self) -> float:
        """Seconds since recording started (0 if not recording)."""
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time
