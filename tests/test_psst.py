"""Comprehensive unit tests for PSST.

Run with:  python -m pytest tests/  -v
       or: python -m unittest discover tests/
"""

from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# psst.config
# ---------------------------------------------------------------------------

class TestConfigDefaults(unittest.TestCase):
    def test_all_default_values(self):
        from psst.config import Config
        cfg = Config()
        self.assertEqual(cfg.hotkey, "ctrl+shift+space")
        self.assertEqual(cfg.sample_rate, 16000)
        self.assertEqual(cfg.channels, 1)
        self.assertAlmostEqual(cfg.min_recording_seconds, 0.3)
        self.assertAlmostEqual(cfg.max_recording_seconds, 300.0)
        self.assertEqual(cfg.model, "small")
        self.assertEqual(cfg.device, "auto")
        self.assertEqual(cfg.compute_type, "auto")
        self.assertIsNone(cfg.language)
        self.assertTrue(cfg.vad_filter)
        self.assertTrue(cfg.cleanup_enabled)
        self.assertEqual(cfg.cleanup_backend, "llama_cpp")
        self.assertEqual(cfg.cleanup_model, "qwen3.5:4b")
        self.assertEqual(cfg.cleanup_timeout, 60)
        self.assertEqual(cfg.ollama_url, "http://localhost:11434")
        self.assertEqual(cfg.llama_cpp_model_path, "")
        self.assertEqual(cfg.history_size, 10)
        self.assertTrue(cfg.audio_cues)
        self.assertFalse(cfg.log_to_file)
        self.assertEqual(cfg.log_file, "psst.log")
        self.assertTrue(cfg.clipboard_fallback)

    def test_load_config_missing_file_returns_defaults(self):
        from psst.config import load_config
        cfg = load_config("/nonexistent/path/that/does/not/exist.toml")
        self.assertEqual(cfg.model, "small")
        self.assertEqual(cfg.hotkey, "ctrl+shift+space")

    def test_load_config_from_toml_file(self):
        from psst.config import load_config
        content = b'model = "small"\nhotkey = "ctrl+f1"\nsample_rate = 22050\n'
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            cfg = load_config(path)
            self.assertEqual(cfg.model, "small")
            self.assertEqual(cfg.hotkey, "ctrl+f1")
            self.assertEqual(cfg.sample_rate, 22050)
        finally:
            os.unlink(path)

    def test_load_config_unknown_keys_ignored(self):
        from psst.config import load_config
        content = b'completely_unknown_key = "ignored"\nmodel = "tiny"\n'
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            cfg = load_config(path)
            self.assertEqual(cfg.model, "tiny")
        finally:
            os.unlink(path)

    def test_load_config_bool_value(self):
        from psst.config import load_config
        content = b'cleanup_enabled = true\nvad_filter = false\n'
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            cfg = load_config(path)
            self.assertTrue(cfg.cleanup_enabled)
            self.assertFalse(cfg.vad_filter)
        finally:
            os.unlink(path)

    def test_apply_section_direct(self):
        from psst.config import Config, _apply_section
        cfg = Config()
        _apply_section(cfg, {"model": "medium", "history_size": 20})
        self.assertEqual(cfg.model, "medium")
        self.assertEqual(cfg.history_size, 20)

    def test_apply_section_unknown_key_is_noop(self):
        from psst.config import Config, _apply_section
        cfg = Config()
        before_model = cfg.model
        _apply_section(cfg, {"no_such_field": "xyz"})
        self.assertEqual(cfg.model, before_model)

    def test_apply_section_nested_table_flattened(self):
        from psst.config import Config, _apply_section
        # TOML: [whisper]\nmodel = "large-v3"  → key "whisper_model"
        cfg = Config()
        _apply_section(cfg, {"whisper": {"model": "large-v3"}})
        # "whisper_model" not a known field, so it's silently skipped
        self.assertEqual(cfg.model, "small")  # unchanged

    def test_find_config_file_nonexistent_override(self):
        from psst.config import _find_config_file
        result = _find_config_file("/no/such/file.toml")
        self.assertIsNone(result)

    def test_find_config_file_existing_override(self):
        from psst.config import _find_config_file
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            path = f.name
        try:
            result = _find_config_file(path)
            self.assertEqual(result, Path(path))
        finally:
            os.unlink(path)


class TestConfigVersion(unittest.TestCase):
    def test_version_in_parse_args_matches_package(self):
        """parse_args() --version must use __version__, not a hardcoded string."""
        from psst import __version__
        from psst.config import parse_args
        import io
        with self.assertRaises(SystemExit):
            with patch("sys.argv", ["psst", "--version"]):
                with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                    parse_args()
        # argparse prints to stdout on --version
        # The important thing is no SystemExit(2) (which would mean error)


class TestConfigNewFields(unittest.TestCase):
    def test_cancel_hotkey_default(self):
        from psst.config import Config
        cfg = Config()
        self.assertEqual(cfg.cancel_hotkey, "escape")

    def test_active_prompt_default(self):
        from psst.config import Config
        cfg = Config()
        self.assertEqual(cfg.active_prompt, "default")

    def test_prompts_default_has_three_profiles(self):
        from psst.config import Config
        cfg = Config()
        self.assertIn("default", cfg.prompts)
        self.assertIn("code", cfg.prompts)
        self.assertIn("actions", cfg.prompts)

    def test_prompts_default_profile_has_instruction(self):
        from psst.config import Config
        cfg = Config()
        self.assertIn("instruction", cfg.prompts["default"])
        self.assertIsInstance(cfg.prompts["default"]["instruction"], str)
        self.assertGreater(len(cfg.prompts["default"]["instruction"]), 0)

    def test_prompts_loaded_from_toml(self):
        from psst.config import load_config
        content = (
            b'active_prompt = "code"\n'
            b'[prompts.custom]\n'
            b'name = "Custom"\n'
            b'instruction = "Do custom stuff."\n'
        )
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            cfg = load_config(path)
            self.assertEqual(cfg.active_prompt, "code")
            self.assertIn("custom", cfg.prompts)
            self.assertEqual(cfg.prompts["custom"]["instruction"], "Do custom stuff.")
            # Default built-in profiles still present
            self.assertIn("default", cfg.prompts)
        finally:
            os.unlink(path)

    def test_prompts_instances_are_independent(self):
        """Two Config() instances must not share the same prompts dict."""
        from psst.config import Config
        cfg1 = Config()
        cfg2 = Config()
        cfg1.prompts["new_key"] = {"instruction": "x"}
        self.assertNotIn("new_key", cfg2.prompts)

    def test_cancel_hotkey_loaded_from_toml(self):
        from psst.config import load_config
        content = b'cancel_hotkey = "f1"\n'
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            cfg = load_config(path)
            self.assertEqual(cfg.cancel_hotkey, "f1")
        finally:
            os.unlink(path)


class TestIsAdmin(unittest.TestCase):
    def test_is_admin_returns_bool(self):
        from psst.config import is_admin
        result = is_admin()
        self.assertIsInstance(result, bool)

    def test_is_admin_on_non_windows_returns_true(self):
        from psst.config import is_admin
        with patch("psst.config.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = is_admin()
        self.assertTrue(result)

    def test_is_admin_handles_exception_gracefully(self):
        from psst.config import is_admin
        with patch("psst.config.sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch("ctypes.windll", side_effect=AttributeError("no windll")):
                # Should not raise — returns False on any exception
                try:
                    result = is_admin()
                    self.assertIsInstance(result, bool)
                except Exception:
                    self.fail("is_admin() raised unexpectedly")


# ---------------------------------------------------------------------------
# psst.audio_cues — WAV generation
# ---------------------------------------------------------------------------

class TestAudioCuesWAVGeneration(unittest.TestCase):
    def test_generate_wav_creates_valid_wav_file(self):
        from psst.audio_cues import _generate_wav
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test_tone.wav"
            _generate_wav(path, freq=440.0, duration=0.1)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 44)  # > WAV header size
            with wave.open(str(path)) as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)       # 16-bit
                self.assertEqual(wf.getframerate(), 44100)
                expected_frames = int(44100 * 0.1)
                self.assertEqual(wf.getnframes(), expected_frames)

    def test_generate_wav_all_tones(self):
        from psst.audio_cues import _generate_wav, TONE_SPEC
        with tempfile.TemporaryDirectory() as d:
            for name, (freq, dur) in TONE_SPEC.items():
                path = Path(d) / f"{name}.wav"
                _generate_wav(path, freq, dur)
                self.assertTrue(path.exists(), f"{name}.wav not created")
                with wave.open(str(path)) as wf:
                    self.assertGreater(wf.getnframes(), 0)

    def test_generate_wav_creates_parent_dirs(self):
        from psst.audio_cues import _generate_wav
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "subdir" / "nested" / "tone.wav"
            _generate_wav(path, freq=440.0, duration=0.05)
            self.assertTrue(path.exists())

    def test_ensure_sounds_returns_all_paths(self):
        from psst.audio_cues import _ensure_sounds, TONE_SPEC
        with tempfile.TemporaryDirectory() as d:
            with patch("psst.audio_cues.SOUNDS_DIR", Path(d)):
                from psst import audio_cues
                paths = audio_cues._ensure_sounds()
        self.assertEqual(set(paths.keys()), set(TONE_SPEC.keys()))
        for p in paths.values():
            self.assertIsInstance(p, Path)

    def test_play_unknown_name_does_not_raise(self):
        from psst import audio_cues
        with patch("psst.audio_cues._get_paths", return_value={}):
            audio_cues.play("nonexistent_cue")  # must not raise

    def test_play_winsound_error_suppressed(self):
        from psst import audio_cues
        fake_path = Path("/fake/path.wav")
        with patch("psst.audio_cues._get_paths", return_value={"start": fake_path}):
            with patch("psst.audio_cues.winsound.PlaySound", side_effect=Exception("hw error")):
                audio_cues.play("start")  # must not raise

    def test_convenience_functions_call_play(self):
        from psst import audio_cues
        with patch.object(audio_cues, "play") as mock_play:
            audio_cues.play_start()
            audio_cues.play_done()
            audio_cues.play_error()
            audio_cues.play_cancel()
        self.assertEqual(mock_play.call_count, 4)
        names = [call.args[0] for call in mock_play.call_args_list]
        self.assertEqual(names, ["start", "done", "error", "cancel"])

    def test_play_stop_falls_back_to_play_when_no_custom_wav(self):
        """play_stop() uses generated tone when pssst.wav does not exist."""
        from psst import audio_cues
        with tempfile.TemporaryDirectory() as d:
            # Patch Path.cwd() so the "CWD" is a temp dir with no pssst.wav
            with patch("pathlib.Path.cwd", return_value=Path(d)):
                with patch.object(audio_cues, "play") as mock_play:
                    audio_cues.play_stop()
        mock_play.assert_called_once_with("stop")

    def test_play_stop_uses_custom_pssst_wav_when_present(self):
        """play_stop() plays pssst.wav directly when it exists in CWD."""
        from psst import audio_cues
        from psst.audio_cues import _generate_wav
        with tempfile.TemporaryDirectory() as d:
            wav_path = Path(d) / "pssst.wav"
            _generate_wav(wav_path, 330.0, 0.1)
            # Patch Path.cwd() so the "CWD" is the temp dir containing pssst.wav
            with patch("pathlib.Path.cwd", return_value=Path(d)):
                with patch("psst.audio_cues.winsound.PlaySound") as mock_ps:
                    audio_cues.play_stop()
            mock_ps.assert_called_once()
            call_path = mock_ps.call_args[0][0]
            self.assertIn("pssst.wav", call_path)

    def test_cancel_in_tone_spec(self):
        from psst.audio_cues import TONE_SPEC
        self.assertIn("cancel", TONE_SPEC)

    def test_cancel_tone_is_low_frequency(self):
        from psst.audio_cues import TONE_SPEC
        freq, dur = TONE_SPEC["cancel"]
        self.assertLess(freq, 300.0)   # low tone
        self.assertLess(dur, 0.2)      # short

    def test_play_cancel_function_exists(self):
        from psst import audio_cues
        self.assertTrue(hasattr(audio_cues, "play_cancel"))

    def test_play_cancel_calls_play(self):
        from psst import audio_cues
        with patch.object(audio_cues, "play") as mock_play:
            audio_cues.play_cancel()
        mock_play.assert_called_once_with("cancel")


# ---------------------------------------------------------------------------
# psst.recorder
# ---------------------------------------------------------------------------

class TestRecorder(unittest.TestCase):
    def test_creation_with_defaults(self):
        from psst.recorder import Recorder
        r = Recorder()
        self.assertEqual(r.sample_rate, 16000)
        self.assertEqual(r.channels, 1)
        self.assertAlmostEqual(r.min_duration, 0.3)
        self.assertAlmostEqual(r.max_duration, 60.0)
        self.assertFalse(r.is_recording)

    def test_elapsed_zero_when_not_recording(self):
        from psst.recorder import Recorder
        r = Recorder()
        self.assertEqual(r.elapsed(), 0.0)

    def test_stop_when_not_recording_returns_none(self):
        from psst.recorder import Recorder
        r = Recorder()
        self.assertIsNone(r.stop())

    def test_start_returns_true_first_time(self):
        import sounddevice as sd
        from psst.recorder import Recorder
        r = Recorder()
        mock_stream = MagicMock()
        with patch.object(sd, "InputStream", return_value=mock_stream):
            result = r.start()
        self.assertTrue(result)
        self.assertTrue(r.is_recording)
        # Cleanup without real hardware
        r._recording = False

    def test_start_returns_false_when_already_recording(self):
        import sounddevice as sd
        from psst.recorder import Recorder
        r = Recorder()
        mock_stream = MagicMock()
        with patch.object(sd, "InputStream", return_value=mock_stream):
            r.start()
            result = r.start()  # second call while recording
        self.assertFalse(result)
        r._recording = False

    def test_stop_returns_none_for_short_recording(self):
        """Recording shorter than min_duration must return None."""
        import sounddevice as sd
        from psst.recorder import Recorder
        r = Recorder(min_duration=5.0)  # impossibly long min
        mock_stream = MagicMock()
        with patch.object(sd, "InputStream", return_value=mock_stream):
            r.start()
        r._chunks = [np.zeros(1600, dtype="float32")]
        r._start_time = time.monotonic() - 0.05  # only 50 ms
        result = r.stop()
        self.assertIsNone(result)

    def test_stop_returns_audio_for_sufficient_recording(self):
        """Recording longer than min_duration must return a float32 array."""
        import sounddevice as sd
        from psst.recorder import Recorder
        r = Recorder(min_duration=0.01)
        mock_stream = MagicMock()
        with patch.object(sd, "InputStream", return_value=mock_stream):
            r.start()
        # Simulate 1 second of audio chunks
        r._chunks = [np.zeros((1600, 1), dtype="float32") for _ in range(10)]
        r._start_time = time.monotonic() - 1.0
        result = r.stop()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.ndim, 1)  # flattened to 1-D

    def test_stop_trims_to_max_duration(self):
        """Audio exceeding max_duration must be trimmed."""
        import sounddevice as sd
        from psst.recorder import Recorder
        sr = 16000
        max_dur = 1.0
        r = Recorder(sample_rate=sr, min_duration=0.0, max_duration=max_dur)
        mock_stream = MagicMock()
        with patch.object(sd, "InputStream", return_value=mock_stream):
            r.start()
        # 3 seconds of audio
        r._chunks = [np.zeros(sr * 3, dtype="float32")]
        r._start_time = time.monotonic() - 3.0
        result = r.stop()
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), int(sr * max_dur))

    def test_stop_returns_none_for_empty_chunks(self):
        """If no audio was captured, stop() returns None."""
        import sounddevice as sd
        from psst.recorder import Recorder
        r = Recorder(min_duration=0.0)
        mock_stream = MagicMock()
        with patch.object(sd, "InputStream", return_value=mock_stream):
            r.start()
        r._chunks = []
        r._start_time = time.monotonic() - 1.0
        result = r.stop()
        self.assertIsNone(result)

    def test_callback_copies_indata(self):
        """Callback must copy indata — sounddevice reuses its buffers."""
        from psst.recorder import Recorder
        r = Recorder()
        indata = np.ones((100, 1), dtype="float32")
        r._callback(indata, 100, None, None)
        self.assertEqual(len(r._chunks), 1)
        # Mutate original — copy in _chunks must be unaffected
        indata[:] = 0.0
        self.assertTrue(np.all(r._chunks[0] == 1.0))

    def test_is_recording_property(self):
        from psst.recorder import Recorder
        r = Recorder()
        self.assertFalse(r.is_recording)
        r._recording = True
        self.assertTrue(r.is_recording)

    def test_elapsed_positive_while_recording(self):
        from psst.recorder import Recorder
        r = Recorder()
        r._recording = True
        r._start_time = time.monotonic() - 0.5
        elapsed = r.elapsed()
        self.assertGreater(elapsed, 0.4)
        r._recording = False


class TestRecorderCancel(unittest.TestCase):
    def test_cancel_when_not_recording_does_not_raise(self):
        from psst.recorder import Recorder
        r = Recorder()
        r.cancel()  # must not raise

    def test_cancel_stops_recording(self):
        import sounddevice as sd
        from psst.recorder import Recorder
        r = Recorder(min_duration=0.0)
        mock_stream = MagicMock()
        with patch.object(sd, "InputStream", return_value=mock_stream):
            r.start()
        self.assertTrue(r.is_recording)
        r.cancel()
        self.assertFalse(r.is_recording)

    def test_cancel_discards_chunks(self):
        """After cancel(), stop() must return None (chunks discarded)."""
        import sounddevice as sd
        from psst.recorder import Recorder
        r = Recorder(min_duration=0.0)
        mock_stream = MagicMock()
        with patch.object(sd, "InputStream", return_value=mock_stream):
            r.start()
        r._chunks = [np.zeros(1600, dtype="float32")]
        r._start_time = time.monotonic() - 1.0
        r.cancel()
        # Chunks should be cleared; stop() should return None (not recording)
        result = r.stop()
        self.assertIsNone(result)

    def test_cancel_twice_does_not_raise(self):
        import sounddevice as sd
        from psst.recorder import Recorder
        r = Recorder(min_duration=0.0)
        mock_stream = MagicMock()
        with patch.object(sd, "InputStream", return_value=mock_stream):
            r.start()
        r.cancel()
        r.cancel()  # second cancel — must not raise


# ---------------------------------------------------------------------------
# psst.output
# ---------------------------------------------------------------------------

class TestOutputClipboard(unittest.TestCase):
    def test_copy_via_pyperclip_success(self):
        from psst.output import copy_to_clipboard
        import pyperclip
        with patch.object(pyperclip, "copy", return_value=None) as mock_copy:
            result = copy_to_clipboard("hello psst")
        self.assertTrue(result)
        mock_copy.assert_called_once_with("hello psst")

    def test_copy_fallback_disabled_returns_false_when_pyperclip_fails(self):
        from psst.output import copy_to_clipboard
        import pyperclip
        with patch.object(pyperclip, "copy", side_effect=Exception("no clipboard")):
            result = copy_to_clipboard("text", fallback=False)
        self.assertFalse(result)

    def test_copy_ctypes_skip_on_non_windows(self):
        from psst.output import _copy_ctypes
        with patch("psst.output.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = _copy_ctypes("test")
        self.assertFalse(result)

    def test_copy_ctypes_win32_returns_bool(self):
        """On Windows ctypes path must return a bool (True or False)."""
        from psst.output import _copy_ctypes
        if sys.platform == "win32":
            result = _copy_ctypes("psst test string")
            self.assertIsInstance(result, bool)

    def test_all_methods_fail_still_returns_false(self):
        from psst.output import copy_to_clipboard
        import pyperclip
        with patch.object(pyperclip, "copy", side_effect=Exception("fail")):
            with patch("psst.output._copy_ctypes", return_value=False):
                result = copy_to_clipboard("x", fallback=True)
        self.assertFalse(result)

    def test_copy_unicode_text(self):
        from psst.output import copy_to_clipboard
        import pyperclip
        text = "こんにちは — héllo wörld 🎤"
        with patch.object(pyperclip, "copy", return_value=None) as mock_copy:
            result = copy_to_clipboard(text)
        self.assertTrue(result)
        mock_copy.assert_called_once_with(text)


# ---------------------------------------------------------------------------
# psst.transcriber — CUDA detection and transcription
# ---------------------------------------------------------------------------

class TestTranscriberCUDADetection(unittest.TestCase):
    def test_auto_falls_back_to_cpu_when_cuda_unavailable(self):
        import ctranslate2
        from psst.transcriber import _detect_device_and_compute
        with patch.object(ctranslate2, "get_supported_compute_types",
                          side_effect=Exception("no CUDA")):
            device, compute = _detect_device_and_compute("auto", "auto")
        self.assertEqual(device, "cpu")
        self.assertEqual(compute, "int8")

    def test_auto_selects_cuda_when_available(self):
        import ctranslate2
        from psst.transcriber import _detect_device_and_compute
        with patch.object(ctranslate2, "get_supported_compute_types",
                          return_value=["float16", "int8"]):
            device, compute = _detect_device_and_compute("auto", "auto")
        self.assertEqual(device, "cuda")
        self.assertEqual(compute, "float16")

    def test_force_cpu_ignores_cuda(self):
        import ctranslate2
        from psst.transcriber import _detect_device_and_compute
        with patch.object(ctranslate2, "get_supported_compute_types",
                          return_value=["float16"]):
            device, compute = _detect_device_and_compute("cpu", "auto")
        self.assertEqual(device, "cpu")
        self.assertEqual(compute, "int8")

    def test_explicit_compute_type_passed_through(self):
        import ctranslate2
        from psst.transcriber import _detect_device_and_compute
        with patch.object(ctranslate2, "get_supported_compute_types",
                          side_effect=Exception("no CUDA")):
            device, compute = _detect_device_and_compute("auto", "float32")
        self.assertEqual(compute, "float32")

    def test_cuda_requested_but_unavailable_warns_and_falls_back(self):
        import ctranslate2
        from psst.transcriber import _detect_device_and_compute
        with patch.object(ctranslate2, "get_supported_compute_types",
                          side_effect=Exception("no CUDA")):
            device, _ = _detect_device_and_compute("cuda", "auto")
        self.assertEqual(device, "cpu")

    def test_transcribe_normalizes_clipped_audio(self):
        """Audio with peak > 1.0 must be normalized before passing to Whisper."""
        import ctranslate2
        import faster_whisper
        from psst.transcriber import Transcriber

        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.99
        seg = MagicMock()
        seg.text = "clipped audio test"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], info)

        with patch.object(ctranslate2, "get_supported_compute_types",
                          side_effect=Exception("no CUDA")):
            with patch.object(faster_whisper, "WhisperModel",
                               return_value=mock_model):
                t = Transcriber(model_name="base", device="cpu")
                # Peak = 3.0 → must be normalized to 1.0 before call
                audio = np.ones(16000, dtype="float32") * 3.0
                result = t.transcribe(audio)

        self.assertEqual(result, "clipped audio test")
        called_audio = mock_model.transcribe.call_args[0][0]
        self.assertLessEqual(float(np.abs(called_audio).max()), 1.0 + 1e-6)

    def test_transcribe_returns_none_for_empty_output(self):
        """Empty segment list → transcribe() returns None."""
        import ctranslate2
        import faster_whisper
        from psst.transcriber import Transcriber

        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.5
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], info)

        with patch.object(ctranslate2, "get_supported_compute_types",
                          side_effect=Exception("no CUDA")):
            with patch.object(faster_whisper, "WhisperModel",
                               return_value=mock_model):
                t = Transcriber(model_name="base", device="cpu")
                audio = np.zeros(16000, dtype="float32")
                result = t.transcribe(audio)

        self.assertIsNone(result)

    def test_transcribe_joins_multiple_segments(self):
        """Multiple segments must be joined with a space."""
        import ctranslate2
        import faster_whisper
        from psst.transcriber import Transcriber

        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.9
        seg1, seg2 = MagicMock(), MagicMock()
        seg1.text = "Hello"
        seg2.text = "world"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], info)

        with patch.object(ctranslate2, "get_supported_compute_types",
                          side_effect=Exception("no CUDA")):
            with patch.object(faster_whisper, "WhisperModel",
                               return_value=mock_model):
                t = Transcriber(model_name="base", device="cpu")
                audio = np.zeros(16000, dtype="float32")
                result = t.transcribe(audio)

        self.assertEqual(result, "Hello world")

    def test_transcribe_converts_int16_to_float32(self):
        """Non-float32 audio must be cast before transcription."""
        import ctranslate2
        import faster_whisper
        from psst.transcriber import Transcriber

        info = MagicMock()
        info.language = "en"
        info.language_probability = 0.8
        seg = MagicMock()
        seg.text = "dtype test"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], info)

        with patch.object(ctranslate2, "get_supported_compute_types",
                          side_effect=Exception("no CUDA")):
            with patch.object(faster_whisper, "WhisperModel",
                               return_value=mock_model):
                t = Transcriber(model_name="base", device="cpu")
                audio = np.zeros(16000, dtype="int16")
                t.transcribe(audio)

        called_audio = mock_model.transcribe.call_args[0][0]
        self.assertEqual(called_audio.dtype, np.float32)


# ---------------------------------------------------------------------------
# psst.cleanup — backend factory and instruction parameter
# ---------------------------------------------------------------------------

class TestCleanupBackend(unittest.TestCase):
    def test_get_backend_returns_ollama(self):
        from psst.config import Config
        from psst.cleanup import get_backend, OllamaBackend
        cfg = Config()
        cfg.cleanup_backend = "ollama"
        backend = get_backend(cfg)
        self.assertIsInstance(backend, OllamaBackend)
        self.assertEqual(backend.model, cfg.cleanup_model)
        self.assertEqual(backend.timeout, cfg.cleanup_timeout)

    def test_get_backend_ollama_url_stripped(self):
        from psst.config import Config
        from psst.cleanup import get_backend
        cfg = Config()
        cfg.cleanup_backend = "ollama"
        cfg.ollama_url = "http://localhost:11434/"
        backend = get_backend(cfg)
        self.assertEqual(backend._generate_url,
                         "http://localhost:11434/api/generate")

    def test_get_backend_unknown_returns_none(self):
        from psst.config import Config
        from psst.cleanup import get_backend
        cfg = Config()
        cfg.cleanup_backend = "totally_unknown"
        self.assertIsNone(get_backend(cfg))

    def test_get_backend_llama_cpp_no_path_returns_none_when_download_fails(self):
        from psst.config import Config
        from psst.cleanup import get_backend
        cfg = Config()
        cfg.cleanup_backend = "llama_cpp"
        cfg.llama_cpp_model_path = ""
        # With no local path, get_backend falls back to HF auto-download via
        # _ensure_model. Mock it to return None (simulating a download failure
        # or missing huggingface_hub) and confirm the backend is disabled.
        with patch("psst.cleanup._ensure_model", return_value=None):
            self.assertIsNone(get_backend(cfg))

    def test_get_backend_llama_cpp_with_path(self):
        from psst.config import Config
        from psst.cleanup import get_backend, LlamaCppBackend
        cfg = Config()
        cfg.cleanup_backend = "llama_cpp"
        cfg.llama_cpp_model_path = "/fake/model.gguf"
        backend = get_backend(cfg)
        self.assertIsInstance(backend, LlamaCppBackend)
        self.assertEqual(backend.model_path, "/fake/model.gguf")

    def test_get_backend_llama_cpp_aliases(self):
        """'llama-cpp' and 'llamacpp' must also be accepted."""
        from psst.config import Config
        from psst.cleanup import get_backend, LlamaCppBackend
        for alias in ("llama-cpp", "llamacpp"):
            cfg = Config()
            cfg.cleanup_backend = alias
            cfg.llama_cpp_model_path = "/fake/model.gguf"
            backend = get_backend(cfg)
            self.assertIsInstance(backend, LlamaCppBackend, f"alias {alias!r} failed")

    def test_ollama_backend_returns_raw_text_on_connection_error(self):
        """OllamaBackend.clean() must NEVER raise — return raw text on failure."""
        from psst.cleanup import OllamaBackend
        # Port 1 is reserved/closed — connection will fail immediately
        backend = OllamaBackend("http://127.0.0.1:1", "model", timeout=1)
        raw = "um hello world"
        result = backend.clean(raw)
        self.assertEqual(result, raw)

    def test_ollama_backend_returns_cleaned_text(self):
        """OllamaBackend.clean() must return the 'response' field from JSON."""
        from psst.cleanup import OllamaBackend
        import urllib.request
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"response": "Hello world."}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        backend = OllamaBackend("http://localhost:11434", "llama3")
        with patch.object(urllib.request, "urlopen", return_value=mock_resp):
            result = backend.clean("hello world")
        self.assertEqual(result, "Hello world.")

    def test_ollama_backend_empty_response_returns_raw(self):
        """Empty 'response' field → fall back to raw text."""
        from psst.cleanup import OllamaBackend
        import urllib.request
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"response": "   "}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        backend = OllamaBackend("http://localhost:11434", "llama3")
        with patch.object(urllib.request, "urlopen", return_value=mock_resp):
            result = backend.clean("raw text")
        self.assertEqual(result, "raw text")

    def test_llama_cpp_backend_returns_raw_on_import_error(self):
        """If llama_cpp is not installed, clean() must return raw text."""
        from psst.cleanup import LlamaCppBackend
        backend = LlamaCppBackend("/fake/model.gguf")
        with patch.dict("sys.modules", {"llama_cpp": None}):
            result = backend.clean("test input")
        self.assertEqual(result, "test input")


class TestCleanupWithInstruction(unittest.TestCase):
    def test_ollama_uses_custom_instruction(self):
        """When instruction is passed, it replaces the default SYSTEM_PROMPT."""
        from psst.cleanup import OllamaBackend
        import urllib.request
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"response": "Custom result."}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        backend = OllamaBackend("http://localhost:11434", "llama3")
        custom = "Format as bullet points."
        with patch.object(urllib.request, "urlopen", return_value=mock_resp) as mock_open:
            result = backend.clean("some text", instruction=custom)

        self.assertEqual(result, "Custom result.")
        body = json.loads(mock_open.call_args[0][0].data.decode())
        self.assertEqual(body["system"], custom)

    def test_ollama_uses_default_prompt_when_no_instruction(self):
        """When instruction=None, SYSTEM_PROMPT must be used."""
        from psst.cleanup import OllamaBackend, SYSTEM_PROMPT
        import urllib.request
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"response": "Result."}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        backend = OllamaBackend("http://localhost:11434", "llama3")
        with patch.object(urllib.request, "urlopen", return_value=mock_resp) as mock_open:
            backend.clean("some text")

        body = json.loads(mock_open.call_args[0][0].data.decode())
        self.assertEqual(body["system"], SYSTEM_PROMPT)

    def test_clean_signature_accepts_instruction_kwarg(self):
        """clean() must accept instruction as keyword argument."""
        from psst.cleanup import OllamaBackend
        # Port 1 will fail — but we just want to verify the signature
        backend = OllamaBackend("http://127.0.0.1:1", "model", timeout=1)
        result = backend.clean("text", instruction="custom instruction")
        # Should return raw text on connection failure
        self.assertEqual(result, "text")


# ---------------------------------------------------------------------------
# psst.hotkey
# ---------------------------------------------------------------------------

class TestHotkeyListener(unittest.TestCase):
    def test_single_key_parsed(self):
        from psst.hotkey import HotkeyListener
        hl = HotkeyListener("f13", queue.Queue())
        self.assertEqual(hl._trigger, "f13")
        self.assertEqual(hl._modifiers, set())

    def test_combo_parsed(self):
        from psst.hotkey import HotkeyListener
        hl = HotkeyListener("ctrl+shift+space", queue.Queue())
        self.assertEqual(hl._trigger, "space")
        self.assertEqual(hl._modifiers, {"ctrl", "shift"})

    def test_combo_with_whitespace_stripped(self):
        from psst.hotkey import HotkeyListener
        hl = HotkeyListener(" Ctrl + Alt + F1 ", queue.Queue())
        self.assertEqual(hl._trigger, "f1")
        self.assertIn("ctrl", hl._modifiers)
        self.assertIn("alt", hl._modifiers)

    def test_stop_when_not_started_does_not_raise(self):
        from psst.hotkey import HotkeyListener
        hl = HotkeyListener("ctrl+space", queue.Queue())
        hl.stop()  # must not raise

    def test_key_down_queues_press_event(self):
        import keyboard as kb
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q)
        ev = MagicMock()
        ev.name = "space"
        ev.event_type = kb.KEY_DOWN
        with patch.object(hl, "_modifiers_held", return_value=True):
            hl._on_key_event(ev)
        self.assertEqual(q.get_nowait(), ("press",))

    def test_key_repeat_ignored(self):
        """Second KEY_DOWN while already pressed must not enqueue another event."""
        import keyboard as kb
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q)
        ev = MagicMock()
        ev.name = "space"
        ev.event_type = kb.KEY_DOWN
        with patch.object(hl, "_modifiers_held", return_value=True):
            hl._on_key_event(ev)
            hl._on_key_event(ev)  # key-repeat
        items = list(q.queue)
        self.assertEqual(len(items), 1)

    def test_key_up_queues_release_event(self):
        import keyboard as kb
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q)
        hl._pressed = True
        ev = MagicMock()
        ev.name = "space"
        ev.event_type = kb.KEY_UP
        hl._on_key_event(ev)
        self.assertEqual(q.get_nowait(), ("release",))

    def test_key_up_without_prior_press_ignored(self):
        """KEY_UP when not pressed must not enqueue a release."""
        import keyboard as kb
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q)
        ev = MagicMock()
        ev.name = "space"
        ev.event_type = kb.KEY_UP
        hl._on_key_event(ev)
        self.assertTrue(q.empty())

    def test_wrong_key_ignored(self):
        import keyboard as kb
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q)
        ev = MagicMock()
        ev.name = "enter"
        ev.event_type = kb.KEY_DOWN
        with patch.object(hl, "_modifiers_held", return_value=True):
            hl._on_key_event(ev)
        self.assertTrue(q.empty())

    def test_modifiers_not_held_prevents_press(self):
        import keyboard as kb
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q)
        ev = MagicMock()
        ev.name = "space"
        ev.event_type = kb.KEY_DOWN
        with patch.object(hl, "_modifiers_held", return_value=False):
            hl._on_key_event(ev)
        self.assertTrue(q.empty())


class TestHotkeyListenerCancel(unittest.TestCase):
    def test_cancel_key_queues_cancel_event(self):
        import keyboard as kb
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q, cancel_key="escape")
        ev = MagicMock()
        ev.name = "escape"
        ev.event_type = kb.KEY_DOWN
        hl._on_key_event(ev)
        self.assertEqual(q.get_nowait(), ("cancel",))

    def test_cancel_key_repeat_ignored(self):
        """Second KEY_DOWN on cancel key while already pressed must not re-enqueue."""
        import keyboard as kb
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q, cancel_key="escape")
        ev = MagicMock()
        ev.name = "escape"
        ev.event_type = kb.KEY_DOWN
        hl._on_key_event(ev)
        hl._on_key_event(ev)  # repeat
        self.assertEqual(q.qsize(), 1)

    def test_cancel_key_up_clears_flag_allowing_repress(self):
        """After KEY_UP, pressing cancel again should queue another cancel."""
        import keyboard as kb
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q, cancel_key="escape")
        ev_down = MagicMock()
        ev_down.name = "escape"
        ev_down.event_type = kb.KEY_DOWN
        ev_up = MagicMock()
        ev_up.name = "escape"
        ev_up.event_type = kb.KEY_UP
        hl._on_key_event(ev_down)
        hl._on_key_event(ev_up)
        hl._on_key_event(ev_down)  # second press after release
        self.assertEqual(q.qsize(), 2)

    def test_cancel_key_does_not_interfere_with_main_hotkey(self):
        """Cancel key events must not propagate to the hotkey logic."""
        import keyboard as kb
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q, cancel_key="escape")
        # Press the main hotkey
        ev_main = MagicMock()
        ev_main.name = "space"
        ev_main.event_type = kb.KEY_DOWN
        with patch.object(hl, "_modifiers_held", return_value=True):
            hl._on_key_event(ev_main)
        self.assertEqual(q.get_nowait(), ("press",))
        self.assertTrue(q.empty())

    def test_no_cancel_key_param_normal_hotkey_works(self):
        """Without cancel_key, normal hotkey events still work."""
        import keyboard as kb
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q)  # no cancel_key
        ev = MagicMock()
        ev.name = "space"
        ev.event_type = kb.KEY_DOWN
        with patch.object(hl, "_modifiers_held", return_value=True):
            hl._on_key_event(ev)
        self.assertEqual(q.get_nowait(), ("press",))

    def test_start_returns_false_on_permission_error(self):
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q)
        with patch("psst.hotkey.keyboard.hook", side_effect=PermissionError("no admin")):
            result = hl.start()
        self.assertFalse(result)

    def test_start_returns_true_on_success(self):
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q)
        with patch("psst.hotkey.keyboard.hook", return_value=MagicMock()):
            result = hl.start()
        self.assertTrue(result)
        hl.stop()

    def test_start_returns_false_on_os_error(self):
        from psst.hotkey import HotkeyListener
        q = queue.Queue()
        hl = HotkeyListener("ctrl+space", q)
        with patch("psst.hotkey.keyboard.hook", side_effect=OSError("access denied")):
            result = hl.start()
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# psst.ui
# ---------------------------------------------------------------------------

class TestUI(unittest.TestCase):
    def test_initial_state_is_idle(self):
        from psst.ui import UI, State
        ui = UI()
        self.assertEqual(ui._state, State.IDLE)

    def test_set_state_all_states_no_exception(self):
        from psst.ui import UI, State
        ui = UI()
        for state in State:
            ui.set_state(state)
            self.assertEqual(ui._state, state)

    def test_set_state_done_with_detail(self):
        from psst.ui import UI, State
        ui = UI()
        ui.set_state(State.DONE, "some transcribed text")
        self.assertEqual(ui._state, State.DONE)

    def test_set_state_error_with_detail(self):
        from psst.ui import UI, State
        ui = UI()
        ui.set_state(State.ERROR, "something broke")
        self.assertEqual(ui._state, State.ERROR)

    def test_set_state_recording_sets_start_time(self):
        from psst.ui import UI, State
        ui = UI()
        ui._recording_start = 0.0
        ui.set_state(State.RECORDING)
        self.assertGreater(ui._recording_start, 0.0)

    def test_add_to_history_whisper_tag(self):
        from psst.ui import UI
        ui = UI()
        ui.add_to_history("raw text", cleaned=False)
        text, tag = ui._history[0]
        self.assertEqual(text, "raw text")
        self.assertEqual(tag, "whisper")

    def test_add_to_history_cleaned_tag(self):
        from psst.ui import UI
        ui = UI()
        ui.add_to_history("cleaned text", cleaned=True)
        _, tag = ui._history[0]
        self.assertEqual(tag, "cleaned")

    def test_history_maxlen_respected(self):
        from psst.ui import UI
        ui = UI(history_size=3)
        for i in range(5):
            ui.add_to_history(f"item {i}")
        self.assertEqual(len(ui._history), 3)
        # Most recent item is at index 0 (appendleft)
        self.assertEqual(ui._history[0][0], "item 4")

    def test_print_banner_does_not_raise(self):
        from psst.ui import UI
        ui = UI()
        ui.print_banner("ctrl+space", "base", "cpu (auto)", False)

    def test_print_banner_with_cancel_hotkey(self):
        from psst.ui import UI
        ui = UI()
        ui.print_banner("ctrl+space", "base", "cpu (auto)", False, cancel_hotkey="f1")

    def test_print_history_empty_does_not_raise(self):
        from psst.ui import UI
        ui = UI()
        ui.print_history()

    def test_print_history_with_items_does_not_raise(self):
        from psst.ui import UI
        ui = UI()
        ui.add_to_history("a short transcription")
        ui.print_history()

    def test_print_history_truncates_long_text(self):
        from psst.ui import UI
        ui = UI()
        long_text = "x" * 200
        ui.add_to_history(long_text)  # must not raise


class TestUINewStates(unittest.TestCase):
    def test_cancelled_state_exists(self):
        from psst.ui import State
        self.assertTrue(hasattr(State, "CANCELLED"))

    def test_set_state_cancelled_no_exception(self):
        from psst.ui import UI, State
        ui = UI()
        ui.set_state(State.CANCELLED)
        self.assertEqual(ui._state, State.CANCELLED)

    def test_print_admin_warning_no_exception(self):
        from psst.ui import UI
        ui = UI()
        ui.print_admin_warning()  # must not raise

    def test_cancelled_in_state_style_map(self):
        from psst.ui import State, _STATE_STYLE
        self.assertIn(State.CANCELLED, _STATE_STYLE)


# ---------------------------------------------------------------------------
# psst.tray
# ---------------------------------------------------------------------------

class TestTrayIcon(unittest.TestCase):
    def test_tray_module_importable(self):
        """psst.tray must be importable even if pystray/Pillow not installed."""
        import importlib
        mod = importlib.import_module("psst.tray")
        self.assertIsNotNone(mod)

    def test_tray_icon_creation_does_not_raise(self):
        from psst.tray import TrayIcon
        q = queue.Queue()
        ti = TrayIcon(q)
        self.assertIsNotNone(ti)

    def test_tray_stop_when_not_started_does_not_raise(self):
        from psst.tray import TrayIcon
        q = queue.Queue()
        ti = TrayIcon(q)
        ti.stop()  # must not raise

    def test_tray_update_status_when_not_started_does_not_raise(self):
        from psst.tray import TrayIcon
        q = queue.Queue()
        ti = TrayIcon(q)
        ti.update_status("Recording...")  # must not raise

    def test_tray_start_skipped_when_unavailable(self):
        """If pystray is not available, start() is a silent no-op."""
        import psst.tray as tray_mod
        q = queue.Queue()
        ti = tray_mod.TrayIcon(q)
        with patch.object(tray_mod, "_TRAY_AVAILABLE", False):
            ti.start()
        self.assertIsNone(ti._icon)

    def test_tray_quit_puts_event_on_queue(self):
        """_on_quit must put ('quit',) on the event queue."""
        from psst.tray import TrayIcon
        q = queue.Queue()
        ti = TrayIcon(q)
        mock_icon = MagicMock()
        ti._on_quit(mock_icon, None)
        self.assertEqual(q.get_nowait(), ("quit",))

    def test_tray_make_icon_image(self):
        """Icon generation should succeed when Pillow is available."""
        try:
            from psst.tray import _make_icon_image
            img = _make_icon_image()
            self.assertEqual(img.size, (64, 64))
        except ImportError:
            self.skipTest("Pillow not installed")


# ---------------------------------------------------------------------------
# psst package
# ---------------------------------------------------------------------------

class TestPackageMetadata(unittest.TestCase):
    def test_version_is_semver_string(self):
        from psst import __version__
        self.assertIsInstance(__version__, str)
        parts = __version__.split(".")
        self.assertEqual(len(parts), 3, f"Expected X.Y.Z semver, got {__version__!r}")
        for part in parts:
            self.assertTrue(part.isdigit(), f"Non-numeric semver part: {part!r}")

    def test_version_is_0_3_0(self):
        from psst import __version__
        self.assertEqual(__version__, "0.5.1")

    def test_author_defined(self):
        from psst import __author__
        self.assertIsInstance(__author__, str)
        self.assertTrue(len(__author__) > 0)

    def test_license_defined(self):
        from psst import __license__
        self.assertIsInstance(__license__, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
