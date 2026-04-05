# PSST — Push, Speak, Send Text

A free, open-source, fully local speech-to-text dictation tool.
Privacy-first alternative to Wispr Flow — nothing leaves your machine.

## How it works

1. **Hold** the hotkey (`Ctrl+Shift+Space` by default)
2. **Speak** your text
3. **Release** the hotkey
4. Whisper transcribes locally → text is **copied to your clipboard**
5. **Paste** anywhere

## Requirements

- Windows 10/11 (primary target; Linux/Mac partial support)
- Python 3.10+
- NVIDIA GPU recommended (RTX series with CUDA) — CPU works too, just slower

## Quick start

```powershell
# 1. Clone / download the repo
cd D:\PSST

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Copy and edit config
copy config.example.toml config.toml

# 5. Run  (Administrator recommended for global hotkeys on Windows)
python -m psst
```

> **Tip:** Double-click `start_psst.bat` to activate the virtual environment and
> launch PSST in one step. Run it as Administrator for global hotkey support.

> **Note:** The `keyboard` library requires Administrator privileges on Windows
> to capture global hotkeys.  Right-click your terminal → "Run as administrator".

## Configuration

Copy `config.example.toml` → `config.toml` and edit. Key settings:

| Key | Default | Description |
|-----|---------|-------------|
| `hotkey` | `ctrl+shift+space` | Hold-to-record key combo |
| `model` | `small` | Whisper model (tiny/base/small/medium/large-v3) |
| `device` | `auto` | `auto`, `cuda`, or `cpu` |
| `max_recording_seconds` | `300` | Max recording length in seconds (change in `config.toml`) |
| `cleanup_enabled` | `true` | Enable LLM filler-word cleanup |
| `cleanup_backend` | `llama_cpp` | `llama_cpp` (recommended) or `ollama` |
| `audio_cues` | `true` | Play tones on start/stop/done |

Config search order:
1. `--config PATH` CLI flag
2. `./config.toml` (current directory)
3. `%APPDATA%/psst/config.toml`
4. `~/.psst/config.toml`
5. Built-in defaults

## LLM cleanup

Cleanup is **on by default**. Whisper already suppresses many filler words
naturally via its training data and VAD filter — the session history labels
this output `[whisper]`. When LLM cleanup is also running it adds proper
punctuation, sentence capitalisation, and catches any remaining disfluencies,
labelling the result `[cleaned]`.

If the configured backend cannot be initialised (e.g. no model file set),
PSST silently falls back to Whisper output — **your dictation is never lost**.

> **Tip:** To skip LLM cleanup entirely, use `--no-cleanup` on the CLI or set
> `cleanup_enabled = false` in `config.toml`.

### llama-cpp-python backend (recommended)

The default Qwen3.5-4B model (~2.7 GB) is **auto-downloaded from Hugging Face**
at startup — no manual setup required.

For GPU acceleration, run the bundled build script from a **VS 2022
Developer Command Prompt**:

    cd /d D:\PSST
    build_cuda_llama.bat

The script pins llama-cpp-python to `0.3.19` (the newest release that
supports Qwen3.5 and pre-dates a broken upstream `llama.cpp` submodule
bump), then compiles from source with `-DGGML_CUDA=on`. Takes ~20 minutes.

Requirements:
- [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) installed
- CMake on PATH
- VS 2022 with C++ build tools (the "Developer Command Prompt" supplies `cl.exe`)

For CPU-only use, skip the build script entirely — `pip install llama-cpp-python`
(prebuilt CPU wheel) works fine, just slower at inference.

#### Troubleshooting the CUDA build

- `[!] n_gpu_layers=-1 but llama-cpp-python was built WITHOUT CUDA support`
  with 0.3.x — fixed as of commit `dfb73a7`. Pull latest and rebuild.
- `Cannot open source file: 'deprecation-warning.cpp'` — you're on 0.3.20;
  the build script should pin 0.3.19. Check line 24 of `build_cuda_llama.bat`.
- `cmake: command not found` or wrong cmake — run `where.exe cmake` to see
  which is first on PATH. The build script rejects CMake paths containing
  `WILLOW` (retired project directory).

To use a **different HF model**, change these in `config.toml`:
```toml
llama_cpp_repo_id  = "unsloth/Qwen3.5-4B-GGUF"
llama_cpp_filename = "Qwen3.5-4B-Q4_K_M.gguf"
chat_format        = "chatml"          # match the model's chat template
```

To use a **local GGUF file** instead of auto-download, set `llama_cpp_model_path`:
```toml
llama_cpp_model_path = "C:/models/your-model.gguf"
```

### Ollama backend

```powershell
# Install Ollama from https://ollama.com
ollama pull qwen3.5:4b
```

In `config.toml`:
```toml
cleanup_enabled = true
cleanup_backend = "ollama"
cleanup_model   = "qwen3.5:4b"
```

All LLM failures fall back silently to Whisper text — **your dictation is never lost**.

## CLI flags

```
python -m psst [options]

  --config PATH        Config file path
  --model NAME         Whisper model (tiny/base/small/medium/large-v3)
  --hotkey COMBO       Hotkey combo
  --device {auto,cuda,cpu}
  --cleanup            Enable LLM cleanup
  --no-cleanup         Disable LLM cleanup (overrides config)
  --no-audio-cues      Disable sound feedback
  --language LANG      Force language (e.g. en, fr)
  --log                Write session log to psst.log
  --version
```

## Architecture

```
Thread 1 (main)     — event loop, transcription, LLM, clipboard, UI
Thread 2 (keyboard) — only puts events on queue
Thread 3 (PortAudio)— only appends audio chunks
Thread 4 (tray)     — pystray daemon, only puts ("quit",) on queue
```

Key invariant: keyboard and audio threads **never do heavy work**.

## Models

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| tiny  | ~1 GB | very fast | basic |
| base  | ~1 GB | fast | good |
| small | ~2 GB | moderate | better |
| medium | ~5 GB | slow | great |
| large-v3 | ~10 GB | slowest | best |

Both Whisper and LLM cleanup models are downloaded automatically from Hugging Face.
The Whisper model downloads on first transcription; the cleanup model (~2.7 GB)
downloads at startup so it's ready when you need it.

## Custom icon

Place an `icon.png` or `icon.ico` file in the `assets/` directory to customise
the system tray icon and console window icon (Windows). If no custom icon is
found, PSST uses a generated blue "P" fallback.

## Privacy

- 100% local — no cloud API calls, no telemetry
- Audio is captured in memory only; never written to disk
- Model weights cached in `~/.cache/huggingface/` by default

## License

Apache 2.0 — see [LICENSE](LICENSE).
