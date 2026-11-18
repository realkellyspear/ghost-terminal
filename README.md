<p align="center">
  <img src="assets/ghost_wordmark.png" alt="GHoSt Terminal - GPU Hosted Static Terminal" width="480">
</p>

# GHoSt Terminal

**GHoSt Terminal** (**G**PU **Ho**sted **St**atic **Terminal**) is a streamlined, fully local GGUF model loader and desktop chat app for Windows. Drop in a GGUF model, check how it fits your GPU, mount it, and chat. No cloud, no accounts, no telemetry.

## See It in Action

<p align="center">
  <img src="assets/terminal-chat.png" alt="GHoSt Terminal chat interface" width="48%">
  <img src="assets/chat_config.png" alt="GHoSt Terminal CONFIG modal" width="48%">
</p>

## Features

- **Desktop Window & Local Loopback API**: Opens in a native desktop window (pywebview + Microsoft Edge WebView2) backed by a local `llama.cpp` / FastAPI server bound strictly to `127.0.0.1` (auto-selects the first open port starting at `5555`, or override with `GHOST_API_PORT`).
- **Hardware-Aware VRAM Preflight (`CHECK FIT`)**: Inspects GGUF tensor and architecture metadata alongside live `nvidia-smi` VRAM availability to pick a safe GPU layer count and KV-cache type (`F16` or `Q8_0`). Supports both standard transformers and hybrid SSM/DeltaNet + attention architectures (such as Qwen3.5 and Qwen3.6).
- **Flash Attention Toggle**: Enabled by default in **[ CONFIG ]** for faster inference and to unlock `Q8_0` KV-cache compression when VRAM is tight (uncheck as a compatibility fallback to keep `F16` KV cache).
- **Full Sampler Controls**: Temperature, Top P, Top K, Min P, Repetition Penalty, Presence Penalty, Seed, Maximum Response Tokens, Stop Sequences, Context Length, and Chat Template selection (`auto` or manual override).
- **Clean System Prompt**: Left blank by default so your models behave cleanly out of the box, or enter your own system prompt when needed.
- **Chat Controls**: Streaming responses with per-message **Stop**, **Read aloud** (speaker icon), **Copy**, **Edit**, **Redo**, and **Delete**.
- **Local Read-Aloud Voices**: Powered by [Piper](https://github.com/OHF-Voice/piper1-gpl). Includes **Silas** and **Kara** out of the box (downloaded automatically from Hugging Face on first use, ~63 MB each, then cached offline in `piper_models/`). Drop any custom `.onnx` + `.onnx.json` pair into `piper_models/` to add more voices.
- **Optional Reasoning Disclosure**: `<think>` reasoning blocks are suppressed/hidden by default (including Jinja `enable_thinking` models) and can be toggled on in **[ CONFIG ]** as collapsible disclosures beneath replies.
- **Persistent Settings**: Saved locally in `ghost_terminal_settings.json` beside the executable.

---

## Quick Start

### Option A: Download the Standalone Windows Release (Recommended)

1. Download and extract **`GHoSt-Terminal-v1.0.2-win64.zip`** from Releases.
2. Copy one or more `.gguf` model files into the extracted `ghost_models/` folder.
3. Double-click **`GHoSt-Terminal.exe`**.
4. In **[ CONFIG ]**, select your model, click **CHECK FIT** (optional) and **MOUNT**, then click **CLOSE** (or **SAVE SETTINGS & CLOSE**) to start chatting.

- **UNLOAD** releases the resident model and frees VRAM while keeping your chat on screen.
- **NEW CHAT** clears the transcript and resets the model's active KV/context state without unloading the model.

### Option B: Run From Python Source

Requires Python 3.11 on Windows (install CUDA-enabled `llama-cpp-python` + `nvidia-cublas` if using an NVIDIA GPU).

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python ghost_terminal.py
```

To run only the local OpenAI-compatible API + web UI without opening the desktop window:

```powershell
python ghost_terminal.py --api-only
```

Optional environment variables:
- `GHOST_API_PORT`: override the default port (`5555`, or next free port up to `5574`).
- `GHOST_BACKEND`: force `auto` (default), `cuda`, `vulkan`, or `cpu`.

---

## Build & Package From Source

### Build `GHoSt-Terminal.exe`

```powershell
python build_terminal.py
```

`build_terminal.py` automatically switches to `./.venv` if present and requires CUDA `llama-cpp-python` (`ggml-cuda.dll` + `cublas`) by default (pass `--cpu-only` if you intentionally want a CPU-only build). It outputs `GHoSt-Terminal.exe` and stages external backend DLLs into `backends/`.

Run the focused regression tests without loading a model:

```powershell
python -m unittest discover -s tests -v
```

### Create a Clean Distributable Release `.zip`

```powershell
python package_release.py
```

This stages files in a temporary directory and writes **`../GHoSt-Terminal-v1.0.2-win64.zip`** containing:

```text
GHoSt-Terminal/
  GHoSt-Terminal.exe
  README.md
  LICENSE
  THIRD_PARTY_NOTICES.md
  assets/
  backends/            cpu/ and cuda/ runtime DLLs
  ghost_models/        drop your *.gguf files here
  piper_models/        MODEL_CARDs + README (voices download on first use)
  source/              complete first-party source code
```

No `.gguf` model files, personal settings, or logs are included in the release archive.

### Backend Packaging Status

The builder externalizes the prebuilt DLLs from `llama-cpp-python` and `nvidia-cublas` into `backends/cpu` and `backends/cuda` rather than embedding the CUDA runtime inside the EXE. Verified on NVIDIA Windows hardware (RTX 5060 Ti); CPU-only machines without the linked CUDA/cuBLAS DLLs and Vulkan hardware are not yet certified.

---

## Privacy

The server binds to `127.0.0.1` only with zero telemetry. Nothing leaves your machine except the one-time download of a selected Piper voice from `rhasspy/piper-voices` on Hugging Face the first time you click the speaker button.

Browser requests from other origins are rejected. The local API is not authenticated, so other programs running on the same machine can still access it; only run trusted local software while it is active.

## License

GHoSt Terminal's first-party code is licensed under **GPL-3.0-or-later** — see [LICENSE](LICENSE). Third-party components and voice models retain their respective licenses — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
