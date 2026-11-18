#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kelly Spear
# SPDX-License-Identifier: GPL-3.0-or-later
"""
=======================================================================
 GHoSt Terminal â€” GPU Hosted Static Terminal
 A streamlined, fully-offline GGUF model loader and chat window.
=======================================================================
  - Hardware target : Local GPU (CUDA / Vulkan) with safe CPU fallback
  - Bind target     : 127.0.0.1 only (strict isolation, zero telemetry)
  - Core engine     : llama-cpp-python
  - API schema      : OpenAI-compatible (/v1/chat/completions)
  - Extras          : Piper read-aloud voices (downloaded on first use)
=======================================================================
"""

import argparse
import csv
import importlib.util
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
import gc
import json
import time
import uuid
import logging
import traceback
import threading
import shutil
import subprocess
import tempfile
import io
import re
import secrets
import socket
import urllib.request
import wave
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator
from gguf import GGUFReader, LlamaFileType

IS_FROZEN = bool(getattr(sys, "frozen", False))
APP_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else Path(__file__).resolve().parent
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
BACKEND_ROOT = APP_DIR / "backends"
_DLL_DIRECTORY_HANDLES = []


def _find_nvidia_smi() -> Optional[str]:
    executable = shutil.which("nvidia-smi")
    if executable:
        return executable
    candidates = [
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
        Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _probe_nvidia_runtime() -> bool:
    executable = _find_nvidia_smi()
    if executable is None:
        return False
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def _probe_vulkan_runtime() -> bool:
    executable = shutil.which("vulkaninfo")
    return executable is not None and bool(os.environ.get("VULKAN_SDK"))


def _cuda_runtime_dlls() -> List[Path]:
    roots = [
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia",
        Path(os.environ.get("CUDA_PATH", "")) / "bin",
    ]
    dlls = []
    for root in roots:
        if root.is_dir():
            dlls.extend(root.rglob("cublas*.dll"))
    unique = {path.name.lower(): path for path in dlls}
    return list(unique.values())


def _configure_runtime_backend() -> tuple[str, Path]:
    package_spec = importlib.util.find_spec("llama_cpp")
    if package_spec is None or package_spec.origin is None:
        raise RuntimeError("llama_cpp Python package is unavailable.")
    package_lib = Path(package_spec.origin).parent / "lib"

    candidates = {
        backend: BACKEND_ROOT / backend
        for backend in ("cuda", "vulkan", "cpu")
    }
    if not IS_FROZEN or not any(path.is_dir() for path in candidates.values()):
        candidates = {"cuda": package_lib, "vulkan": package_lib, "cpu": package_lib}

    plugins = {
        "cuda": "ggml-cuda.dll",
        "vulkan": "ggml-vulkan.dll",
        "cpu": "ggml-cpu.dll",
    }
    available = {
        name: directory.is_dir() and (directory / plugins[name]).is_file()
        for name, directory in candidates.items()
    }
    requested = os.environ.get("GHOST_BACKEND", "auto").lower()
    if requested not in {"auto", "cuda", "vulkan", "cpu"}:
        raise RuntimeError("GHOST_BACKEND must be auto, cuda, vulkan, or cpu.")

    if requested == "auto":
        if available["cuda"] and _probe_nvidia_runtime():
            selected = "cuda"
        elif available["vulkan"] and _probe_vulkan_runtime():
            selected = "vulkan"
        else:
            selected = "cpu"
    else:
        selected = requested
        if not available[selected]:
            raise RuntimeError(f"Requested {selected} backend library is not installed.")
        if selected == "cuda" and not _probe_nvidia_runtime():
            raise RuntimeError("CUDA backend was requested, but nvidia-smi found no CUDA device.")
        if selected == "vulkan" and not _probe_vulkan_runtime():
            raise RuntimeError("Vulkan backend was requested, but Vulkan SDK/runtime probing is unavailable.")

    source = candidates[selected]
    if IS_FROZEN:
        runtime_lib = RESOURCE_DIR / "llama_cpp" / "lib"
    else:
        runtime_lib = Path(tempfile.gettempdir()) / "GHoSt-Terminal" / "backends" / selected
    runtime_lib.mkdir(parents=True, exist_ok=True)
    for existing in runtime_lib.glob("*.dll"):
        existing.unlink()

    dlls = list(source.glob("*.dll"))
    if not IS_FROZEN and selected in {"cpu", "cuda"}:
        dlls.extend(_cuda_runtime_dlls())
    required = {"llama.dll", "ggml.dll", "ggml-base.dll", plugins[selected]}
    names = {item.name for item in dlls}
    missing = required - names
    if selected in {"cpu", "cuda"} and (source / "ggml-cuda.dll").is_file():
        missing_cuda = {"ggml-cuda.dll"} - names
        if missing_cuda:
            missing |= missing_cuda
        import pefile
        cuda_imports = {
            entry.dll.decode(errors="replace").lower()
            for entry in pefile.PE(str(source / "ggml-cuda.dll")).DIRECTORY_ENTRY_IMPORT
            if entry.dll.decode(errors="replace").lower().startswith("cublas")
        }
        present = {item.name.lower() for item in dlls}
        missing |= cuda_imports - present
    if missing:
        raise RuntimeError(f"{selected} backend is missing required DLLs: {sorted(missing)}")
    for item in dlls:
        shutil.copy2(item, runtime_lib / item.name)

    if os.name == "nt":
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(runtime_lib)))
        os.environ["PATH"] = str(runtime_lib) + os.pathsep + os.environ.get("PATH", "")
    os.environ["GGML_BACKEND_PATH"] = str(runtime_lib)
    return selected, runtime_lib


BACKEND_NAME, RUNTIME_LLAMA_LIB_DIR = _configure_runtime_backend()

from llama_cpp import Llama, llama_cpp

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

# =====================================================================
# CONFIGURATION
# =====================================================================

def _find_available_port(preferred: int = 5555) -> int:
    """Honor GHOST_API_PORT, otherwise take the first free loopback port from 5555."""
    if "GHOST_API_PORT" in os.environ:
        return int(os.environ["GHOST_API_PORT"])
    for candidate in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", candidate)) == 0:
                continue
        return candidate
    return preferred


def _reserve_server_socket(host: str, preferred: int = 5555) -> tuple[int, socket.socket]:
    """Bind once so another process cannot claim the chosen port before startup."""
    explicit = os.environ.get("GHOST_API_PORT")
    ports = [int(explicit)] if explicit is not None else range(preferred, preferred + 20)
    last_error = None
    for port in ports:
        if not 1 <= port <= 65535:
            raise ValueError("GHOST_API_PORT must be between 1 and 65535.")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if os.name == "nt":
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind((host, port))
            listener.listen(2048)
            return port, listener
        except OSError as exc:
            listener.close()
            last_error = exc
    raise OSError(f"Could not bind GHoSt Terminal to {host}: {last_error}") from last_error


HOST              = "127.0.0.1"
if __name__ == "__main__":
    PORT, SERVER_SOCKET = _reserve_server_socket(HOST)
else:
    PORT, SERVER_SOCKET = _find_available_port(5555), None
MODEL_DIR = APP_DIR / "ghost_models"
PIPER_MODEL_DIR = APP_DIR / "piper_models"
SETTINGS_PATH = APP_DIR / "ghost_terminal_settings.json"
HTML_PATH = RESOURCE_DIR / "ghost_terminal.html"
ASSET_DIR = RESOURCE_DIR / "assets"
LOG_PATH = APP_DIR / "ghost_terminal.log"
DEFAULT_N_CTX = 8192
MIN_N_CTX = 512
MAX_N_CTX = 262144
MAX_UI_MAX_TOKENS = 32768
GIB = 1024 ** 3
GPU_SAFETY_RESERVE_BYTES = 768 * 1024 ** 2
GPU_MAX_UTILIZATION = 0.95
N_THREADS         = os.cpu_count() or 8
FLASH_ATTN        = True      # Flash Attention 2 if supported by GGUF
N_BATCH           = 512

DEFAULT_SETTINGS = {
    "model": "Qwen3-14B-Q5_K_M.gguf",
    "chat_format": "auto",
    "system_prompt": "",
    "tts_voice": "silas",
    "flash_attn": True,
    "show_reasoning": False,
    "temperature": 0.7,
    "top_p": 0.95,
    "min_p": 0.05,
    "top_k": 40,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "seed": None,
    "max_tokens": 512,
    "stop_sequences": "",
    "n_ctx": DEFAULT_N_CTX,
}

# =====================================================================
# LOGGING — [GHoSt-Engine] prefix, no external handlers
# =====================================================================

log_handlers = [logging.FileHandler(LOG_PATH, encoding="utf-8")]
if sys.stdout is not None:
    log_handlers.append(logging.StreamHandler(sys.stdout))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GHoSt-Terminal] %(levelname)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=log_handlers,
)
log = logging.getLogger("ghost")

# =====================================================================
# GHoSt TERMINAL ASCII BANNER
# =====================================================================

GHOST_BANNER = r"""
   ____ _   _       ____  _   
  / ___| | | | ___ / ___|| |_ 
 | |  _| |_| |/ _ \\___ \| __|
 | |_| |  _  | (_) |___) | |_ 
  \____|_| |_|\___/|____/ \__|

     GHoSt Terminal // GPU HOSTED STATIC TERMINAL
     Local. Offline. Yours.
=========================================================
"""

# =====================================================================
# RUNTIME STATE — one model at a time, hard flush between loads
# =====================================================================

class EngineState:
    def __init__(self):
        self.llm: Optional[Llama] = None
        self.loaded_model: Optional[str] = None
        self.loaded_at: Optional[float] = None
        self.load_seconds: Optional[float] = None
        self.loaded_gpu_layers: Optional[int] = None
        self.loaded_context_size: Optional[int] = None
        self.loaded_kv_cache_type: Optional[str] = None
        self.loaded_chat_format: Optional[str] = None
        self.loaded_flash_attn: Optional[bool] = None
        self.prompt_opens_think: bool = False
        self._think_handlers: Dict[bool, Any] = {}
        self.locked = False  # coarse serialization guard
        self.busy = False
        self.streaming = False
        self.generation_id = 0
        self.abort_event = threading.Event()
        self._lease = threading.Lock()
        self._operation_lock = threading.Lock()

    def begin_operation(self) -> bool:
        """Atomically reserve the model for a load, unload, or reset operation."""
        if not self._operation_lock.acquire(blocking=False):
            return False

        with self._lease:
            if self.busy:
                self._operation_lock.release()
                return False
            self.locked = True

        return True

    def end_operation(self) -> None:
        """Release the model-operation reservation."""
        with self._lease:
            self.locked = False
        self._operation_lock.release()

    def begin_generation(self, streaming: bool) -> int:
        with self._lease:
            if self.locked:
                raise HTTPException(status_code=409, detail="Model change in progress.")
            if self.busy:
                raise HTTPException(status_code=409, detail="A generation is already in progress.")
            if self.llm is None:
                raise HTTPException(status_code=409, detail="No model resident. Open CONFIG and click MOUNT first.")
            self.busy = True
            self.streaming = streaming
            self.abort_event.clear()
            self.generation_id += 1
            return self.generation_id

    def end_generation(self, generation_id: Optional[int] = None) -> None:
        with self._lease:
            if generation_id is not None and generation_id != self.generation_id:
                return
            self.busy = False
            self.streaming = False
            self.abort_event.clear()

    def request_abort(self) -> bool:
        with self._lease:
            if not self.busy:
                return False
            self.abort_event.set()
            return True

    @property
    def abort_requested(self) -> bool:
        return self.abort_event.is_set()


STATE = EngineState()

# =====================================================================
# READ ALOUD â€” local Piper TTS (voices download on first use)
# =====================================================================

_PIPER_VOICES: Dict[str, Any] = {}
_PIPER_LOCK = threading.Lock()

DEFAULT_VOICE_URLS = {
    "silas": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33/en/en_US/john/medium/en_US-john-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33/en/en_US/john/medium/en_US-john-medium.onnx.json",
    },
    "kara": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33/en/en_US/kristin/medium/en_US-kristin-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/e7229f7d33/en/en_US/kristin/medium/en_US-kristin-medium.onnx.json",
    },
}


def _clean_text_for_speech(text: str) -> str:
    """Strip <think> blocks, code blocks, and Markdown marks so Piper speaks naturally."""
    cleaned = _strip_think_blocks(text)
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\*+", "", cleaned)
    cleaned = re.sub(r"(?m)^#+\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _strip_think_blocks(text: str) -> str:
    """Remove complete or unterminated reasoning blocks and stray think tags."""
    cleaned = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", str(text), flags=re.IGNORECASE)
    return re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)


def scan_piper_voices() -> List[Dict[str, str]]:
    voices: Dict[str, str] = {"silas": "Silas", "kara": "Kara"}
    for directory in (PIPER_MODEL_DIR, RESOURCE_DIR / "piper_models"):
        if directory.is_dir():
            for f in sorted(directory.glob("*.onnx")):
                vid = f.stem.lower()
                if vid not in voices:
                    voices[vid] = vid.capitalize()
    return [{"id": k, "label": v} for k, v in voices.items()]


def _resolve_piper_model_path(voice_name: str) -> Path:
    safe_voice = Path(voice_name.strip().lower()).stem
    for path in (PIPER_MODEL_DIR / f"{safe_voice}.onnx", RESOURCE_DIR / "piper_models" / f"{safe_voice}.onnx"):
        if path.is_file():
            return path

    if safe_voice in DEFAULT_VOICE_URLS:
        PIPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        target_onnx = PIPER_MODEL_DIR / f"{safe_voice}.onnx"
        target_json = PIPER_MODEL_DIR / f"{safe_voice}.onnx.json"
        urls = DEFAULT_VOICE_URLS[safe_voice]
        log.info(f"Downloading Piper voice '{safe_voice}' from Hugging Face to {PIPER_MODEL_DIR} ...")
        partial_onnx = target_onnx.with_suffix(".onnx.part")
        try:
            urllib.request.urlretrieve(urls["json"], str(target_json))
            urllib.request.urlretrieve(urls["onnx"], str(partial_onnx))
            os.replace(partial_onnx, target_onnx)
        except Exception:
            for leftover in (partial_onnx, target_json):
                try:
                    leftover.unlink()
                except OSError:
                    pass
            raise
        return target_onnx

    raise FileNotFoundError(f"Piper voice model '{safe_voice}.onnx' not found in {PIPER_MODEL_DIR}")


def read_aloud(text: str, voice_name: str = "silas", play_locally: bool = True) -> bytes:
    """Synthesize speech with the local Piper engine and optionally play it on this machine."""
    cleaned_text = _clean_text_for_speech(text)
    if not cleaned_text:
        raise ValueError("No speakable text in message.")

    output_dir = Path(tempfile.gettempdir()) / "GHoSt-Terminal"
    output_dir.mkdir(parents=True, exist_ok=True)

    with _PIPER_LOCK:
        model_path = _resolve_piper_model_path(voice_name)
        wav_bytes: Optional[bytes] = None
        try:
            from piper import PiperVoice

            cache_key = str(model_path.resolve())
            voice = _PIPER_VOICES.get(cache_key)
            if voice is None:
                espeak_dir = RESOURCE_DIR / "piper" / "espeak-ng-data"
                if espeak_dir.is_dir():
                    voice = PiperVoice.load(model_path, espeak_data_dir=espeak_dir)
                else:
                    voice = PiperVoice.load(model_path)
                _PIPER_VOICES[cache_key] = voice

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                voice.synthesize_wav(cleaned_text, wav_file)
            wav_bytes = buf.getvalue()
        except Exception as exc:
            log.warning(f"In-process PiperVoice failed, trying piper CLI: {exc}")
            if not IS_FROZEN:
                with tempfile.NamedTemporaryFile(
                    prefix="speech-", suffix=".wav", dir=output_dir, delete=False
                ) as temp_audio:
                    output_audio = Path(temp_audio.name)
                try:
                    command = [sys.executable, "-m", "piper", "-m", str(model_path), "-f", str(output_audio)]
                    subprocess.run(
                        command,
                        input=cleaned_text,
                        text=True,
                        capture_output=True,
                        check=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                    if output_audio.is_file():
                        wav_bytes = output_audio.read_bytes()
                finally:
                    output_audio.unlink(missing_ok=True)

        if not wav_bytes:
            raise RuntimeError("Piper did not produce audio output. Is piper-tts installed?")

        if play_locally:
            try:
                import sounddevice as sd
                import soundfile as sf

                sd.stop()
                data, fs = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                sd.play(data, fs)
            except Exception as exc:
                log.warning(f"Local audio playback warning: {exc}")

        return wav_bytes


def stop_read_aloud() -> None:
    try:
        import sounddevice as sd

        sd.stop()
    except Exception:
        pass

# =====================================================================
# DIRECTORY SCAN — dynamic model discovery
# =====================================================================

def scan_models() -> List[Dict[str, Any]]:
    """Actively scan ./ghost_models for .gguf files."""
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    for f in sorted(MODEL_DIR.glob("*.gguf")):
        try:
            file_size_bytes = f.stat().st_size
        except OSError:
            file_size_bytes = 0
        models.append({
            "id": f.name,
            "object": "model",
            "created": int(f.stat().st_mtime) if f.exists() else 0,
            "owned_by": "GHoSt-Terminal-Local",
            "size_gb": round(file_size_bytes / (1024 ** 3), 2),
            "file_size_bytes": file_size_bytes,
            "file_size_gib": round(file_size_bytes / GIB, 3),
        })
    return models


def _gguf_value(reader: GGUFReader, key: str) -> Any:
    field = reader.fields.get(key)
    if field is None:
        return None
    value = field.contents()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _probe_nvidia_gpus() -> List[Dict[str, Any]]:
    executable = _find_nvidia_smi()
    if executable is None:
        return []

    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Unable to query NVIDIA VRAM with nvidia-smi: %s", exc)
        return []
    devices = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 3:
            continue
        try:
            devices.append({
                "name": row[0].strip(),
                "total_bytes": int(row[1].strip()) * 1024 ** 2,
                "free_bytes": int(row[2].strip()) * 1024 ** 2,
            })
        except ValueError:
            continue
    return devices


def preflight_model(filename: str, context_size: int, flash_attn: bool = True) -> dict:
    """Estimate a conservative GPU layer count from GGUF and live VRAM data."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = MODEL_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Model '{filename}' not found in ./ghost_models.")

    try:
        reader = GGUFReader(path)
        architecture = str(_gguf_value(reader, "general.architecture") or "unknown")
        layer_count = int(_gguf_value(reader, f"{architecture}.block_count") or 0)
        embedding_length = int(_gguf_value(reader, f"{architecture}.embedding_length") or 0)
        head_count = int(_gguf_value(reader, f"{architecture}.attention.head_count") or 0)
        kv_head_count = int(_gguf_value(reader, f"{architecture}.attention.head_count_kv") or 0)
        key_length = int(
            _gguf_value(reader, f"{architecture}.attention.key_length")
            or _gguf_value(reader, f"{architecture}.rope.dimension_count")
            or 0
        )
        value_length = int(_gguf_value(reader, f"{architecture}.attention.value_length") or key_length)
        sliding_window_value = _gguf_value(reader, f"{architecture}.attention.sliding_window")
        sliding_window = int(sliding_window_value) if sliding_window_value is not None else None
        if sliding_window is not None and sliding_window <= 0:
            sliding_window = None
        raw_model_context = _gguf_value(reader, f"{architecture}.context_length")
        model_context_length = int(raw_model_context) if raw_model_context is not None else None
        if model_context_length is not None and model_context_length <= 0:
            model_context_length = None
        conv_kernel = int(_gguf_value(reader, f"{architecture}.ssm.conv_kernel") or 0)
        ssm_inner_size = int(_gguf_value(reader, f"{architecture}.ssm.inner_size") or 0)
        ssm_state_size = int(_gguf_value(reader, f"{architecture}.ssm.state_size") or 0)
        ssm_group_count = int(_gguf_value(reader, f"{architecture}.ssm.group_count") or 0)
        file_type = _gguf_value(reader, "general.file_type")
        metadata_parameter_count = _gguf_value(reader, "general.parameter_count")
        tensor_parameter_count = sum(int(tensor.n_elements) for tensor in reader.tensors)
        parameter_count = int(metadata_parameter_count or tensor_parameter_count)
        weight_bytes = sum(int(tensor.n_bytes) for tensor in reader.tensors)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read GGUF metadata: {exc}") from exc

    if layer_count <= 0:
        raise HTTPException(status_code=422, detail="GGUF does not declare a valid block count.")
    if key_length <= 0 and head_count > 0 and embedding_length % head_count == 0:
        key_length = embedding_length // head_count
    if value_length <= 0:
        value_length = key_length
    kv_metadata_supported = kv_head_count > 0 and key_length > 0 and value_length > 0

    try:
        quantization = LlamaFileType(int(file_type)).name if file_type is not None else "unknown"
    except ValueError:
        quantization = f"unknown({file_type})"

    layer_weight_bytes = [0] * layer_count
    non_layer_bytes = 0
    key_layers = set()
    value_layers = set()
    fused_qkv_layers = set()
    ssm_layers = set()
    nextn_layers = set()
    ssm_conv1d_dim = 0
    has_ssm_state = any(key.startswith(f"{architecture}.ssm.") for key in reader.fields)
    for tensor in reader.tensors:
        parts = tensor.name.split(".")
        if len(parts) >= 3 and parts[0] == "blk":
            try:
                index = int(parts[1])
            except ValueError:
                non_layer_bytes += int(tensor.n_bytes)
                continue
            if 0 <= index < layer_count:
                layer_weight_bytes[index] += int(tensor.n_bytes)
                if parts[2] == "nextn":
                    nextn_layers.add(index)
                elif parts[2].startswith("ssm_"):
                    ssm_layers.add(index)
                    if parts[2] == "ssm_conv1d" and len(parts) >= 4 and parts[3] == "weight":
                        dims = [int(d) for d in getattr(tensor, "shape", ())]
                        if len(dims) == 2:
                            ssm_conv1d_dim = dims[1] if dims[0] == conv_kernel else max(dims)
                if len(parts) >= 4 and parts[2] == "attn_k" and parts[3] == "weight":
                    key_layers.add(index)
                if len(parts) >= 4 and parts[2] == "attn_v" and parts[3] == "weight":
                    value_layers.add(index)
                if len(parts) >= 4 and parts[2] == "attn_qkv" and parts[3] == "weight":
                    fused_qkv_layers.add(index)
            else:
                non_layer_bytes += int(tensor.n_bytes)
        else:
            non_layer_bytes += int(tensor.n_bytes)

    attention_layers = (key_layers & value_layers) - nextn_layers
    if not attention_layers and architecture in {"qwen2", "qwen3"}:
        attention_layers = set(range(layer_count)) - nextn_layers
    if not attention_layers and architecture == "phi3" and len(fused_qkv_layers) == layer_count:
        attention_layers = fused_qkv_layers - nextn_layers

    ssm_state_bytes = 0
    ssm_supported = True
    if has_ssm_state or ssm_layers:
        conv_dim = ssm_conv1d_dim or (ssm_inner_size + 2 * ssm_group_count * ssm_state_size)
        ssm_supported = bool(ssm_layers) and conv_kernel > 0 and conv_dim > 0 and ssm_inner_size > 0 and ssm_state_size > 0
        if ssm_supported:
            r_bytes_per_layer = max(0, conv_kernel - 1) * conv_dim * 4
            s_bytes_per_layer = ssm_inner_size * ssm_state_size * 4
            ssm_state_bytes = len(ssm_layers) * (r_bytes_per_layer + s_bytes_per_layer)

    covered_layers = attention_layers | ssm_layers | nextn_layers
    kv_supported = (
        sliding_window is None
        and len(covered_layers) == layer_count
        and (not attention_layers or kv_metadata_supported)
        and (not (has_ssm_state or ssm_layers) or ssm_supported)
        and (bool(attention_layers) or bool(ssm_layers))
    )

    kv_f16_bytes = 0
    kv_q8_bytes = 0
    kv_method = "unsupported_cpu_fallback"
    if kv_supported:
        attn_f16_bytes = (
            len(attention_layers)
            * context_size
            * kv_head_count
            * (key_length + value_length)
            * 2  # FP16 uses 2 bytes per K or V element.
        ) if attention_layers else 0
        # Q8_0 stores 32 8-bit values plus a 16-bit scale: 34 bytes per 64 FP16 bytes.
        attn_q8_bytes = (attn_f16_bytes * 34 + 63) // 64
        kv_f16_bytes = attn_f16_bytes + ssm_state_bytes
        kv_q8_bytes = attn_q8_bytes + ssm_state_bytes
        kv_method = "hybrid_ssm_fp16" if ssm_layers else "dense_gqa_fp16"
    compute_reserve_bytes = int(0.5 * GIB + 0.04 * weight_bytes)
    devices = _probe_nvidia_gpus()
    warnings = []
    if model_context_length is not None and context_size > model_context_length:
        warnings.append(
            f"Requested context ({context_size:,}) exceeds this GGUF's declared context_length ({model_context_length:,})."
        )
    gpu_layers = 0
    gpu_weight_bytes = 0
    kv_cache_bytes = kv_f16_bytes
    kv_type = int(llama_cpp.GGML_TYPE_F16)
    kv_cache_type = "F16"
    usable_bytes = 0
    gpu_name = devices[0]["name"] if len(devices) == 1 else None
    total_bytes = devices[0]["total_bytes"] if len(devices) == 1 else 0
    free_bytes = devices[0]["free_bytes"] if len(devices) == 1 else 0

    if BACKEND_NAME != "cuda":
        if devices:
            warnings.append(
                f"{BACKEND_NAME.upper()} backend active; GPU layer offload is disabled. "
                "Rebuild from the CUDA .venv to enable NVIDIA GPU offload."
            )
        else:
            warnings.append(f"{BACKEND_NAME.upper()} backend active; GPU layer offload is disabled.")
    elif len(devices) == 1:
        usable_bytes = min(
            int(total_bytes * GPU_MAX_UTILIZATION),
            max(0, free_bytes - GPU_SAFETY_RESERVE_BYTES),
        )
        if not kv_supported:
            if not kv_metadata_supported or not (attention_layers or ssm_layers):
                warnings.append("GGUF attention metadata is incomplete; GPU offload disabled for safety.")
            elif sliding_window is not None:
                warnings.append("Sliding-window KV allocation is not modeled; GPU offload disabled for safety.")
            elif (has_ssm_state or ssm_layers) and not ssm_supported:
                warnings.append("SSM/recurrent metadata is incomplete; GPU offload disabled for safety.")
            else:
                warnings.append("KV cache layout is not supported by preflight; GPU offload disabled for safety.")
        else:
            def fitting_layers(kv_bytes: int) -> tuple[int, int]:
                available = usable_bytes - kv_bytes - compute_reserve_bytes - non_layer_bytes
                for candidate in range(layer_count, 0, -1):
                    candidate_bytes = sum(layer_weight_bytes[layer_count - candidate:])
                    if candidate_bytes <= available:
                        return candidate, candidate_bytes
                return 0, 0

            f16_layers, f16_weight_bytes = fitting_layers(kv_f16_bytes)
            q8_layers, q8_weight_bytes = fitting_layers(kv_q8_bytes) if flash_attn else (0, 0)
            if flash_attn and q8_layers > f16_layers and attention_layers:
                kv_cache_bytes = kv_q8_bytes
                kv_type = int(llama_cpp.GGML_TYPE_Q8_0)
                kv_cache_type = "Q8_0"
                kv_method = "hybrid_ssm_q8_0" if ssm_layers else "dense_gqa_q8_0"
                gpu_layers = q8_layers
                gpu_weight_bytes = q8_weight_bytes
                warnings.append("KV cache reduced from FP16 to Q8_0 before reducing GPU layer count.")
            else:
                gpu_layers = f16_layers
                gpu_weight_bytes = f16_weight_bytes
            if gpu_layers == 0:
                warnings.append("Model weights, KV estimate, overhead, and safety reserve leave no layer budget for GPU offload.")
    elif not devices:
        warnings.append("nvidia-smi did not report a GPU; selecting CPU-only loading.")
    else:
        warnings.append("Multiple GPUs are detected, but this Phase One estimator only schedules a single GPU; selecting CPU-only loading.")

    predicted_gpu_bytes = 0
    if gpu_layers:
        predicted_gpu_bytes = non_layer_bytes + gpu_weight_bytes + kv_cache_bytes + compute_reserve_bytes

    return {
        "model": filename,
        "backend": BACKEND_NAME,
        "flash_attn": bool(flash_attn and gpu_layers > 0),
        "architecture": architecture,
        "quantization": quantization,
        "parameter_count": parameter_count,
        "weight_bytes": weight_bytes,
        "weight_gib": round(weight_bytes / GIB, 3),
        "context_size": context_size,
        "model_context_length": model_context_length,
        "total_layers": layer_count,
        "attention_layers": len(attention_layers),
        "ssm_layers": len(ssm_layers),
        "n_gpu_layers": gpu_layers,
        "kv_cache_type": kv_cache_type,
        "kv_type": kv_type,
        "kv_method": kv_method,
        "gpu_name": gpu_name,
        "gpu_total_bytes": total_bytes,
        "gpu_free_bytes": free_bytes,
        "usable_vram_bytes": usable_bytes,
        "kv_cache_estimate_bytes": kv_cache_bytes,
        "compute_reserve_bytes": compute_reserve_bytes,
        "non_layer_weight_bytes": non_layer_bytes,
        "gpu_layer_weight_bytes": gpu_weight_bytes,
        "predicted_gpu_bytes": predicted_gpu_bytes,
        "warnings": warnings,
        "estimate_note": "Tensor bytes are read from the GGUF index. Dense KV and hybrid SSM/recurrent state use conservative upper-bound estimates. Runtime allocations can differ; the safety reserve is applied.",
    }

# =====================================================================
# LIFECYCLE — hard unload / load
# =====================================================================

def unload_current_model() -> None:
    """Completely flush the resident model to prevent VRAM fragmentation / OOM."""
    if STATE.llm is not None:
        log.info(f"Unloading resident model '{STATE.loaded_model}' ...")
        try:
            del STATE.llm  # drop reference; llama_cpp frees on __del__
        except Exception as e:
            log.warning(f"Exception during model deletion: {e}")
        STATE.llm = None
        STATE.loaded_model = None
        STATE.loaded_at = None
        STATE.loaded_gpu_layers = None
        STATE.loaded_context_size = None
        STATE.loaded_kv_cache_type = None
        STATE.loaded_chat_format = None
        STATE.loaded_flash_attn = None
        STATE.prompt_opens_think = False
        STATE._think_handlers = {}

        # Aggressive flush: GC cycles + allocator release
        for _ in range(3):
            gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)  # best-effort Linux malloc trim
        except Exception:
            pass  # non-Linux or already trimmed
        time.sleep(0.5)
        log.info("Resident model flushed. VRAM released.")


def _build_think_handlers(llm: Llama) -> tuple[Dict[bool, Any], bool]:
    """For Jinja templates with enable_thinking (e.g. Qwen3.5), create think-on/off handlers."""
    try:
        from llama_cpp import llama_chat_format as lcf
    except Exception:
        return {}, False
    template = llm.metadata.get("tokenizer.chat_template")
    if not template or "enable_thinking" not in template or llm.chat_format != "chat_template.default":
        return {}, False

    eos_id = llm.token_eos()
    bos_id = llm.token_bos()
    eos_token = llm._model.token_get_text(eos_id) if eos_id != -1 else ""
    bos_token = llm._model.token_get_text(bos_id) if bos_id != -1 else ""

    class _ToggleThinkingFormatter(lcf.Jinja2ChatFormatter):
        def __init__(self, *args, enable_thinking: bool, **kwargs):
            super().__init__(*args, **kwargs)
            self.enable_thinking = enable_thinking

        def __call__(self, **kwargs):
            kwargs.setdefault("enable_thinking", self.enable_thinking)
            return super().__call__(**kwargs)

    handlers = {}
    prompt_opens_think = False
    for enabled in (False, True):
        formatter = _ToggleThinkingFormatter(
            template=template,
            eos_token=eos_token,
            bos_token=bos_token,
            stop_token_ids=[eos_id] if eos_id != -1 else None,
            enable_thinking=enabled,
        )
        if enabled:
            try:
                sample = formatter(messages=[{"role": "user", "content": "hi"}]).prompt
                prompt_opens_think = sample.rstrip().endswith("<think>")
            except Exception:
                prompt_opens_think = False
        handlers[enabled] = formatter.to_chat_handler()
    return handlers, prompt_opens_think


def _apply_thinking_mode(show_reasoning: bool) -> bool:
    """Switch Jinja enable_thinking if supported; returns whether prompt pre-opened <think>."""
    if STATE.llm is not None and STATE._think_handlers:
        handler = STATE._think_handlers.get(bool(show_reasoning))
        if handler is not None:
            STATE.llm._chat_handlers["chat_template.default"] = handler
            return bool(show_reasoning and STATE.prompt_opens_think)
    return False


def load_model(
    filename: str,
    context_size: int,
    chat_format: str = "auto",
    flash_attn: bool = True,
) -> tuple[dict, float]:
    """Unload any resident model, preflight the candidate, then load its fit."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = MODEL_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Model '{filename}' not found in ./ghost_models.")
    if not STATE.begin_operation():
        raise HTTPException(status_code=409, detail="A model change is already in progress.")

    try:
        unload_current_model()
        preflight = preflight_model(filename, context_size, flash_attn=flash_attn)
        gpu_layers = int(preflight["n_gpu_layers"])
        llama_gpu_layers = -1 if gpu_layers >= int(preflight["total_layers"]) > 0 else gpu_layers
        use_flash_attn = bool(flash_attn and gpu_layers > 0)
        log.info(
            f"Loading '{filename}' (preflight n_gpu_layers={gpu_layers}, n_ctx={context_size}, flash_attn={use_flash_attn}) ..."
        )
        t0 = time.perf_counter()
        try:
            STATE.llm = Llama(
                model_path=str(path),
                n_gpu_layers=llama_gpu_layers,
                n_ctx=context_size,
                n_threads=N_THREADS,
                n_batch=N_BATCH,
                type_k=int(preflight["kv_type"]),
                type_v=int(preflight["kv_type"]),
                offload_kqv=gpu_layers > 0,
                flash_attn=use_flash_attn,
                chat_format=None if chat_format == "auto" else chat_format,
                verbose=False,
            )
        except Exception as exc:
            log.error(f"Load failed: {exc}")
            log.error(traceback.format_exc())
            unload_current_model()
            raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc

        dt = time.perf_counter() - t0
        STATE.loaded_model = filename
        STATE.loaded_at = time.time()
        STATE.load_seconds = dt
        STATE.loaded_gpu_layers = gpu_layers
        STATE.loaded_context_size = context_size
        STATE.loaded_kv_cache_type = preflight["kv_cache_type"]
        STATE.loaded_chat_format = chat_format
        STATE.loaded_flash_attn = use_flash_attn
        STATE._think_handlers, STATE.prompt_opens_think = _build_think_handlers(STATE.llm)
        log.info(f"Model '{filename}' resident in VRAM. Load time: {dt:.2f}s")
        return preflight, dt
    finally:
        STATE.end_operation()

# =====================================================================
# API SCHEMA — exact OpenAI-compatible request/response
# =====================================================================

class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: StrictStr
    content: Any = None
    name: Optional[StrictStr] = None
    tool_call_id: Optional[StrictStr] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None


class JsonSchemaSpec(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: StrictStr = Field(min_length=1, max_length=128)
    schema_definition: Dict[str, Any] = Field(alias="schema")
    description: Optional[StrictStr] = None
    strict: Optional[bool] = None


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaSpec] = None

    @model_validator(mode="after")
    def require_json_schema_payload(self):
        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError("response_format.json_schema is required for json_schema output")
        return self


class ToolFunctionSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: StrictStr = Field(min_length=1, max_length=128)
    description: Optional[StrictStr] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function"]
    function: ToolFunctionSpec

ChatFormat = Literal[
    "auto", "llama-2", "llama-3", "alpaca", "qwen", "vicuna", "oasst_llama",
    "baichuan-2", "baichuan", "openbuddy", "redpajama-incite", "snoozy",
    "phind", "intel", "open-orca", "mistrallite", "zephyr", "pygmalion",
    "chatml", "mistral-instruct", "chatglm3", "openchat", "saiga", "gemma",
]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: StrictStr
    messages: List[Message]
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[StrictInt] = Field(default=None, ge=0, le=200)
    repeat_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    max_tokens: Optional[StrictInt] = Field(default=None, ge=1, le=MAX_N_CTX)
    max_completion_tokens: Optional[StrictInt] = Field(default=None, ge=1, le=MAX_N_CTX)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[StrictInt] = None
    show_reasoning: Optional[bool] = None
    response_format: Optional[ResponseFormat] = None
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], Dict[str, Any]]] = None


class LoadRequest(BaseModel):
    model: StrictStr = Field(min_length=1, max_length=255)
    n_ctx: StrictInt = Field(default=DEFAULT_N_CTX, ge=MIN_N_CTX, le=MAX_N_CTX)
    chat_format: ChatFormat = "auto"
    flash_attn: bool = True


class PreflightRequest(BaseModel):
    model: StrictStr = Field(min_length=1, max_length=255)
    n_ctx: StrictInt = Field(default=DEFAULT_N_CTX, ge=MIN_N_CTX, le=MAX_N_CTX)
    flash_attn: bool = True


class TTSRequest(BaseModel):
    text: StrictStr = Field(min_length=1, max_length=20000)
    voice: StrictStr = Field(default="silas", max_length=64)
    play_on_host: Optional[bool] = True


class LoaderSettings(BaseModel):
    model: StrictStr = Field(default=DEFAULT_SETTINGS["model"], min_length=1, max_length=255)
    chat_format: ChatFormat = "auto"
    system_prompt: StrictStr = Field(default=DEFAULT_SETTINGS["system_prompt"], max_length=12000)
    tts_voice: StrictStr = Field(default="silas", min_length=1, max_length=64)
    flash_attn: bool = True
    show_reasoning: bool = False
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    min_p: float = Field(default=0.05, ge=0.0, le=1.0)
    top_k: StrictInt = Field(default=40, ge=0, le=200)
    repeat_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    seed: Optional[StrictInt] = None
    max_tokens: StrictInt = Field(default=512, ge=1, le=MAX_UI_MAX_TOKENS)
    stop_sequences: StrictStr = Field(default="", max_length=2000)
    n_ctx: StrictInt = Field(default=DEFAULT_N_CTX, ge=MIN_N_CTX, le=MAX_N_CTX)


def load_settings() -> dict:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return LoaderSettings.model_validate({**DEFAULT_SETTINGS, **data}).model_dump()
    except FileNotFoundError:
        return dict(DEFAULT_SETTINGS)
    except Exception as exc:
        log.warning("Ignoring invalid loader settings: %s", exc)
        return dict(DEFAULT_SETTINGS)


def save_settings(settings: LoaderSettings) -> dict:
    data = settings.model_dump()
    temporary_path = SETTINGS_PATH.with_name(SETTINGS_PATH.name + ".tmp")
    temporary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary_path, SETTINGS_PATH)
    return data


app = FastAPI(title="GHoSt Terminal", version="1.0.2", docs_url=None, redoc_url=None)


@app.get("/")
def loader_page():
    if not HTML_PATH.is_file():
        raise HTTPException(status_code=500, detail=f"Terminal UI not found: {HTML_PATH}")
    return FileResponse(HTML_PATH, media_type="text/html")


@app.get("/assets/{filename}")
def loader_asset(filename: str):
    safe_name = Path(filename).name
    path = ASSET_DIR / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found.")
    return FileResponse(path)


@app.get("/v1/settings")
def get_settings():
    return load_settings()


@app.post("/v1/settings")
def update_settings(settings: LoaderSettings):
    return save_settings(settings)


@app.get("/v1/models")
def list_models():
    models = scan_models()
    log.info(f"GET /v1/models -> {len(models)} GGUF file(s) detected.")
    return {"object": "list", "data": models}


@app.post("/v1/preflight")
def preflight_endpoint(req: PreflightRequest):
    if STATE.locked:
        raise HTTPException(status_code=409, detail="A model change is in progress.")
    plan = preflight_model(req.model, req.n_ctx, flash_attn=req.flash_attn)
    if STATE.loaded_model:
        plan["warnings"].append(
            "A model is currently resident; this preview uses current free VRAM. "
            "Loading will unload it and recalculate before allocation."
        )
    return plan


@app.post("/v1/load")
def load_endpoint(req: LoadRequest):
    preflight, load_seconds = load_model(req.model, req.n_ctx, req.chat_format, flash_attn=req.flash_attn)
    return {
        "status": "loaded",
        "model": req.model,
        "load_seconds": round(load_seconds, 2),
        "gpu_layers": preflight["n_gpu_layers"],
        "context": req.n_ctx,
        "kv_cache_type": preflight["kv_cache_type"],
        "chat_format": req.chat_format,
        "flash_attn": preflight["flash_attn"],
        "preflight": preflight,
    }


@app.post("/v1/unload")
def unload_endpoint():
    if not STATE.begin_operation():
        raise HTTPException(
            status_code=409,
            detail="A model change or generation is in progress.",
        )
    try:
        unload_current_model()
    finally:
        STATE.end_operation()
    return {"status": "unloaded"}


@app.post("/v1/reset")
def reset_endpoint():
    if not STATE.begin_operation():
        raise HTTPException(
            status_code=409,
            detail="Wait for the current model operation to finish.",
        )
    try:
        if STATE.llm is None:
            raise HTTPException(status_code=409, detail="No model is loaded.")
        STATE.llm.reset()
        return {"status": "reset", "model": STATE.loaded_model}
    finally:
        STATE.end_operation()


@app.get("/v1/status")
def status():
    return {
        "engine": "GHoSt-Terminal",
        "backend": BACKEND_NAME,
        "resident_model": STATE.loaded_model,
        "n_gpu_layers": STATE.loaded_gpu_layers,
        "n_ctx": STATE.loaded_context_size,
        "kv_cache_type": STATE.loaded_kv_cache_type,
        "chat_format": STATE.loaded_chat_format,
        "flash_attn": STATE.loaded_flash_attn,
        "loading": STATE.locked,
        "generating": STATE.busy,
        "models_available": len(scan_models()),
        "offline": True,
    }

app.add_middleware(
    CORSMiddleware,
    # The bundled UI is served from this exact origin.
    allow_origins=[f"http://{HOST}:{PORT}"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class LocalOriginMiddleware:
    """Reject cross-origin browser requests, including simple requests without preflight."""

    def __init__(self, app, allowed_origin: str):
        self.app = app
        self.allowed_origin = allowed_origin

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = scope.get("headers", [])
            origins = [value for name, value in headers if name.lower() == b"origin"]
            fetch_sites = [value.lower() for name, value in headers if name.lower() == b"sec-fetch-site"]
            if any(origin != self.allowed_origin.encode("ascii") for origin in origins) or b"cross-site" in fetch_sites:
                response = Response("Cross-origin request rejected.", status_code=403, media_type="text/plain")
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


app.add_middleware(LocalOriginMiddleware, allowed_origin=f"http://{HOST}:{PORT}")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=[HOST])

# =====================================================================
# READ-ALOUD ENDPOINTS
# =====================================================================

@app.get("/v1/tts/voices")
def tts_voices_endpoint():
    return {"voices": scan_piper_voices()}


@app.post("/v1/tts")
def tts_endpoint(req: TTSRequest):
    # The server is bound to loopback, so audio always plays on this machine.
    try:
        wav_bytes = read_aloud(req.text, req.voice, play_locally=bool(req.play_on_host))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.error(f"TTS error: {exc}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store", "X-Played-On-Host": "true" if req.play_on_host else "false"},
    )


@app.post("/v1/tts/stop")
def tts_stop_endpoint():
    stop_read_aloud()
    return {"status": "stopped"}


# =====================================================================
# CHAT COMPLETIONS — streaming (abortable) + non-streaming
# =====================================================================

def _prepare_messages(req: ChatRequest) -> tuple[List[Dict[str, Any]], bool, bool]:
    settings = load_settings()
    show_reasoning = req.show_reasoning if req.show_reasoning is not None else bool(settings.get("show_reasoning", False))
    messages = [message.model_dump(exclude_none=True) for message in req.messages]
    prepend_open_think = _apply_thinking_mode(show_reasoning)

    # Classic Qwen3 models without Jinja enable_thinking accept /no_think to stay direct.
    if not STATE._think_handlers and STATE.loaded_model and "qwen3" in STATE.loaded_model.lower() and not show_reasoning:
        for msg in messages:
            if msg.get("role") in ("system", "user") and isinstance(msg.get("content"), str):
                if "/no_think" not in msg["content"]:
                    msg["content"] = msg["content"].rstrip() + "\n/no_think"
    return messages, show_reasoning, prepend_open_think


def _chat_completion_args(req: ChatRequest, messages: List[Dict[str, Any]]) -> dict:
    settings = load_settings()

    def pick(name: str, fallback: Any) -> Any:
        value = getattr(req, name)
        if name in req.model_fields_set and value is not None:
            return value
        stored = settings.get(name)
        return stored if stored is not None else fallback

    stop = req.stop
    if stop is None:
        raw_stop = str(settings.get("stop_sequences") or "").strip()
        if raw_stop:
            stop = [line.strip() for line in raw_stop.splitlines() if line.strip()]

    chosen_seed = pick("seed", None)
    if chosen_seed is None:
        chosen_seed = secrets.randbelow(2**31 - 1) + 1

    resolved_max_tokens = req.max_tokens
    if resolved_max_tokens is None and req.max_completion_tokens is not None:
        resolved_max_tokens = req.max_completion_tokens
    if resolved_max_tokens is None:
        stored_max_tokens = settings.get("max_tokens")
        resolved_max_tokens = stored_max_tokens if stored_max_tokens is not None else 512

    args = {
        "messages": messages,
        "max_tokens": int(resolved_max_tokens),
        "temperature": float(pick("temperature", 0.7)),
        "top_p": float(pick("top_p", 0.95)),
        "min_p": float(pick("min_p", 0.05)),
        "top_k": int(pick("top_k", 40)),
        "repeat_penalty": float(pick("repeat_penalty", 1.0)),
        "presence_penalty": float(pick("presence_penalty", 0.0)),
        "stop": stop,
        "seed": chosen_seed,
    }
    if req.response_format is not None:
        args["response_format"] = req.response_format.model_dump(by_alias=True, exclude_none=True)
    if req.tools is not None:
        args["tools"] = [tool.model_dump(exclude_none=True) for tool in req.tools]
    if req.tool_choice is not None:
        args["tool_choice"] = req.tool_choice
    return args


class _ThinkGate:
    """Hides <think>...</think> spans from a token stream unless reasoning is enabled."""
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, show_reasoning: bool):
        self.show_reasoning = show_reasoning
        self.inside = False
        self.pending = ""

    @staticmethod
    def _partial_tail(text: str, tag: str) -> int:
        for size in range(min(len(text), len(tag) - 1), 0, -1):
            if text.endswith(tag[:size]):
                return size
        return 0

    def feed(self, text: str) -> str:
        if self.show_reasoning or not text:
            return text
        buf = self.pending + text
        lowered = buf.lower()
        out: List[str] = []
        cursor = 0
        self.pending = ""
        while cursor < len(buf):
            tag = self.CLOSE if self.inside else self.OPEN
            hit = lowered.find(tag, cursor)
            if hit < 0:
                tail = buf[cursor:]
                hold = self._partial_tail(tail.lower(), tag)
                if not self.inside:
                    out.append(tail[:-hold] if hold else tail)
                self.pending = tail[-hold:] if hold else ""
                break
            if not self.inside:
                out.append(buf[cursor:hit])
            cursor = hit + len(tag)
            self.inside = not self.inside
        return "".join(out)

    def flush(self) -> str:
        remainder = "" if self.inside else self.pending
        self.pending = ""
        return remainder


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _stream_generator(
    req: ChatRequest,
    messages: List[Dict[str, Any]],
    generation_id: int,
    show_reasoning: bool,
    prepend_open_think: bool = False,
):
    start = time.perf_counter()
    token_count = 0
    aborted = False
    failed = False
    stream = None
    think_gate = _ThinkGate(show_reasoning)
    opened_think = False

    def delta_chunk(content: str = "", finish: Optional[str] = None, **extra) -> str:
        choice: Dict[str, Any] = {"index": 0, "delta": {"content": content} if content else {}}
        if finish:
            choice["finish_reason"] = finish
        return _sse({
            "id": f"chatcmpl-{generation_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [choice],
            **extra,
        })

    try:
        stream_args = _chat_completion_args(req, messages)
        stream_args["stream"] = True
        llm = STATE.llm
        if llm is None:
            raise RuntimeError("No model resident.")
        stream = llm.create_chat_completion(**stream_args)
        for chunk in stream:
            if STATE.abort_requested:
                aborted = True
                break
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                delta.pop("reasoning_content", None)
                if isinstance(content, str) and content:
                    if prepend_open_think and not opened_think:
                        opened_think = True
                        if not content.lstrip().lower().startswith("<think>"):
                            content = "<think>\n" + content
                    filtered = think_gate.feed(content)
                    if filtered:
                        delta["content"] = filtered
                        token_count += 1
                    else:
                        delta.pop("content", None)
                if not delta and choices[0].get("finish_reason") is None:
                    continue
            yield _sse(chunk)

        residual = think_gate.flush()
        if residual:
            yield delta_chunk(residual)
        if aborted:
            yield delta_chunk(finish="stop", ghost_aborted=True)
        yield "data: [DONE]\n\n"
    except Exception as exc:
        failed = True
        log.error(f"Streaming generation error: {exc}")
        yield _sse({"error": {"message": str(exc)}})
        yield "data: [DONE]\n\n"
    finally:
        if stream is not None:
            closer = getattr(stream, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as exc:
                    log.warning(f"Could not close generation stream: {exc}")
        STATE.end_generation(generation_id)
        elapsed = time.perf_counter() - start
        rate = token_count / elapsed if elapsed > 0 else 0.0
        outcome = "ABORTED" if aborted else ("FAILED" if failed else "DONE")
        log.info(
            f"STREAM {outcome} :: gen={generation_id} model={req.model} chunks={token_count} "
            f"time={elapsed:.2f}s speed={rate:.2f} chunk/s"
        )


@app.post("/v1/abort")
def abort_generation():
    active = STATE.request_abort()
    streaming = bool(active and STATE.streaming)
    log.info(f"ABORT REQUEST :: gen={STATE.generation_id} active={active} streaming={streaming}")
    return {
        "status": "aborting" if active else "idle",
        "generation_id": STATE.generation_id,
        "interruptible": streaming,
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if STATE.llm is None:
        raise HTTPException(status_code=409, detail="No model resident. Open CONFIG and click MOUNT first.")

    generation_id = STATE.begin_generation(bool(req.stream))
    try:
        messages, show_reasoning, prepend_open_think = _prepare_messages(req)
    except Exception:
        STATE.end_generation(generation_id)
        raise
    log.info(f"REQUEST :: gen={generation_id} model={req.model} stream={req.stream} msgs={len(messages)}")

    if req.stream:
        return StreamingResponse(
            _stream_generator(req, messages, generation_id, show_reasoning, prepend_open_think),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            background=BackgroundTask(STATE.end_generation, generation_id),
        )

    # ---- Non-streaming ----
    start = time.perf_counter()
    try:
        completion_args = _chat_completion_args(req, messages)
        completion_args["stream"] = False
        llm = STATE.llm
        if llm is None:
            raise RuntimeError("No model resident.")
        out = llm.create_chat_completion(**completion_args)
        try:
            message = out["choices"][0]["message"]
            if not show_reasoning:
                message.pop("reasoning_content", None)
            content = message.get("content")
            if isinstance(content, str):
                if not show_reasoning:
                    out["choices"][0]["message"]["content"] = _strip_think_blocks(content).strip()
                elif prepend_open_think and not content.lstrip().lower().startswith("<think>"):
                    out["choices"][0]["message"]["content"] = "<think>\n" + content
        except (KeyError, IndexError, TypeError):
            pass
    except Exception as e:
        log.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        STATE.end_generation(generation_id)

    elapsed = time.perf_counter() - start
    usage = out.get("usage", {})
    tps = usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0.0
    log.info(
        f"COMPLETION :: model={req.model} prompt_tok={usage.get('prompt_tokens', '?')} "
        f"completion_tok={usage.get('completion_tokens', '?')} time={elapsed:.2f}s speed={tps:.2f} tok/s"
    )
    return out

# =====================================================================
# STARTUP
# =====================================================================

def startup_banner():
    if sys.stdout is not None:
        print(GHOST_BANNER)
    log.info("GHoSt Terminal boot sequence initiated.")
    log.info(f"Model directory : {MODEL_DIR.resolve()}")
    log.info(f"Settings file   : {SETTINGS_PATH.resolve()}")
    log.info(f"Selected backend: {BACKEND_NAME} ({RUNTIME_LLAMA_LIB_DIR})")
    log.info(f"Bind address    : http://{HOST}:{PORT} (loopback-only, zero telemetry)")
    detected = scan_models()
    log.info(f"Default context : {DEFAULT_N_CTX}; GPU layers are selected by preflight")
    log.info(f"GGUF models detected: {len(detected)}")
    for m in detected:
        log.info(f"  -> {m['id']} ({m['size_gb']} GB)")
    if not detected:
        log.warning("No GGUF files found. Drop .gguf files into ./ghost_models and reload.")
    log.info("GHoSt Terminal online. Awaiting lifecycle commands.")


@app.on_event("startup")
def on_startup():
    startup_banner()


@app.on_event("shutdown")
def on_shutdown():
    log.info("Shutdown signal received. Flushing VRAM ...")
    unload_current_model()
    log.info("GHoSt Terminal terminated cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GHoSt Terminal - GPU Hosted Static Terminal")
    parser.add_argument("--api-only", action="store_true", help="Run the API without a desktop window")
    args = parser.parse_args()

    try:
        if args.api_only:
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=HOST,
                    port=PORT,
                    log_level="warning",
                    access_log=False,
                    log_config=None,
                )
            )
            server.run(sockets=[SERVER_SOCKET])
        else:
            import webview

            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=HOST,
                    port=PORT,
                    log_level="warning",
                    access_log=False,
                    log_config=None,
                )
            )
            server_thread = threading.Thread(target=server.run, kwargs={"sockets": [SERVER_SOCKET]}, daemon=True)
            server_thread.start()
            deadline = time.monotonic() + 15
            while not server.started and server_thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not server.started:
                server.should_exit = True
                server_thread.join(timeout=5)
                raise RuntimeError(f"GHoSt Terminal API did not start on {HOST}:{PORT}.")

            try:
                webview.create_window(
                    "GHoSt Terminal",
                    f"http://{HOST}:{PORT}/",
                    width=1440,
                    height=960,
                    min_size=(900, 640),
                    background_color="#060c12",
                    resizable=True,
                )
                webview.start()
            finally:
                server.should_exit = True
                server_thread.join(timeout=10)
    except Exception as exc:
        log.exception("GHoSt Terminal failed to start")
        message = f"GHoSt Terminal could not start.\n\n{exc}\n\nDetails: {LOG_PATH}"
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "GHoSt Terminal startup error", 0x10)
        else:
            print(message, file=sys.stderr)
        raise
    finally:
        if SERVER_SOCKET is not None:
            SERVER_SOCKET.close()
