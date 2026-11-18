"""Focused regression tests that do not load a model or initialize native backends."""
import ast
import asyncio
import io
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import wave
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from unittest import mock

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError, model_validator

import build_terminal
import package_release


def load_server_constants(*names):
    source = Path(__file__).resolve().parents[1] / "ghost_terminal.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return {name: namespace[name] for name in names}


def load_server_parts(*names, extra=None):
    source = Path(__file__).resolve().parents[1] / "ghost_terminal.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    wanted = set(names)
    nodes = [
        node for node in tree.body
        if (
            isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
        ) or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in wanted for target in node.targets)
        )
    ]
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node.decorator_list = []
    namespace = {
        "Any": Any, "Dict": Dict, "List": List, "Literal": Literal, "Optional": Optional, "Union": Union,
        "BaseModel": BaseModel, "ConfigDict": ConfigDict, "Field": Field,
        "StrictInt": StrictInt, "StrictStr": StrictStr, "model_validator": model_validator,
        "Path": Path, "io": io, "log": logging.getLogger("tests"), "os": os,
        "re": re, "subprocess": subprocess, "sys": sys, "tempfile": tempfile,
        "threading": threading, "wave": wave,
    }
    namespace.update(extra or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class HTTPError(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        super().__init__(detail)


class RegressionTests(unittest.TestCase):
    def test_reasoning_filter(self):
        ns = load_server_parts("_strip_think_blocks", "_ThinkGate")
        strip = ns["_strip_think_blocks"]
        for source, expected in [
            ("<think>secret</think>", ""),
            ("before <THINK>secret</THINK> after", "before  after"),
            ("before <think>unfinished", "before "),
        ]:
            with self.subTest(source=source):
                self.assertEqual(strip(source), expected)
        gate = ns["_ThinkGate"](False)
        self.assertEqual("".join(gate.feed(part) for part in ["hi <th", "ink>secret</th", "ink> there"]) + gate.flush(), "hi  there")

    def test_generation_and_model_operations_exclude_one_another(self):
        ns = load_server_parts("EngineState", extra={"Llama": object, "HTTPException": HTTPError})
        state = ns["EngineState"]()
        state.llm = object()
        self.assertTrue(state.begin_operation())
        self.assertFalse(state.begin_operation())
        with self.assertRaises(HTTPError):
            state.begin_generation(True)
        state.end_operation()
        generation = state.begin_generation(False)
        self.assertFalse(state.begin_operation())
        state.end_generation(generation)
        self.assertTrue(state.begin_operation())
        state.end_operation()

    def test_non_streaming_think_only_response_is_empty(self):
        state = types.SimpleNamespace(
            llm=types.SimpleNamespace(create_chat_completion=lambda **kwargs: {
                "choices": [{"message": {"content": "<think>secret</think>", "reasoning_content": "secret"}}],
                "usage": {},
            }),
            begin_generation=lambda streaming: 1,
            end_generation=lambda generation_id: None,
        )
        ns = load_server_parts("_strip_think_blocks", "chat_completions", extra={
            "STATE": state, "ChatRequest": object, "HTTPException": HTTPError, "time": time,
            "_prepare_messages": lambda req: ([], False, False),
            "_chat_completion_args": lambda req, messages: {},
        })
        result = ns["chat_completions"](types.SimpleNamespace(stream=False, model="test"))
        self.assertEqual(result["choices"][0]["message"], {"content": ""})

    def test_port_is_reserved_until_socket_is_closed(self):
        ns = load_server_parts("_reserve_server_socket", extra={"socket": socket})
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            port = occupied.getsockname()[1]
            with mock.patch.dict(os.environ, {"GHOST_API_PORT": str(port)}):
                with self.assertRaises(OSError):
                    ns["_reserve_server_socket"]("127.0.0.1")
        with mock.patch.dict(os.environ, {"GHOST_API_PORT": str(port)}):
            chosen, listener = ns["_reserve_server_socket"]("127.0.0.1")
            try:
                self.assertEqual(chosen, port)
                with socket.socket() as other:
                    with self.assertRaises(OSError):
                        other.bind(("127.0.0.1", port))
            finally:
                listener.close()

    def test_failed_tts_cli_does_not_return_stale_audio(self):
        ns = load_server_parts("read_aloud", extra={
            "_clean_text_for_speech": lambda text: text,
            "_resolve_piper_model_path": lambda voice: Path("voice.onnx"),
            "_PIPER_LOCK": threading.Lock(), "_PIPER_VOICES": {},
            "RESOURCE_DIR": Path("."), "IS_FROZEN": False,
        })
        piper = types.ModuleType("piper")
        piper.PiperVoice = types.SimpleNamespace(load=mock.Mock(side_effect=RuntimeError("failure")))
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "GHoSt-Terminal"
            output_dir.mkdir()
            (output_dir / "current_speech.wav").write_bytes(b"old audio")
            with mock.patch.dict(sys.modules, {"piper": piper}), \
                 mock.patch.object(tempfile, "gettempdir", return_value=directory), \
                 mock.patch.object(subprocess, "run", side_effect=subprocess.CalledProcessError(1, "piper")):
                with self.assertRaises(subprocess.CalledProcessError):
                    ns["read_aloud"]("hello", play_locally=False)
            self.assertEqual([p.name for p in output_dir.iterdir()], ["current_speech.wav"])

    def test_origin_guard_rejects_cross_origin_and_allows_native_clients(self):
        class FakeResponse:
            def __init__(self, body, status_code, media_type):
                self.status_code = status_code

            async def __call__(self, scope, receive, send):
                await send({"status": self.status_code})

        ns = load_server_parts("LocalOriginMiddleware", extra={"Response": FakeResponse})

        async def app(scope, receive, send):
            await send({"status": 200})

        middleware = ns["LocalOriginMiddleware"](app, "http://127.0.0.1:5555")

        async def check(headers):
            result = []
            async def send(message):
                result.append(message["status"])
            await middleware({"type": "http", "headers": headers}, None, send)
            return result[0]

        self.assertEqual(asyncio.run(check([])), 200)
        self.assertEqual(asyncio.run(check([(b"origin", b"http://127.0.0.1:5555")])), 200)
        self.assertEqual(asyncio.run(check([(b"origin", b"https://evil.example")])), 403)
        self.assertEqual(asyncio.run(check([(b"sec-fetch-site", b"cross-site")])), 403)

    def test_failed_build_preserves_existing_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "assets").mkdir()
            (root / "assets" / "ghost_icon.ico").touch()
            (root / "ghost_terminal.html").touch()
            executable = root / "GHoSt-Terminal.exe"
            executable.write_bytes(b"working executable")
            with mock.patch.object(build_terminal, "HERE", root), \
                 mock.patch.object(build_terminal, "_copy_external_backends", return_value=["cpu"]), \
                 mock.patch.object(build_terminal.subprocess, "check_call", side_effect=RuntimeError("build failed")):
                with self.assertRaises(RuntimeError):
                    build_terminal.build(allow_cpu_only=True)
            self.assertEqual(executable.read_bytes(), b"working executable")

    def test_release_staging_refuses_existing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            important = stage / "important.txt"
            important.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                package_release.stage_clean_release(stage)
            self.assertEqual(important.read_text(encoding="utf-8"), "keep")

    def test_failed_zip_preserves_existing_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release.zip"
            release.write_bytes(b"existing release")
            with mock.patch.object(package_release.zipfile, "ZipFile", side_effect=RuntimeError("zip failed")):
                with self.assertRaises(RuntimeError):
                    package_release.build_release_zip(root, release)
            self.assertEqual(release.read_bytes(), b"existing release")

    def test_high_context_and_token_limits(self):
        constants = load_server_constants(
            "DEFAULT_N_CTX", "MIN_N_CTX", "MAX_N_CTX", "MAX_UI_MAX_TOKENS", "DEFAULT_SETTINGS"
        )
        ns = load_server_parts(
            "ChatFormat", "Message", "JsonSchemaSpec", "ResponseFormat",
            "ToolFunctionSpec", "ToolSpec", "ChatRequest",
            "LoadRequest", "PreflightRequest", "LoaderSettings", "_chat_completion_args",
            extra={
                **constants,
                "load_settings": lambda: {"max_tokens": 512},
                "secrets": types.SimpleNamespace(randbelow=lambda n: 0),
            },
        )
        max_n_ctx = constants["MAX_N_CTX"]
        self.assertGreaterEqual(max_n_ctx, 262144)

        for cls in (ns["LoadRequest"], ns["PreflightRequest"], ns["LoaderSettings"]):
            for ctx in (65000, 65536, 131072, max_n_ctx):
                obj = cls(model="test.gguf", n_ctx=ctx)
                self.assertEqual(obj.n_ctx, ctx)
            for bad_ctx in (511, max_n_ctx + 1):
                with self.assertRaises(ValidationError):
                    cls(model="test.gguf", n_ctx=bad_ctx)

        req_completion = ns["ChatRequest"](
            model="test.gguf",
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=65536,
        )
        args = ns["_chat_completion_args"](req_completion, [{"role": "user", "content": "hi"}])
        self.assertEqual(args["max_tokens"], 65536)

        req_both = ns["ChatRequest"](
            model="test.gguf",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096,
            max_completion_tokens=65536,
        )
        args_both = ns["_chat_completion_args"](req_both, [{"role": "user", "content": "hi"}])
        self.assertEqual(args_both["max_tokens"], 4096)

    def test_preflight_reports_and_warns_on_declared_context_length(self):
        class FakeField:
            def __init__(self, value):
                self._value = value

            def contents(self):
                return self._value

        class FakeTensor:
            def __init__(self, name, n_elements, n_bytes, shape=()):
                self.name = name
                self.n_elements = n_elements
                self.n_bytes = n_bytes
                self.shape = shape

        declared_ctx = {"value": 40960}

        class FakeGGUFReader:
            def __init__(self, path):
                self.fields = {
                    "general.architecture": FakeField("qwen35"),
                    "qwen35.block_count": FakeField(2),
                    "qwen35.embedding_length": FakeField(1024),
                    "qwen35.attention.head_count": FakeField(4),
                    "qwen35.attention.head_count_kv": FakeField(2),
                    "qwen35.attention.key_length": FakeField(256),
                    "qwen35.attention.value_length": FakeField(256),
                    "qwen35.context_length": FakeField(declared_ctx["value"]),
                    "qwen35.ssm.conv_kernel": FakeField(4),
                    "qwen35.ssm.inner_size": FakeField(1024),
                    "qwen35.ssm.state_size": FakeField(64),
                    "qwen35.ssm.group_count": FakeField(4),
                    "general.file_type": FakeField(17),
                }
                self.tensors = [
                    FakeTensor("blk.0.ssm_conv1d.weight", 4 * 1536, 1024 * 1024, (4, 1536)),
                    FakeTensor("blk.1.attn_k.weight", 1024 * 512, 1024 * 1024),
                    FakeTensor("blk.1.attn_v.weight", 1024 * 512, 1024 * 1024),
                ]

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "test.gguf").write_bytes(b"GGUF")
            ns = load_server_parts("_gguf_value", "preflight_model", extra={
                "MODEL_DIR": model_dir,
                "GGUFReader": FakeGGUFReader,
                "LlamaFileType": lambda x: types.SimpleNamespace(name="MOSTLY_Q5_K_M"),
                "llama_cpp": types.SimpleNamespace(GGML_TYPE_F16=1, GGML_TYPE_Q8_0=8),
                "BACKEND_NAME": "cuda",
                "GIB": 1024 ** 3,
                "GPU_MAX_UTILIZATION": 0.95,
                "GPU_SAFETY_RESERVE_BYTES": 768 * 1024 ** 2,
                "_probe_nvidia_gpus": lambda: [{"name": "RTX Test", "total_bytes": 16 * 1024 ** 3, "free_bytes": 14 * 1024 ** 3}],
                "HTTPException": HTTPError,
            })

            over_plan = ns["preflight_model"]("test.gguf", 65000, flash_attn=True)
            self.assertEqual(over_plan["model_context_length"], 40960)
            self.assertEqual(over_plan["n_gpu_layers"], 2)
            self.assertTrue(any("exceeds this GGUF's declared context_length (40,960)" in w for w in over_plan["warnings"]))

            declared_ctx["value"] = 262144
            ok_plan = ns["preflight_model"]("test.gguf", 65000, flash_attn=True)
            self.assertEqual(ok_plan["model_context_length"], 262144)
            self.assertEqual(ok_plan["warnings"], [])

    def test_ui_context_input_and_error_formatting(self):
        html = (Path(__file__).resolve().parents[1] / "ghost_terminal.html").read_text(encoding="utf-8")
        self.assertIn('id="context-size" min="512" max="262144" step="1"', html)
        self.assertIn("function extractApiError(data, status)", html)


if __name__ == "__main__":
    unittest.main()
