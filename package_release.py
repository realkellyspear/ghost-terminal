# SPDX-FileCopyrightText: 2026 Kelly Spear
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage a clean standalone folder and build the distributable Windows release .zip."""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent


def link_or_copy(src: Path, dst: Path) -> None:
    real_src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except OSError:
            old_dst = dst.with_suffix(dst.suffix + ".old")
            if old_dst.exists():
                try:
                    old_dst.unlink()
                except OSError:
                    pass
            dst.rename(old_dst)
    try:
        os.link(real_src, dst)
    except OSError:
        shutil.copy2(real_src, dst)


def stage_clean_release(stage_dir: Path) -> None:
    """Create a clean first-time-user folder using hardlinks where possible."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    exe_path = ROOT / "GHoSt-Terminal.exe"
    if not exe_path.is_file():
        raise FileNotFoundError(f"Missing {exe_path}. Run `python build_terminal.py` first.")

    link_or_copy(exe_path, stage_dir / "GHoSt-Terminal.exe")
    for doc in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        doc_path = ROOT / doc
        if doc_path.is_file():
            link_or_copy(doc_path, stage_dir / doc)

    # Complete first-party corresponding source for GPL-3.0-or-later compliance
    for name in (
        "ghost_terminal.py",
        "ghost_terminal.html",
        "build_terminal.py",
        "package_release.py",
        "requirements.txt",
        ".gitignore",
    ):
        src = ROOT / name
        if src.is_file():
            link_or_copy(src, stage_dir / "source" / name)

    # Assets
    for f in (ROOT / "assets").glob("*"):
        if f.is_file():
            link_or_copy(f, stage_dir / "assets" / f.name)

    # External llama.cpp / CUDA backends
    cuda_dir = ROOT / "backends" / "cuda"
    if not cuda_dir.is_dir() or not (cuda_dir / "ggml-cuda.dll").is_file():
        raise FileNotFoundError("Missing CUDA backend DLLs in ./backends/cuda. Run `python build_terminal.py` first.")
    for backend_sub in ("cpu", "cuda", "vulkan"):
        bdir = ROOT / "backends" / backend_sub
        if bdir.is_dir():
            for f in bdir.glob("*.dll"):
                link_or_copy(f, stage_dir / "backends" / backend_sub / f.name)

    # Empty ghost_models folder (README only; never include user .gguf files)
    link_or_copy(ROOT / "ghost_models" / "README.md", stage_dir / "ghost_models" / "README.md")

    # piper_models metadata (voices download on first use)
    link_or_copy(ROOT / "piper_models" / "README.md", stage_dir / "piper_models" / "README.md")
    for f in (ROOT / "piper_models").glob("*.MODEL_CARD"):
        if f.is_file():
            link_or_copy(f, stage_dir / "piper_models" / f.name)


def build_release_zip(stage_dir: Path, zip_path: Path) -> None:
    """Create the distributable Windows release .zip from the staged directory."""
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in sorted(stage_dir.rglob("*")):
            if file_path.is_dir():
                continue
            if file_path.suffix.lower() in (".gguf", ".old", ".log", ".onnx", ".wav"):
                continue
            if file_path.name.endswith(".onnx.json") or file_path.name.startswith("ghost_terminal_settings"):
                continue
            rel = file_path.relative_to(stage_dir)
            zf.write(file_path, str(Path("GHoSt-Terminal") / rel))


if __name__ == "__main__":
    stage = ROOT.parent / "GHoSt-Terminal-Release"
    print(f"Staging clean release folder at: {stage}")
    stage_clean_release(stage)

    release_zip = ROOT.parent / f"GHoSt-Terminal-v{VERSION}-win64.zip"
    print(f"Packaging distributable release archive: {release_zip} ...")
    build_release_zip(stage, release_zip)
    print(f"Done! Release zip: {release_zip} ({release_zip.stat().st_size / (1024 ** 2):.1f} MB)")
