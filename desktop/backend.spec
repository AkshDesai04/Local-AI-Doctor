# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


repo_root = Path(SPECPATH).parent
backend_root = repo_root / "backend"
# collect_all() resolves packages before Analysis applies its pathex. Prefer the
# checked-out backend so local builds cannot copy stale package data from an
# older installation in the selected Python environment.
sys.path.insert(0, str(backend_root))
datas = []
binaries = []
hiddenimports = []

# Transformers and its media/model helpers use lazy imports that static
# analysis cannot reliably discover. Collecting these packages keeps the
# portable application feature-equivalent to the native Python installation.
# PyInstaller's maintained hooks already collect Torch and TorchVision,
# including their runtime DLLs, while excluding development-only archives and
# headers, so do not collect those two packages a second time here.
for package in (
    "accelerate",
    "av",
    "local_ai_doctor",
    "PIL",
    "qwen_vl_utils",
    "safetensors",
    "sentence_transformers",
    "transformers",
    "uvicorn",
):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    [str(repo_root / "desktop" / "backend_launcher.py")],
    pathex=[str(backend_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["pytest", "mypy", "ruff"],
    module_collection_mode={
        # Torch performs runtime source inspection even outside TorchScript and
        # torch.compile. Keeping its Python modules only in the PYZ makes
        # inspect.getsource() fail inside the frozen worker and can prevent a
        # real model from loading. Keep importable bytecode in the PYZ for
        # startup speed while also retaining source for runtime inspection.
        "accelerate": "pyz",
        "sentence_transformers": "pyz",
        "torch": "pyz+py",
        "torchvision": "pyz+py",
        "transformers": "pyz",
    },
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="local-ai-doctor-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="local-ai-doctor-backend",
)
