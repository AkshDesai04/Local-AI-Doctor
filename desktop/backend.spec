# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import re
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
# bitsandbytes is collected when the packaging environment has it.
packages = [
    "accelerate",
    "av",
    "local_ai_doctor",
    "PIL",
    "qwen_vl_utils",
    "safetensors",
    "sentence_transformers",
    "transformers",
    "uvicorn",
]
if importlib.util.find_spec("bitsandbytes") is not None:
    packages.append("bitsandbytes")
# Test-only subpackages that collect_all() would otherwise bundle as sources.
test_only = {"accelerate": ("test_utils",)}
for package in packages:
    skipped = tuple(f"{package}.{name}" for name in test_only.get(package, ()))
    package_datas, package_binaries, package_hiddenimports = collect_all(
        package,
        filter_submodules=lambda name, skipped=skipped: not any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in skipped
        ),
        exclude_datas=list(test_only.get(package, ())),
    )
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


def _package_dir(name):
    spec = importlib.util.find_spec(name)
    return Path(next(iter(spec.submodule_search_locations))).resolve() if spec else None


# CUDA runtime DLLs may only come from the pinned wheels. Anything else, such
# as a CUDA Toolkit found on PATH, is machine-dependent and version-mismatched.
cuda_runtime_dll = re.compile(r"^(nvrtc|cudart|cublas|cudnn|cufft|cusparse|cusolver|curand|nvjitlink)")
torch_dir = _package_dir("torch")
cuda_dll_sources = {
    directory
    for directory in (
        torch_dir / "lib" if torch_dir else None,
        _package_dir("torchvision"),
        _package_dir("bitsandbytes"),
    )
    if directory is not None
}
# Shipped by the CUDA wheel but never imported or loaded by name by any bundled
# binary or Python module (checked against PE import tables and a string scan of
# the whole bundle). The packaged self-check still runs cuBLAS, SDPA, and
# NVRTC-compiled kernels without them.
unused_dlls = {"cusolvermg64_11.dll", "nvrtc64_120_0.alt.dll"}


def _keep_binary(destination, source):
    name = Path(destination).name.lower()
    if name in unused_dlls:
        return False
    if name.startswith("libbitsandbytes_"):
        # Keep only the CPU library and the build matching torch's CUDA 12.8.
        return name.startswith(("libbitsandbytes_cpu", "libbitsandbytes_cuda128"))
    if cuda_runtime_dll.match(name):
        return Path(source).resolve().parent in cuda_dll_sources
    return True


a.binaries = [entry for entry in a.binaries if _keep_binary(entry[0], entry[1])]
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
