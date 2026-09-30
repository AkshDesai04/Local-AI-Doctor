"""PyInstaller entrypoint with an explicit, file-signaled graceful shutdown."""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

if os.environ.get("LOCAL_AI_DOCTOR_STARTUP_TRACE") == "1":
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True)

import uvicorn

# PyTorch local-version suffix and torch.version.cuda for each packaged flavor.
TORCH_VARIANTS: dict[str, tuple[str, str | None]] = {
    "cuda": ("+cu128", "12.8"),
    "cpu": ("+cpu", None),
}


async def _watch_shutdown_file(server: uvicorn.Server, shutdown_file: Path) -> None:
    """Request ASGI lifespan shutdown when Electron creates its private marker."""

    while not server.should_exit:
        try:
            if await asyncio.to_thread(shutdown_file.is_file):
                server.should_exit = True
                return
        except OSError:
            # A transient filesystem error must not take down a healthy API.
            pass
        await asyncio.sleep(0.25)


async def _serve() -> None:
    shutdown_file_value = os.environ.pop("LAD_DESKTOP_SHUTDOWN_FILE", "")
    shutdown_file = Path(shutdown_file_value) if shutdown_file_value else None
    host = os.environ.get("LAD_SERVER__HOST", "127.0.0.1")
    port = int(os.environ.get("LAD_SERVER__PORT", "6767"))
    config = uvicorn.Config(
        "local_ai_doctor.main:create_app",
        factory=True,
        host=host,
        port=port,
        workers=1,
        log_level=os.environ.get("LAD_LOGGING__LEVEL", "info"),
    )
    # Load the application before starting any lifecycle watcher so native
    # extension initialization is complete before the server begins startup.
    config.load()
    server = uvicorn.Server(config)
    watcher = (
        asyncio.create_task(_watch_shutdown_file(server, shutdown_file))
        if shutdown_file is not None
        else None
    )
    try:
        await server.serve()
    finally:
        if watcher is not None:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher


def _bitsandbytes_version() -> str | None:
    if importlib.util.find_spec("bitsandbytes") is None:
        return None
    try:
        return importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _cuda_numerics(torch: Any) -> list[str]:
    """Compare small CUDA kernels against the CPU reference."""

    problems: list[str] = []
    functional = torch.nn.functional
    generator = torch.Generator().manual_seed(0)
    left = torch.randn(64, 64, generator=generator)
    right = torch.randn(64, 64, generator=generator)
    if not torch.allclose((left.cuda() @ right.cuda()).cpu(), left @ right, rtol=1e-3, atol=1e-3):
        problems.append("CUDA float32 matmul does not match the CPU result")
    query, key, value = (torch.randn(1, 2, 16, 32, generator=generator) for _ in range(3))
    expected = functional.scaled_dot_product_attention(query, key, value)
    actual = functional.scaled_dot_product_attention(
        query.cuda().bfloat16(), key.cuda().bfloat16(), value.cuda().bfloat16()
    )
    if not torch.allclose(actual.float().cpu(), expected, rtol=5e-2, atol=5e-2):
        problems.append("CUDA bfloat16 scaled_dot_product_attention does not match the CPU result")
    # A runtime-compiled kernel proves that the bundled NVRTC libraries load.
    doubled = torch.cuda.jiterator._create_jit_fn(
        "template <typename T> T lad_double(T x) { return x * T(2); }"
    )(torch.arange(4.0, device="cuda"))
    if doubled.cpu().tolist() != [0.0, 2.0, 4.0, 6.0]:
        problems.append("NVRTC-compiled CUDA kernel returned an incorrect result")
    return problems


def self_check(variant: str | None, require_cuda: bool) -> tuple[dict[str, Any], list[str]]:
    """Describe the bundled PyTorch runtime and list every unmet expectation."""

    import torch

    cuda_available = bool(torch.cuda.is_available())
    report: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "device_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        "bitsandbytes_version": _bitsandbytes_version(),
        "variant_expected": variant,
    }
    problems: list[str] = []
    expected = TORCH_VARIANTS.get(variant) if variant else None
    if variant and expected is None:
        problems.append(f"unknown variant {variant!r}")
    elif expected is not None and (
        not report["torch_version"].endswith(expected[0])
        or report["torch_cuda_build"] != expected[1]
    ):
        problems.append(
            f"expected a {variant} PyTorch build (*{expected[0]}, CUDA {expected[1] or 'none'}) but found "
            f"{report['torch_version']} (CUDA {report['torch_cuda_build'] or 'none'})"
        )
    if require_cuda:
        if not cuda_available:
            problems.append("CUDA is required but torch.cuda.is_available() is False")
        else:
            try:
                problems += _cuda_numerics(torch)
            except Exception as exc:
                problems.append(f"CUDA kernels failed ({type(exc).__name__}: {exc})")
    return report, problems


def _run_self_check(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="local-ai-doctor-backend --self-check")
    parser.add_argument("--self-check", action="store_true", required=True)
    parser.add_argument("--variant", choices=sorted(TORCH_VARIANTS))
    parser.add_argument("--require-cuda", action="store_true")
    options = parser.parse_args(arguments)
    report, problems = self_check(options.variant, options.require_cuda)
    print(json.dumps({**report, "problems": problems}), flush=True)
    return 1 if problems else 0


def main() -> int:
    # PyInstaller replaces this function so spawned model workers re-enter the
    # multiprocessing bootstrap instead of launching a second API server.
    multiprocessing.freeze_support()
    if "--self-check" in sys.argv[1:]:
        return _run_self_check(sys.argv[1:])
    try:
        asyncio.run(_serve())
    finally:
        faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
