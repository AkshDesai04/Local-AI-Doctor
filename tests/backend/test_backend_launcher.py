from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

LAUNCHER_PATH = Path(__file__).resolve().parents[2] / "desktop" / "backend_launcher.py"


def _load_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lad_backend_launcher", LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load_launcher()


def _fake_torch(version: str, cuda: str | None, available: bool) -> Any:
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_name=lambda index: "Fixture GPU",
            get_device_capability=lambda index: (8, 9),
        ),
    )


@pytest.mark.parametrize(
    ("variant", "version", "cuda"),
    [("cuda", "2.8.0+cu128", "12.8"), ("cpu", "2.8.0+cpu", None)],
)
def test_self_check_accepts_the_expected_variant(
    monkeypatch: pytest.MonkeyPatch, variant: str, version: str, cuda: str | None
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(version, cuda, False))
    report, problems = launcher.self_check(variant, require_cuda=False)
    assert problems == []
    assert report["torch_version"] == version
    assert report["torch_cuda_build"] == cuda
    assert report["cuda_available"] is False
    assert report["device_name"] is None
    assert report["variant_expected"] == variant
    assert set(report) == {
        "torch_version",
        "torch_cuda_build",
        "cuda_available",
        "device_name",
        "device_capability",
        "bitsandbytes_version",
        "variant_expected",
    }


@pytest.mark.parametrize(
    ("variant", "version", "cuda"),
    [
        ("cuda", "2.8.0", None),
        ("cuda", "2.8.0+cpu", None),
        ("cuda", "2.8.0+cu126", "12.6"),
        ("cpu", "2.8.0+cu128", "12.8"),
    ],
)
def test_self_check_rejects_a_mismatched_variant(
    monkeypatch: pytest.MonkeyPatch, variant: str, version: str, cuda: str | None
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(version, cuda, False))
    _, problems = launcher.self_check(variant, require_cuda=False)
    assert len(problems) == 1
    assert f"expected a {variant} PyTorch build" in problems[0]


def test_require_cuda_fails_without_a_usable_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("2.8.0+cu128", "12.8", False))
    _, problems = launcher.self_check("cuda", require_cuda=True)
    assert problems == ["CUDA is required but torch.cuda.is_available() is False"]


def test_require_cuda_reports_device_and_kernel_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("2.8.0+cu128", "12.8", True))
    monkeypatch.setattr(launcher, "_cuda_numerics", lambda torch: [])
    assert launcher._run_self_check(["--self-check", "--variant", "cuda", "--require-cuda"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["device_name"] == "Fixture GPU"
    assert report["device_capability"] == [8, 9]
    assert report["problems"] == []

    monkeypatch.setattr(launcher, "_cuda_numerics", lambda torch: ["matmul mismatch"])
    assert launcher._run_self_check(["--self-check", "--variant", "cuda", "--require-cuda"]) == 1
    assert json.loads(capsys.readouterr().out)["problems"] == ["matmul mismatch"]
