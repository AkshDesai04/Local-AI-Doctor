from __future__ import annotations

from pathlib import Path

from local_ai_doctor.errors import ModelInvalidError


def test_structured_errors_redact_paths_and_unknown_objects() -> None:
    error = ModelInvalidError(
        "model cannot be loaded",
        hint="Inspect the model metadata.",
        details={"path": Path("/private/models/model"), "exception": RuntimeError("secret")},
    )
    payload = error.to_dict()
    assert payload["code"] == "model_invalid"
    assert payload["details"]["path"] == "<path:model>"
    assert payload["details"]["exception"] == "RuntimeError"
    assert "secret" not in str(payload)


def test_worker_insufficient_memory_becomes_a_507_with_its_byte_counts() -> None:
    from local_ai_doctor.errors import worker_failure_response

    status, payload = worker_failure_response(
        {
            "code": "insufficient_memory",
            "message": "the model does not fit the available GPU memory",
            "hint": "Turn off Strict VRAM.",
            "memory_kind": "vram",
            "required_bytes": 10,
            "available_bytes": 4,
            "estimate": {"weights": 8, "kv_reserve": 2, "margin": 1},
        }
    )

    assert status == 507
    assert payload["code"] == "out_of_memory"
    assert payload["details"] == {
        "memory_kind": "vram",
        "required_bytes": 10,
        "available_bytes": 4,
        "estimate": {"weights": 8, "kv_reserve": 2, "margin": 1},
        "worker_code": "insufficient_memory",
    }
    assert worker_failure_response({"code": "model_not_resident"})[0] == 409
    assert worker_failure_response({"code": "worker_busy"})[0] == 409
