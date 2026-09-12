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
