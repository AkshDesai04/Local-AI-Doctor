"""Command-line entrypoint; all overrides are delegated to SettingsLoader."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from .config import AppSettings, SettingsLoader
from .discovery.scanner import ModelScanner


def _settings(arguments: Sequence[str]) -> AppSettings:
    root = Path.cwd()
    return SettingsLoader().load_from_cli(
        arguments,
        fallback_config=root / "config" / "default.yaml",
        fallback_user_config=(root / "config" / "local.yaml")
        if (root / "config" / "local.yaml").is_file()
        else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="local-ai-doctor")
    parser.add_argument("command", nargs="?", choices=("serve", "scan", "config"), default="serve")
    known, remaining = parser.parse_known_args(arguments)
    settings = _settings(remaining)
    if known.command == "config":
        print(json.dumps(settings.redacted_effective_config(), indent=2))
        return 0
    if known.command == "scan":
        report = ModelScanner(
            settings.paths.model_roots,
            conservative_context_limit=settings.inference.conservative_context_limit,
        ).scan()
        print(json.dumps({"models": [model.public_dict() for model in report.models]}, indent=2))
        return 0
    uvicorn.run(
        "local_ai_doctor.main:create_app",
        factory=True,
        host=settings.server.host,
        port=settings.server.port,
        workers=1,
        log_level=settings.logging.level.value,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
