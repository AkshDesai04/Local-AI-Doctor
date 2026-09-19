#!/usr/bin/env python3
"""Report the context length discovery selects for every model under a root.

The selection logic is not duplicated here: this calls the same ``ModelScanner`` the
service uses, so the output cannot drift from what the app reports.

Usage:
    python scripts/verify_model_context.py [ROOT ...] [--fallback N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from local_ai_doctor.discovery import ModelScanner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("models")])
    parser.add_argument(
        "--fallback",
        type=int,
        default=4096,
        help="context length used only for checkpoints that declare none",
    )
    args = parser.parse_args()

    roots = [root for root in (args.roots or [Path("models")])]
    missing = [root for root in roots if not root.is_dir()]
    for root in missing:
        print(f"not a directory: {root}", file=sys.stderr)
    roots = [root for root in roots if root.is_dir()]
    if not roots:
        return 1

    report = ModelScanner(roots, conservative_context_limit=args.fallback).scan()
    if not report.models:
        print("no models discovered")
        return 0

    width = max(len(model.display_name) for model in report.models)
    for model in sorted(report.models, key=lambda item: item.display_name.casefold()):
        selected = next(
            (
                value.source
                for value in model.context_values
                if value.value == model.effective_context_limit
            ),
            "unknown",
        )
        print(f"{model.display_name:<{width}}  {model.effective_context_limit:>9}  {selected}")
        conflict = next(
            (item for item in model.diagnostics if item.code == "conflicting_context_metadata"),
            None,
        )
        if conflict is not None:
            declared = conflict.evidence.get("discovered_values")
            print(f"{'':<{width}}  {'':>9}  conflicting declared values: {declared}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
