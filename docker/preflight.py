"""Fail fast when container mounts are unsafe or unusable."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

MODEL_ROOT = Path("/models")
CONFIG_PATHS = (Path("/app/config/default.yaml"), Path("/app/config/local.yaml"))
WRITABLE_ROOTS = (
    Path("/data/database"),
    Path("/data/uploads"),
    Path("/data/cache"),
    Path("/data/exports"),
    Path("/data/backups"),
)


def main() -> None:
    if not MODEL_ROOT.is_dir() or not os.access(MODEL_ROOT, os.R_OK | os.X_OK):
        raise SystemExit("/models must be a readable directory")
    if not os.statvfs(MODEL_ROOT).f_flag & os.ST_RDONLY:
        raise SystemExit("/models is not mounted read-only")

    for path in CONFIG_PATHS:
        if not path.is_file() or not os.access(path, os.R_OK):
            raise SystemExit(f"required configuration is not readable: {path}")

    for root in WRITABLE_ROOTS:
        if not root.is_dir():
            raise SystemExit(f"persistent directory is missing: {root}")
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=".write-check-", delete=True):
                pass
        except OSError as exc:
            raise SystemExit(f"persistent directory is not writable: {root}: {exc}") from exc

    print("container mount preflight passed")


if __name__ == "__main__":
    main()
