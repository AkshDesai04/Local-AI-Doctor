"""Online SQLite backup and guarded offline restore for Compose volumes."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3
from pathlib import Path

DATABASE = Path("/data/database/workbench.sqlite3")
BACKUP_ROOT = Path("/data/backups")


def _inside_backup_root(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = BACKUP_ROOT / candidate
    resolved = candidate.resolve(strict=False)
    root = BACKUP_ROOT.resolve(strict=True)
    if resolved.parent != root:
        raise argparse.ArgumentTypeError("backup file must be directly inside /data/backups")
    return resolved


def backup() -> Path:
    if not DATABASE.is_file():
        raise SystemExit(f"database does not exist: {DATABASE}")
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = BACKUP_ROOT / f"workbench-{stamp}.sqlite3"
    with (
        sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as source,
        sqlite3.connect(destination) as target,
    ):
        source.backup(target)
        result = target.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise SystemExit(f"backup integrity check failed: {result!r}")
    destination.chmod(0o640)
    return destination


def restore(source: Path) -> Path:
    if not source.is_file():
        raise SystemExit(f"backup does not exist: {source}")
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    temporary = DATABASE.with_suffix(".restore.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as backup_db:
            result = backup_db.execute("PRAGMA integrity_check").fetchone()
            if result != ("ok",):
                raise SystemExit(f"source integrity check failed: {result!r}")
            with sqlite3.connect(temporary) as restored_db:
                backup_db.backup(restored_db)
        temporary.chmod(0o640)
        os.replace(temporary, DATABASE)
        DATABASE.with_name(DATABASE.name + "-wal").unlink(missing_ok=True)
        DATABASE.with_name(DATABASE.name + "-shm").unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return DATABASE


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("backup", help="create and integrity-check an online backup")
    restore_parser = subcommands.add_parser("restore", help="replace the stopped app database")
    restore_parser.add_argument("source", type=_inside_backup_root)
    arguments = parser.parse_args()
    output = backup() if arguments.command == "backup" else restore(arguments.source)
    print(output)


if __name__ == "__main__":
    main()
