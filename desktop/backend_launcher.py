"""PyInstaller entrypoint with an explicit, file-signaled graceful shutdown."""

from __future__ import annotations

import asyncio
import faulthandler
import multiprocessing
import os
from contextlib import suppress
from pathlib import Path

if os.environ.get("LOCAL_AI_DOCTOR_STARTUP_TRACE") == "1":
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True)

import uvicorn


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


def main() -> int:
    # PyInstaller replaces this function so spawned model workers re-enter the
    # multiprocessing bootstrap instead of launching a second API server.
    multiprocessing.freeze_support()
    try:
        asyncio.run(_serve())
    finally:
        faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
