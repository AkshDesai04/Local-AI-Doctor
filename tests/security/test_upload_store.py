from __future__ import annotations

import io
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from local_ai_doctor.errors import PathSecurityError, PayloadTooLargeError, WorkbenchError
from local_ai_doctor.persistence import Database, WorkspaceRepository
from local_ai_doctor.services.uploads import UploadStore


def _upload(data: bytes, filename: str, content_type: str) -> UploadFile:
    return UploadFile(
        io.BytesIO(data),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


@pytest.fixture
async def upload_store(
    tmp_path: Path,
) -> AsyncIterator[tuple[UploadStore, WorkspaceRepository, Database, Path]]:
    database = Database(tmp_path / "state.sqlite3", tmp_path / "backups")
    await database.initialize()
    repository = WorkspaceRepository(database)
    root = tmp_path / "uploads"
    store = UploadStore(root, repository, maximum_bytes=1024)
    try:
        yield store, repository, database, root
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_traversal_filename_is_metadata_only_and_content_stays_confined(
    upload_store: tuple[UploadStore, WorkspaceRepository, Database, Path], tmp_path: Path
) -> None:
    store, _repository, _database, root = upload_store
    attachment = await store.save(
        _upload(b"safe text", "../../outside.txt", "text/plain"),
        supported_modalities=frozenset({"text"}),
    )

    assert attachment["original_name"] == "outside.txt"
    stored = root / attachment["storage_name"]
    assert stored.parent == root
    assert stored.read_bytes() == b"safe text"
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.asyncio
async def test_mime_signature_mismatch_is_rejected_and_temp_file_is_removed(
    upload_store: tuple[UploadStore, WorkspaceRepository, Database, Path],
) -> None:
    store, _repository, database, root = upload_store
    fake_png = b"\x89PNG\r\n\x1a\n" + bytes(32)
    with pytest.raises(WorkbenchError, match="declared MIME type"):
        await store.save(
            _upload(fake_png, "not-text.txt", "text/plain"),
            supported_modalities=frozenset({"image"}),
        )

    assert list(root.iterdir()) == []
    assert await database.fetch_one("SELECT COUNT(*) AS count FROM attachments") == {"count": 0}


@pytest.mark.asyncio
async def test_oversized_upload_is_rejected_without_partial_file_or_metadata(
    upload_store: tuple[UploadStore, WorkspaceRepository, Database, Path],
) -> None:
    _store, repository, database, root = upload_store
    store = UploadStore(root, repository, maximum_bytes=8)
    with pytest.raises(WorkbenchError, match="size limit"):
        await store.save(
            _upload(b"nine-byte", "large.txt", "text/plain"),
            supported_modalities=frozenset({"text"}),
        )

    assert list(root.iterdir()) == []
    assert await database.fetch_one("SELECT COUNT(*) AS count FROM attachments") == {"count": 0}


@pytest.mark.asyncio
async def test_plain_text_signature_validates_the_entire_utf8_payload(
    upload_store: tuple[UploadStore, WorkspaceRepository, Database, Path],
) -> None:
    _store, repository, database, root = upload_store
    store = UploadStore(root, repository, maximum_bytes=8192)
    malformed = b"a" * 4096 + b"\xff"
    with pytest.raises(WorkbenchError, match="UTF-8"):
        await store.save(
            _upload(malformed, "malformed.txt", "text/plain"),
            supported_modalities=frozenset({"text"}),
        )

    assert list(root.iterdir()) == []
    assert await database.fetch_one("SELECT COUNT(*) AS count FROM attachments") == {"count": 0}


@pytest.mark.asyncio
async def test_decoded_image_pixel_limit_rejects_compression_bomb_before_verify(
    upload_store: tuple[UploadStore, WorkspaceRepository, Database, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, repository, database, root = upload_store
    verify_called = False

    class FakeImage:
        size = (20, 20)
        n_frames = 1
        format = "PNG"

        def __enter__(self) -> FakeImage:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def verify(self) -> None:
            nonlocal verify_called
            verify_called = True

    image_module = ModuleType("PIL.Image")
    image_module.open = lambda _path: FakeImage()  # type: ignore[attr-defined]
    image_module.DecompressionBombError = RuntimeError  # type: ignore[attr-defined]
    pillow_module = ModuleType("PIL")
    pillow_module.Image = image_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", pillow_module)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_module)

    store = UploadStore(root, repository, maximum_bytes=1024, maximum_image_pixels=100)
    with pytest.raises(PayloadTooLargeError, match="pixel limit"):
        await store.save(
            _upload(b"\x89PNG\r\n\x1a\n" + bytes(32), "bomb.png", "image/png"),
            supported_modalities=frozenset({"image"}),
        )

    assert verify_called is False
    assert list(root.iterdir()) == []
    assert await database.fetch_one("SELECT COUNT(*) AS count FROM attachments") == {"count": 0}


@pytest.mark.asyncio
async def test_tampered_storage_name_cannot_be_resolved_outside_upload_root(
    upload_store: tuple[UploadStore, WorkspaceRepository, Database, Path],
) -> None:
    store, repository, _database, _root = upload_store
    attachment = await repository.create_attachment(
        {
            "id": "tampered",
            "sha256": "b" * 64,
            "storage_name": "../outside.txt",
            "original_name": "outside.txt",
            "media_type": "text/plain",
            "size_bytes": 1,
        }
    )

    with pytest.raises(PathSecurityError, match="failed validation"):
        await store.resolve(attachment["id"])


@pytest.mark.asyncio
async def test_symlink_escape_is_rejected_when_platform_supports_symlinks(
    upload_store: tuple[UploadStore, WorkspaceRepository, Database, Path], tmp_path: Path
) -> None:
    store, repository, _database, root = upload_store
    outside = tmp_path / "private.txt"
    outside.write_text("private", encoding="utf-8")
    storage_name = f"{'c' * 64}.txt"
    link = root / storage_name
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("this platform does not permit creating test symlinks")
    attachment = await repository.create_attachment(
        {
            "id": "symlink",
            "sha256": "c" * 64,
            "storage_name": storage_name,
            "original_name": "link.txt",
            "media_type": "text/plain",
            "size_bytes": 7,
        }
    )

    with pytest.raises(PathSecurityError, match="escaped"):
        await store.resolve(attachment["id"])
