"""Content-addressed uploads with signature checks and path confinement."""

from __future__ import annotations

import codecs
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from ..errors import (
    AttachmentNotFoundError,
    CapabilityUnavailableError,
    InvalidUploadError,
    PathSecurityError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    WorkbenchError,
)
from ..persistence import WorkspaceRepository

_STORAGE_NAME = re.compile(r"^[0-9a-f]{64}\.[a-z0-9]{1,10}$")


def _detect_media_type(header: bytes) -> tuple[str, str] | None:
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", ".jpg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", ".gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp", ".webp"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return "video/mp4", ".mp4"
    if header.startswith(b"\x1aE\xdf\xa3"):
        return "video/webm", ".webm"
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "audio/wav", ".wav"
    if header.startswith(b"fLaC"):
        return "audio/flac", ".flac"
    if header.startswith(b"ID3") or header[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return "audio/mpeg", ".mp3"
    if header.startswith(b"%PDF-"):
        return "application/pdf", ".pdf"
    try:
        header.decode("utf-8")
        return "text/plain", ".txt"
    except UnicodeDecodeError:
        return None


def _media_family(media_type: str) -> str:
    return media_type.split("/", maxsplit=1)[0]


def _validate_utf8(path: Path) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            decoder.decode(chunk)
        decoder.decode(b"", final=True)


class UploadStore:
    def __init__(
        self,
        root: Path,
        repository: WorkspaceRepository,
        maximum_bytes: int,
        *,
        maximum_image_pixels: int = 40_000_000,
        maximum_video_frames: int = 256,
        maximum_video_frame_pixels: int = 8_500_000,
        maximum_decoded_media_pixels: int = 500_000_000,
        maximum_media_duration_seconds: float = 600.0,
    ) -> None:
        self.root = root.resolve()
        self.repository = repository
        self.maximum_bytes = maximum_bytes
        self.maximum_image_pixels = maximum_image_pixels
        self.maximum_video_frames = maximum_video_frames
        self.maximum_video_frame_pixels = maximum_video_frame_pixels
        self.maximum_decoded_media_pixels = maximum_decoded_media_pixels
        self.maximum_media_duration_seconds = maximum_media_duration_seconds
        self.root.mkdir(parents=True, exist_ok=True)

    def _validate_image(self, path: Path) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise CapabilityUnavailableError(
                "image decoder validation is unavailable",
                hint="Install the locked ML dependencies before accepting images.",
            ) from exc
        try:
            with Image.open(path) as image:
                width, height = image.size
                frame_count = int(getattr(image, "n_frames", 1))
                image_format = image.format
                pixels = width * height
                decoded_pixels = pixels * frame_count
                if pixels > self.maximum_image_pixels:
                    raise PayloadTooLargeError(
                        "decoded image dimensions exceed the configured pixel limit",
                        details={
                            "pixels": pixels,
                            "maximum_pixels": self.maximum_image_pixels,
                        },
                    )
                if frame_count > self.maximum_video_frames:
                    raise PayloadTooLargeError(
                        "animated image exceeds the configured frame limit",
                        details={
                            "frames": frame_count,
                            "maximum_frames": self.maximum_video_frames,
                        },
                    )
                if decoded_pixels > self.maximum_decoded_media_pixels:
                    raise PayloadTooLargeError(
                        "decoded image exceeds the configured total-pixel limit",
                        details={
                            "decoded_pixels": decoded_pixels,
                            "maximum_decoded_pixels": self.maximum_decoded_media_pixels,
                        },
                    )
                image.verify()
            return {
                "width": width,
                "height": height,
                "frames": frame_count,
                "decoded_pixels": decoded_pixels,
                "format": image_format,
            }
        except PayloadTooLargeError:
            raise
        except Image.DecompressionBombError as exc:
            raise PayloadTooLargeError(
                "decoded image exceeds Pillow's decompression safety limit"
            ) from exc
        except (OSError, ValueError) as exc:
            raise InvalidUploadError("image failed decoder validation") from exc

    def _validate_av_media(self, path: Path, family: str) -> dict[str, Any]:
        try:
            import av
        except ImportError as exc:
            raise CapabilityUnavailableError(
                "media decoder validation is unavailable",
                hint="Install the locked ML dependencies before accepting video or audio.",
            ) from exc

        try:
            with av.open(str(path), mode="r") as container:
                streams = [stream for stream in container.streams if stream.type == family]
                if not streams:
                    raise InvalidUploadError(f"file does not contain a decodable {family} stream")
                stream = streams[0]
                duration: float | None = None
                if stream.duration is not None and stream.time_base is not None:
                    duration = float(stream.duration * stream.time_base)
                elif container.duration is not None:
                    duration = float(container.duration / av.time_base)
                if duration is not None and duration > self.maximum_media_duration_seconds:
                    raise PayloadTooLargeError(
                        "media duration exceeds the configured limit",
                        details={
                            "duration_seconds": duration,
                            "maximum_duration_seconds": self.maximum_media_duration_seconds,
                        },
                    )

                decoded_units = 0
                decoded_pixels = 0
                sample_count = 0
                for frame in container.decode(stream):
                    decoded_units += 1
                    if family == "video":
                        pixels = int(getattr(frame, "width", 0)) * int(getattr(frame, "height", 0))
                        if pixels > self.maximum_video_frame_pixels:
                            raise PayloadTooLargeError(
                                "decoded video frame exceeds the configured pixel limit",
                                details={
                                    "frame_pixels": pixels,
                                    "maximum_frame_pixels": self.maximum_video_frame_pixels,
                                },
                            )
                        decoded_pixels += pixels
                        if decoded_units > self.maximum_video_frames:
                            raise PayloadTooLargeError(
                                "decoded video exceeds the configured frame limit",
                                details={
                                    "frames": decoded_units,
                                    "maximum_frames": self.maximum_video_frames,
                                },
                            )
                        if decoded_pixels > self.maximum_decoded_media_pixels:
                            raise PayloadTooLargeError(
                                "decoded video exceeds the configured total-pixel limit",
                                details={
                                    "decoded_pixels": decoded_pixels,
                                    "maximum_decoded_pixels": self.maximum_decoded_media_pixels,
                                },
                            )
                    else:
                        sample_count += int(getattr(frame, "samples", 0))
                        sample_rate = int(
                            getattr(frame, "sample_rate", None)
                            or getattr(stream, "sample_rate", None)
                            or 0
                        )
                        if sample_rate > 0 and (
                            sample_count / sample_rate > self.maximum_media_duration_seconds
                        ):
                            raise PayloadTooLargeError(
                                "decoded audio exceeds the configured duration limit",
                                details={
                                    "maximum_duration_seconds": self.maximum_media_duration_seconds
                                },
                            )
                if decoded_units == 0:
                    raise InvalidUploadError(f"file contains no decodable {family} frames")
                return {
                    "duration_seconds": duration,
                    "decoded_frames": decoded_units,
                    "decoded_pixels": decoded_pixels if family == "video" else None,
                    "decoded_samples": sample_count if family == "audio" else None,
                }
        except (PayloadTooLargeError, WorkbenchError):
            raise
        except Exception as exc:
            raise InvalidUploadError(f"{family} failed decoder validation") from exc

    async def save(
        self,
        upload: UploadFile,
        *,
        supported_modalities: frozenset[str],
        allow_extracted_text: bool = True,
    ) -> dict[str, Any]:
        original_name = Path(upload.filename or "upload").name
        if not original_name or "\x00" in original_name:
            raise InvalidUploadError("upload filename is invalid")
        temporary = self.root / f".upload-{uuid.uuid4().hex}.tmp"
        digest = hashlib.sha256()
        size = 0
        header = b""
        try:
            with temporary.open("xb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.maximum_bytes:
                        raise PayloadTooLargeError(
                            "upload exceeds the configured size limit",
                            details={"maximum_bytes": self.maximum_bytes},
                        )
                    if len(header) < 4096:
                        header += chunk[: 4096 - len(header)]
                    digest.update(chunk)
                    handle.write(chunk)
            detected = _detect_media_type(header)
            if detected is None:
                raise UnsupportedMediaTypeError(
                    "upload file signature is not supported",
                    hint="Use a supported image, video, audio, plain-text, or PDF file.",
                )
            media_type, extension = detected
            if media_type == "text/plain":
                try:
                    _validate_utf8(temporary)
                except UnicodeDecodeError as exc:
                    raise InvalidUploadError("plain-text upload is not valid UTF-8") from exc
            claimed = (upload.content_type or "").lower()
            if (
                claimed
                and claimed != "application/octet-stream"
                and _media_family(claimed) != _media_family(media_type)
            ):
                raise InvalidUploadError(
                    "declared MIME type does not match the file signature",
                    details={"declared": claimed, "detected": media_type},
                )
            family = _media_family(media_type)
            native_modality = {"image": "image", "video": "video", "audio": "audio"}.get(family)
            preprocessing: dict[str, Any]
            if native_modality:
                if native_modality not in supported_modalities:
                    raise CapabilityUnavailableError(
                        f"{native_modality} input is not supported by the selected model",
                        details={"supported_modalities": sorted(supported_modalities)},
                    )
                preprocessing = {"path": "native_model_processor", "modality": native_modality}
            elif media_type == "text/plain" and allow_extracted_text:
                preprocessing = {"path": "extracted_text", "encoding": "utf-8"}
            elif media_type == "application/pdf" and allow_extracted_text:
                raise CapabilityUnavailableError(
                    "PDF text extraction is not installed in this runtime",
                    hint="Convert the document to plain text or install a reviewed document extractor adapter.",
                )
            else:
                raise CapabilityUnavailableError("the selected model cannot process this upload")

            sha256 = digest.hexdigest()
            storage_name = f"{sha256}{extension}"
            destination = (self.root / storage_name).resolve()
            try:
                destination.relative_to(self.root)
            except ValueError as exc:
                raise PathSecurityError("upload destination escaped its configured root") from exc
            created_destination = not destination.exists()
            if not created_destination:
                temporary.unlink(missing_ok=True)
            else:
                os.replace(temporary, destination)
            metadata: dict[str, Any] = {"detected_media_type": media_type}
            if family == "image":
                try:
                    metadata.update(self._validate_image(destination))
                except Exception:
                    if created_destination:
                        destination.unlink(missing_ok=True)
                    raise
            elif family in {"video", "audio"}:
                try:
                    metadata.update(self._validate_av_media(destination, family))
                except Exception:
                    if created_destination:
                        destination.unlink(missing_ok=True)
                    raise
            return await self.repository.create_attachment(
                {
                    "id": str(uuid.uuid4()),
                    "sha256": sha256,
                    "storage_name": storage_name,
                    "original_name": original_name[:255],
                    "media_type": media_type,
                    "size_bytes": size,
                    "metadata": metadata,
                    "preprocessing": preprocessing,
                }
            )
        finally:
            temporary.unlink(missing_ok=True)
            await upload.close()

    async def resolve(self, attachment_id: str) -> tuple[dict[str, Any], Path]:
        attachment = await self.repository.get_attachment(attachment_id)
        if attachment is None:
            raise AttachmentNotFoundError(
                "attachment not found", details={"attachment_id": attachment_id}
            )
        storage_name = str(attachment["storage_name"])
        if not _STORAGE_NAME.fullmatch(storage_name):
            raise PathSecurityError("stored attachment name failed validation")
        candidate = self.root / storage_name
        if candidate.is_symlink():
            raise PathSecurityError("attachment symlinks are not allowed")
        path = candidate.resolve(strict=True)
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise PathSecurityError("attachment path escaped its configured root") from exc
        if not path.is_file():
            raise PathSecurityError("attachment is not a regular confined file")
        return attachment, path
