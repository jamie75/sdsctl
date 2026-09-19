"""Read-only, lossless SDS200 Favorites FTP backup."""

from __future__ import annotations

import ftplib
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Protocol

from .favorites_catalog import (
    FavoritesCatalog,
    FavoritesCatalogEntry,
    FavoritesCatalogError,
    project_favorites_catalog,
)
from .favorites_file import (
    FavoritesFileParseError,
    FavoritesSourceFile,
    parse_favorites_file,
)
from .favorites_storage_ftp import (
    FAVORITES_FTP_DEFAULT_MAX_CATALOG_BYTES,
    FAVORITES_FTP_DEFAULT_MAX_DOCUMENT_BYTES,
    FAVORITES_FTP_DEFAULT_MAX_LISTING_ENTRIES,
    FAVORITES_FTP_DEFAULT_MAX_SNAPSHOT_BYTES,
    FAVORITES_FTP_DEFAULT_PORT,
    FAVORITES_FTP_DEFAULT_TIMEOUT,
)

FAVORITES_BACKUP_DEFAULT_FTP_PORT = FAVORITES_FTP_DEFAULT_PORT
FAVORITES_BACKUP_DEFAULT_FTP_TIMEOUT = FAVORITES_FTP_DEFAULT_TIMEOUT
FAVORITES_BACKUP_REMOTE_DIRECTORY = "/favorites_lists"
FAVORITES_BACKUP_CATALOG_NAME = "f_list.cfg"
FAVORITES_BACKUP_SCHEMA = "sdsctl.favorites-backup"
FAVORITES_BACKUP_SCHEMA_VERSION = 1
_TRANSFER_BLOCK_SIZE = 64 * 1024


class FavoritesBackupError(OSError):
    """Report a safe, stage-specific Favorites backup failure."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"Favorites backup {stage} failed: {message}")


class FavoritesBackupFtp(Protocol):
    """The small FTP surface used by the read-only backup operation."""

    def connect(self, host: str, port: int, timeout: float) -> str:
        ...

    def login(self, user: str, passwd: str) -> str:
        ...

    def set_pasv(self, val: bool) -> None:
        ...

    def cwd(self, dirname: str) -> str:
        ...

    def nlst(self) -> list[str]:
        ...

    def retrbinary(
        self,
        cmd: str,
        callback: Callable[[bytes], None],
        blocksize: int,
    ) -> str:
        ...

    def quit(self) -> str:
        ...

    def close(self) -> None:
        ...


def _default_ftp_factory() -> FavoritesBackupFtp:
    return ftplib.FTP()


FavoritesBackupFtpFactory = Callable[[], FavoritesBackupFtp]


@dataclass(frozen=True, slots=True)
class FavoritesBackupDocument:
    """One exact referenced HPD and its generic lossless parsed records."""

    filename: str
    content: bytes
    source: FavoritesSourceFile | None
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class FavoritesBackupList:
    """One F-List entry bound to its referenced HPD, preserving raw fields."""

    name: str
    filename: str
    raw_fields: tuple[str, ...]
    source_index: int
    document: FavoritesBackupDocument


@dataclass(frozen=True, slots=True)
class FavoritesBackupInventory:
    """All read-only Favorites data needed for one local backup."""

    host: str
    requested_model: str
    target_model: str | None
    format_version: str | None
    catalog_bytes: bytes
    catalog_source: FavoritesSourceFile
    catalog: FavoritesCatalog
    lists: tuple[FavoritesBackupList, ...]
    documents: tuple[FavoritesBackupDocument, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FavoritesBackupResult:
    """Published local backup directory and its completed inventory."""

    directory: Path
    manifest_path: Path
    inventory: FavoritesBackupInventory


def _validate_host(host: str) -> str:
    if (
        not isinstance(host, str)
        or not host
        or host.strip() != host
        or "://" in host
        or "/" in host
        or "\\" in host
        or "@" in host
        or any(character.isspace() for character in host)
    ):
        raise ValueError("Favorites backup host must be one host name or address.")
    return host


def _validate_filename(filename: str) -> str:
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or bool(PureWindowsPath(filename).drive)
        or any(ord(character) < 32 or ord(character) == 127 for character in filename)
        or not filename.endswith(".hpd")
    ):
        raise FavoritesBackupError(
            "catalog",
            f"referenced Favorites filename is unsafe: {filename!r}",
        )
    return filename


def _metadata_value(
    source: FavoritesSourceFile,
    command: str,
    warnings: list[str],
) -> str | None:
    matches = [record for record in source.records if record.command == command]
    if len(matches) > 1:
        warnings.append(f"duplicate {command} records: {len(matches)}")
    if not matches:
        warnings.append(f"missing {command} record")
        return None
    if not matches[0].fields:
        warnings.append(f"{command} record has no value")
        return None
    return matches[0].fields[0]


def _ftp_failure(stage: str, error: BaseException) -> FavoritesBackupError:
    del error
    return FavoritesBackupError(stage, "the scanner FTP operation could not be completed")


@contextmanager
def _anonymous_ftp(
    host: str,
    port: int,
    timeout: float,
    ftp_factory: FavoritesBackupFtpFactory,
) -> Iterator[FavoritesBackupFtp]:
    try:
        ftp = ftp_factory()
    except (ftplib.Error, OSError, EOFError, UnicodeError) as error:
        raise _ftp_failure("connect", error) from None

    connected = False
    operation_error: BaseException | None = None
    try:
        try:
            ftp.connect(host, port, timeout)
            connected = True
        except (ftplib.Error, OSError, EOFError, UnicodeError) as error:
            raise _ftp_failure("connect", error) from None

        try:
            ftp.login("anonymous", "anonymous@")
            ftp.set_pasv(True)
            ftp.cwd(FAVORITES_BACKUP_REMOTE_DIRECTORY)
        except (ftplib.Error, OSError, EOFError, UnicodeError) as error:
            raise _ftp_failure("session", error) from None

        yield ftp
    except BaseException as error:
        operation_error = error
        raise
    finally:
        if connected:
            try:
                ftp.quit()
            except (ftplib.Error, OSError, EOFError, UnicodeError) as error:
                with suppress(OSError):
                    ftp.close()
                if operation_error is None:
                    raise _ftp_failure("close", error) from None
        else:
            with suppress(OSError):
                ftp.close()


class _TransferLimitExceeded(RuntimeError):
    pass


def _retrieve(
    ftp: FavoritesBackupFtp,
    filename: str,
    *,
    max_bytes: int,
) -> bytes:
    content = bytearray()

    def consume(chunk: bytes) -> None:
        if len(content) + len(chunk) > max_bytes:
            raise _TransferLimitExceeded
        content.extend(chunk)

    try:
        ftp.retrbinary(
            f"RETR {filename}",
            consume,
            blocksize=_TRANSFER_BLOCK_SIZE,
        )
    except _TransferLimitExceeded:
        raise FavoritesBackupError(
            "retrieve",
            f"{filename} exceeds the configured byte limit",
        ) from None
    except (ftplib.Error, OSError, EOFError, UnicodeError) as error:
        raise _ftp_failure("retrieve", error) from None
    return bytes(content)


def _parse_catalog(data: bytes) -> tuple[FavoritesSourceFile, FavoritesCatalog]:
    try:
        source = parse_favorites_file(data)
    except FavoritesFileParseError as error:
        raise FavoritesBackupError("parse", str(error)) from None
    try:
        catalog = project_favorites_catalog(source)
    except FavoritesCatalogError as error:
        raise FavoritesBackupError("parse", str(error)) from None
    return source, catalog


def read_favorites_inventory(
    host: str,
    *,
    requested_model: str = "SDS200",
    port: int = FAVORITES_BACKUP_DEFAULT_FTP_PORT,
    timeout: float = FAVORITES_BACKUP_DEFAULT_FTP_TIMEOUT,
    ftp_factory: FavoritesBackupFtpFactory = _default_ftp_factory,
) -> FavoritesBackupInventory:
    """Read and validate the referenced Favorites files without scanner writes."""

    host = _validate_host(host)
    if requested_model != "SDS200":
        raise ValueError("Favorites backup supports requested model SDS200 only.")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("Favorites backup FTP port must be between 1 and 65535.")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("Favorites backup FTP timeout must be greater than zero.")

    warnings: list[str] = []
    errors: list[str] = []
    with _anonymous_ftp(host, port, float(timeout), ftp_factory) as ftp:
        try:
            raw_listing = ftp.nlst()
        except (ftplib.Error, OSError, EOFError, UnicodeError) as error:
            raise _ftp_failure("listing", error) from None
        if not isinstance(raw_listing, (list, tuple)):
            raise FavoritesBackupError("listing", "scanner FTP listing was not a sequence")
        if len(raw_listing) > FAVORITES_FTP_DEFAULT_MAX_LISTING_ENTRIES:
            raise FavoritesBackupError("listing", "scanner FTP listing is too large")
        listing = tuple(raw_listing)
        if any(not isinstance(name, str) for name in listing):
            raise FavoritesBackupError("listing", "scanner FTP listing contained non-text data")
        if FAVORITES_BACKUP_CATALOG_NAME not in listing:
            raise FavoritesBackupError("catalog", "f_list.cfg was not present")

        catalog_bytes = _retrieve(
            ftp,
            FAVORITES_BACKUP_CATALOG_NAME,
            max_bytes=FAVORITES_FTP_DEFAULT_MAX_CATALOG_BYTES,
        )
        catalog_source, catalog = _parse_catalog(catalog_bytes)
        target_model = _metadata_value(catalog_source, "TargetModel", warnings)
        format_version = _metadata_value(catalog_source, "FormatVersion", warnings)
        if not catalog.entries:
            warnings.append("f_list.cfg contains no Favorites List records")

        references: list[tuple[FavoritesCatalogEntry, str]] = []
        seen: set[str] = set()
        for entry in catalog.entries:
            filename = _validate_filename(entry.filename)
            if filename in seen:
                warnings.append(f"duplicate Favorites document reference: {filename}")
            else:
                seen.add(filename)
                references.append((entry, filename))

        documents: list[FavoritesBackupDocument] = []
        documents_by_filename: dict[str, FavoritesBackupDocument] = {}
        total_bytes = len(catalog_bytes)
        for _entry, filename in references:
            if filename not in listing:
                raise FavoritesBackupError(
                    "retrieve",
                    f"referenced Favorites document is missing: {filename}",
                )
            content = _retrieve(
                ftp,
                filename,
                max_bytes=FAVORITES_FTP_DEFAULT_MAX_DOCUMENT_BYTES,
            )
            total_bytes += len(content)
            if total_bytes > FAVORITES_FTP_DEFAULT_MAX_SNAPSHOT_BYTES:
                raise FavoritesBackupError("retrieve", "Favorites snapshot is too large")
            source: FavoritesSourceFile | None
            parse_error: str | None
            try:
                source = parse_favorites_file(content)
                parse_error = None
            except FavoritesFileParseError as error:
                source = None
                parse_error = str(error)
                errors.append(f"{filename}: {parse_error}")
            document = FavoritesBackupDocument(
                filename=filename,
                content=content,
                source=source,
                parse_error=parse_error,
            )
            documents.append(document)
            documents_by_filename[filename] = document

        lists = tuple(
            FavoritesBackupList(
                name=entry.name,
                filename=filename,
                raw_fields=entry.source.fields,
                source_index=entry.source_index,
                document=documents_by_filename[filename],
            )
            for entry, filename in (
                (entry, _validate_filename(entry.filename))
                for entry in catalog.entries
            )
        )

    return FavoritesBackupInventory(
        host=host,
        requested_model=requested_model,
        target_model=target_model,
        format_version=format_version,
        catalog_bytes=catalog_bytes,
        catalog_source=catalog_source,
        catalog=catalog,
        lists=lists,
        documents=tuple(documents),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary_name)
        raise


def _manifest(inventory: FavoritesBackupInventory) -> bytes:
    document_lists: dict[str, list[str]] = {}
    for item in inventory.lists:
        document_lists.setdefault(item.filename, []).append(item.name)
    payload = {
        "schema": FAVORITES_BACKUP_SCHEMA,
        "schema_version": FAVORITES_BACKUP_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "host": inventory.host,
        "requested_model": inventory.requested_model,
        "target_model": inventory.target_model,
        "format_version": inventory.format_version,
        "catalog": {
            "filename": FAVORITES_BACKUP_CATALOG_NAME,
            "bytes": len(inventory.catalog_bytes),
            "sha256": _sha256(inventory.catalog_bytes),
            "records": len(inventory.catalog_source.records),
        },
        "favorites_lists": [
            {
                "name": item.name,
                "filename": item.filename,
                "raw_fields": list(item.raw_fields),
                "source_index": item.source_index,
            }
            for item in inventory.lists
        ],
        "documents": [
            {
                "filename": document.filename,
                "list_names": document_lists[document.filename],
                "bytes": len(document.content),
                "sha256": _sha256(document.content),
                "parse_error": document.parse_error,
                "records": (
                    len(document.source.records)
                    if document.source is not None
                    else None
                ),
            }
            for document in inventory.documents
        ],
        "warnings": list(inventory.warnings),
        "errors": list(inventory.errors),
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def backup_favorites(
    host: str,
    output_directory: Path,
    *,
    requested_model: str = "SDS200",
    port: int = FAVORITES_BACKUP_DEFAULT_FTP_PORT,
    timeout: float = FAVORITES_BACKUP_DEFAULT_FTP_TIMEOUT,
    ftp_factory: FavoritesBackupFtpFactory = _default_ftp_factory,
    now: datetime | None = None,
) -> FavoritesBackupResult:
    """Read the scanner Favorites tree and publish one exact local backup."""

    inventory = read_favorites_inventory(
        host,
        requested_model=requested_model,
        port=port,
        timeout=timeout,
        ftp_factory=ftp_factory,
    )
    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory)
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    stamp = (datetime.now(UTC) if now is None else now.astimezone(UTC)).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    base_name = f"sds200-favorites-{stamp}"
    staging_directory = output_directory / (
        f".{base_name}.tmp-{uuid.uuid4().hex}"
    )
    staging_directory.mkdir()
    try:
        _atomic_write(staging_directory / FAVORITES_BACKUP_CATALOG_NAME, inventory.catalog_bytes)
        for document in inventory.documents:
            _atomic_write(staging_directory / document.filename, document.content)
        _atomic_write(staging_directory / "manifest.json", _manifest(inventory))
        suffix = 0
        while True:
            directory_name = base_name if suffix == 0 else f"{base_name}-{suffix:02d}"
            final_directory = output_directory / directory_name
            if final_directory.exists():
                suffix += 1
                continue
            try:
                staging_directory.rename(final_directory)
            except FileExistsError:
                suffix += 1
                continue
            break
    except BaseException:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise

    return FavoritesBackupResult(
        directory=final_directory,
        manifest_path=final_directory / "manifest.json",
        inventory=inventory,
    )


__all__ = [
    "FAVORITES_BACKUP_CATALOG_NAME",
    "FAVORITES_BACKUP_DEFAULT_FTP_PORT",
    "FAVORITES_BACKUP_DEFAULT_FTP_TIMEOUT",
    "FAVORITES_BACKUP_REMOTE_DIRECTORY",
    "FAVORITES_BACKUP_SCHEMA",
    "FAVORITES_BACKUP_SCHEMA_VERSION",
    "FavoritesBackupDocument",
    "FavoritesBackupError",
    "FavoritesBackupFtp",
    "FavoritesBackupFtpFactory",
    "FavoritesBackupInventory",
    "FavoritesBackupList",
    "FavoritesBackupResult",
    "backup_favorites",
    "read_favorites_inventory",
]
