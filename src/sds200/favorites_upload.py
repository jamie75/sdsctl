"""Safely preflight and optionally upload a validated SDS200 Favorites image."""

from __future__ import annotations

import ftplib
import hashlib
import io
import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from .favorites_backup import (
    FAVORITES_BACKUP_DEFAULT_FTP_PORT,
    FAVORITES_BACKUP_DEFAULT_FTP_TIMEOUT,
    FAVORITES_BACKUP_REMOTE_DIRECTORY,
    FavoritesBackupResult,
    backup_favorites,
)
from .favorites_programming import (
    FavoritesProgrammingError,
    FavoritesProgrammingSource,
    FavoritesProgrammingValidation,
    load_programming_source,
    validate_candidate_image,
)

logger = logging.getLogger(__name__)

FAVORITES_UPLOAD_DEFAULT_SCANNER_TIMEOUT = 5.0
FAVORITES_UPLOAD_REMOTE_DIRECTORY = FAVORITES_BACKUP_REMOTE_DIRECTORY
_TRANSFER_BLOCK_SIZE = 64 * 1024


class FavoritesProgrammingScanner(Protocol):
    def enter_ftp_mode(self, *, timeout: float = 5.0) -> None: ...

    def exit_ftp_mode(self, *, timeout: float = 5.0) -> None: ...


class FavoritesProgrammingFtp(Protocol):
    def connect(self, host: str, port: int, timeout: float) -> str: ...

    def login(self, user: str, passwd: str) -> str: ...

    def set_pasv(self, val: bool) -> None: ...

    def cwd(self, dirname: str) -> str: ...

    def storbinary(self, cmd: str, fp: io.BytesIO, blocksize: int = 8192) -> str: ...

    def retrbinary(
        self,
        cmd: str,
        callback: Callable[[bytes], object],
        blocksize: int = 8192,
    ) -> str: ...

    def quit(self) -> str: ...

    def close(self) -> None: ...


FavoritesProgrammingFtpFactory = Callable[[], FavoritesProgrammingFtp]
FavoritesBackupFunction = Callable[..., FavoritesBackupResult]


@dataclass(frozen=True, slots=True)
class FavoritesProgrammingPreflight:
    """Validated local candidate plus the fresh scanner safety backup."""

    host: str
    candidate_directory: Path
    source: FavoritesProgrammingSource
    candidate_manifest_sha256: str
    fresh_safety_backup: FavoritesBackupResult
    validation: FavoritesProgrammingValidation
    changed_files: tuple[str, ...]
    prewrite_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class FavoritesProgrammingFileResult:
    filename: str
    expected_sha256: str
    prewrite_sha256: str
    retrieved_sha256: str | None
    status: str


@dataclass(frozen=True, slots=True)
class FavoritesProgrammingResult:
    """Redacted, structured outcome suitable for CLI or future UI callers."""

    success: bool
    dry_run: bool
    host: str
    candidate_directory: Path
    candidate_manifest_sha256: str
    source_backup_directory: Path
    source_manifest_sha256: str
    fresh_safety_backup_path: Path
    source_validation_succeeded: bool
    scanner_stale_check_succeeded: bool
    gfm_entered: bool
    ftp_authenticated: bool
    files_attempted: tuple[str, ...]
    files_successfully_written: tuple[str, ...]
    file_results: tuple[FavoritesProgrammingFileResult, ...]
    ftp_closed: bool
    efm_attempted: bool
    efm_succeeded: bool
    primary_error: str | None
    cleanup_errors: tuple[str, ...]
    receipt_path: Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _redacted_error(error: BaseException, password: str | None = None) -> str:
    detail = str(error).strip() or error.__class__.__name__
    if password:
        detail = detail.replace(password, "<redacted>")
    return f"{error.__class__.__name__}: {detail}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FavoritesProgrammingError(
            f"cannot read candidate manifest: {error}"
        ) from None
    if not isinstance(value, dict):
        raise FavoritesProgrammingError("candidate manifest must contain an object")
    return value


def _candidate_source(
    candidate_directory: Path,
) -> tuple[FavoritesProgrammingSource, str]:
    directory = candidate_directory.expanduser().resolve()
    if not directory.is_dir():
        raise FavoritesProgrammingError("candidate directory does not exist")
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    source_meta = manifest.get("source_backup")
    if not isinstance(source_meta, Mapping):
        raise FavoritesProgrammingError("candidate source_backup metadata is missing")
    source_path = source_meta.get("path")
    manifest_sha = source_meta.get("manifest_sha256")
    if not isinstance(source_path, str) or not source_path:
        raise FavoritesProgrammingError("candidate source backup path is missing")
    if not isinstance(manifest_sha, str) or not manifest_sha:
        raise FavoritesProgrammingError("candidate source manifest identity is missing")
    source = load_programming_source(Path(source_path))
    if source.backup_directory != Path(source_path).expanduser().resolve():
        raise FavoritesProgrammingError("candidate source backup path is invalid")
    return source, _sha256(manifest_path.read_bytes())


def _candidate_files(
    source: FavoritesProgrammingSource,
    validation: FavoritesProgrammingValidation,
) -> tuple[str, ...]:
    if validation.files_added or validation.files_removed:
        raise FavoritesProgrammingError(
            "Favorites programming candidate must not add or remove files"
        )
    if "f_list.cfg" in validation.files_changed:
        raise FavoritesProgrammingError(
            "Favorites programming candidate must not change f_list.cfg"
        )
    changed = tuple(sorted(validation.files_changed))
    if any(not name.lower().endswith(".hpd") for name in changed):
        raise FavoritesProgrammingError(
            "Favorites programming candidate may change only HPD files"
        )
    expected = set(source.programming_filenames)
    if any(name not in expected for name in changed):
        raise FavoritesProgrammingError(
            "Favorites programming candidate contains an unexpected file"
        )
    return changed


def _candidate_bytes(directory: Path, filenames: tuple[str, ...]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for filename in filenames:
        try:
            result[filename] = (directory / filename).read_bytes()
        except OSError as error:
            raise FavoritesProgrammingError(
                f"cannot read candidate file {filename}: {error}"
            ) from None
    return result


def _fresh_matches_source(
    source: FavoritesProgrammingSource,
    fresh: FavoritesBackupResult,
) -> dict[str, str]:
    fresh_source = load_programming_source(fresh.directory)
    source_bytes = {
        "f_list.cfg": source.snapshot.catalog_bytes,
        **{document.filename: document.content for document in source.snapshot.documents},
    }
    fresh_bytes = {
        "f_list.cfg": fresh_source.snapshot.catalog_bytes,
        **{document.filename: document.content for document in fresh_source.snapshot.documents},
    }
    if set(source_bytes) != set(fresh_bytes):
        raise FavoritesProgrammingError(
            "scanner Favorites files do not match the candidate source"
        )
    mismatched = tuple(
        filename
        for filename in sorted(source_bytes)
        if source_bytes[filename] != fresh_bytes[filename]
    )
    if mismatched:
        details = ", ".join(
            f"{filename} expected={_sha256(source_bytes[filename])} "
            f"actual={_sha256(fresh_bytes[filename])}"
            for filename in mismatched
        )
        raise FavoritesProgrammingError(
            f"scanner Favorites state is stale or changed: {details}"
        )
    return {filename: _sha256(data) for filename, data in fresh_bytes.items()}


def prepare_favorites_programming(
    candidate_directory: Path,
    *,
    host: str,
    safety_backup_output: Path,
    ftp_port: int = FAVORITES_BACKUP_DEFAULT_FTP_PORT,
    ftp_timeout: float = FAVORITES_BACKUP_DEFAULT_FTP_TIMEOUT,
    backup_function: FavoritesBackupFunction = backup_favorites,
) -> FavoritesProgrammingPreflight:
    """Validate a candidate and compare it with a mandatory fresh backup."""
    candidate = candidate_directory.expanduser().resolve()
    source, candidate_manifest_sha = _candidate_source(candidate)
    validation = validate_candidate_image(source, candidate)
    changed_files = _candidate_files(source, validation)
    candidate_data = _candidate_bytes(candidate, changed_files)
    if any(_sha256(data) == "" for data in candidate_data.values()):
        raise FavoritesProgrammingError("candidate file hash calculation failed")

    logger.info(
        "Favorites programming preflight candidate=%s host=%s files=%s",
        candidate,
        host,
        ",".join(changed_files) or "none",
    )
    fresh = backup_function(
        host,
        safety_backup_output,
        requested_model="SDS200",
        port=ftp_port,
        timeout=ftp_timeout,
    )
    prewrite_sha = _fresh_matches_source(source, fresh)
    return FavoritesProgrammingPreflight(
        host=host,
        candidate_directory=candidate,
        source=source,
        candidate_manifest_sha256=candidate_manifest_sha,
        fresh_safety_backup=fresh,
        validation=validation,
        changed_files=changed_files,
        prewrite_sha256=prewrite_sha,
    )


def _receipt_path(preflight: FavoritesProgrammingPreflight) -> Path:
    return preflight.fresh_safety_backup.directory.parent / (
        f"{preflight.fresh_safety_backup.directory.name}-programming-result.json"
    )


def _atomic_write(path: Path, data: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _result_payload(result: FavoritesProgrammingResult) -> dict[str, Any]:
    return {
        "schema": "sdsctl.favorites-programming-result",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "scanner": {"host": result.host, "model": "SDS200"},
        "candidate": {
            "directory": str(result.candidate_directory),
            "manifest_sha256": result.candidate_manifest_sha256,
        },
        "source_backup": {
            "directory": str(result.source_backup_directory),
            "manifest_sha256": result.source_manifest_sha256,
        },
        "fresh_safety_backup": str(result.fresh_safety_backup_path),
        "result": {
            "success": result.success,
            "dry_run": result.dry_run,
            "source_validation_succeeded": result.source_validation_succeeded,
            "scanner_stale_check_succeeded": result.scanner_stale_check_succeeded,
            "gfm_entered": result.gfm_entered,
            "ftp_authenticated": result.ftp_authenticated,
            "ftp_closed": result.ftp_closed,
            "efm_attempted": result.efm_attempted,
            "efm_succeeded": result.efm_succeeded,
        },
        "files": [
            {
                "filename": item.filename,
                "expected_sha256": item.expected_sha256,
                "prewrite_sha256": item.prewrite_sha256,
                "retrieved_sha256": item.retrieved_sha256,
                "status": item.status,
            }
            for item in result.file_results
        ],
        "files_attempted": list(result.files_attempted),
        "files_successfully_written": list(result.files_successfully_written),
        "primary_error": result.primary_error,
        "cleanup_errors": list(result.cleanup_errors),
    }


def _write_receipt(result: FavoritesProgrammingResult) -> None:
    _atomic_write(
        result.receipt_path,
        (json.dumps(_result_payload(result), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _dry_run_result(
    preflight: FavoritesProgrammingPreflight,
) -> FavoritesProgrammingResult:
    files = tuple(
        FavoritesProgrammingFileResult(
            filename=filename,
            expected_sha256=_sha256(
                (preflight.candidate_directory / filename).read_bytes()
            ),
            prewrite_sha256=preflight.prewrite_sha256[filename],
            retrieved_sha256=None,
            status="would_write",
        )
        for filename in preflight.changed_files
    )
    return FavoritesProgrammingResult(
        success=True,
        dry_run=True,
        host=preflight.host,
        candidate_directory=preflight.candidate_directory,
        candidate_manifest_sha256=preflight.candidate_manifest_sha256,
        source_backup_directory=preflight.source.backup_directory,
        source_manifest_sha256=preflight.source.manifest_sha256,
        fresh_safety_backup_path=preflight.fresh_safety_backup.directory,
        source_validation_succeeded=True,
        scanner_stale_check_succeeded=True,
        gfm_entered=False,
        ftp_authenticated=False,
        files_attempted=(),
        files_successfully_written=(),
        file_results=files,
        ftp_closed=False,
        efm_attempted=False,
        efm_succeeded=False,
        primary_error=None,
        cleanup_errors=(),
        receipt_path=_receipt_path(preflight),
    )


def execute_favorites_programming(
    preflight: FavoritesProgrammingPreflight,
    scanner: FavoritesProgrammingScanner,
    *,
    username: str,
    password: str,
    ftp_port: int = FAVORITES_BACKUP_DEFAULT_FTP_PORT,
    ftp_timeout: float = FAVORITES_BACKUP_DEFAULT_FTP_TIMEOUT,
    scanner_timeout: float = FAVORITES_UPLOAD_DEFAULT_SCANNER_TIMEOUT,
    ftp_factory: FavoritesProgrammingFtpFactory | None = None,
) -> FavoritesProgrammingResult:
    """Write only validated HPDs, verify each by RETR, and always exit GFM."""
    if not username:
        raise ValueError("Favorites programming FTP username is required")
    if not password:
        raise ValueError("Favorites programming FTP password is required")
    if not preflight.changed_files:
        result = _dry_run_result(preflight)
        result = replace(result, dry_run=False)
        _write_receipt(result)
        return result

    ftp: FavoritesProgrammingFtp | None = None
    gfm_entered = False
    ftp_authenticated = False
    ftp_closed = False
    efm_attempted = False
    efm_succeeded = False
    primary_error: str | None = None
    cleanup_errors: list[str] = []
    attempted: list[str] = []
    written: list[str] = []
    file_results: list[FavoritesProgrammingFileResult] = []

    try:
        logger.info("Favorites programming stage=enter_gfm host=%s", preflight.host)
        scanner.enter_ftp_mode(timeout=scanner_timeout)
        gfm_entered = True
        factory = ftp_factory or cast(
            FavoritesProgrammingFtpFactory,
            ftplib.FTP,
        )
        ftp = factory()
        ftp.connect(preflight.host, ftp_port, ftp_timeout)
        ftp.set_pasv(True)
        ftp.login(username, password)
        ftp_authenticated = True
        ftp.cwd(FAVORITES_UPLOAD_REMOTE_DIRECTORY)

        for filename in preflight.changed_files:
            attempted.append(filename)
            candidate_bytes = (preflight.candidate_directory / filename).read_bytes()
            expected_sha = _sha256(candidate_bytes)
            prewrite_sha = preflight.prewrite_sha256[filename]
            logger.info(
                "Favorites programming stage=write_verify host=%s file=%s "
                "prewrite_sha256=%s candidate_sha256=%s",
                preflight.host,
                filename,
                prewrite_sha,
                expected_sha,
            )
            ftp.storbinary(
                f"STOR {filename}",
                io.BytesIO(candidate_bytes),
                blocksize=_TRANSFER_BLOCK_SIZE,
            )
            received = bytearray()
            ftp.retrbinary(
                f"RETR {filename}",
                received.extend,
                blocksize=_TRANSFER_BLOCK_SIZE,
            )
            retrieved_sha = _sha256(bytes(received))
            if bytes(received) != candidate_bytes:
                raise FavoritesProgrammingError(
                    f"post-write byte verification failed for {filename}: "
                    f"expected={expected_sha} actual={retrieved_sha}"
                )
            file_results.append(
                FavoritesProgrammingFileResult(
                    filename,
                    expected_sha,
                    prewrite_sha,
                    retrieved_sha,
                    "verified",
                )
            )
            written.append(filename)
    except Exception as error:
        primary_error = _redacted_error(error, password)
        logger.error(
            "Favorites programming failed host=%s files_written=%s error=%s",
            preflight.host,
            ",".join(written) or "none",
            primary_error,
        )
    finally:
        if ftp is not None:
            try:
                ftp.quit()
            except Exception as error:
                cleanup_errors.append(_redacted_error(error, password))
                try:
                    ftp.close()
                except Exception as close_error:
                    cleanup_errors.append(_redacted_error(close_error, password))
                else:
                    ftp_closed = True
            else:
                ftp_closed = True
        if gfm_entered:
            efm_attempted = True
            try:
                scanner.exit_ftp_mode(timeout=scanner_timeout)
            except Exception as error:
                cleanup_errors.append(_redacted_error(error, password))
            else:
                efm_succeeded = True

    success = (
        primary_error is None
        and len(written) == len(preflight.changed_files)
        and ftp_closed
        and efm_succeeded
    )
    result = FavoritesProgrammingResult(
        success=success,
        dry_run=False,
        host=preflight.host,
        candidate_directory=preflight.candidate_directory,
        candidate_manifest_sha256=preflight.candidate_manifest_sha256,
        source_backup_directory=preflight.source.backup_directory,
        source_manifest_sha256=preflight.source.manifest_sha256,
        fresh_safety_backup_path=preflight.fresh_safety_backup.directory,
        source_validation_succeeded=True,
        scanner_stale_check_succeeded=True,
        gfm_entered=gfm_entered,
        ftp_authenticated=ftp_authenticated,
        files_attempted=tuple(attempted),
        files_successfully_written=tuple(written),
        file_results=tuple(file_results),
        ftp_closed=ftp_closed,
        efm_attempted=efm_attempted,
        efm_succeeded=efm_succeeded,
        primary_error=primary_error,
        cleanup_errors=tuple(cleanup_errors),
        receipt_path=_receipt_path(preflight),
    )
    try:
        _write_receipt(result)
    except OSError as error:
        cleanup_errors.append(f"receipt: {_redacted_error(error)}")
        result = replace(
            result,
            success=False,
            cleanup_errors=tuple(cleanup_errors),
        )
    return result


def run_favorites_programming(
    candidate_directory: Path,
    *,
    host: str,
    safety_backup_output: Path,
    execute: bool = False,
    scanner: FavoritesProgrammingScanner | None = None,
    username: str = "uniden",
    password: str | None = None,
    ftp_port: int = FAVORITES_BACKUP_DEFAULT_FTP_PORT,
    ftp_timeout: float = FAVORITES_BACKUP_DEFAULT_FTP_TIMEOUT,
    scanner_timeout: float = FAVORITES_UPLOAD_DEFAULT_SCANNER_TIMEOUT,
    backup_function: FavoritesBackupFunction = backup_favorites,
    ftp_factory: FavoritesProgrammingFtpFactory | None = None,
) -> FavoritesProgrammingResult:
    """Run preflight by default; require execute plus scanner and password to write."""
    preflight = prepare_favorites_programming(
        candidate_directory,
        host=host,
        safety_backup_output=safety_backup_output,
        ftp_port=ftp_port,
        ftp_timeout=ftp_timeout,
        backup_function=backup_function,
    )
    if not execute:
        result = _dry_run_result(preflight)
        _write_receipt(result)
        return result
    if scanner is None:
        raise ValueError("scanner is required for --execute")
    if password is None:
        raise ValueError("password is required for --execute")
    return execute_favorites_programming(
        preflight,
        scanner,
        username=username,
        password=password,
        ftp_port=ftp_port,
        ftp_timeout=ftp_timeout,
        scanner_timeout=scanner_timeout,
        ftp_factory=ftp_factory,
    )


__all__ = [
    "FAVORITES_UPLOAD_DEFAULT_SCANNER_TIMEOUT",
    "FAVORITES_UPLOAD_REMOTE_DIRECTORY",
    "FavoritesProgrammingFileResult",
    "FavoritesProgrammingFtp",
    "FavoritesProgrammingFtpFactory",
    "FavoritesProgrammingPreflight",
    "FavoritesProgrammingResult",
    "FavoritesProgrammingScanner",
    "execute_favorites_programming",
    "prepare_favorites_programming",
    "run_favorites_programming",
]
