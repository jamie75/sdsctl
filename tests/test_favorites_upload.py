from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sds200.favorites_backup import FavoritesBackupResult, backup_favorites
from sds200.favorites_file import parse_favorites_file
from sds200.favorites_programming import (
    FavoritesProgrammingEdit,
    FavoritesProgrammingError,
    FavoritesProgrammingPlan,
    generate_candidate_image,
    load_programming_source,
)
from sds200.favorites_upload import (
    FavoritesProgrammingPreflight,
    execute_favorites_programming,
    prepare_favorites_programming,
    run_favorites_programming,
)

FIXTURES = Path(__file__).parent / "fixtures" / "favorites"
CATALOG = (FIXTURES / "synthetic-f_list.cfg").read_bytes()
HPD = (FIXTURES / "synthetic-favorites.hpd").read_bytes()


def _backup(root: Path, *, hpd: bytes = HPD) -> Path:
    root.mkdir()
    (root / "f_list.cfg").write_bytes(CATALOG)
    (root / "f_000001.hpd").write_bytes(hpd)
    catalog_source = parse_favorites_file(CATALOG)
    hpd_source = parse_favorites_file(hpd)
    manifest = {
        "schema": "sdsctl.favorites-backup",
        "schema_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "host": "fixture",
        "requested_model": "SDS200",
        "target_model": "BCDx36HP",
        "format_version": "1.00",
        "catalog": {
            "filename": "f_list.cfg",
            "bytes": len(CATALOG),
            "sha256": hashlib.sha256(CATALOG).hexdigest(),
            "records": len(catalog_source.records),
        },
        "favorites_lists": [
            {
                "name": "Synthetic Favorites",
                "filename": "f_000001.hpd",
                "raw_fields": list(catalog_source.records[2].fields),
                "source_index": 2,
            }
        ],
        "documents": [
            {
                "filename": "f_000001.hpd",
                "list_names": ["Synthetic Favorites"],
                "bytes": len(hpd),
                "sha256": hashlib.sha256(hpd).hexdigest(),
                "parse_error": None,
                "records": len(hpd_source.records),
            }
        ],
        "warnings": [],
        "errors": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _candidate(tmp_path: Path) -> Path:
    source = load_programming_source(_backup(tmp_path / "source"))
    plan = FavoritesProgrammingPlan(
        (
            FavoritesProgrammingEdit(
                favorites_list="Synthetic Favorites",
                filename="f_000001.hpd",
                record_index=14,
                record_type="TGID",
                field="name",
                expected_old_value="Synthetic Dispatch",
                new_value="Temporary Dispatch",
            ),
        )
    )
    return generate_candidate_image(source, tmp_path / "candidate", plan).directory


class _BackupFtp:
    def __init__(self, *, hpd: bytes = HPD) -> None:
        self.files = {"f_list.cfg": CATALOG, "f_000001.hpd": hpd}
        self.calls: list[tuple[object, ...]] = []

    def connect(self, host: str, port: int, timeout: float) -> str:
        self.calls.append(("connect", host, port, timeout))
        return "connected"

    def login(self, user: str, passwd: str) -> str:
        self.calls.append(("login", user, passwd))
        return "logged in"

    def set_pasv(self, val: bool) -> None:
        self.calls.append(("set_pasv", val))

    def cwd(self, dirname: str) -> str:
        self.calls.append(("cwd", dirname))
        return "directory changed"

    def nlst(self) -> list[str]:
        self.calls.append(("nlst",))
        return list(self.files)

    def retrbinary(
        self,
        cmd: str,
        callback: Callable[[bytes], object],
        blocksize: int = 8192,
    ) -> str:
        self.calls.append(("retrbinary", cmd))
        callback(self.files[cmd.removeprefix("RETR ")])
        return "retrieved"

    def quit(self) -> str:
        self.calls.append(("quit",))
        return "closed"

    def close(self) -> None:
        self.calls.append(("close",))


class _WriteFtp:
    def __init__(self, *, fail: str | None = None, retrieved: bytes | None = None) -> None:
        self.fail = fail
        self.retrieved = retrieved
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple[object, ...]] = []

    def connect(self, host: str, port: int, timeout: float) -> str:
        self.calls.append(("connect", host, port, timeout))
        if self.fail == "connect":
            raise OSError("connect failed")
        return "connected"

    def login(self, user: str, passwd: str) -> str:
        self.calls.append(("login", user, passwd))
        if self.fail == "login":
            raise RuntimeError("password=do-not-log")
        return "logged in"

    def set_pasv(self, val: bool) -> None:
        self.calls.append(("set_pasv", val))

    def cwd(self, dirname: str) -> str:
        self.calls.append(("cwd", dirname))
        return "directory changed"

    def storbinary(self, cmd: str, fp: io.BytesIO, blocksize: int = 8192) -> str:
        self.calls.append(("storbinary", cmd))
        if self.fail == "store":
            raise RuntimeError("store failed")
        self.files[cmd.removeprefix("STOR ")] = fp.read()
        return "stored"

    def retrbinary(
        self,
        cmd: str,
        callback: Callable[[bytes], object],
        blocksize: int = 8192,
    ) -> str:
        self.calls.append(("retrbinary", cmd))
        if self.fail == "retrieve":
            raise RuntimeError("retrieve failed")
        filename = cmd.removeprefix("RETR ")
        callback(self.retrieved if self.retrieved is not None else self.files[filename])
        return "retrieved"

    def quit(self) -> str:
        self.calls.append(("quit",))
        if self.fail == "quit":
            raise RuntimeError("quit failed")
        return "closed"

    def close(self) -> None:
        self.calls.append(("close",))


class _Scanner:
    def __init__(self, *, enter_error: Exception | None = None) -> None:
        self.calls: list[str] = []
        self.enter_error = enter_error

    def enter_ftp_mode(self, *, timeout: float = 5.0) -> None:
        self.calls.append("enter")
        if self.enter_error is not None:
            raise self.enter_error

    def exit_ftp_mode(self, *, timeout: float = 5.0) -> None:
        self.calls.append("exit")


def _preflight(
    tmp_path: Path,
    *,
    scanner_hpd: bytes = HPD,
) -> FavoritesProgrammingPreflight:
    candidate = _candidate(tmp_path)
    backup_ftp = _BackupFtp(hpd=scanner_hpd)

    def backup(
        host: str,
        output: Path,
        **kwargs: Any,
    ) -> FavoritesBackupResult:
        return backup_favorites(
            host,
            output,
            ftp_factory=lambda: backup_ftp,
            **kwargs,
        )

    return prepare_favorites_programming(
        candidate,
        host="192.0.2.5",
        safety_backup_output=tmp_path / "safety",
        backup_function=backup,
    )


def test_dry_run_requires_no_scanner_or_password_and_performs_only_backup(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)
    backup_ftp = _BackupFtp()

    def backup(
        host: str,
        output: Path,
        **kwargs: Any,
    ) -> FavoritesBackupResult:
        return backup_favorites(host, output, ftp_factory=lambda: backup_ftp, **kwargs)

    result = run_favorites_programming(
        candidate,
        host="192.0.2.5",
        safety_backup_output=tmp_path / "safety",
        backup_function=backup,
    )

    assert result.success is True
    assert result.dry_run is True
    assert result.gfm_entered is False
    assert result.ftp_authenticated is False
    assert result.files_attempted == ()
    assert not any(call[0] == "storbinary" for call in backup_ftp.calls)
    assert "password" not in result.receipt_path.read_text().lower()


def test_cli_defaults_to_dry_run_and_accepts_explicit_execute() -> None:
    from sds200.cli import build_parser

    dry_run = build_parser().parse_args(
        [
            "--host",
            "192.0.2.5",
            "favorites-program",
            "--candidate",
            "/tmp/candidate",
            "--safety-backup-output",
            "/tmp/safety",
            "--dry-run",
        ]
    )
    execute = build_parser().parse_args(
        [
            "--host",
            "192.0.2.5",
            "favorites-program",
            "--candidate",
            "/tmp/candidate",
            "--safety-backup-output",
            "/tmp/safety",
            "--execute",
        ]
    )

    assert dry_run.action == "favorites-program"
    assert dry_run.dry_run is True
    assert dry_run.execute is False
    assert execute.execute is True


def test_candidate_tampering_is_rejected_before_fresh_backup(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    (candidate / "f_000001.hpd").write_bytes(
        (candidate / "f_000001.hpd").read_bytes() + b"tampered"
    )
    called = False

    def backup(*args: object, **kwargs: object) -> FavoritesBackupResult:
        nonlocal called
        called = True
        raise AssertionError("fresh backup should not run")

    with pytest.raises(FavoritesProgrammingError):
        prepare_favorites_programming(
            candidate,
            host="192.0.2.5",
            safety_backup_output=tmp_path / "safety",
            backup_function=backup,
        )
    assert called is False


def test_fresh_scanner_hash_mismatch_stops_before_gfm(tmp_path: Path) -> None:
    scanner_hpd = HPD.replace(b"Synthetic Dispatch", b"Changed by scanner")
    candidate = _candidate(tmp_path)
    backup_ftp = _BackupFtp(hpd=scanner_hpd)

    def backup(
        host: str,
        output: Path,
        **kwargs: Any,
    ) -> FavoritesBackupResult:
        return backup_favorites(host, output, ftp_factory=lambda: backup_ftp, **kwargs)

    with pytest.raises(FavoritesProgrammingError, match="stale or changed"):
        prepare_favorites_programming(
            candidate,
            host="192.0.2.5",
            safety_backup_output=tmp_path / "safety",
            backup_function=backup,
        )


def test_successful_execution_is_passive_and_byte_verified(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    scanner = _Scanner()
    ftp = _WriteFtp()

    result = execute_favorites_programming(
        preflight,
        scanner,
        username="uniden",
        password="secret-not-for-logs",
        ftp_factory=lambda: ftp,
    )

    assert result.success is True
    assert result.gfm_entered is True
    assert result.ftp_authenticated is True
    assert result.files_attempted == ("f_000001.hpd",)
    assert result.files_successfully_written == ("f_000001.hpd",)
    assert result.file_results[0].retrieved_sha256 == result.file_results[0].expected_sha256
    assert result.ftp_closed is True
    assert result.efm_attempted is True
    assert result.efm_succeeded is True
    assert scanner.calls == ["enter", "exit"]
    assert ftp.calls == [
        ("connect", "192.0.2.5", 21, 10.0),
        ("set_pasv", True),
        ("login", "uniden", "secret-not-for-logs"),
        ("cwd", "/favorites_lists"),
        ("storbinary", "STOR f_000001.hpd"),
        ("retrbinary", "RETR f_000001.hpd"),
        ("quit",),
    ]
    assert "secret-not-for-logs" not in result.receipt_path.read_text()


@pytest.mark.parametrize("failure", ["connect", "login", "store", "retrieve"])
def test_execution_failures_stop_and_exit_ftp_mode(
    tmp_path: Path,
    failure: str,
) -> None:
    preflight = _preflight(tmp_path)
    scanner = _Scanner()
    ftp = _WriteFtp(fail=failure)

    result = execute_favorites_programming(
        preflight,
        scanner,
        username="uniden",
        password="secret-not-for-logs",
        ftp_factory=lambda: ftp,
    )

    assert result.success is False
    assert result.primary_error is not None
    assert "secret-not-for-logs" not in result.primary_error
    assert scanner.calls == ["enter", "exit"]
    assert result.efm_attempted is True


def test_byte_mismatch_stops_and_reports_no_successful_write(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    scanner = _Scanner()
    ftp = _WriteFtp(retrieved=b"wrong")

    result = execute_favorites_programming(
        preflight,
        scanner,
        username="uniden",
        password="secret",
        ftp_factory=lambda: ftp,
    )

    assert result.success is False
    assert result.files_successfully_written == ()
    assert result.primary_error is not None
    assert "post-write byte verification failed" in result.primary_error
    assert scanner.calls == ["enter", "exit"]


def test_cleanup_error_does_not_hide_programming_error(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)
    scanner = _Scanner()
    ftp = _WriteFtp(fail="store")

    def broken_quit() -> str:
        ftp.calls.append(("quit",))
        raise RuntimeError("quit cleanup failed")

    ftp.quit = broken_quit  # type: ignore[method-assign]
    result = execute_favorites_programming(
        preflight,
        scanner,
        username="uniden",
        password="secret",
        ftp_factory=lambda: ftp,
    )

    assert result.success is False
    assert "store failed" in (result.primary_error or "")
    assert any("quit cleanup failed" in error for error in result.cleanup_errors)
    assert result.efm_succeeded is True
