from __future__ import annotations

import ftplib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import sds200.cli as cli
from sds200.favorites_backup import (
    FAVORITES_BACKUP_REMOTE_DIRECTORY,
    FavoritesBackupError,
    backup_favorites,
    read_favorites_inventory,
)

CATALOG = (
    b"TargetModel\tSDS200\r\n"
    b"FormatVersion\t1.00\r\n"
    b"F-List\tNorth Carolina\tf_000001.hpd\textra\r\n"
    b"UnknownRecord\tkeep\tthis\n"
)
HPD = (
    b"TargetModel\tSDS200\r\n"
    b"FormatVersion\t1.00\r\n"
    b"Conventional\tGaston\r\n"
)


class FakeFtp:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.calls: list[tuple[str, object]] = []
        self.closed = False

    def connect(self, host: str, port: int, timeout: float) -> str:
        self.calls.append(("connect", (host, port, timeout)))
        return "connected"

    def login(self, user: str, passwd: str) -> str:
        self.calls.append(("login", (user, passwd)))
        return "logged in"

    def set_pasv(self, val: bool) -> None:
        self.calls.append(("set_pasv", val))

    def cwd(self, dirname: str) -> str:
        self.calls.append(("cwd", dirname))
        return "directory changed"

    def nlst(self) -> list[str]:
        self.calls.append(("nlst", None))
        return list(self.files)

    def retrbinary(self, cmd: str, callback, blocksize: int) -> str:
        self.calls.append(("retrbinary", (cmd, blocksize)))
        filename = cmd.removeprefix("RETR ")
        try:
            content = self.files[filename]
        except KeyError:
            raise ftplib.error_perm("550 missing") from None
        callback(content)
        return "transfer complete"

    def quit(self) -> str:
        self.calls.append(("quit", None))
        self.closed = True
        return "221 goodbye"

    def close(self) -> None:
        self.calls.append(("close", None))
        self.closed = True


def _factory(ftp: FakeFtp):
    return lambda: ftp


def test_parser_preserves_catalog_bytes_and_reports_inventory() -> None:
    ftp = FakeFtp({"f_list.cfg": CATALOG, "f_000001.hpd": HPD})
    inventory = read_favorites_inventory(
        "192.0.2.5",
        ftp_factory=_factory(ftp),
    )

    assert inventory.catalog_source.to_bytes() == CATALOG
    assert inventory.catalog_source.records[-1].raw_bytes == (
        b"UnknownRecord\tkeep\tthis\n"
    )
    assert inventory.target_model == "SDS200"
    assert inventory.format_version == "1.00"
    assert inventory.lists[0].name == "North Carolina"
    assert inventory.lists[0].raw_fields == (
        "North Carolina",
        "f_000001.hpd",
        "extra",
    )
    assert inventory.documents[0].source is not None
    assert inventory.documents[0].source.to_bytes() == HPD
    assert ("login", ("anonymous", "anonymous@")) in ftp.calls
    assert ("set_pasv", True) in ftp.calls
    assert ("cwd", FAVORITES_BACKUP_REMOTE_DIRECTORY) in ftp.calls
    assert ftp.closed


def test_backup_writes_exact_files_hashes_and_manifest_atomically(
    tmp_path: Path,
) -> None:
    ftp = FakeFtp({"f_list.cfg": CATALOG, "f_000001.hpd": HPD})
    result = backup_favorites(
        "192.0.2.5",
        tmp_path,
        ftp_factory=_factory(ftp),
        now=datetime(2026, 9, 15, 19, 0, tzinfo=UTC),
    )

    assert result.directory == tmp_path / "sds200-favorites-20260915T190000Z"
    assert (result.directory / "f_list.cfg").read_bytes() == CATALOG
    assert (result.directory / "f_000001.hpd").read_bytes() == HPD
    manifest = json.loads((result.directory / "manifest.json").read_text())
    assert manifest["schema"] == "sdsctl.favorites-backup"
    assert manifest["target_model"] == "SDS200"
    assert manifest["documents"][0]["filename"] == "f_000001.hpd"
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_duplicate_reference_is_warned_and_document_retrieved_once() -> None:
    catalog = CATALOG.replace(
        b"UnknownRecord\tkeep\tthis\n",
        b"F-List\tSecond\tf_000001.hpd\r\n",
    )
    ftp = FakeFtp({"f_list.cfg": catalog, "f_000001.hpd": HPD})
    inventory = read_favorites_inventory(
        "192.0.2.5",
        ftp_factory=_factory(ftp),
    )

    assert len(inventory.documents) == 1
    assert any("duplicate" in warning for warning in inventory.warnings)
    retrievals = [
        item for item in ftp.calls
        if item[0] == "retrbinary" and "f_000001.hpd" in str(item[1])
    ]
    assert len(retrievals) == 1


def test_missing_referenced_document_fails_without_backup_directory(
    tmp_path: Path,
) -> None:
    ftp = FakeFtp({"f_list.cfg": CATALOG})
    with pytest.raises(FavoritesBackupError, match="missing"):
        backup_favorites(
            "192.0.2.5",
            tmp_path,
            ftp_factory=_factory(ftp),
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("filename", ("../bad.hpd", "/bad.hpd", "dir/bad.hpd", "bad.txt"))
def test_unsafe_referenced_filename_is_rejected(filename: str) -> None:
    catalog = CATALOG.replace(b"f_000001.hpd", filename.encode("ascii"))
    ftp = FakeFtp({"f_list.cfg": catalog, filename: HPD})
    with pytest.raises(FavoritesBackupError, match="unsafe"):
        read_favorites_inventory(
            "192.0.2.5",
            ftp_factory=_factory(ftp),
        )


def test_malformed_hpd_is_preserved_and_reported() -> None:
    malformed = b"TargetModel\t\xff\r\n"
    ftp = FakeFtp({"f_list.cfg": CATALOG, "f_000001.hpd": malformed})
    inventory = read_favorites_inventory(
        "192.0.2.5",
        ftp_factory=_factory(ftp),
    )

    assert inventory.documents[0].content == malformed
    assert inventory.documents[0].source is None
    assert inventory.documents[0].parse_error is not None
    assert inventory.errors


def test_cli_parser_exposes_read_only_backup_without_scanner_options() -> None:
    args = cli.build_parser().parse_args(
        [
            "--host",
            "192.0.2.5",
            "favorites-backup",
            "--output",
            "/tmp/backups",
        ]
    )

    assert args.action == "favorites-backup"
    assert args.output == Path("/tmp/backups")
    assert args.ftp_port == 21
