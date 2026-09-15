from __future__ import annotations

import io
from collections.abc import Callable

import pytest

from sds200.cli import build_parser
from sds200.ftp_write_test import (
    FTP_WRITE_TEST_PAYLOAD,
    FtpWriteTestResult,
    run_ftp_write_test,
)


class _FakeScanner:
    def __init__(
        self,
        *,
        enter_error: Exception | None = None,
        exit_error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, float]] = []
        self.enter_error = enter_error
        self.exit_error = exit_error

    def enter_ftp_mode(self, *, timeout: float = 5.0) -> None:
        self.calls.append(("enter", timeout))
        if self.enter_error is not None:
            raise self.enter_error

    def exit_ftp_mode(self, *, timeout: float = 5.0) -> None:
        self.calls.append(("exit", timeout))
        if self.exit_error is not None:
            raise self.exit_error


class _FakeFtp:
    def __init__(
        self,
        *,
        fail: str | None = None,
        initial_files: list[str] | None = None,
        retrieved: bytes = FTP_WRITE_TEST_PAYLOAD,
    ) -> None:
        self.fail = fail
        self.files = {
            name: b"existing"
            for name in (initial_files or [])
        }
        self.retrieved = retrieved
        self.calls: list[tuple[object, ...]] = []

    def connect(
        self,
        host: str,
        port: int = 0,
        timeout: float | None = None,
    ) -> str:
        self.calls.append(("connect", host, port, timeout))
        if self.fail == "connect":
            raise OSError("connection failed")
        return "connected"

    def login(
        self,
        user: str = "anonymous",
        passwd: str = "",
        acct: str = "",
    ) -> str:
        self.calls.append(("login", user, passwd))
        if self.fail == "login":
            raise RuntimeError("authentication failed")
        return "logged in"

    def set_pasv(self, val: bool) -> None:
        self.calls.append(("set_pasv", val))

    def nlst(self, *args: str) -> list[str]:
        self.calls.append(("nlst",))
        if self.fail == "list":
            raise RuntimeError("listing failed")
        return list(self.files)

    def storbinary(
        self,
        cmd: str,
        fp: io.BytesIO,
        blocksize: int = 8192,
        callback: Callable[[bytes], object] | None = None,
        rest: int | None = None,
    ) -> str:
        self.calls.append(("storbinary", cmd))
        if self.fail == "store":
            raise RuntimeError("store failed")
        filename = cmd.removeprefix("STOR ")
        self.files[filename] = fp.read()
        return "stored"

    def retrbinary(
        self,
        cmd: str,
        callback: Callable[[bytes], object],
        blocksize: int = 8192,
        rest: int | None = None,
    ) -> str:
        self.calls.append(("retrbinary", cmd))
        if self.fail == "retrieve":
            raise RuntimeError("retrieve failed")
        callback(self.retrieved)
        return "retrieved"

    def delete(self, filename: str) -> str:
        self.calls.append(("delete", filename))
        if self.fail == "delete":
            raise RuntimeError("delete failed")
        self.files.pop(filename, None)
        return "deleted"

    def quit(self) -> str:
        self.calls.append(("quit",))
        if self.fail == "quit":
            raise RuntimeError("quit failed")
        return "closed"

    def close(self) -> None:
        self.calls.append(("close",))


def _run(
    ftp: _FakeFtp,
    scanner: _FakeScanner | None = None,
) -> FtpWriteTestResult:
    return run_ftp_write_test(
        scanner or _FakeScanner(),
        host="192.0.2.10",
        username="uniden",
        password="not-for-logs",
        ftp_factory=lambda: ftp,
        filename_factory=lambda: "sdsctl-write-test-fixed.txt",
    )


def test_parser_exposes_disposable_ftp_write_test() -> None:
    args = build_parser().parse_args(
        [
            "--host",
            "192.0.2.10",
            "ftp-write-test",
            "--ftp-username",
            "uniden",
            "--ftp-port",
            "21",
        ]
    )

    assert args.action == "ftp-write-test"
    assert args.ftp_username == "uniden"
    assert args.ftp_port == 21
    assert args.ftp_timeout == 10.0
    assert args.scanner_timeout == 5.0


def test_successful_round_trip_is_passive_and_cleans_up() -> None:
    scanner = _FakeScanner()
    ftp = _FakeFtp()

    result = _run(ftp, scanner)

    assert result.succeeded is True
    assert result.cleanup_succeeded is True
    assert scanner.calls == [("enter", 5.0), ("exit", 5.0)]
    assert ftp.files == {}
    assert ftp.calls == [
        ("connect", "192.0.2.10", 21, 10.0),
        ("set_pasv", True),
        ("login", "uniden", "not-for-logs"),
        ("nlst",),
        ("storbinary", "STOR sdsctl-write-test-fixed.txt"),
        ("retrbinary", "RETR sdsctl-write-test-fixed.txt"),
        ("nlst",),
        ("delete", "sdsctl-write-test-fixed.txt"),
        ("nlst",),
        ("quit",),
    ]


@pytest.mark.parametrize("failure", ["connect", "login", "store", "retrieve"])
def test_transfer_failure_still_exits_ftp_mode(failure: str) -> None:
    scanner = _FakeScanner()
    result = _run(_FakeFtp(fail=failure), scanner)

    assert result.succeeded is False
    assert scanner.calls[-1] == ("exit", 5.0)
    assert any(
        stage.name == "exit FTP mode" and stage.succeeded
        for stage in result.stages
    )


def test_delete_failure_is_reported_and_ftp_mode_is_exited() -> None:
    scanner = _FakeScanner()
    result = _run(_FakeFtp(fail="delete"), scanner)

    assert result.succeeded is False
    assert result.cleanup_succeeded is False
    assert scanner.calls == [("enter", 5.0), ("exit", 5.0)]
    assert any(
        stage.name == "delete disposable file" and not stage.succeeded
        for stage in result.stages
    )


def test_enter_failure_still_attempts_exit_without_ftp_connection() -> None:
    scanner = _FakeScanner(enter_error=TimeoutError("GFM timed out"))

    result = _run(_FakeFtp(), scanner)

    assert result.succeeded is False
    assert scanner.calls == [("enter", 5.0), ("exit", 5.0)]
    assert any(
        stage.name == "enter FTP mode" and not stage.succeeded
        for stage in result.stages
    )


def test_exit_failure_is_reported_without_leaking_password(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scanner = _FakeScanner(exit_error=RuntimeError("password not-for-logs"))
    result = _run(_FakeFtp(), scanner)

    assert result.succeeded is False
    assert any(
        stage.name == "exit FTP mode" and not stage.succeeded
        for stage in result.stages
    )
    assert "not-for-logs" not in caplog.text


@pytest.mark.parametrize(
    "filename",
    [
        "f_list.cfg",
        "profile.cfg",
        "scanner.inf",
        "existing.hpd",
        "../unsafe.txt",
    ],
)
def test_generated_filename_cannot_target_scanner_file(filename: str) -> None:
    scanner = _FakeScanner()
    ftp = _FakeFtp()

    with pytest.raises(ValueError):
        run_ftp_write_test(
            scanner,
            host="192.0.2.10",
            username="uniden",
            password="secret",
            ftp_factory=lambda: ftp,
            filename_factory=lambda: filename,
        )

    assert scanner.calls == []
    assert ftp.calls == []
    assert ftp.files == {}


def test_existing_generated_filename_is_never_overwritten() -> None:
    scanner = _FakeScanner()
    ftp = _FakeFtp(initial_files=["sdsctl-write-test-fixed.txt"])

    result = _run(ftp, scanner)

    assert result.succeeded is False
    assert ftp.files == {"sdsctl-write-test-fixed.txt": b"existing"}
    assert not any(call[0] == "storbinary" for call in ftp.calls)
    assert scanner.calls[-1] == ("exit", 5.0)


def test_retrieved_bytes_must_match_exact_payload() -> None:
    result = _run(
        _FakeFtp(retrieved=FTP_WRITE_TEST_PAYLOAD + b"extra"),
    )

    assert result.succeeded is False
    assert any(
        stage.name == "verify retrieved bytes" and not stage.succeeded
        for stage in result.stages
    )
