from __future__ import annotations

import ftplib
import io
import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast

logger = logging.getLogger(__name__)

FTP_WRITE_TEST_DEFAULT_PORT = 21
FTP_WRITE_TEST_DEFAULT_TIMEOUT = 10.0
FTP_WRITE_TEST_PAYLOAD = b"sdsctl FTP write validation\n"
_PROTECTED_FILENAMES = frozenset(
    {
        "f_list.cfg",
        "profile.cfg",
        "scanner.inf",
    }
)


class FtpWriteTestScanner(Protocol):
    def enter_ftp_mode(self, *, timeout: float = 5.0) -> None: ...

    def exit_ftp_mode(self, *, timeout: float = 5.0) -> None: ...


class FtpWriteTestClient(Protocol):
    def connect(
        self,
        host: str,
        port: int = 0,
        timeout: float | None = None,
    ) -> str: ...

    def login(
        self,
        user: str = "anonymous",
        passwd: str = "",
        acct: str = "",
    ) -> str: ...

    def set_pasv(self, val: bool) -> None: ...

    def nlst(self, *args: str) -> list[str]: ...

    def storbinary(
        self,
        cmd: str,
        fp: io.BytesIO,
        blocksize: int = 8192,
        callback: Callable[[bytes], object] | None = None,
        rest: int | None = None,
    ) -> str: ...

    def retrbinary(
        self,
        cmd: str,
        callback: Callable[[bytes], object],
        blocksize: int = 8192,
        rest: int | None = None,
    ) -> str: ...

    def delete(self, filename: str) -> str: ...

    def quit(self) -> str: ...

    def close(self) -> None: ...


FtpWriteTestFactory = Callable[[], FtpWriteTestClient]
FtpWriteTestFilenameFactory = Callable[[], str]


@dataclass(frozen=True)
class FtpWriteTestStage:
    name: str
    succeeded: bool
    detail: str | None = None


@dataclass(frozen=True)
class FtpWriteTestResult:
    filename: str
    stages: tuple[FtpWriteTestStage, ...]
    succeeded: bool

    @property
    def cleanup_succeeded(self) -> bool:
        cleanup_names = {
            "delete disposable file",
            "verify disposable file is absent",
            "close FTP session",
            "exit FTP mode",
        }
        cleanup_stages = [
            stage
            for stage in self.stages
            if stage.name in cleanup_names
        ]
        return bool(cleanup_stages) and all(
            stage.succeeded for stage in cleanup_stages
        )


class _ValidationStopped(RuntimeError):
    pass


def _new_filename() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"sdsctl-write-test-{timestamp}-{secrets.token_hex(4)}.txt"


def _validate_filename(filename: str) -> None:
    if not filename or filename in {".", ".."}:
        raise ValueError(
            "FTP validation filename must not be empty or relative"
        )
    lower_filename = filename.lower()
    if lower_filename in _PROTECTED_FILENAMES or lower_filename.endswith(".hpd"):
        raise ValueError(
            f"FTP validation refuses protected filename {filename}"
        )
    if any(
        character in filename
        for character in ("/", "\\", "\r", "\n", "\x00")
    ):
        raise ValueError(
            "FTP validation filename must be a single plain filename"
        )


def _failure_detail(error: BaseException, password: str) -> str:
    detail = str(error).strip() or error.__class__.__name__
    if password:
        detail = detail.replace(password, "<redacted>")
    return f"{error.__class__.__name__}: {detail}"


def run_ftp_write_test(
    scanner: FtpWriteTestScanner,
    *,
    host: str,
    username: str,
    password: str,
    port: int = FTP_WRITE_TEST_DEFAULT_PORT,
    timeout: float = FTP_WRITE_TEST_DEFAULT_TIMEOUT,
    scanner_timeout: float = 5.0,
    ftp_factory: FtpWriteTestFactory | None = None,
    filename_factory: FtpWriteTestFilenameFactory = _new_filename,
) -> FtpWriteTestResult:
    """Validate one disposable FTP round trip without scanner file changes."""
    if not host:
        raise ValueError("FTP validation host is required")
    if not username:
        raise ValueError("FTP validation username is required")
    if not password:
        raise ValueError("FTP validation password is required")
    if not 1 <= port <= 65535:
        raise ValueError("FTP validation port must be between 1 and 65535")
    if timeout <= 0:
        raise ValueError("FTP validation timeout must be greater than zero")
    if scanner_timeout <= 0:
        raise ValueError("FTP mode timeout must be greater than zero")

    filename = filename_factory()
    _validate_filename(filename)
    stages: list[FtpWriteTestStage] = []
    ftp: FtpWriteTestClient | None = None
    write_attempted = False
    deletion_verified = False
    failure_seen = False

    def succeed(name: str, detail: str | None = None) -> None:
        stages.append(FtpWriteTestStage(name, True, detail))
        logger.info(
            "FTP write validation stage=%s result=success file=%s",
            name,
            filename,
        )

    def fail(name: str, error: BaseException) -> None:
        nonlocal failure_seen
        failure_seen = True
        detail = _failure_detail(error, password)
        stages.append(FtpWriteTestStage(name, False, detail))
        logger.warning(
            "FTP write validation stage=%s result=failed file=%s error=%s",
            name,
            filename,
            detail,
        )

    def run_stage(name: str, action: Callable[[], object]) -> None:
        try:
            action()
        except Exception as error:
            fail(name, error)
            raise _ValidationStopped from error
        else:
            succeed(name)

    def connect() -> None:
        nonlocal ftp
        ftp = (
            ftp_factory()
            if ftp_factory is not None
            else cast(FtpWriteTestClient, ftplib.FTP())
        )
        ftp.connect(host, port, timeout)

    try:
        run_stage(
            "enter FTP mode",
            lambda: scanner.enter_ftp_mode(timeout=scanner_timeout),
        )
        run_stage("connect FTP", connect)
        assert ftp is not None
        client = ftp

        run_stage("enable passive FTP", lambda: client.set_pasv(True))
        run_stage("authenticate FTP", lambda: client.login(username, password))

        def verify_unused() -> None:
            if filename in client.nlst():
                raise FileExistsError(
                    f"generated FTP validation filename already exists: {filename}"
                )

        run_stage("verify disposable filename is unused", verify_unused)

        def upload() -> None:
            nonlocal write_attempted
            write_attempted = True
            client.storbinary(
                f"STOR {filename}",
                io.BytesIO(FTP_WRITE_TEST_PAYLOAD),
            )

        run_stage("upload disposable file", upload)

        received = bytearray()
        run_stage(
            "retrieve disposable file",
            lambda: client.retrbinary(
                f"RETR {filename}",
                received.extend,
            ),
        )

        def verify_payload() -> None:
            if bytes(received) != FTP_WRITE_TEST_PAYLOAD:
                raise ValueError(
                    "retrieved payload does not match uploaded bytes"
                )

        run_stage("verify retrieved bytes", verify_payload)
    except _ValidationStopped:
        pass
    finally:
        if write_attempted and ftp is not None and not deletion_verified:
            try:
                if filename in ftp.nlst():
                    ftp.delete(filename)
                    succeed("delete disposable file")
                else:
                    succeed("delete disposable file", "already absent")
                if filename in ftp.nlst():
                    raise FileExistsError(
                        f"disposable FTP file remains after deletion: {filename}"
                    )
                deletion_verified = True
                succeed("verify disposable file is absent")
            except Exception as error:
                fail("delete disposable file", error)

        if ftp is not None:
            try:
                ftp.quit()
            except Exception as error:
                fail("close FTP session", error)
                try:
                    ftp.close()
                except Exception as close_error:
                    fail("close FTP session fallback", close_error)
            else:
                succeed("close FTP session")

        try:
            scanner.exit_ftp_mode(timeout=scanner_timeout)
        except Exception as error:
            fail("exit FTP mode", error)
        else:
            succeed("exit FTP mode")

    return FtpWriteTestResult(
        filename=filename,
        stages=tuple(stages),
        succeeded=not failure_seen and deletion_verified,
    )


__all__ = [
    "FTP_WRITE_TEST_DEFAULT_PORT",
    "FTP_WRITE_TEST_DEFAULT_TIMEOUT",
    "FTP_WRITE_TEST_PAYLOAD",
    "FtpWriteTestResult",
    "FtpWriteTestStage",
    "run_ftp_write_test",
]
