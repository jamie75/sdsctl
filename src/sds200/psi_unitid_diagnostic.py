"""Optional, bounded PSI evidence; names have unknown scanner-side provenance."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import stat
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import BinaryIO

from .models import ScannerInfo

logger = logging.getLogger(__name__)
MAX_SECONDS = 300.0
MAX_RECORDS = 1000
MAX_BYTES = 1024 * 1024
QUEUE_CAPACITY = 64
_RESERVE = 4096  # Header/summary are included in both output limits.
_UID = re.compile(r"(?:UID:\s*)?([0-9]{1,10})", re.IGNORECASE)


class _NonWaitingQueue(queue.Queue[dict[str, object]]):
    def put_nowait(self, item: dict[str, object]) -> None:
        # Queue.put_nowait normally still waits for its mutex. Never do that
        # on the scanner reader, even if another thread stalls holding it.
        if not self.mutex.acquire(blocking=False):
            raise queue.Full
        try:
            if self._qsize() >= self.maxsize:
                raise queue.Full
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()
        finally:
            self.mutex.release()


def _numeric(value: str | None) -> int | None:
    if value is None:
        return None
    match = _UID.fullmatch(value.strip())
    return int(match[1]) if match and int(match[1]) > 0 else None


def _private_file(path: Path) -> BinaryIO:
    """Walk directories using no-follow descriptors; create only the leaf dir."""
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("diagnostic path must be absolute without parent traversal")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(path.parent.parts[1:]):
            last = index == len(path.parent.parts[1:]) - 1
            try:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
            except FileNotFoundError:
                if not last:
                    raise
                os.mkdir(part, mode=0o700, dir_fd=directory)
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
            os.close(directory)
            directory = child
        info = os.fstat(directory)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise PermissionError("diagnostic directory must be owned and private")
        fd = os.open(
            path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
        )
        try:
            os.fchmod(fd, 0o600)
            return os.fdopen(fd, "wb", buffering=0)
        except BaseException:
            os.close(fd)
            raise
    finally:
        os.close(directory)


class PsiUnitIdDiagnostic:
    """One five-minute session. No scanner methods or production state writes."""

    def __init__(
        self,
        path: Path,
        *,
        firmware: Callable[[], str | None] = lambda: None,
        interval_ms: int = 250,
        clock: Callable[[], float] = monotonic,
        opener: Callable[[Path], BinaryIO] = _private_file,
    ) -> None:
        self.path = path
        self._firmware = firmware
        self._clock = clock
        self._opener = opener
        self._gap = max(1.0, interval_ms * 4 / 1000)
        self._queue = _NonWaitingQueue(QUEUE_CAPACITY)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._start = clock()
        self._started_utc = datetime.now(UTC).isoformat()
        self._last_time: float | None = None
        self._key: tuple[object, ...] | None = None
        self._first_uid = 0.0
        self._first_name: float | None = None
        self._episode = 0
        self._signature: object = None
        self._duplicates = 0
        self._drops = 0
        self._seen = 0
        self._enqueued = 0
        self._written = 0
        self._reason = "closed"
        self._error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._active = True
        self._thread = threading.Thread(
            target=self._write, name="psi-unitid-diagnostic", daemon=True
        )
        try:
            self._thread.start()
        except Exception:
            self._active = False
            self._thread = None
            logger.warning("PSI UnitID diagnostic disabled: writer could not start")

    def close(self) -> None:
        self._active = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.25)

    def observe(self, info: object) -> None:
        # on_psi also publishes non-XML acknowledgments such as PSI,OK.
        if not isinstance(info, ScannerInfo) or not self._active:
            return
        self._seen += 1
        if not self._lock.acquire(blocking=False):
            self._drops += 1
            return
        try:
            now = self._clock()
            if now - self._start >= MAX_SECONDS:
                self._reason = "duration_limit"
                self._active = False
                return
            self._observe(info, now)
        except Exception as error:
            # No logging/file I/O here: the writer reports a sanitized type.
            self._error = type(error).__name__
            self._reason = "observer_error"
            self._active = False
        finally:
            self._lock.release()

    def _observe(self, info: ScannerInfo, now: float) -> None:
        truncated: list[str] = []

        def text(value: str | None, field: str) -> str | None:
            if value is not None and len(value) > 256:
                truncated.append(field)
                return value[:256]
            return value

        def attribute(tags: tuple[str, ...], key: str, field: str) -> str | None:
            for tag in tags:
                candidate = info.node(tag)
                if candidate is not None and candidate.get(key) is not None:
                    value = text(candidate.get(key), field)
                    return value.strip() if value is not None else None
            return None

        def index(tag: str, field: str) -> int | None:
            value = attribute((tag,), "Index", field)
            if field in truncated or value is None:
                return None
            try:
                return int(value)
            except ValueError:
                return None

        # Bounded traversal, including detection of an incomplete oversized
        # response. Do not retain ScannerInfo, nodes, XML, or arbitrary attrs.
        nodes = []
        if len(info.records) <= 128:
            nodes = list(info.records_by_tag("UnitID"))
        else:
            truncated.append("psi_records")
        ambiguous = len(nodes) > 1 or bool(truncated)
        node = nodes[0] if len(nodes) == 1 and not ambiguous else None
        raw_uid = text(node.get("U_Id") if node else None, "uid_raw")
        uid = _numeric(raw_uid) if "uid_raw" not in truncated else None
        name = text(node.get("Name") if node else None, "name_raw")
        name_present = node is not None and "Name" in node.attributes
        context = {
            "system": attribute(("System",), "Name", "system"),
            "system_index": index("System", "system_index"),
            "site": attribute(("Site",), "Name", "site"),
            "site_index": index("Site", "site_index"),
            "tgid_raw": attribute(("TGID", "ConvFrequency", "SrchFrequency"), "TGID", "tgid_raw"),
            "mute": attribute(("Property",), "Mute", "mute"),
            "p25_status": attribute(("Property",), "P25Status", "p25_status"),
        }
        voice = (
            context["mute"] == "Unmute"
            and uid is not None
            and not ambiguous
            and bool(context["system"])
            and bool(context["tgid_raw"])
            and not truncated
        )
        key = (context["system"], context["system_index"], context["tgid_raw"], uid)
        gap = self._last_time is not None and now - self._last_time >= self._gap
        # Break episodes even when boundary records would otherwise deduplicate.
        if not voice:
            self._key = None
        elif self._key != key or gap:
            self._episode += 1
            self._key = key
            self._first_uid = now
            self._first_name = None
            self._signature = None
        self._last_time = now
        elapsed = now - self._first_uid if voice else None
        if voice and name and name.strip() and self._first_name is None:
            self._first_name = elapsed
        record: dict[str, object] = {
            "type": "observation",
            **context,
            "unitid_present": None if "psi_records" in truncated else bool(nodes),
            "unitid_count": None if "psi_records" in truncated else len(nodes),
            "ambiguous": ambiguous,
            "uid_present": node is not None and "U_Id" in node.attributes,
            "uid_raw": raw_uid,
            "uid_numeric": uid,
            "uid_valid": uid is not None,
            "name_present": name_present,
            "name_raw": name,
            "name_is_uid_echo": uid is not None and _numeric(name) == uid,
            "episode": self._episode if voice else None,
            "first_nonempty_name_ms": (
                round(self._first_name * 1000, 3)
                if voice and self._first_name is not None and self._first_name > 0
                else None
            ),
            "name_present_on_first_observation": bool(voice and self._first_name == 0),
            "name_observed_later": bool(
                voice and self._first_name is not None and self._first_name > 0
            ),
            "truncated_fields": truncated,
            "provenance": "scanner_reported_unknown",
        }
        signature = tuple((k, repr(v)) for k, v in record.items())
        if signature == self._signature:
            self._duplicates += 1
            return
        record["received_at"] = info.received_at.astimezone(UTC).isoformat()
        record["elapsed_ms"] = round((now - self._start) * 1000, 3)
        record["uid_elapsed_ms"] = round(elapsed * 1000, 3) if elapsed is not None else None
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._drops += 1
        else:
            self._signature = signature
            self._enqueued += 1

    def _write(self) -> None:
        try:
            with self._opener(self.path) as output:
                # Wait for the first parsed PSI so cached firmware can have
                # arrived through ordinary startup, never a new VER request.
                first = self._next()
                firmware = self._firmware()
                header = {
                    "type": "session",
                    "version": 1,
                    "started_at": self._started_utc,
                    "firmware": firmware[:256] if firmware else None,
                    "firmware_truncated": bool(firmware and len(firmware) > 256),
                    "provenance": "scanner_reported_unknown",
                    "limits": {
                        "seconds": MAX_SECONDS,
                        "records": MAX_RECORDS,
                        "bytes": MAX_BYTES,
                        "queue": QUEUE_CAPACITY,
                    },
                }
                used = self._emit(output, header)
                record = first
                while record is not None:
                    data = self._encode(record)
                    if self._written >= MAX_RECORDS - 2:
                        self._reason = "record_limit"
                        break
                    if used + len(data) + _RESERVE > MAX_BYTES:
                        self._reason = "byte_limit"
                        break
                    if output.write(data) != len(data):
                        raise OSError("short diagnostic write")
                    used += len(data)
                    self._written += 1
                    if self._written >= MAX_RECORDS - 2:
                        self._reason = "record_limit"
                        break
                    record = self._next()
                self._active = False
                self._stop.set()
                # Bounded synchronization with a callback already in flight.
                if self._lock.acquire(timeout=0.05):
                    self._lock.release()
                summary = {
                    "type": "summary",
                    "elapsed_seconds": self._clock() - self._start,
                    "observations_seen": self._seen,
                    "observations_written": self._written,
                    "duplicates_suppressed": self._duplicates,
                    "queue_drops": self._drops,
                    "queued_not_written": max(0, self._enqueued - self._written),
                    "stop_reason": self._reason,
                    "observer_error": self._error,
                    "output_error": None,
                }
                self._emit(output, summary)
                if self._error:
                    logger.warning("PSI UnitID diagnostic disabled: observer error")
        except Exception as error:
            self._error = type(error).__name__
            self._active = False
            self._stop.set()
            logger.warning("PSI UnitID diagnostic disabled: output error (%s)", self._error)

    def _next(self) -> dict[str, object] | None:
        while self._active and not self._stop.is_set():
            remaining = MAX_SECONDS - (self._clock() - self._start)
            if remaining <= 0:
                self._reason = "duration_limit"
                return None
            try:
                return self._queue.get(timeout=min(0.05, remaining))
            except queue.Empty:
                pass
        return None

    @staticmethod
    def _encode(record: dict[str, object]) -> bytes:
        return (json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n").encode()

    def _emit(self, output: BinaryIO, record: dict[str, object]) -> int:
        data = self._encode(record)
        if output.write(data) != len(data):
            raise OSError("short diagnostic write")
        return len(data)
