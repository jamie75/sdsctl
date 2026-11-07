from __future__ import annotations

import io
import json
import queue
import threading
from pathlib import Path
from typing import Any

import pytest

from sds200.events import EventBus
from sds200.psi_unitid_diagnostic import PsiUnitIdDiagnostic, _private_file
from sds200.xml_protocol import ScannerInfoParser


class Clock:
    value = 0.0

    def __call__(self) -> float:
        return self.value


def psi(
    unit: str = '<UnitID U_Id="UID:12345" Name="Example" />',
    *,
    system: str = "System A",
    tgid: str = "1",
    mute: str = "Unmute",
) -> Any:
    return ScannerInfoParser().parse(
        "PSI",
        f'<ScannerInfo><System Name="{system}" Index="1" />'
        f'<Site Name="Site" Index="2" /><TGID TGID="TGID:{tgid}" />'
        f'<Property Mute="{mute}" P25Status="P25" />{unit}</ScannerInfo>',
    )


def observer(tmp_path: Path) -> tuple[PsiUnitIdDiagnostic, Clock]:
    clock = Clock()
    item = PsiUnitIdDiagnostic(tmp_path / "private" / "evidence.jsonl", clock=clock)
    item._active = True  # Deterministic callback-only tests: no thread or hardware.
    return item, clock


def drain(item: PsiUnitIdDiagnostic) -> list[dict[str, Any]]:
    result = []
    while True:
        try:
            result.append(item._queue.get_nowait())
        except queue.Empty:
            return result


@pytest.mark.parametrize(
    "unit,expected",
    [
        ("", {"unitid_present": False, "uid_valid": False, "name_present": False}),
        ('<UnitID Name="Named" />', {"uid_present": False, "uid_valid": False}),
        ('<UnitID U_Id="UID None" />', {"uid_present": True, "uid_valid": False}),
        ('<UnitID U_Id="UID:12345" />', {"uid_valid": True, "name_present": False}),
        ('<UnitID U_Id="UID:12345" Name="" />', {"name_present": True, "name_raw": ""}),
        ('<UnitID U_Id="UID:12345" Name="Example" />', {"name_raw": "Example"}),
        ('<UnitID U_Id="UID:12345" Name="UID:12345" />', {"name_is_uid_echo": True}),
        ('<UnitID U_Id="UID:12345" Name="12345" />', {"name_is_uid_echo": True}),
        (
            '<UnitID U_Id="UID:12345" Name="Example" /><UnitID U_Id="UID:9" />',
            {"ambiguous": True, "unitid_count": 2, "uid_numeric": None, "episode": None},
        ),
    ],
)
def test_fields(tmp_path: Path, unit: str, expected: dict[str, object]) -> None:
    item, _ = observer(tmp_path)
    item.observe(psi(unit))
    record = drain(item)[0]
    assert expected.items() <= record.items()
    assert record["provenance"] == "scanner_reported_unknown"
    assert "xml" not in json.dumps(record).lower()
    assert "call_id" not in record


@pytest.mark.parametrize("boundary", ["uid", "system", "tgid", "mute", "missing", "gap"])
def test_episode_boundaries_before_dedup(tmp_path: Path, boundary: str) -> None:
    item, clock = observer(tmp_path)
    item.observe(psi())
    if boundary == "uid":
        item.observe(psi('<UnitID U_Id="UID:54321" />'))
    elif boundary == "system":
        item.observe(psi(system="System B"))
    elif boundary == "tgid":
        item.observe(psi(tgid="2"))
    elif boundary == "mute":
        item.observe(psi(mute="Mute"))
    elif boundary == "missing":
        item.observe(psi(""))
    else:
        clock.value = 2.0
    item.observe(psi())
    records = drain(item)
    assert records[0]["episode"] == 1
    assert records[-1]["episode"] > 1
    assert records[-1]["uid_elapsed_ms"] == 0


def test_late_name_and_name_changes(tmp_path: Path) -> None:
    item, clock = observer(tmp_path)
    item.observe(psi('<UnitID U_Id="UID:12345" />'))
    clock.value = 0.75
    item.observe(psi())
    clock.value = 0.9
    item.observe(psi('<UnitID U_Id="UID:12345" Name="Changed" />'))
    records = drain(item)
    assert [r["episode"] for r in records] == [1, 1, 1]
    assert records[1]["name_observed_later"] is True
    assert records[1]["first_nonempty_name_ms"] == 750
    assert records[2]["name_raw"] == "Changed"
    assert records[2]["first_nonempty_name_ms"] == 750


def test_first_name_is_not_zero_decoding_latency_and_duplicates(tmp_path: Path) -> None:
    item, clock = observer(tmp_path)
    item.observe(psi())
    clock.value = 0.25
    item.observe(psi())
    records = drain(item)
    assert len(records) == 1 and item._duplicates == 1
    assert records[0]["name_present_on_first_observation"] is True
    assert records[0]["first_nonempty_name_ms"] is None
    assert records[0]["name_observed_later"] is False


def test_bounded_text_and_no_fallback_pairing(tmp_path: Path) -> None:
    item, _ = observer(tmp_path)
    item.observe(psi('<UnitID Name="' + "x" * 2000 + '" />'))
    record = drain(item)[0]
    assert len(record["name_raw"]) == 256
    assert record["truncated_fields"] == ["name_raw"]
    assert record["uid_numeric"] is None and record["episode"] is None


def test_queue_full_and_contended_never_wait(tmp_path: Path) -> None:
    item, _ = observer(tmp_path)
    for i in range(70):
        item.observe(psi(f'<UnitID U_Id="UID:{i + 1}" />'))
    assert item._queue.qsize() == 64 and item._drops == 6
    item._queue.mutex.acquire()
    try:
        item.observe(psi())
    finally:
        item._queue.mutex.release()
    item._lock.acquire()
    try:
        item.observe(psi())
    finally:
        item._lock.release()
    assert item._drops == 8


def test_disabled_callback(tmp_path: Path) -> None:
    item = PsiUnitIdDiagnostic(tmp_path / "unused")
    item.observe(psi())
    assert item._thread is None and item._queue.empty()
    assert not item.path.exists()


def test_callback_does_not_serialize_write_or_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import sds200.psi_unitid_diagnostic as module

    item, _ = observer(tmp_path)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("reader callback performed diagnostic I/O or serialization")

    item._opener = forbidden
    monkeypatch.setattr(module.json, "dumps", forbidden)
    item.observe(psi())
    assert len(drain(item)) == 1 and not caplog.records


def test_callback_error_and_expiry_do_not_acquire_event_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    item, clock = observer(tmp_path)

    def forbidden() -> None:
        pytest.fail("reader callback attempted blocking Event.set")

    monkeypatch.setattr(item._stop, "set", forbidden)
    item.observe(object())
    assert item._reason == "observer_error"
    item._active = True
    clock.value = 301
    item.observe(psi())
    assert item._reason == "duration_limit"


def test_missing_system_has_no_episode(tmp_path: Path) -> None:
    item, _ = observer(tmp_path)
    item.observe(psi(system=""))
    assert drain(item)[0]["episode"] is None


def test_oversized_record_list_is_unknown_not_absent(tmp_path: Path) -> None:
    item, _ = observer(tmp_path)
    item.observe(psi('<UnitID U_Id="UID:12345" />' + "<Extra />" * 128))
    record = drain(item)[0]
    assert record["unitid_present"] is None
    assert record["ambiguous"] is True and record["truncated_fields"] == ["psi_records"]


def test_observer_error_isolated_and_other_callbacks_continue(tmp_path: Path) -> None:
    item, _ = observer(tmp_path)
    bus = EventBus()
    received = []
    bus.subscribe("psi", item.observe)
    bus.subscribe("psi", received.append)
    bus.emit("psi", object())
    assert len(received) == 1 and not item._active
    assert item._reason == "observer_error"


def test_malformed_xml_never_reaches_observer(tmp_path: Path) -> None:
    from sds200.exceptions import ProtocolError

    item, _ = observer(tmp_path)
    with pytest.raises(ProtocolError):
        item.observe(ScannerInfoParser().parse("PSI", "<ScannerInfo>"))
    assert item._seen == 0


def write_records(
    item: PsiUnitIdDiagnostic, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    records = iter([*drain(item), None])
    monkeypatch.setattr(item, "_next", lambda: next(records))
    item._write()
    return [json.loads(line) for line in item.path.read_text().splitlines()]


def test_jsonl_permissions_and_selected_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, _ = observer(tmp_path)
    item.observe(psi())
    rows = write_records(item, monkeypatch)
    assert [r["type"] for r in rows] == ["session", "observation", "summary"]
    assert rows[-1]["observations_written"] == 1
    assert item.path.stat().st_mode & 0o777 == 0o600
    assert item.path.parent.stat().st_mode & 0o777 == 0o700
    assert item.path.stat().st_size <= 1024 * 1024


@pytest.mark.parametrize(
    "kind", ["existing", "symlink", "parent_symlink", "unsafe_dir", "relative"]
)
def test_file_rejection(tmp_path: Path, kind: str) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    target = private / "capture.jsonl"
    if kind == "existing":
        target.write_bytes(b"existing")
    elif kind == "symlink":
        target.symlink_to(private / "other")
    elif kind == "parent_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(private, target_is_directory=True)
        target = alias / "capture.jsonl"
    elif kind == "unsafe_dir":
        private.chmod(0o755)
    else:
        target = Path("relative/file")
    with pytest.raises((OSError, ValueError)):
        _private_file(target)
    if kind == "existing":
        assert target.read_bytes() == b"existing"


@pytest.mark.parametrize("which", ["records", "bytes"])
def test_output_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str) -> None:
    import sds200.psi_unitid_diagnostic as module

    item, _ = observer(tmp_path)
    if which == "records":
        monkeypatch.setattr(module, "MAX_RECORDS", 4)
    else:
        monkeypatch.setattr(module, "MAX_BYTES", 6000)
    for i in range(8):
        item.observe(psi(f'<UnitID U_Id="UID:{i + 1}" Name="' + "x" * 256 + '" />'))
    rows = write_records(item, monkeypatch)
    assert rows[-1]["stop_reason"] == ("record_limit" if which == "records" else "byte_limit")
    assert len(rows) <= module.MAX_RECORDS
    assert item.path.stat().st_size <= module.MAX_BYTES
    assert not item._active


def test_duration_expiry_without_traffic(tmp_path: Path) -> None:
    item, clock = observer(tmp_path)
    clock.value = 301
    assert item._next() is None
    item.observe(psi())
    assert not item._active and item._reason == "duration_limit"


def test_writer_thread_expires_without_psi_and_uses_cached_firmware(tmp_path: Path) -> None:
    item, clock = observer(tmp_path)
    item._firmware = lambda: "Version 1.27.00"
    item.start()
    clock.value = 301
    assert item._thread is not None
    item._thread.join(1)
    assert not item._thread.is_alive() and not item._active
    rows = [json.loads(line) for line in item.path.read_text().splitlines()]
    assert rows[0]["firmware"] == "Version 1.27.00"
    assert rows[-1]["stop_reason"] == "duration_limit"
    assert rows[-1]["observations_written"] == 0


@pytest.mark.parametrize("failure", ["open", "write"])
def test_output_failure_sanitized(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, failure: str
) -> None:
    item, _ = observer(tmp_path)
    item.observe(psi())

    class Broken(io.BytesIO):
        def write(self, data: Any) -> int:
            raise OSError("SECRET error")

    def opener(path: Path) -> Any:
        if failure == "open":
            raise OSError("SECRET path")
        return Broken()

    item._opener = opener
    item._write()
    assert not item._active
    assert "SECRET" not in caplog.text
    assert len(caplog.records) == 1


def test_shutdown_bounded_with_stuck_writer(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    def opener(path: Path) -> Any:
        entered.set()
        release.wait(2)
        raise OSError("released")

    item = PsiUnitIdDiagnostic(tmp_path / "unused", opener=opener)
    item.start()
    assert entered.wait(1)
    try:
        item.close()
        assert item._thread is not None and item._thread.is_alive()
    finally:
        release.set()
        item._thread.join(1)
