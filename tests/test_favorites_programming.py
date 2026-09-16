from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sds200.favorites_file import parse_favorites_file
from sds200.favorites_programming import (
    FavoritesProgrammingEdit,
    FavoritesProgrammingError,
    FavoritesProgrammingPlan,
    generate_candidate_image,
    load_programming_source,
    validate_candidate_image,
)

FIXTURES = Path(__file__).parent / "fixtures" / "favorites"
CATALOG = (FIXTURES / "synthetic-f_list.cfg").read_bytes()
HPD = (FIXTURES / "synthetic-favorites.hpd").read_bytes()


def _backup(root: Path, *, catalog: bytes = CATALOG, hpd: bytes = HPD) -> Path:
    root.mkdir()
    (root / "f_list.cfg").write_bytes(catalog)
    (root / "f_000001.hpd").write_bytes(hpd)
    catalog_source = parse_favorites_file(catalog)
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
            "bytes": len(catalog),
            "sha256": hashlib.sha256(catalog).hexdigest(),
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


def _edit(
    old: str = "Synthetic Dispatch", new: str = "Temporary Dispatch"
) -> FavoritesProgrammingPlan:
    return FavoritesProgrammingPlan(
        (
            FavoritesProgrammingEdit(
                favorites_list="Synthetic Favorites",
                filename="f_000001.hpd",
                record_index=14,
                record_type="TGID",
                field="name",
                expected_old_value=old,
                new_value=new,
            ),
        )
    )


def test_load_programming_source_verifies_real_shape(tmp_path: Path) -> None:
    source = load_programming_source(_backup(tmp_path / "backup"))
    assert source.target_model == "BCDx36HP"
    assert source.format_version == "1.00"
    assert source.programming_filenames == ("f_list.cfg", "f_000001.hpd")


@pytest.mark.parametrize("mutation", ("manifest", "catalog", "hpd", "missing_hpd", "unsafe"))
def test_invalid_backup_is_rejected(tmp_path: Path, mutation: str) -> None:
    backup = _backup(tmp_path / "backup")
    if mutation == "manifest":
        (backup / "manifest.json").write_text("[]", encoding="utf-8")
    elif mutation == "catalog":
        (backup / "f_list.cfg").write_bytes(CATALOG + b"x")
    elif mutation == "hpd":
        (backup / "f_000001.hpd").write_bytes(HPD + b"x")
    elif mutation == "missing_hpd":
        (backup / "f_000001.hpd").unlink()
    else:
        manifest = json.loads((backup / "manifest.json").read_text())
        manifest["documents"][0]["filename"] = "../bad.hpd"
        (backup / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FavoritesProgrammingError):
        load_programming_source(backup)


def test_no_edit_candidate_is_lossless_and_separate(tmp_path: Path) -> None:
    backup = _backup(tmp_path / "backup")
    original = {
        name: (backup / name).read_bytes()
        for name in ("f_list.cfg", "f_000001.hpd", "manifest.json")
    }
    source = load_programming_source(backup)
    candidate = generate_candidate_image(
        source,
        tmp_path / "out",
        FavoritesProgrammingPlan.empty(),
        now=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert candidate.validation.changed_file_count == 0
    assert candidate.validation.changed_record_count == 0
    assert candidate.validation.changed_field_count == 0
    assert (candidate.directory / "f_list.cfg").read_bytes() == original["f_list.cfg"]
    assert (candidate.directory / "f_000001.hpd").read_bytes() == original["f_000001.hpd"]
    assert {name: (backup / name).read_bytes() for name in original} == original


def test_exact_tgid_edit_reports_one_lossless_field_change(tmp_path: Path) -> None:
    source_dir = _backup(tmp_path / "backup")
    source = load_programming_source(source_dir)
    candidate = generate_candidate_image(
        source, tmp_path / "out", _edit(), now=datetime(2026, 1, 2, tzinfo=UTC)
    )
    assert candidate.validation.files_changed == ("f_000001.hpd",)
    assert candidate.validation.changed_record_count == 1
    assert candidate.validation.changed_field_count == 1
    change = candidate.validation.field_changes[0]
    assert (change.filename, change.record_index, change.record_type, change.field) == (
        "f_000001.hpd",
        14,
        "TGID",
        "name",
    )
    assert (change.old_value, change.new_value) == ("Synthetic Dispatch", "Temporary Dispatch")
    assert (candidate.directory / "f_list.cfg").read_bytes() == CATALOG
    assert json.loads(candidate.changes_path.read_text())["summary"] == {
        "changed_files": 1,
        "changed_records": 1,
        "changed_fields": 1,
    }


def test_stale_target_and_source_overlap_are_rejected(tmp_path: Path) -> None:
    source_dir = _backup(tmp_path / "backup")
    source = load_programming_source(source_dir)
    with pytest.raises(FavoritesProgrammingError, match="stale"):
        generate_candidate_image(
            source, tmp_path / "out", _edit(old="wrong"), now=datetime(2026, 1, 2, tzinfo=UTC)
        )
    with pytest.raises(FavoritesProgrammingError, match="inside"):
        generate_candidate_image(source, source_dir / "candidate", FavoritesProgrammingPlan.empty())


def test_candidate_validation_rejects_unexpected_extra_file(tmp_path: Path) -> None:
    source = load_programming_source(_backup(tmp_path / "backup"))
    candidate = generate_candidate_image(
        source,
        tmp_path / "out",
        FavoritesProgrammingPlan.empty(),
        now=datetime(2026, 1, 2, tzinfo=UTC),
    )
    (candidate.directory / "unexpected.txt").write_text("no", encoding="ascii")
    with pytest.raises(FavoritesProgrammingError):
        validate_candidate_image(source, candidate.directory)


def test_cli_prepare_does_not_require_scanner_host() -> None:
    import sds200.cli as cli

    args = cli.build_parser().parse_args(
        ["favorites-prepare", "--backup", "/tmp/backup", "--output", "/tmp/output"]
    )
    assert args.action == "favorites-prepare"
    assert args.host is None


def _avoid_edit(
    old: str = "Off", new: str = "On"
) -> FavoritesProgrammingPlan:
    return FavoritesProgrammingPlan(
        (
            FavoritesProgrammingEdit(
                favorites_list="Synthetic Favorites",
                filename="f_000001.hpd",
                record_index=14,
                record_type="TGID",
                field="avoid",
                expected_old_value=old,
                new_value=new,
            ),
        )
    )


def test_tgid_avoid_edit_reports_one_lossless_field_change(tmp_path: Path) -> None:
    source_dir = _backup(tmp_path / "backup")
    source = load_programming_source(source_dir)
    candidate = generate_candidate_image(
        source, tmp_path / "out", _avoid_edit(), now=datetime(2026, 1, 2, tzinfo=UTC)
    )

    assert candidate.validation.files_changed == ("f_000001.hpd",)
    assert candidate.validation.changed_record_count == 1
    assert candidate.validation.changed_field_count == 1
    change = candidate.validation.field_changes[0]
    assert (change.filename, change.record_index, change.record_type, change.field) == (
        "f_000001.hpd",
        14,
        "TGID",
        "avoid",
    )
    assert (change.old_value, change.new_value) == ("Off", "On")
    assert (candidate.directory / "f_list.cfg").read_bytes() == CATALOG
    assert json.loads(candidate.changes_path.read_text())["edits"][0]["field"] == "avoid"


@pytest.mark.parametrize(("old", "new"), (("Off", "On"), ("On", "Off")))
def test_tgid_avoid_edit_supports_both_persistent_values(
    tmp_path: Path, old: str, new: str
) -> None:
    hpd = HPD.replace(
        b"TGID\t\t\tSynthetic Dispatch\tOff\t",
        f"TGID\t\t\tSynthetic Dispatch\t{old}\t".encode(),
    )
    source = load_programming_source(_backup(tmp_path / "backup", hpd=hpd))
    candidate = generate_candidate_image(
        source, tmp_path / "out", _avoid_edit(old, new), now=datetime(2026, 1, 2, tzinfo=UTC)
    )
    records = parse_favorites_file((candidate.directory / "f_000001.hpd").read_bytes()).records
    assert records[14].fields[3] == new


def test_tgid_avoid_edit_preserves_everything_except_target_field(tmp_path: Path) -> None:
    source_dir = _backup(tmp_path / "backup")
    source = load_programming_source(source_dir)
    candidate = generate_candidate_image(
        source, tmp_path / "out", _avoid_edit(), now=datetime(2026, 1, 2, tzinfo=UTC)
    )
    before = parse_favorites_file(HPD).records
    after = parse_favorites_file((candidate.directory / "f_000001.hpd").read_bytes()).records
    assert len(before) == len(after)
    assert all(before[index] == after[index] for index in range(len(before)) if index != 14)
    assert before[14].fields[:3] == after[14].fields[:3]
    assert before[14].fields[4:] == after[14].fields[4:]
    assert before[14].line_ending == after[14].line_ending
    assert (candidate.directory / "f_000001.hpd").read_bytes().count(b"\r\n") == HPD.count(b"\r\n")


def test_tgid_avoid_edit_rejects_stale_and_noop_targets(tmp_path: Path) -> None:
    source = load_programming_source(_backup(tmp_path / "backup"))
    with pytest.raises(FavoritesProgrammingError, match="stale"):
        generate_candidate_image(
            source, tmp_path / "stale", _avoid_edit(old="On"), now=datetime(2026, 1, 2, tzinfo=UTC)
        )
    with pytest.raises(FavoritesProgrammingError, match="does not change"):
        generate_candidate_image(
            source,
            tmp_path / "noop",
            _avoid_edit(old="Off", new="Off"),
            now=datetime(2026, 1, 2, tzinfo=UTC),
        )


@pytest.mark.parametrize(("expected", "new"), (("Maybe", "On"), ("Off", "Maybe")))
def test_tgid_avoid_edit_rejects_unsupported_values(
    tmp_path: Path, expected: str, new: str
) -> None:
    with pytest.raises(FavoritesProgrammingError, match="Avoid values"):
        _avoid_edit(expected, new)


def test_programming_plan_rejects_duplicate_avoid_target() -> None:
    edit = _avoid_edit().edits[0]
    with pytest.raises(FavoritesProgrammingError, match="duplicate"):
        FavoritesProgrammingPlan(edits=(edit, edit))
