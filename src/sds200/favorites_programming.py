"""Build and validate local-only SDS200 Favorites programming images."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from .favorites_catalog import project_favorites_catalog
from .favorites_editing import (
    FavoritesRecordEditError,
    rename_favorites_record,
    select_favorites_record_target,
    set_tgid_avoid,
)
from .favorites_file import FavoritesSourceFile, parse_favorites_file
from .favorites_schema import FavoritesSchemaSeverity, validate_favorites_workspace
from .favorites_storage import (
    FavoritesStorageDocument,
    FavoritesStorageSnapshot,
    project_favorites_storage_snapshot,
)

PROGRAMMING_IMAGE_SCHEMA = "sdsctl.favorites-programming-image"
PROGRAMMING_IMAGE_SCHEMA_VERSION = 1
PROGRAMMING_CHANGES_SCHEMA = "sdsctl.favorites-programming-changes"
PROGRAMMING_CHANGES_SCHEMA_VERSION = 1
_BACKUP_SCHEMA = "sdsctl.favorites-backup"
_CATALOG_FILENAME = "f_list.cfg"
_NAME_FIELD_INDEX = 2
_AVOID_FIELD_INDEX = 3
_SUPPORTED_FIELDS = frozenset({"name", "avoid"})


class FavoritesProgrammingError(ValueError):
    """Report an unsafe, incomplete, stale, or unexpected local image."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_filename(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or bool(PureWindowsPath(value).drive)
    ):
        raise FavoritesProgrammingError(f"unsafe Favorites filename: {value!r}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FavoritesProgrammingError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FavoritesProgrammingError(f"{label} must be a list")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FavoritesProgrammingError(f"cannot read JSON {path.name}: {error}") from None
    if not isinstance(value, dict):
        raise FavoritesProgrammingError(f"JSON {path.name} must contain an object")
    return value


def _metadata_value(source: FavoritesSourceFile, command: str) -> str | None:
    for record in source.records:
        if record.command == command and record.fields:
            return record.fields[0]
    return None


@dataclass(frozen=True, slots=True)
class FavoritesProgrammingEdit:
    """One stale-protected semantic TGID name or Avoid edit."""

    favorites_list: str
    filename: str
    record_index: int
    record_type: str
    field: str
    expected_old_value: str
    new_value: str

    def __post_init__(self) -> None:
        values = (
            self.favorites_list,
            self.filename,
            self.record_type,
            self.field,
            self.expected_old_value,
            self.new_value,
        )
        if any(type(value) is not str or not value for value in values):
            raise FavoritesProgrammingError("Favorites edit text fields must be non-empty strings")
        if type(self.record_index) is not int or self.record_index < 0:
            raise FavoritesProgrammingError("Favorites edit record_index must be non-negative")
        if self.record_type != "TGID" or self.field not in _SUPPORTED_FIELDS:
            raise FavoritesProgrammingError("only TGID name or Avoid edits are supported")
        if self.field == "avoid" and (
            self.expected_old_value not in {"Off", "On"}
            or self.new_value not in {"Off", "On"}
        ):
            raise FavoritesProgrammingError("TGID Avoid values must be 'Off' or 'On'")
        _safe_filename(self.filename)


@dataclass(frozen=True, slots=True)
class FavoritesProgrammingPlan:
    edits: tuple[FavoritesProgrammingEdit, ...]

    def __post_init__(self) -> None:
        if type(self.edits) is not tuple or any(
            not isinstance(edit, FavoritesProgrammingEdit) for edit in self.edits
        ):
            raise TypeError("Favorites programming plan edits must be a tuple")
        targets = [
            (edit.favorites_list, edit.filename, edit.record_index, edit.field)
            for edit in self.edits
        ]
        if len(targets) != len(set(targets)):
            raise FavoritesProgrammingError("edit plan contains duplicate record targets")

    @classmethod
    def empty(cls) -> FavoritesProgrammingPlan:
        return cls(edits=())


@dataclass(frozen=True, slots=True)
class FavoritesProgrammingSource:
    backup_directory: Path
    snapshot: FavoritesStorageSnapshot
    manifest: Mapping[str, Any]
    manifest_sha256: str
    target_model: str
    format_version: str

    @property
    def programming_filenames(self) -> tuple[str, ...]:
        return (_CATALOG_FILENAME, *(document.filename for document in self.snapshot.documents))


@dataclass(frozen=True, slots=True)
class FavoritesProgrammingFieldChange:
    filename: str
    record_index: int
    record_type: str
    field: str
    old_value: str | None
    new_value: str | None


@dataclass(frozen=True, slots=True)
class FavoritesProgrammingRecordChange:
    filename: str
    record_index: int
    record_type: str
    field_changes: tuple[FavoritesProgrammingFieldChange, ...]
    line_ending_changed: bool


@dataclass(frozen=True, slots=True)
class FavoritesProgrammingValidation:
    files_added: tuple[str, ...]
    files_removed: tuple[str, ...]
    files_changed: tuple[str, ...]
    record_changes: tuple[FavoritesProgrammingRecordChange, ...]
    field_changes: tuple[FavoritesProgrammingFieldChange, ...]
    schema_warnings: tuple[str, ...]
    schema_errors: tuple[str, ...]

    @property
    def changed_file_count(self) -> int:
        return len(self.files_changed)

    @property
    def changed_record_count(self) -> int:
        return len(self.record_changes)

    @property
    def changed_field_count(self) -> int:
        return len(self.field_changes)


@dataclass(frozen=True, slots=True)
class FavoritesProgrammingCandidate:
    directory: Path
    manifest_path: Path
    changes_path: Path
    validation: FavoritesProgrammingValidation


def _verify_manifest_file(directory: Path, entry: Mapping[str, Any]) -> bytes:
    filename = _safe_filename(entry.get("filename"))
    path = directory / filename
    if not path.is_file():
        raise FavoritesProgrammingError(f"manifest references missing file: {filename}")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise FavoritesProgrammingError(f"cannot read {filename}: {error}") from None
    if type(entry.get("bytes")) is not int or entry["bytes"] != len(data):
        raise FavoritesProgrammingError(f"size mismatch for {filename}")
    if not isinstance(entry.get("sha256"), str) or entry["sha256"] != _sha256(data):
        raise FavoritesProgrammingError(f"SHA-256 mismatch for {filename}")
    return data


def _schema_diagnostics(
    snapshot: FavoritesStorageSnapshot,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    result = validate_favorites_workspace(project_favorites_storage_snapshot(snapshot))
    warnings = tuple(
        d.message for d in result.diagnostics if d.severity is FavoritesSchemaSeverity.WARNING
    )
    errors = tuple(
        d.message for d in result.diagnostics if d.severity is FavoritesSchemaSeverity.ERROR
    )
    return warnings, errors


def load_programming_source(backup_directory: Path) -> FavoritesProgrammingSource:
    """Load and verify an existing immutable read-only Favorites backup."""
    directory = backup_directory.resolve()
    if not directory.is_dir():
        raise FavoritesProgrammingError(f"backup directory does not exist: {backup_directory}")
    manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != _BACKUP_SCHEMA or manifest.get("schema_version") != 1:
        raise FavoritesProgrammingError("unsupported or malformed Favorites backup manifest")
    catalog_meta = _mapping(manifest.get("catalog"), "manifest catalog")
    if _safe_filename(catalog_meta.get("filename")) != _CATALOG_FILENAME:
        raise FavoritesProgrammingError("manifest catalog must be f_list.cfg")
    catalog_bytes = _verify_manifest_file(directory, catalog_meta)
    try:
        catalog_source = parse_favorites_file(catalog_bytes)
        catalog = project_favorites_catalog(catalog_source)
    except (ValueError, TypeError) as error:
        raise FavoritesProgrammingError(f"catalog validation failed: {error}") from None
    document_map: dict[str, bytes] = {}
    documents: list[FavoritesStorageDocument] = []
    for raw in _list(manifest.get("documents"), "manifest documents"):
        entry = _mapping(raw, "manifest document")
        filename = _safe_filename(entry.get("filename"))
        if filename in document_map or not filename.endswith(".hpd"):
            raise FavoritesProgrammingError(f"invalid or duplicate HPD document: {filename}")
        data = _verify_manifest_file(directory, entry)
        try:
            parse_favorites_file(data)
        except (ValueError, TypeError) as error:
            raise FavoritesProgrammingError(
                f"document validation failed for {filename}: {error}"
            ) from None
        document_map[filename] = data
        documents.append(FavoritesStorageDocument(filename=filename, content=data))
    references = tuple(entry.filename for entry in catalog.entries)
    if len(references) != len(set(references)) or set(references) != set(document_map):
        raise FavoritesProgrammingError("manifest documents do not match catalog references")
    manifest_lists = []
    for raw in _list(manifest.get("favorites_lists"), "manifest favorites_lists"):
        entry = _mapping(raw, "manifest Favorites List")
        manifest_lists.append(
            (_safe_filename(entry.get("filename")), entry.get("name"), entry.get("source_index"))
        )
    expected_lists = [(entry.filename, entry.name, entry.source_index) for entry in catalog.entries]
    if manifest_lists != expected_lists:
        raise FavoritesProgrammingError("manifest Favorites Lists do not match f_list.cfg")
    snapshot = FavoritesStorageSnapshot(catalog_bytes=catalog_bytes, documents=tuple(documents))
    _, errors = _schema_diagnostics(snapshot)
    if errors:
        raise FavoritesProgrammingError("Favorites schema validation failed: " + "; ".join(errors))
    target_model = _metadata_value(catalog_source, "TargetModel")
    format_version = _metadata_value(catalog_source, "FormatVersion")
    if target_model is None or format_version is None:
        raise FavoritesProgrammingError("Favorites catalog is missing TargetModel or FormatVersion")
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_files != set(document_map) | {_CATALOG_FILENAME, "manifest.json"}:
        raise FavoritesProgrammingError("backup contains unexpected or missing files")
    try:
        manifest_digest = _sha256(manifest_path.read_bytes())
    except OSError as error:
        raise FavoritesProgrammingError(f"cannot read manifest: {error}") from None
    return FavoritesProgrammingSource(
        directory, snapshot, manifest, manifest_digest, target_model, format_version
    )


def load_edit_plan(path: Path) -> FavoritesProgrammingPlan:
    value = _read_json(path)
    edits: list[FavoritesProgrammingEdit] = []
    for raw in _list(value.get("edits"), "edit plan edits"):
        item = _mapping(raw, "edit plan entry")
        try:
            edits.append(
                FavoritesProgrammingEdit(
                    favorites_list=item["favorites_list"],
                    filename=item["filename"],
                    record_index=item["record_index"],
                    record_type=item["record_type"],
                    field=item["field"],
                    expected_old_value=item["expected_old_value"],
                    new_value=item["new_value"],
                )
            )
        except (KeyError, TypeError) as error:
            raise FavoritesProgrammingError(f"invalid edit plan entry: {error}") from None
    return FavoritesProgrammingPlan(edits=tuple(edits))


def _edit_dict(edit: FavoritesProgrammingEdit) -> dict[str, Any]:
    return {
        "favorites_list": edit.favorites_list,
        "filename": edit.filename,
        "record_index": edit.record_index,
        "record_type": edit.record_type,
        "field": edit.field,
        "expected_old_value": edit.expected_old_value,
        "new_value": edit.new_value,
    }


def _snapshot_to_map(snapshot: FavoritesStorageSnapshot) -> dict[str, bytes]:
    return {
        _CATALOG_FILENAME: snapshot.catalog_bytes,
        **{d.filename: d.content for d in snapshot.documents},
    }


def _apply_plan(
    source: FavoritesProgrammingSource, plan: FavoritesProgrammingPlan
) -> FavoritesStorageSnapshot:
    snapshot = source.snapshot
    for edit in plan.edits:
        workspace = project_favorites_storage_snapshot(snapshot)
        matches = [
            binding
            for binding in workspace.bindings
            if binding.name == edit.favorites_list and binding.filename == edit.filename
        ]
        if len(matches) != 1:
            raise FavoritesProgrammingError(
                f"Favorites List/file binding is not unique: {edit.favorites_list}/{edit.filename}"
            )
        document_index = next(
            (i for i, d in enumerate(snapshot.documents) if d.filename == edit.filename), None
        )
        if document_index is None:
            raise FavoritesProgrammingError(f"missing edit document: {edit.filename}")
        source_file = parse_favorites_file(snapshot.documents[document_index].content)
        if edit.record_index >= len(source_file.records):
            raise FavoritesProgrammingError(f"record index out of range: {edit.record_index}")
        record = source_file.records[edit.record_index]
        field_index = _NAME_FIELD_INDEX if edit.field == "name" else _AVOID_FIELD_INDEX
        if record.command != edit.record_type or len(record.fields) <= field_index:
            raise FavoritesProgrammingError(
                f"record target mismatch at {edit.filename}[{edit.record_index}]"
            )
        if record.fields[field_index] != edit.expected_old_value:
            raise FavoritesProgrammingError(
                f"stale edit target at {edit.filename}[{edit.record_index}]"
            )
        if edit.new_value == edit.expected_old_value:
            raise FavoritesProgrammingError("edit does not change the target value")
        try:
            target = select_favorites_record_target(
                snapshot, edit.record_index, document_index=document_index
            )
            if edit.field == "name":
                snapshot = rename_favorites_record(snapshot, target, edit.new_value)
            else:
                snapshot = set_tgid_avoid(snapshot, target, edit.new_value)
        except (FavoritesRecordEditError, TypeError, ValueError) as error:
            raise FavoritesProgrammingError(str(error)) from None
    return snapshot


def _field_label(record_type: str, index: int) -> str:
    if record_type == "TGID" and index == _AVOID_FIELD_INDEX:
        return "avoid"
    return "name" if index == _NAME_FIELD_INDEX else f"field_{index}"


def _record_differences(
    source: FavoritesStorageSnapshot, candidate: FavoritesStorageSnapshot
) -> tuple[
    tuple[FavoritesProgrammingRecordChange, ...], tuple[FavoritesProgrammingFieldChange, ...]
]:
    old_map, new_map = _snapshot_to_map(source), _snapshot_to_map(candidate)
    records: list[FavoritesProgrammingRecordChange] = []
    fields: list[FavoritesProgrammingFieldChange] = []
    for filename in sorted(set(old_map) & set(new_map)):
        if old_map[filename] == new_map[filename]:
            continue
        old_file, new_file = (
            parse_favorites_file(old_map[filename]),
            parse_favorites_file(new_map[filename]),
        )
        if len(old_file.records) != len(new_file.records):
            raise FavoritesProgrammingError(f"record count changed unexpectedly in {filename}")
        for index, (old, new) in enumerate(zip(old_file.records, new_file.records, strict=True)):
            if old == new:
                continue
            if old.command != new.command or len(old.fields) != len(new.fields):
                raise FavoritesProgrammingError(
                    f"record shape changed unexpectedly in {filename}[{index}]"
                )
            changed: list[FavoritesProgrammingFieldChange] = []
            for field_index, (before, after) in enumerate(zip(old.fields, new.fields, strict=True)):
                if before != after:
                    change = FavoritesProgrammingFieldChange(
                        filename,
                        index,
                        old.command,
                        _field_label(old.command, field_index),
                        before,
                        after,
                    )
                    changed.append(change)
                    fields.append(change)
            records.append(
                FavoritesProgrammingRecordChange(
                    filename, index, old.command, tuple(changed), old.line_ending != new.line_ending
                )
            )
    return tuple(records), tuple(fields)


def _candidate_snapshot(
    source: FavoritesProgrammingSource, directory: Path
) -> FavoritesStorageSnapshot:
    return FavoritesStorageSnapshot(
        catalog_bytes=(directory / _CATALOG_FILENAME).read_bytes(),
        documents=tuple(
            FavoritesStorageDocument(filename=name, content=(directory / name).read_bytes())
            for name in source.programming_filenames
            if name != _CATALOG_FILENAME
        ),
    )


def compare_programming_image(
    source: FavoritesProgrammingSource, candidate_directory: Path
) -> FavoritesProgrammingValidation:
    candidate_files = {
        path.name
        for path in candidate_directory.iterdir()
        if path.name not in {"manifest.json", "changes.json"}
    }
    source_files = set(source.programming_filenames)
    added, removed = (
        tuple(sorted(candidate_files - source_files)),
        tuple(sorted(source_files - candidate_files)),
    )
    if added or removed:
        return FavoritesProgrammingValidation(added, removed, (), (), (), (), ())
    candidate = _candidate_snapshot(source, candidate_directory)
    records, fields = _record_differences(source.snapshot, candidate)
    old_map, new_map = _snapshot_to_map(source.snapshot), _snapshot_to_map(candidate)
    changed = tuple(sorted(name for name in source_files if old_map[name] != new_map[name]))
    return FavoritesProgrammingValidation((), (), changed, records, fields, (), ())


def _atomic_write(path: Path, data: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _manifest_files(
    source: FavoritesProgrammingSource, snapshot: FavoritesStorageSnapshot
) -> list[dict[str, Any]]:
    old_map, new_map = _snapshot_to_map(source.snapshot), _snapshot_to_map(snapshot)
    return [
        {
            "filename": name,
            "bytes": len(data),
            "sha256": _sha256(data),
            "source_sha256": _sha256(old_map[name]),
            "changed": data != old_map[name],
        }
        for name, data in sorted(new_map.items())
    ]


def _validate_candidate_metadata(source: FavoritesProgrammingSource, directory: Path) -> None:
    manifest = _read_json(directory / "manifest.json")
    if (
        manifest.get("schema") != PROGRAMMING_IMAGE_SCHEMA
        or manifest.get("schema_version") != PROGRAMMING_IMAGE_SCHEMA_VERSION
    ):
        raise FavoritesProgrammingError("candidate manifest schema is invalid")
    source_meta = _mapping(manifest.get("source_backup"), "candidate source_backup")
    if source_meta.get("manifest_sha256") != source.manifest_sha256:
        raise FavoritesProgrammingError("candidate source manifest identity mismatch")
    if manifest.get("scanner_model") != "SDS200":
        raise FavoritesProgrammingError("candidate scanner model is invalid")
    if (
        manifest.get("target_model") != source.target_model
        or manifest.get("format_version") != source.format_version
    ):
        raise FavoritesProgrammingError("candidate Favorites metadata does not match source")
    entries = _list(manifest.get("files"), "candidate files")
    filenames = {
        _safe_filename(_mapping(entry, "candidate file").get("filename")) for entry in entries
    }
    if filenames != set(source.programming_filenames) or len(entries) != len(filenames):
        raise FavoritesProgrammingError("candidate manifest file list is incomplete or duplicated")
    source_map = _snapshot_to_map(source.snapshot)
    for raw in entries:
        entry = _mapping(raw, "candidate file")
        filename = _safe_filename(entry.get("filename"))
        data = _verify_manifest_file(directory, entry)
        if entry.get("source_sha256") != _sha256(source_map[filename]):
            raise FavoritesProgrammingError(f"candidate source SHA-256 mismatch for {filename}")
        if entry.get("changed") is not (data != source_map[filename]):
            raise FavoritesProgrammingError(f"candidate changed flag mismatch for {filename}")


def _validate_changes_metadata(
    directory: Path,
    validation: FavoritesProgrammingValidation,
) -> None:
    changes = _read_json(directory / "changes.json")
    if (
        changes.get("schema") != PROGRAMMING_CHANGES_SCHEMA
        or changes.get("schema_version") != PROGRAMMING_CHANGES_SCHEMA_VERSION
    ):
        raise FavoritesProgrammingError("changes manifest schema is invalid")
    plan = load_edit_plan(directory / "changes.json")
    expected_fields = tuple(
        sorted(
            (
                edit.filename,
                edit.record_index,
                edit.record_type,
                edit.field,
                edit.expected_old_value,
                edit.new_value,
            )
            for edit in plan.edits
        )
    )
    actual_fields = tuple(
        sorted(
            (
                change.filename,
                change.record_index,
                change.record_type,
                change.field,
                change.old_value,
                change.new_value,
            )
            for change in validation.field_changes
        )
    )
    if expected_fields != actual_fields:
        raise FavoritesProgrammingError("changes edits do not match candidate diff")
    summary = _mapping(changes.get("summary"), "changes summary")
    expected = {
        "changed_files": len({change.filename for change in validation.record_changes}),
        "changed_records": validation.changed_record_count,
        "changed_fields": validation.changed_field_count,
    }
    if dict(summary) != expected:
        raise FavoritesProgrammingError("changes summary does not match candidate diff")


def validate_candidate_image(
    source: FavoritesProgrammingSource, candidate_directory: Path
) -> FavoritesProgrammingValidation:
    directory = candidate_directory.resolve()
    if not directory.is_dir():
        raise FavoritesProgrammingError("candidate directory does not exist")
    _validate_candidate_metadata(source, directory)
    result = compare_programming_image(source, directory)
    if result.files_added or result.files_removed:
        raise FavoritesProgrammingError("candidate has unexpected added or removed files")
    if any(change.line_ending_changed for change in result.record_changes):
        raise FavoritesProgrammingError("candidate changed a source line ending")
    warnings, errors = _schema_diagnostics(_candidate_snapshot(source, directory))
    if errors:
        raise FavoritesProgrammingError("candidate schema validation failed: " + "; ".join(errors))
    result = FavoritesProgrammingValidation(
        result.files_added,
        result.files_removed,
        result.files_changed,
        result.record_changes,
        result.field_changes,
        warnings,
        errors,
    )
    _validate_changes_metadata(directory, result)
    return result


def generate_candidate_image(
    source: FavoritesProgrammingSource,
    output_directory: Path,
    plan: FavoritesProgrammingPlan,
    *,
    now: datetime | None = None,
) -> FavoritesProgrammingCandidate:
    if not isinstance(source, FavoritesProgrammingSource) or not isinstance(
        plan, FavoritesProgrammingPlan
    ):
        raise TypeError("candidate generation requires a programming source and plan")
    root = output_directory.resolve()
    try:
        root.relative_to(source.backup_directory)
    except ValueError:
        pass
    else:
        raise FavoritesProgrammingError(
            "candidate output must not be inside the immutable source backup"
        )
    instant = now or datetime.now(UTC)
    directory = root / f"sds200-programming-{instant.strftime('%Y%m%d-%H%M%S')}"
    if directory.exists():
        raise FavoritesProgrammingError(f"candidate directory already exists: {directory}")
    snapshot = _apply_plan(source, plan)
    directory.mkdir(parents=True)
    for filename, data in _snapshot_to_map(snapshot).items():
        _atomic_write(directory / filename, data)
    record_changes, field_changes = _record_differences(source.snapshot, snapshot)
    created_at = instant.isoformat()
    manifest = {
        "schema": PROGRAMMING_IMAGE_SCHEMA,
        "schema_version": PROGRAMMING_IMAGE_SCHEMA_VERSION,
        "created_at": created_at,
        "scanner_model": "SDS200",
        "target_model": source.target_model,
        "format_version": source.format_version,
        "source_backup": {
            "path": str(source.backup_directory),
            "manifest_sha256": source.manifest_sha256,
        },
        "files": _manifest_files(source, snapshot),
        "validation": {"status": "pending", "warnings": [], "errors": []},
    }
    changes = {
        "schema": PROGRAMMING_CHANGES_SCHEMA,
        "schema_version": PROGRAMMING_CHANGES_SCHEMA_VERSION,
        "created_at": created_at,
        "edits": [_edit_dict(edit) for edit in plan.edits],
        "summary": {
            "changed_files": len({change.filename for change in record_changes}),
            "changed_records": len(record_changes),
            "changed_fields": len(field_changes),
        },
    }
    _atomic_write(
        directory / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    _atomic_write(
        directory / "changes.json", (json.dumps(changes, indent=2, sort_keys=True) + "\n").encode()
    )
    validation = validate_candidate_image(source, directory)
    manifest["validation"] = {
        "status": "valid",
        "warnings": list(validation.schema_warnings),
        "errors": [],
    }
    _atomic_write(
        directory / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return FavoritesProgrammingCandidate(
        directory, directory / "manifest.json", directory / "changes.json", validation
    )


__all__ = [
    "FavoritesProgrammingCandidate",
    "FavoritesProgrammingEdit",
    "FavoritesProgrammingError",
    "FavoritesProgrammingFieldChange",
    "FavoritesProgrammingPlan",
    "FavoritesProgrammingRecordChange",
    "FavoritesProgrammingSource",
    "FavoritesProgrammingValidation",
    "compare_programming_image",
    "generate_candidate_image",
    "load_edit_plan",
    "load_programming_source",
    "validate_candidate_image",
]
