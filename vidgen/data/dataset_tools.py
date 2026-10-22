"""Dataset inspection and packing tools for FolderClipDataset directories.

The trainers consume a flat directory of uint8 ``.npy`` clips plus a
``manifest.jsonl`` file.  This module provides the reusable logic behind
``scripts/vidgin_helper.py``:

* :func:`check_dataset` gives a trainer-ready health report without
  loading whole clips into RAM (arrays are memory-mapped and brightness
  is sampled from the first/middle/last frames).
* :func:`pack_datasets` merges several prepared directories into one,
  prefixing file names so chunks with identical ``clip_000001.npy``
  names cannot overwrite each other.

No external model or network access is used here.  Caption generation
itself stays in ``scripts/caption_clips.py``; the helper CLI merely
exposes it alongside these local tools.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

MANIFEST_NAME = "manifest.jsonl"
DEFAULT_DARK_THRESHOLD = 25.0


@dataclass
class DatasetIssue:
    """One problem or warning found in a dataset."""

    severity: str  # "error" or "warning"
    code: str
    message: str
    file: str | None = None
    entry_index: int | None = None  # 1-based manifest entry number


@dataclass
class DatasetReport:
    """JSON-friendly summary returned by :func:`check_dataset`."""

    data_dir: str
    manifest_exists: bool = False
    manifest_lines: int = 0
    manifest_entries: int = 0
    referenced_clips: int = 0
    clip_files_on_disk: int = 0
    depth_files_on_disk: int = 0
    clean_clips: int = 0
    error_clips: int = 0
    warning_clips: int = 0
    missing_clips: int = 0
    readable_clips: int = 0
    depth_referenced: int = 0
    depth_present: int = 0
    depth_missing: int = 0
    depth_valid: int = 0
    clip_shape: list[int] | None = None
    depth_shape: list[int] | None = None
    shapes: dict[str, int] = field(default_factory=dict)
    depth_shapes: dict[str, int] = field(default_factory=dict)
    total_bytes: int = 0
    orphan_files: list[str] = field(default_factory=list)
    unreferenced_depth_files: list[str] = field(default_factory=list)
    dark_clips: list[str] = field(default_factory=list)
    duplicate_file_references: dict[str, list[int]] = field(default_factory=dict)
    exact_duplicate_groups: list[list[str]] = field(default_factory=list)
    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[DatasetIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[DatasetIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when the dataset has entries and no error-level issues."""

        return self.manifest_exists and self.manifest_entries > 0 and not self.errors

    def to_dict(self) -> dict:
        data = asdict(self)
        data["ok"] = self.ok
        data["error_count"] = len(self.errors)
        data["warning_count"] = len(self.warnings)
        return data


@dataclass
class PackedSource:
    label: str
    source: str
    entries: int
    depth_files: int
    bytes_copied: int


@dataclass
class PackReport:
    output_dir: str
    sources: list[PackedSource]
    entries: int
    clip_files: int
    depth_files: int
    total_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_join(base: str, name: str) -> str | None:
    """Join ``name`` under ``base``; return None if it escapes the folder."""

    if not isinstance(name, str) or not name:
        return None
    base_abs = os.path.abspath(base)
    path = os.path.abspath(os.path.join(base_abs, name))
    try:
        if os.path.commonpath([base_abs, path]) != base_abs:
            return None
    except ValueError:
        return None
    return path


def _array_header(path: str) -> tuple[tuple[int, ...], np.dtype, np.ndarray]:
    """Memory-map an .npy array and force a tiny first/last-value read."""

    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise ValueError("file does not contain a NumPy array")
    if arr.size:
        flat = arr.reshape(-1)
        _ = flat[0]
        _ = flat[-1]
    return tuple(int(x) for x in arr.shape), arr.dtype, arr


def _sampled_mean(arr: np.ndarray) -> float:
    """Mean brightness estimated from first/middle/last frames."""

    if arr.size == 0:
        return 0.0
    indices = sorted({0, len(arr) // 2, len(arr) - 1})
    return float(arr[indices].astype(np.float64, copy=False).mean())


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest_for_check(
    data_dir: str, report: DatasetReport, add_issue
) -> list[tuple[int, dict]]:
    """Read manifest lines, recording malformed lines as report issues."""

    manifest_path = os.path.join(data_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        add_issue("error", "manifest_missing", f"No {MANIFEST_NAME} in {data_dir}")
        return []

    report.manifest_exists = True
    entries: list[tuple[int, dict]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            report.manifest_lines += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                add_issue(
                    "error",
                    "manifest_invalid_json",
                    f"manifest line {line_no}: invalid JSON: {exc}",
                )
                continue
            if not isinstance(rec, dict):
                add_issue(
                    "error",
                    "manifest_entry_not_object",
                    f"manifest line {line_no}: expected a JSON object",
                )
                continue
            missing = [key for key in ("file", "caption") if key not in rec]
            if missing:
                add_issue(
                    "error",
                    "manifest_missing_keys",
                    f"manifest line {line_no}: missing keys {missing}",
                )
                continue
            entries.append((len(entries) + 1, rec))

    report.manifest_entries = len(entries)
    report.referenced_clips = len(entries)
    if not entries:
        add_issue("error", "manifest_empty", "manifest contains no usable entries")
    return entries


def check_dataset(
    data_dir: str,
    *,
    expected_frames: int | None = None,
    expected_height: int | None = None,
    expected_width: int | None = None,
    require_depth: bool = False,
    dark_threshold: float = DEFAULT_DARK_THRESHOLD,
    detect_duplicates: bool = False,
) -> DatasetReport:
    """Inspect a prepared clip directory and return a detailed report.

    The check is intentionally read-only.  It catches missing and
    unreadable files, wrong dtype/shape, inconsistent clip geometry,
    missing or mismatched depth maps, duplicate manifest references,
    orphan files, and (from sampled frames) very dark clips.
    """

    data_dir = str(data_dir)
    report = DatasetReport(data_dir=data_dir)
    entry_marks: dict[int, set[str]] = defaultdict(set)

    def add_issue(
        severity: str,
        code: str,
        message: str,
        file: str | None = None,
        entry_index: int | None = None,
    ) -> DatasetIssue:
        issue = DatasetIssue(severity, code, message, file, entry_index)
        report.issues.append(issue)
        if entry_index is not None:
            entry_marks[entry_index].add(severity)
        return issue

    if not os.path.isdir(data_dir):
        add_issue("error", "dataset_missing", f"Dataset directory not found: {data_dir}")
        return report

    # Directory inventory is cheap and helps distinguish missing files
    # from files that exist but are not named by the manifest.
    try:
        disk_names = sorted(
            name for name in os.listdir(data_dir) if name.endswith(".npy")
        )
    except OSError as exc:
        add_issue("error", "dataset_unreadable", f"Cannot list dataset: {exc}")
        return report
    disk_depth_names = [name for name in disk_names if name.endswith("_depth.npy")]
    disk_clip_names = [name for name in disk_names if not name.endswith("_depth.npy")]
    report.clip_files_on_disk = len(disk_clip_names)
    report.depth_files_on_disk = len(disk_depth_names)

    entries = _read_manifest_for_check(data_dir, report, add_issue)

    file_positions: dict[str, list[int]] = defaultdict(list)
    depth_positions: dict[str, list[int]] = defaultdict(list)
    for entry_index, rec in entries:
        if isinstance(rec.get("file"), str):
            file_positions[rec["file"]].append(entry_index)
        if isinstance(rec.get("depth"), str):
            depth_positions[rec["depth"]].append(entry_index)

    report.duplicate_file_references = {
        name: positions for name, positions in file_positions.items() if len(positions) > 1
    }
    for name, positions in report.duplicate_file_references.items():
        for entry_index in positions:
            add_issue(
                "error",
                "duplicate_file_reference",
                f"clip file is referenced by manifest entries {positions}",
                name,
                entry_index,
            )
    for name, positions in depth_positions.items():
        if len(positions) > 1:
            for entry_index in positions:
                add_issue(
                    "error",
                    "duplicate_depth_reference",
                    f"depth file is referenced by manifest entries {positions}",
                    name,
                    entry_index,
                )

    expected_shape: list[int | None] = [
        expected_frames,
        expected_height,
        expected_width,
        3,
    ]
    baseline_shape: tuple[int, ...] | None = None
    baseline_depth_shape: tuple[int, ...] | None = None
    referenced_clip_names: set[str] = set()
    referenced_depth_names: set[str] = set()
    hashes: dict[str, list[str]] = defaultdict(list)

    for entry_index, rec in entries:
        file_name = rec.get("file")
        caption = rec.get("caption")
        if not isinstance(caption, str):
            add_issue(
                "error",
                "caption_not_text",
                "caption must be a string",
                file_name if isinstance(file_name, str) else None,
                entry_index,
            )
        elif not caption.strip():
            add_issue(
                "warning",
                "caption_empty",
                "caption is empty; trainers will substitute a generic label",
                file_name,
                entry_index,
            )

        if not isinstance(file_name, str) or not file_name:
            add_issue(
                "error",
                "file_name_invalid",
                "manifest 'file' must be a non-empty string",
                None,
                entry_index,
            )
            continue
        referenced_clip_names.add(file_name)
        clip_path = _safe_join(data_dir, file_name)
        if clip_path is None:
            add_issue(
                "error",
                "file_outside_dataset",
                "clip path escapes the dataset directory",
                file_name,
                entry_index,
            )
            continue
        if not os.path.isfile(clip_path):
            report.missing_clips += 1
            add_issue(
                "error", "clip_missing", "referenced clip file is missing",
                file_name, entry_index,
            )
            continue

        report.total_bytes += os.path.getsize(clip_path)
        clip_shape: tuple[int, ...] | None = None
        try:
            shape, dtype, arr = _array_header(clip_path)
            clip_shape = shape
            report.readable_clips += 1
            report.shapes[str(shape)] = report.shapes.get(str(shape), 0) + 1
            if dtype != np.uint8:
                add_issue(
                    "error", "clip_wrong_dtype",
                    f"expected uint8 clip, got {dtype}", file_name, entry_index,
                )
            if len(shape) != 4:
                add_issue(
                    "error", "clip_wrong_dimensions",
                    f"expected 4D (T,H,W,3) clip, got shape {shape}",
                    file_name, entry_index,
                )
            else:
                if shape[3] != 3:
                    add_issue(
                        "error", "clip_wrong_channels",
                        f"expected 3 colour channels, got shape {shape}",
                        file_name, entry_index,
                    )
                if baseline_shape is None:
                    baseline_shape = shape
                    report.clip_shape = list(shape)
                # Fill unspecified expected dimensions from the first
                # valid clip, then enforce consistency afterwards.
                for axis, value in enumerate(shape):
                    if expected_shape[axis] is None:
                        expected_shape[axis] = value
                mismatched = [
                    axis for axis, value in enumerate(shape)
                    if expected_shape[axis] is not None and value != expected_shape[axis]
                ]
                if mismatched:
                    add_issue(
                        "error", "clip_shape_mismatch",
                        f"expected clip shape {tuple(expected_shape)}, got {shape}",
                        file_name, entry_index,
                    )
                if shape[0] > 1 and (shape[0] - 1) % 4 != 0:
                    add_issue(
                        "warning", "clip_frame_rule",
                        "frame count does not follow the Vidgin T = 4k+1 rule",
                        file_name, entry_index,
                    )
                if dtype == np.uint8 and shape[3] == 3:
                    mean = _sampled_mean(arr)
                    if mean < dark_threshold:
                        report.dark_clips.append(file_name)
                        add_issue(
                            "warning", "clip_too_dark",
                            f"sampled mean brightness {mean:.1f} is below {dark_threshold:.1f}",
                            file_name, entry_index,
                        )
            if detect_duplicates:
                hashes[_file_sha256(clip_path)].append(file_name)
        except Exception as exc:  # unreadable/corrupt .npy
            add_issue(
                "error", "clip_unreadable",
                f"cannot read clip array: {exc}", file_name, entry_index,
            )

        # Depth is optional unless the caller explicitly requires it.
        depth_name = rec.get("depth")
        if depth_name is None:
            if require_depth:
                add_issue(
                    "error", "depth_not_referenced",
                    "depth is required but this manifest entry has no 'depth' key",
                    file_name, entry_index,
                )
            continue
        if not isinstance(depth_name, str) or not depth_name:
            add_issue(
                "error", "depth_name_invalid",
                "manifest 'depth' must be a non-empty string",
                file_name, entry_index,
            )
            continue

        report.depth_referenced += 1
        referenced_depth_names.add(depth_name)
        depth_path = _safe_join(data_dir, depth_name)
        if depth_path is None:
            add_issue(
                "error", "depth_outside_dataset",
                "depth path escapes the dataset directory",
                depth_name, entry_index,
            )
            continue
        if not os.path.isfile(depth_path):
            report.depth_missing += 1
            add_issue(
                "error", "depth_missing",
                "referenced depth file is missing", depth_name, entry_index,
            )
            continue

        report.depth_present += 1
        report.total_bytes += os.path.getsize(depth_path)
        try:
            dshape, ddtype, _ = _array_header(depth_path)
            report.depth_shapes[str(dshape)] = report.depth_shapes.get(str(dshape), 0) + 1
            valid_depth = True
            if ddtype != np.uint8:
                valid_depth = False
                add_issue(
                    "error", "depth_wrong_dtype",
                    f"expected uint8 depth, got {ddtype}", depth_name, entry_index,
                )
            if len(dshape) != 3:
                valid_depth = False
                add_issue(
                    "error", "depth_wrong_dimensions",
                    f"expected 3D (T,H,W) depth, got shape {dshape}",
                    depth_name, entry_index,
                )
            else:
                if baseline_depth_shape is None:
                    baseline_depth_shape = dshape
                    report.depth_shape = list(dshape)
                if clip_shape is not None and len(clip_shape) == 4 and dshape != clip_shape[:3]:
                    valid_depth = False
                    add_issue(
                        "error", "depth_shape_mismatch",
                        f"expected depth shape {clip_shape[:3]}, got {dshape}",
                        depth_name, entry_index,
                    )
                elif baseline_depth_shape is not None and dshape != baseline_depth_shape:
                    valid_depth = False
                    add_issue(
                        "error", "depth_shape_mismatch",
                        f"expected consistent depth shape {baseline_depth_shape}, got {dshape}",
                        depth_name, entry_index,
                    )
            if valid_depth:
                report.depth_valid += 1
        except Exception as exc:
            add_issue(
                "error", "depth_unreadable",
                f"cannot read depth array: {exc}", depth_name, entry_index,
            )

    if detect_duplicates:
        report.exact_duplicate_groups = [
            sorted(names) for names in hashes.values() if len(names) > 1
        ]
        for group in report.exact_duplicate_groups:
            for name in group:
                positions = file_positions.get(name, [])
                add_issue(
                    "warning", "clip_exact_duplicate",
                    f"clip content is duplicated by {group}", name,
                    positions[0] if positions else None,
                )

    report.orphan_files = sorted(set(disk_clip_names) - referenced_clip_names)
    report.unreferenced_depth_files = sorted(
        set(disk_depth_names) - referenced_depth_names
    )
    for name in report.orphan_files:
        add_issue(
            "warning", "orphan_clip_file",
            "clip file exists on disk but is not in the manifest", name,
        )
    for name in report.unreferenced_depth_files:
        add_issue(
            "warning", "unreferenced_depth_file",
            "depth file exists on disk but is not in the manifest", name,
        )

    for entry_index, _ in entries:
        marks = entry_marks.get(entry_index, set())
        if "error" in marks:
            report.error_clips += 1
        elif "warning" in marks:
            report.warning_clips += 1
        else:
            report.clean_clips += 1

    return report


def format_dataset_report(report: DatasetReport, *, max_issues: int = 50) -> str:
    """Render a short human-readable dataset health report."""

    size_mb = report.total_bytes / (1024 * 1024)
    lines = [
        f"Dataset: {report.data_dir}",
        f"Manifest: {report.manifest_entries} entries ({report.manifest_lines} lines)",
        (
            f"Clips: {report.referenced_clips} referenced, {report.clean_clips} clean, "
            f"{report.missing_clips} missing, {report.clip_files_on_disk} clip files on disk"
        ),
        (
            f"Depth: {report.depth_referenced} referenced, {report.depth_present} present, "
            f"{report.depth_valid} valid, {report.depth_missing} missing"
        ),
    ]
    if report.clip_shape:
        lines.append(f"Clip shape: {tuple(report.clip_shape)} (T,H,W,3)")
    if report.depth_shape:
        lines.append(f"Depth shape: {tuple(report.depth_shape)} (T,H,W)")
    lines.extend(
        [
            f"Referenced size: {size_mb:.1f} MB",
            f"Dark clips (sampled): {len(report.dark_clips)}",
            f"Orphan clip files: {len(report.orphan_files)}",
            f"Errors: {len(report.errors)}",
            f"Warnings: {len(report.warnings)}",
            f"Result: {'OK - ready' if report.ok else 'NOT READY'}",
        ]
    )
    if report.issues:
        lines.append("Issues:")
        for issue in report.issues[:max_issues]:
            where = f" {issue.file}" if issue.file else ""
            lines.append(
                f"  - {issue.severity.upper()} [{issue.code}]{where}: {issue.message}"
            )
        if len(report.issues) > max_issues:
            lines.append(f"  ... and {len(report.issues) - max_issues} more")
    return "\n".join(lines)


def _read_manifest_strict(data_dir: str) -> list[dict]:
    manifest_path = os.path.join(data_dir, MANIFEST_NAME)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"No {MANIFEST_NAME} in {data_dir}")
    entries: list[dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(rec, dict) or "file" not in rec or "caption" not in rec:
                raise ValueError(
                    f"{manifest_path}:{line_no}: each line must contain 'file' and 'caption'"
                )
            entries.append(rec)
    if not entries:
        raise ValueError(f"{manifest_path}: manifest is empty")
    return entries


def _sanitize_label(value: str, fallback: str = "dataset") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    return cleaned or fallback


def _unique_label(base: str, used: set[str]) -> str:
    label = base
    suffix = 2
    while label in used:
        label = f"{base}_{suffix}"
        suffix += 1
    used.add(label)
    return label


def _is_within(path: str, parent: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(parent)]) == os.path.abspath(parent)
    except ValueError:
        return False


def _publish_staging(staging: str, output_dir: str, overwrite: bool) -> None:
    """Move a completed staging directory into place as safely as possible."""

    if not os.path.exists(output_dir):
        os.replace(staging, output_dir)
        return
    if not os.path.isdir(output_dir):
        raise FileExistsError(f"output exists and is not a directory: {output_dir}")
    if not os.listdir(output_dir):
        os.rmdir(output_dir)
        os.replace(staging, output_dir)
        return
    if not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir} (use overwrite to replace it)"
        )

    parent = os.path.dirname(output_dir) or "."
    backup = tempfile.mkdtemp(prefix=".vidgin-pack-backup-", dir=parent)
    os.rmdir(backup)
    os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except Exception:
        os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup)


def pack_datasets(
    sources: list[str],
    output_dir: str,
    *,
    labels: list[str] | None = None,
    overwrite: bool = False,
) -> PackReport:
    """Merge prepared dataset directories into one trainer-ready pack.

    Every source is health-checked first.  A source with error-level
    problems, or a mixture of clip shapes across sources, aborts the
    pack before anything is published.  Files are copied into a staging
    directory and only moved into place after the merged manifest and
    counts verify, so a failed pack does not leave a half-pack behind.
    """

    if not sources:
        raise ValueError("at least one source dataset is required")
    if labels is not None and len(labels) != len(sources):
        raise ValueError("--labels must provide exactly one label per source dataset")

    source_paths = [os.path.abspath(str(path)) for path in sources]
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("the same source dataset was given more than once")
    output_abs = os.path.abspath(str(output_dir))

    for source in source_paths:
        if _is_within(output_abs, source) or _is_within(source, output_abs):
            raise ValueError("output directory must not overlap a source dataset")

    if labels is None:
        raw_labels = [os.path.basename(path.rstrip(os.sep)) or "dataset" for path in source_paths]
        used: set[str] = set()
        pack_labels = [
            _unique_label(_sanitize_label(label), used) for label in raw_labels
        ]
    else:
        pack_labels = [_sanitize_label(label) for label in labels]
        if len(set(pack_labels)) != len(pack_labels):
            raise ValueError("pack labels must be unique after cleaning")

    # Preflight every source before copying a single large clip.
    source_entries: list[list[dict]] = []
    source_shapes: set[tuple[int, ...]] = set()
    for source, label in zip(source_paths, pack_labels):
        report = check_dataset(source)
        if not report.ok:
            first = report.errors[0]
            raise ValueError(
                f"source {label!r} ({source}) is not ready: "
                f"{len(report.errors)} error(s); first: [{first.code}] {first.message}"
            )
        if report.clip_shape is not None:
            source_shapes.add(tuple(report.clip_shape))
        source_entries.append(_read_manifest_strict(source))
    if len(source_shapes) > 1:
        raise ValueError(f"sources have different clip shapes: {sorted(source_shapes)}")

    output_parent = os.path.dirname(output_abs) or "."
    os.makedirs(output_parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".vidgin-pack-", dir=output_parent)
    packed_sources: list[PackedSource] = []
    merged_entries: list[dict] = []
    output_names: set[str] = set()
    total_bytes = 0
    clip_files = 0
    depth_files = 0

    try:
        for source, label, entries in zip(source_paths, pack_labels, source_entries):
            source_bytes = 0
            source_depth = 0
            copied: dict[str, str] = {}

            def copy_one(name: str, *, is_depth: bool = False) -> str:
                if name in copied:
                    return copied[name]
                src = _safe_join(source, name)
                if src is None or not os.path.isfile(src):
                    raise FileNotFoundError(f"{source}: referenced file missing: {name}")
                out_name = f"{label}__{os.path.basename(name)}"
                if out_name in output_names:
                    raise ValueError(f"packed file name collision: {out_name}")
                output_names.add(out_name)
                dest = os.path.join(staging, out_name)
                shutil.copy2(src, dest)
                copied[name] = out_name
                return out_name

            for rec in entries:
                new_rec = dict(rec)
                clip_source_name = rec["file"]
                clip_out_name = copy_one(clip_source_name)
                size = os.path.getsize(os.path.join(staging, clip_out_name))
                source_bytes += size
                total_bytes += size
                clip_files += 1
                new_rec["file"] = clip_out_name

                if rec.get("depth") is not None:
                    depth_out_name = copy_one(str(rec["depth"]), is_depth=True)
                    size = os.path.getsize(os.path.join(staging, depth_out_name))
                    source_bytes += size
                    total_bytes += size
                    depth_files += 1
                    source_depth += 1
                    new_rec["depth"] = depth_out_name

                new_rec["pack_source"] = label
                merged_entries.append(new_rec)

            packed_sources.append(
                PackedSource(
                    label=label,
                    source=source,
                    entries=len(entries),
                    depth_files=source_depth,
                    bytes_copied=source_bytes,
                )
            )

        manifest_path = os.path.join(staging, MANIFEST_NAME)
        with open(manifest_path, "w", encoding="utf-8") as f:
            for rec in merged_entries:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        verify = check_dataset(staging)
        if not verify.ok:
            first = verify.errors[0]
            raise ValueError(
                f"packed dataset failed verification: [{first.code}] {first.message}"
            )
        if verify.manifest_entries != len(merged_entries):
            raise ValueError("packed manifest count does not match the source total")
        if verify.depth_referenced != depth_files:
            raise ValueError("packed depth count does not match the source total")

        _publish_staging(staging, output_abs, overwrite)
        staging = ""  # published; no cleanup needed
    finally:
        if staging and os.path.isdir(staging):
            shutil.rmtree(staging)

    return PackReport(
        output_dir=output_abs,
        sources=packed_sources,
        entries=len(merged_entries),
        clip_files=clip_files,
        depth_files=depth_files,
        total_bytes=total_bytes,
    )


def format_pack_report(report: PackReport) -> str:
    lines = [f"Packed dataset: {report.output_dir}"]
    for source in report.sources:
        lines.append(
            f"  - {source.label}: {source.entries} clips, "
            f"{source.depth_files} depth maps from {source.source}"
        )
    lines.append(
        f"Total: {report.entries} clips, {report.depth_files} depth maps, "
        f"{report.total_bytes / (1024 * 1024):.1f} MB"
    )
    lines.append("Result: OK - ready for training")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic train/val/test splits (grouped by source, no leakage)
# ---------------------------------------------------------------------------

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitReport:
    """Result of :func:`make_splits`: counts per split + written files."""

    data_dir: str
    seed: int
    ratios: dict[str, float]
    counts: dict[str, int]
    groups: dict[str, int]
    files: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


def split_group_key(entry: dict) -> str:
    """Group key deciding which clips must stay in the SAME split.

    Clips cut from one source video share a ``group``/``source``/
    ``source_url`` manifest field when the prep tool recorded one;
    without such a field each clip is its own group (nothing to leak).
    """
    for key in ("group", "source", "source_url"):
        value = entry.get(key)
        if value:
            return str(value)
    return str(entry["file"])


def make_splits(
    data_dir: str,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 1234,
    overwrite: bool = False,
) -> SplitReport:
    """Write deterministic ``manifest_<split>.jsonl`` files into a dataset.

    Groups (see :func:`split_group_key`) are ordered by a seeded hash and
    assigned greedily to the split furthest below its target share, so a
    group never straddles two splits and re-running with the same seed
    reproduces the identical split. The trainers consume a split via
    ``--data-dir <dir> --data-split train`` (likewise ``val``/``test``).
    Also writes ``splits.json`` describing the assignment.
    """
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must be three non-negative numbers summing to 1, got {ratios}")
    entries = _read_manifest_strict(data_dir)
    total = len(entries)

    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        groups[split_group_key(entry)].append(entry)
    ordered_keys = sorted(
        groups,
        key=lambda k: hashlib.sha256(f"{seed}:{k}".encode("utf-8")).hexdigest(),
    )

    targets = {name: ratios[i] * total for i, name in enumerate(SPLIT_NAMES)}
    counts = {name: 0 for name in SPLIT_NAMES}
    assigned: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    group_counts = {name: 0 for name in SPLIT_NAMES}
    for key in ordered_keys:
        members = groups[key]
        # Split with the largest remaining deficit; ties keep SPLIT_NAMES order.
        name = max(SPLIT_NAMES, key=lambda n: (targets[n] - counts[n], -SPLIT_NAMES.index(n)))
        assigned[name].extend(members)
        counts[name] += len(members)
        group_counts[name] += 1

    files: dict[str, str] = {}
    for name in SPLIT_NAMES:
        if ratios[SPLIT_NAMES.index(name)] == 0 and not assigned[name]:
            continue
        path = os.path.join(data_dir, f"manifest_{name}.jsonl")
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"{path} already exists; pass overwrite=True to regenerate the splits"
            )
        # Keep each split's manifest in the original dataset order.
        wanted = {id(e) for e in assigned[name]}
        ordered = [e for e in entries if id(e) in wanted]
        with open(path, "w", encoding="utf-8") as f:
            for entry in ordered:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        files[name] = path

    report = SplitReport(
        data_dir=data_dir,
        seed=seed,
        ratios={name: ratios[i] for i, name in enumerate(SPLIT_NAMES)},
        counts=counts,
        groups=group_counts,
        files=files,
    )
    splits_json = os.path.join(data_dir, "splits.json")
    if os.path.exists(splits_json) and not overwrite:
        raise FileExistsError(
            f"{splits_json} already exists; pass overwrite=True to regenerate the splits"
        )
    with open(splits_json, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return report


def format_split_report(report: SplitReport) -> str:
    lines = [f"Splits for {report.data_dir} (seed {report.seed}):"]
    for name in SPLIT_NAMES:
        if name in report.counts:
            lines.append(
                f"  {name:5s}: {report.counts[name]:6d} clips "
                f"in {report.groups[name]} groups -> {report.files.get(name, '-')}"
            )
    lines.append("Groups never straddle splits (no source leakage).")
    return "\n".join(lines)


# ------------------------------------------------------- caption auditing
GENERIC_CAPTIONS = {
    "", "stock video", "stock video clip", "video", "clip", "a video",
    "stock footage", "people", "city", "city and people",
}


@dataclass
class CaptionAudit:
    """Caption-quality statistics for one manifest (Phase 3, 2026-10-22).

    Captions are the ONLY teaching signal for text conditioning, and the
    prep pipeline historically stored the *search query* as the caption —
    a query is not a description of the actual footage. This audit does
    not judge truth (only a human or a captioning model looking at the
    frames can), but it flags the mechanical failure modes: empty,
    missing, duplicated, overly generic, and very short captions, plus
    coverage of the caption_source field that records HOW each caption
    was produced (search query vs Gemini-from-frames vs manual).
    """

    total: int
    missing: int
    empty: int
    duplicated: int
    generic: int
    very_short: int
    sources: dict[str, int]
    flagged_files: dict[str, list[str]]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def ok(self) -> bool:
        return self.missing == 0 and self.empty == 0


def audit_captions(entries: list[dict], *, min_words: int = 3,
                   max_flagged: int = 50) -> CaptionAudit:
    """Audit manifest entries' captions; see :class:`CaptionAudit`."""
    counts = Counter(str(e.get("caption") or "").strip() for e in entries)
    sources: Counter = Counter()
    flagged: dict[str, list[str]] = defaultdict(list)
    missing = empty = duplicated = generic = very_short = 0

    def flag(kind: str, fname: str) -> None:
        if len(flagged[kind]) < max_flagged:
            flagged[kind].append(fname)

    for e in entries:
        fname = str(e.get("file", "?"))
        sources[str(e.get("caption_source", "unknown"))] += 1
        if "caption" not in e:
            missing += 1
            flag("missing", fname)
            continue
        cap = str(e.get("caption") or "").strip()
        if not cap:
            empty += 1
            flag("empty", fname)
            continue
        if counts[cap] > 1:
            duplicated += 1
            flag("duplicated", fname)
        if cap.lower() in GENERIC_CAPTIONS:
            generic += 1
            flag("generic", fname)
        if len(cap.split()) < min_words:
            very_short += 1
            flag("very_short", fname)
    return CaptionAudit(
        total=len(entries), missing=missing, empty=empty,
        duplicated=duplicated, generic=generic, very_short=very_short,
        sources=dict(sources), flagged_files=dict(flagged),
    )


def format_caption_audit(audit: CaptionAudit) -> str:
    lines = [
        f"Caption audit: {audit.total} entries",
        f"  missing: {audit.missing}   empty: {audit.empty}   "
        f"duplicated: {audit.duplicated}   generic: {audit.generic}   "
        f"very short: {audit.very_short}",
        f"  caption sources: {audit.sources}",
    ]
    if audit.generic or audit.duplicated:
        lines.append(
            "  NOTE: duplicated/generic captions are usually search "
            "queries, not descriptions of the footage — run the caption "
            "tool (helper `captions`) before serious training.")
    return "\n".join(lines)


# ------------------------------------------- perceptual duplicate hunting
def clip_signature(arr: np.ndarray, size: int = 16) -> np.ndarray:
    """Small perceptual signature of a clip: mean-pooled grayscale frames.

    Two clips with near-identical signatures are near-duplicates even if
    their bytes differ (re-encode, slight crop). Used for leakage-safe
    dedupe checks; exact byte duplicates remain the job of content
    hashes in the prep tools.
    """
    a = np.asarray(arr)
    if a.ndim == 4:  # (T,H,W,3) uint8
        gray = a.astype(np.float32).mean(axis=3)
    else:  # already (T,H,W)
        gray = a.astype(np.float32)
    t, h, w = gray.shape
    gh, gw = min(size, h), min(size, w)
    ys = (np.linspace(0, h, gh, endpoint=False)).astype(int)
    xs = (np.linspace(0, w, gw, endpoint=False)).astype(int)
    return gray[:, ys][:, :, xs].reshape(t, -1).mean(axis=0) / 255.0


def find_near_duplicates(data_dir: str, *, threshold: float = 0.01,
                         max_clips: int = 512) -> list[tuple[str, str, float]]:
    """Pairs of clips whose perceptual signatures are within ``threshold``.

    O(n^2) over at most ``max_clips`` manifest entries — a spot check
    for big datasets, a full check for small ones. Returns
    (file_a, file_b, mean_abs_signature_distance) sorted by distance.
    """
    entries = _read_manifest_strict(data_dir)[:max_clips]
    sigs: list[tuple[str, np.ndarray]] = []
    for e in entries:
        path = _safe_join(data_dir, str(e["file"]))
        if path is None or not os.path.isfile(path):
            continue
        arr = np.load(path, mmap_mode="r")
        sigs.append((str(e["file"]), clip_signature(np.asarray(arr))))
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            d = float(np.abs(sigs[i][1] - sigs[j][1]).mean())
            if d <= threshold:
                pairs.append((sigs[i][0], sigs[j][0], d))
    return sorted(pairs, key=lambda p: p[2])


# ------------------------------------------------------ split verification
def verify_splits(data_dir: str) -> dict:
    """Verify written splits are group-disjoint and cover the manifest.

    Returns a report dict: per-split counts, groups per split, any group
    appearing in more than one split (leakage), manifest entries missing
    from every split, and split entries not in the manifest. Raises
    FileNotFoundError when no split manifests exist.
    """
    base = _read_manifest_strict(data_dir)
    base_files = {str(e["file"]) for e in base}
    group_owner: dict[str, str] = {}
    leaked: set[str] = set()
    seen_files: set[str] = set()
    counts: dict[str, int] = {}
    groups: dict[str, int] = {}
    unknown: list[str] = []
    found = False
    for name in SPLIT_NAMES:
        path = os.path.join(data_dir, f"manifest_{name}.jsonl")
        if not os.path.isfile(path):
            continue
        found = True
        entries = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        counts[name] = len(entries)
        gset = set()
        for e in entries:
            g = split_group_key(e)
            gset.add(g)
            if g in group_owner and group_owner[g] != name:
                leaked.add(g)
            group_owner.setdefault(g, name)
            fname = str(e["file"])
            seen_files.add(fname)
            if fname not in base_files:
                unknown.append(fname)
        groups[name] = len(gset)
    if not found:
        raise FileNotFoundError(
            f"no manifest_<split>.jsonl files in {data_dir}; run make_splits first")
    return {
        "data_dir": data_dir,
        "counts": counts,
        "groups": groups,
        "leaked_groups": sorted(leaked),
        "missing_files": sorted(base_files - seen_files),
        "unknown_files": unknown,
        "ok": not leaked and not unknown,
        "small_dataset_warning": len(base) < 50,
    }
