"""Applies approved label corrections from a CSV manifest.

This script supports two dataset layouts:
1) Split layout: data/train|val|test/<class>/...
2) Master layout: data_master/<class>/...

Safety features:
- Dry-run mode (enabled by default).
- Strict class validation.
- Path resolution with ambiguity checks.
- Collision-safe destination naming.
- Full audit CSV + JSON summary.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# =========================
# Configuration
# =========================
DATASET_ROOT = Path('data')
CLASS_NAMES = ('y', 'n')

# If MANIFEST_PATH is None, the script auto-discovers the newest
# review manifest from review_candidates/*/review_manifest_template.csv.
MANIFEST_PATH: Path | None = None

# Default is dry-run for safety. Set to False to execute moves.
DRY_RUN = False

# If True, 'old_label' in CSV must match detected current label (if provided).
STRICT_OLD_LABEL_MATCH = True

# If True, rows with empty new_label are ignored.
SKIP_EMPTY_NEW_LABEL = True

ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


@dataclass
class FileRecord:
    path: Path
    rel: Path
    label: str
    split: str | None


@dataclass
class DatasetLayout:
    mode: str  # 'split' | 'master'
    root: Path


@dataclass
class ApplyResult:
    status: str
    message: str
    src: str | None
    dst: str | None
    image_ref: str
    old_label: str | None
    new_label: str | None


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return text


def detect_layout(root: Path) -> DatasetLayout:
    split_dirs = ['train', 'val', 'test']
    has_split = all((root / d).exists() for d in split_dirs)
    has_master = all((root / c).exists() for c in CLASS_NAMES)

    if has_split:
        return DatasetLayout(mode='split', root=root)
    if has_master:
        return DatasetLayout(mode='master', root=root)

    raise FileNotFoundError(
        'Could not detect dataset layout. Expected either:\n'
        f'- split layout under {root}/train|val|test/<class>\n'
        f'- master layout under {root}/<class>'
    )


def discover_manifest() -> Path:
    candidates = sorted(Path('review_candidates').glob('review_candidates_*/review_manifest_template.csv'))
    if candidates:
        return candidates[-1]

    fallback = Path('label_corrections.csv')
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        'No manifest found. Set MANIFEST_PATH or create one from 09_review_candidates.py output.'
    )


def gather_files(layout: DatasetLayout) -> list[FileRecord]:
    records: list[FileRecord] = []

    if layout.mode == 'split':
        for split in ('train', 'val', 'test'):
            split_dir = layout.root / split
            if not split_dir.exists():
                continue
            for label in CLASS_NAMES:
                class_dir = split_dir / label
                if not class_dir.exists():
                    continue
                for file_path in class_dir.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in ALLOWED_IMAGE_EXTS:
                        records.append(
                            FileRecord(
                                path=file_path,
                                rel=file_path.relative_to(layout.root),
                                label=label,
                                split=split,
                            )
                        )
    else:
        for label in CLASS_NAMES:
            class_dir = layout.root / label
            if not class_dir.exists():
                continue
            for file_path in class_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in ALLOWED_IMAGE_EXTS:
                    records.append(
                        FileRecord(
                            path=file_path,
                            rel=file_path.relative_to(layout.root),
                            label=label,
                            split=None,
                        )
                    )

    return records


def build_lookup(records: list[FileRecord]) -> tuple[dict[str, FileRecord], dict[str, list[FileRecord]]]:
    by_rel: dict[str, FileRecord] = {}
    by_basename: dict[str, list[FileRecord]] = {}

    for rec in records:
        by_rel[rec.rel.as_posix()] = rec
        by_basename.setdefault(rec.path.name, []).append(rec)

    return by_rel, by_basename


def resolve_record_for_image_ref(
    image_ref: str,
    dataset_root: Path,
    by_rel: dict[str, FileRecord],
    by_basename: dict[str, list[FileRecord]],
) -> FileRecord:
    ref = image_ref.strip()
    if not ref:
        raise ValueError('Empty image_ref')

    p = Path(ref)

    # 1) absolute path
    if p.is_absolute() and p.exists():
        try:
            rel = p.relative_to(dataset_root).as_posix()
        except ValueError as exc:
            raise ValueError(f'Absolute path outside DATASET_ROOT: {p}') from exc
        if rel in by_rel:
            return by_rel[rel]

    # 2) direct relative match
    ref_norm = p.as_posix().lstrip('./')
    if ref_norm in by_rel:
        return by_rel[ref_norm]

    # 3) try stripping leading data/ if present
    if ref_norm.startswith('data/'):
        alt = ref_norm[len('data/'):]
        if alt in by_rel:
            return by_rel[alt]

    # 4) basename fallback only if unique
    basename = p.name
    matches = by_basename.get(basename, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f'Ambiguous basename "{basename}" (found {len(matches)} matches). '
            'Use full relative path in image_ref.'
        )

    raise FileNotFoundError(f'Could not resolve image_ref: {image_ref}')


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f'{stem}__relabel_{counter}{suffix}'
        if not candidate.exists():
            return candidate
        counter += 1


def build_destination(layout: DatasetLayout, rec: FileRecord, new_label: str) -> Path:
    if layout.mode == 'split':
        assert rec.split is not None
        src_root = layout.root / rec.split / rec.label
        rel_under_label = rec.path.relative_to(src_root)
        dst = layout.root / rec.split / new_label / rel_under_label
    else:
        src_root = layout.root / rec.label
        rel_under_label = rec.path.relative_to(src_root)
        dst = layout.root / new_label / rel_under_label
    return dst


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f'Manifest file not found: {path}')

    with path.open('r', encoding='utf-8-sig', newline='') as fp:
        reader = csv.DictReader(fp)
        rows = [dict(r) for r in reader]

    if not rows:
        raise ValueError(f'Manifest is empty: {path}')

    if 'image_ref' not in rows[0]:
        raise ValueError('Manifest must include "image_ref" column.')

    return rows


def should_process_row(row: dict[str, str]) -> bool:
    new_label = normalize_label(row.get('new_label'))
    if SKIP_EMPTY_NEW_LABEL and new_label is None:
        return False
    return new_label is not None


def apply_corrections() -> None:
    dataset_root = DATASET_ROOT.resolve()
    layout = detect_layout(dataset_root)

    manifest_path = MANIFEST_PATH if MANIFEST_PATH is not None else discover_manifest()
    manifest_path = manifest_path.resolve()

    rows = read_manifest_rows(manifest_path)
    records = gather_files(layout)
    by_rel, by_basename = build_lookup(records)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('label_correction_logs')
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, Any]] = []
    stats = {
        'total_rows': len(rows),
        'rows_skipped_no_new_label': 0,
        'rows_processed': 0,
        'moved': 0,
        'noop_same_label': 0,
        'errors': 0,
    }

    for i, row in enumerate(rows, start=1):
        if not should_process_row(row):
            stats['rows_skipped_no_new_label'] += 1
            continue

        image_ref = str(row.get('image_ref', '')).strip()
        requested_old_label = normalize_label(row.get('old_label') or row.get('current_label'))
        new_label = normalize_label(row.get('new_label'))

        result = ApplyResult(
            status='error',
            message='',
            src=None,
            dst=None,
            image_ref=image_ref,
            old_label=requested_old_label,
            new_label=new_label,
        )

        try:
            if new_label not in CLASS_NAMES:
                raise ValueError(
                    f'Invalid new_label "{new_label}". Allowed: {CLASS_NAMES}'
                )

            rec = resolve_record_for_image_ref(image_ref, dataset_root, by_rel, by_basename)
            detected_old_label = rec.label

            if requested_old_label and STRICT_OLD_LABEL_MATCH and requested_old_label != detected_old_label:
                raise ValueError(
                    f'old_label mismatch for {image_ref}: requested={requested_old_label}, detected={detected_old_label}'
                )

            if new_label == detected_old_label:
                result.status = 'noop'
                result.message = 'new_label equals current label; nothing to move.'
                result.src = str(rec.path)
                result.dst = str(rec.path)
                stats['noop_same_label'] += 1
                stats['rows_processed'] += 1
            else:
                dst = build_destination(layout, rec, new_label)
                dst = unique_destination(dst)

                result.src = str(rec.path)
                result.dst = str(dst)

                if DRY_RUN:
                    result.status = 'dry_run'
                    result.message = 'planned move only'
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(rec.path), str(dst))
                    result.status = 'moved'
                    result.message = 'move applied'
                    stats['moved'] += 1

                stats['rows_processed'] += 1

        except Exception as exc:  # noqa: BLE001
            result.status = 'error'
            result.message = str(exc)
            stats['errors'] += 1

        audit_rows.append(
            {
                'row_index': i,
                'status': result.status,
                'message': result.message,
                'image_ref': result.image_ref,
                'old_label': result.old_label,
                'new_label': result.new_label,
                'src': result.src,
                'dst': result.dst,
            }
        )

    audit_csv = out_dir / f'label_corrections_audit_{timestamp}.csv'
    with audit_csv.open('w', encoding='utf-8', newline='') as fp:
        fieldnames = [
            'row_index',
            'status',
            'message',
            'image_ref',
            'old_label',
            'new_label',
            'src',
            'dst',
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    summary = {
        'timestamp': timestamp,
        'dry_run': DRY_RUN,
        'dataset_root': str(dataset_root),
        'layout_mode': layout.mode,
        'manifest_path': str(manifest_path),
        'audit_csv': str(audit_csv),
        'stats': stats,
    }

    summary_json = out_dir / f'label_corrections_summary_{timestamp}.json'
    summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    apply_corrections()
