"""Rebuilds deterministic train/val/test splits from a master dataset.

Expected master layout:
- data_master/y/...
- data_master/n/...

Generated split layout:
- data/train/y|n/...
- data/val/y|n/...
- data/test/y|n/...

Notes:
- This script preserves nested subfolders under each class.
- It can check cross-split duplicate content via SHA1 hashes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


# =========================
# Configuration
# =========================
MASTER_ROOT = Path('data_master')
OUTPUT_ROOT = Path('data')
CLASS_NAMES = ('y', 'n')

SPLITS = {
    'train': 0.70,
    'val': 0.15,
    'test': 0.15,
}
SEED = 77

# copy | hardlink | symlink
COPY_MODE = 'copy'

# Safety guard:
# - if True, existing output split dirs are deleted and recreated.
# - if False, script fails when output split dirs already exist.
OVERWRITE_OUTPUT_SPLITS = True

ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

CHECK_DUPLICATE_CONTENT_ACROSS_SPLITS = True
MAX_DUPLICATE_ROWS_IN_REPORT = 500

LOG_DIR = Path('split_build_logs')


def validate_config() -> None:
    total = sum(float(v) for v in SPLITS.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f'Split ratios must sum to 1.0, got {total}')

    if COPY_MODE not in {'copy', 'hardlink', 'symlink'}:
        raise ValueError(f'Invalid COPY_MODE: {COPY_MODE}')

    if MASTER_ROOT.resolve() == OUTPUT_ROOT.resolve():
        raise ValueError('MASTER_ROOT and OUTPUT_ROOT must be different directories.')


def gather_class_files(class_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in class_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTS:
            files.append(path)
    files.sort(key=lambda p: p.as_posix())
    return files


def split_files(files: list[Path], rng: random.Random) -> dict[str, list[Path]]:
    shuffled = list(files)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * SPLITS['train'])
    n_val = int(n * SPLITS['val'])
    n_test = n - n_train - n_val

    return {
        'train': shuffled[:n_train],
        'val': shuffled[n_train:n_train + n_val],
        'test': shuffled[n_train + n_val:n_train + n_val + n_test],
    }


def ensure_master_layout() -> None:
    if not MASTER_ROOT.exists():
        raise FileNotFoundError(
            f'Master dataset root not found: {MASTER_ROOT}. '\
            'Create data_master/<class>/... first.'
        )

    missing = [c for c in CLASS_NAMES if not (MASTER_ROOT / c).exists()]
    if missing:
        raise FileNotFoundError(
            f'Missing class folders in master dataset: {missing}. '\
            f'Expected {MASTER_ROOT}/<class>/...'
        )


def prepare_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_dir = OUTPUT_ROOT / split
        if split_dir.exists():
            if OVERWRITE_OUTPUT_SPLITS:
                shutil.rmtree(split_dir)
            else:
                raise FileExistsError(
                    f'Output split directory already exists: {split_dir}. '\
                    'Set OVERWRITE_OUTPUT_SPLITS=True to replace it.'
                )

        for cls in CLASS_NAMES:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)


def copy_item(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if COPY_MODE == 'copy':
        shutil.copy2(src, dst)
        return 'copied'

    if COPY_MODE == 'hardlink':
        try:
            os.link(src, dst)
            return 'hardlinked'
        except OSError:
            shutil.copy2(src, dst)
            return 'copied_fallback_from_hardlink'

    # symlink
    try:
        os.symlink(src.resolve(), dst)
        return 'symlinked'
    except OSError:
        shutil.copy2(src, dst)
        return 'copied_fallback_from_symlink'


def sha1_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open('rb') as fp:
        while True:
            chunk = fp.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def rebuild_splits() -> None:
    validate_config()
    ensure_master_layout()
    prepare_output_dirs()

    rng = random.Random(SEED)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    assignments: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {
        split: {cls: 0 for cls in CLASS_NAMES} for split in SPLITS
    }

    for cls in CLASS_NAMES:
        class_dir = MASTER_ROOT / cls
        files = gather_class_files(class_dir)
        if not files:
            raise ValueError(f'No images found in class directory: {class_dir}')

        class_split = split_files(files, rng=rng)

        for split, split_files_list in class_split.items():
            for src in split_files_list:
                # Preserve nested structure below class dir.
                rel_under_class = src.relative_to(class_dir)
                dst = OUTPUT_ROOT / split / cls / rel_under_class

                if dst.exists():
                    raise FileExistsError(
                        f'Destination already exists unexpectedly: {dst}. '\
                        'Clear output directory and retry.'
                    )

                transfer_mode = copy_item(src, dst)
                counts[split][cls] += 1
                assignments.append(
                    {
                        'split': split,
                        'class': cls,
                        'src': str(src),
                        'dst': str(dst),
                        'transfer_mode': transfer_mode,
                    }
                )

    manifest_csv = LOG_DIR / f'split_manifest_{timestamp}.csv'
    with manifest_csv.open('w', encoding='utf-8', newline='') as fp:
        fieldnames = ['split', 'class', 'src', 'dst', 'transfer_mode']
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(assignments)

    duplicate_rows: list[dict[str, Any]] = []
    duplicate_summary = {
        'checked': False,
        'num_hashes_with_cross_split_duplicates': 0,
        'num_rows_reported': 0,
    }

    if CHECK_DUPLICATE_CONTENT_ACROSS_SPLITS:
        hash_map: dict[str, dict[str, Any]] = {}
        for rec in assignments:
            dst = Path(rec['dst'])
            digest = sha1_file(dst)
            entry = hash_map.setdefault(digest, {'splits': set(), 'examples': []})
            entry['splits'].add(rec['split'])
            if len(entry['examples']) < 5:
                entry['examples'].append(rec)

        for digest, info in hash_map.items():
            splits = sorted(info['splits'])
            if len(splits) > 1:
                for ex in info['examples']:
                    duplicate_rows.append(
                        {
                            'sha1': digest,
                            'splits': '|'.join(splits),
                            'split': ex['split'],
                            'class': ex['class'],
                            'dst': ex['dst'],
                            'src': ex['src'],
                        }
                    )

        duplicate_rows = duplicate_rows[:MAX_DUPLICATE_ROWS_IN_REPORT]
        duplicate_summary = {
            'checked': True,
            'num_hashes_with_cross_split_duplicates': len(
                {
                    row['sha1']
                    for row in duplicate_rows
                }
            ),
            'num_rows_reported': len(duplicate_rows),
        }

    duplicates_csv = LOG_DIR / f'split_cross_split_duplicates_{timestamp}.csv'
    with duplicates_csv.open('w', encoding='utf-8', newline='') as fp:
        fieldnames = ['sha1', 'splits', 'split', 'class', 'dst', 'src']
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(duplicate_rows)

    summary = {
        'timestamp': timestamp,
        'master_root': str(MASTER_ROOT),
        'output_root': str(OUTPUT_ROOT),
        'class_names': list(CLASS_NAMES),
        'splits': SPLITS,
        'seed': SEED,
        'copy_mode': COPY_MODE,
        'overwrite_output_splits': OVERWRITE_OUTPUT_SPLITS,
        'counts': counts,
        'num_assigned_files': len(assignments),
        'manifest_csv': str(manifest_csv),
        'duplicates_csv': str(duplicates_csv),
        'duplicate_summary': duplicate_summary,
    }

    summary_json = LOG_DIR / f'split_build_summary_{timestamp}.json'
    summary_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    rebuild_splits()
