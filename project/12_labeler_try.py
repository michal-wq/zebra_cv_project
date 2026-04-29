from pathlib import Path
import re

def delete_augmented_files(
    root: str | Path,
    dry_run: bool = True,
    include_dedup_and_relabel: bool = True,
):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    patterns = [
        re.compile(r".*_spin(?:90|180|270)(?:_mirror)?$", re.IGNORECASE),
        re.compile(r".*_orig_mirror$", re.IGNORECASE),
    ]

    if include_dedup_and_relabel:
        patterns.extend([
            re.compile(r".*__dedup\d+$", re.IGNORECASE),
            re.compile(r".*__relabel_\d+$", re.IGNORECASE),
        ])

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

    to_delete = []
    for p in root.rglob("*"):  # recursive: train/y, train/n, val, test, ...
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        stem = p.stem
        if any(rx.match(stem) for rx in patterns):
            to_delete.append(p)

    for p in to_delete:
        if dry_run:
            print(f"[DRY RUN] {p}")
        else:
            p.unlink()

    return {
        "root": str(root),
        "matched": len(to_delete),
        "deleted": 0 if dry_run else len(to_delete),
        "dry_run": dry_run,
    }

# 1) preview first
#print(delete_augmented_files("data", dry_run=True))

# 2) really delete
print(delete_augmented_files("data", dry_run=False))
