"""
- Die Daten werden gezählt bzw. es wird auch gezählt wie viele Representaten die jeweiligen Klassen haben
- Die Daten werden so gesplittet, dass die Gruppen in den jeweiligen Datensätzen glech represäntiert werden.
- Die Bilder werden in die richtige Ordner verschoben / kopiert
- Es wird nochmal gezählt und ein log wird geprintet
"""

from pathlib import Path
from typing import Union, Iterable, Dict, Any
import random
import shutil


def count_data_in_folder(root: Union[str, Path], exts: Iterable[str] = None) -> Dict[str, Any]:
    """
    Zähle Bilddateien in allen Unterordnern von `root` (rekursiv).

    Args:
      root: Pfad zum Wurzelordner.
      exts: Iterable von Dateiendungen (z. B. ['.jpg', '.png']). Standardmäßig typische Bildformate.

    Returns:
      Dict mit Schlüsseln:
        - 'counts': dict mapping Unterordner (relativ zu root) -> Anzahl Bilder
        - 'total': gesamtsumme über alle Unterordner
    """
    root = Path(root)
    if exts is None:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}
    else:
        exts = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in exts}

    counts: Dict[str, int] = {}
    total = 0
    if not root.exists():
        return {'counts': counts, 'total': total}

    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in exts:
            rel_parent = file_path.parent.relative_to(root).as_posix()
            counts[rel_parent] = counts.get(rel_parent, 0) + 1
            total += 1

    return {'counts': counts, 'total': total}

from pathlib import Path
import random
import shutil

def split_dataset_by_class(
    source_root,
    output_root="data",
    class_names=("y", "n"),
    splits=(0.7, 0.15, 0.15),
    seed=42,
    copy_files=True,
    allowed_exts={".jpg", ".jpeg", ".png", ".bmp", ".webp"}
):
    source_root = Path(source_root)
    output_root = Path(output_root)

    if abs(sum(splits) - 1.0) > 1e-9:
        raise ValueError("splits must sum to 1.0")

    train_ratio, val_ratio, test_ratio = splits
    rng = random.Random(seed)

    # Only keep requested class folders (e.g., y and n), ignore everything else (e.g., crops)
    class_dirs = []
    for cname in class_names:
        d = source_root / cname
        if d.is_dir():
            class_dirs.append(d)
        else:
            raise ValueError(f"Expected class folder not found: {d}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in allowed_exts
        ]

        rng.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        split_map = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:n_train + n_val + n_test],
        }

        for split_name, split_files in split_map.items():
            target_dir = output_root / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for src in split_files:
                dst = target_dir / src.name
                if copy_files:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)

    return output_root

def main():
    n = 150

    # Zähle die Daten
    print('='*n)
    path = '../raw_data/data'
    counts = count_data_in_folder(path)
    print(counts)
    print('='*n)

    # Ratios
    train_ratio = 0.7
    val_ratio = 0.15

    train_size = int(train_ratio*counts['total'])
    val_size = int(val_ratio*counts['total'])
    test_size = val_size
    print('Current sizes:\n')
    print(f'Train size: {train_size}')
    print(f'Val size: {val_size}')
    print(f'Test size: {test_size}')
    print('='*n)
    #print(f'Trainset grösse: {}')
    cities = ['luzern', 'st gallen']
    for city in cities:
        split_dataset_by_class(
        source_root=f"../raw_data/data/{city}",  # has: crops/, y/, n/
        output_root="data",
        class_names=("y", "n")             # crops is ignored
        )

main()