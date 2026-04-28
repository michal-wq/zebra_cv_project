"""Creates model-assisted review candidates across train/val/test.

Workflow:
1) Loads the CNN->ViT hybrid checkpoint.
2) Runs inference on each available split in data/train|val|test.
3) Exports candidate lists to review likely label issues quickly.

Outputs:
- review_candidates/<timestamp>/all_predictions_<split>.csv
- review_candidates/<timestamp>/candidates_<split>.csv
- review_candidates/<timestamp>/candidates_all_splits.csv
- review_candidates/<timestamp>/review_manifest_template.csv
- review_candidates/<timestamp>/candidate_images/*
- review_candidates/<timestamp>/summary.json
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


# =========================
# Configuration
# =========================
SEED = 77
IMAGE_SIZE = 224
BATCH_SIZE = 256
NUM_WORKERS = 8
USE_AMP = True

DATA_ROOT = Path('data')
SPLITS_TO_SCAN = ('train', 'val', 'test')
CLASS_NAMES = ('y', 'n')

# Optional explicit source for model weights.
# Supported:
# - trained_models/cnn_vit_seq_cnn9924_best.pt
# - trained_models/<artifact>/model_state_dict.pt
# - trained_models/<artifact>
CHECKPOINT_SOURCE: Path | None = None

# Candidate selection budget per split.
MAX_MISCLASSIFIED_PER_SPLIT = 300
MAX_LOW_CONFIDENCE_CORRECT_PER_SPLIT = 300
MAX_COPIED_IMAGES_PER_SPLIT = 250

REVIEW_ROOT = Path('review_candidates')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_module_from_path(script_path: Path, module_name: str):
    if not script_path.exists():
        raise FileNotFoundError(f'Module file not found: {script_path}')

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not import module spec from {script_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_split_loader(analysis08: Any, split_name: str) -> tuple[DataLoader, Any]:
    split_dir = DATA_ROOT / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f'Split directory not found: {split_dir}')

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = analysis08.PathImageFolder(split_dir, transform=eval_tfms)
    loader_kwargs: dict[str, Any] = {
        'num_workers': NUM_WORKERS,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': NUM_WORKERS > 0,
    }
    if NUM_WORKERS > 0:
        loader_kwargs['prefetch_factor'] = 2

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )
    return loader, dataset


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text('', encoding='utf-8')
        return

    fieldnames = list(rows[0].keys())
    with output_path.open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relativize_to_data_root(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        try:
            return p.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return p.as_posix()
    return p.as_posix()


def build_candidate_rows(pred_rows: list[dict[str, Any]], split_name: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    misclassified: list[dict[str, Any]] = []
    low_conf_correct: list[dict[str, Any]] = []

    for row in pred_rows:
        row = dict(row)
        row['split'] = split_name
        row['path'] = relativize_to_data_root(str(row['path']))

        if int(row['is_misclassified']) == 1:
            # Higher score = model is more confidently contradicting current label.
            score = float(row['pred_prob']) + float(row['confidence_gap_pred_minus_true'])
            row['candidate_reason'] = 'misclassified_high_confidence'
            row['suspicion_score'] = score
            misclassified.append(row)
        else:
            # Lower true_prob = less confidence in current label despite correct argmax.
            score = 1.0 - float(row['true_prob'])
            row['candidate_reason'] = 'low_confidence_correct'
            row['suspicion_score'] = score
            low_conf_correct.append(row)

    misclassified_sorted = sorted(misclassified, key=lambda r: float(r['suspicion_score']), reverse=True)
    low_conf_sorted = sorted(low_conf_correct, key=lambda r: float(r['true_prob']))

    selected = misclassified_sorted[:MAX_MISCLASSIFIED_PER_SPLIT]
    selected.extend(low_conf_sorted[:MAX_LOW_CONFIDENCE_CORRECT_PER_SPLIT])

    stats = {
        'num_all': len(pred_rows),
        'num_misclassified_all': len(misclassified),
        'num_low_conf_correct_all': len(low_conf_correct),
        'num_candidates_selected': len(selected),
    }
    return selected, stats


def copy_candidate_images(candidate_rows: list[dict[str, Any]], output_dir: Path, limit: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for rank, row in enumerate(candidate_rows[:limit], start=1):
        src = Path(row['path'])
        if not src.is_absolute():
            src = Path.cwd() / src
        if not src.exists():
            continue

        dst_name = (
            f"{rank:04d}_split-{row['split']}_true-{row['true_label']}_pred-{row['pred_label']}"
            f"_score-{float(row['suspicion_score']):.4f}{src.suffix.lower() or '.jpg'}"
        )
        shutil.copy2(src, output_dir / dst_name)
        copied += 1

    return copied


def build_review_manifest_template(candidate_rows_all: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for row in candidate_rows_all:
        manifest_rows.append(
            {
                'image_ref': row['path'],
                'split': row['split'],
                'current_label': row['true_label'],
                'new_label': '',
                'review_decision': '',
                'candidate_reason': row['candidate_reason'],
                'suspicion_score': row['suspicion_score'],
                'model_pred_label': row['pred_label'],
                'model_pred_prob': row['pred_prob'],
                'current_label_prob': row['true_prob'],
                'reviewer': '',
                'notes': '',
            }
        )
    return manifest_rows


def main() -> None:
    set_seed(SEED)
    device = get_device()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    analysis08 = load_module_from_path(Path('08_misclassification_analysis.py'), 'misclassification_analysis_08')
    cnn_vit07 = analysis08.load_cnn_vit_module(Path('07_cnn_vit_seq.py'))

    resolved_source = analysis08.resolve_checkpoint_source(CHECKPOINT_SOURCE)
    metadata = analysis08.load_json_if_exists(resolved_source.metadata_path)
    metadata['_source_dir'] = (
        resolved_source.metadata_path.parent if resolved_source.metadata_path is not None else Path.cwd()
    )

    raw_checkpoint = torch.load(resolved_source.state_dict_path, map_location='cpu')
    state_dict, payload = analysis08.extract_state_dict_and_payload(raw_checkpoint)
    state_dict = analysis08.strip_module_prefix(state_dict)

    per_split_rows: dict[str, list[dict[str, Any]]] = {}
    class_to_idx_ref: dict[str, int] | None = None
    idx_to_class_ref: dict[int, str] | None = None

    available_splits: list[str] = []
    for split_name in SPLITS_TO_SCAN:
        split_dir = DATA_ROOT / split_name
        if split_dir.exists():
            available_splits.append(split_name)

    if not available_splits:
        raise FileNotFoundError(f'No split directories found under {DATA_ROOT}')

    for split_name in available_splits:
        loader, dataset = build_split_loader(analysis08, split_name)
        class_to_idx = dataset.class_to_idx

        if class_to_idx_ref is None:
            class_to_idx_ref = class_to_idx
            idx_to_class_ref = {idx: cls_name for cls_name, idx in class_to_idx.items()}
        elif class_to_idx != class_to_idx_ref:
            raise ValueError(
                f'class_to_idx mismatch in split "{split_name}". '
                f'Expected {class_to_idx_ref}, got {class_to_idx}.'
            )

        if set(class_to_idx.keys()) != set(CLASS_NAMES):
            raise ValueError(
                f'Split "{split_name}" classes {set(class_to_idx.keys())} differ from expected {set(CLASS_NAMES)}.'
            )

        per_split_rows[split_name] = []
        # Temporarily keep datasets for stats only.
        _ = loader

    assert class_to_idx_ref is not None
    assert idx_to_class_ref is not None

    if isinstance(payload.get('class_to_idx'), dict) and payload['class_to_idx'] != class_to_idx_ref:
        raise ValueError(
            'Checkpoint class_to_idx differs from dataset class_to_idx. '
            'Use matching checkpoint and dataset.'
        )

    hybrid_cfg = analysis08.resolve_hybrid_config(cnn_vit07, metadata)
    model, model_info = analysis08.build_model_for_inference(
        cnn_module=cnn_vit07,
        num_classes=len(class_to_idx_ref),
        hybrid_cfg=hybrid_cfg,
        device=device,
    )

    model.load_state_dict(state_dict, strict=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = REVIEW_ROOT / f'review_candidates_{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=False)

    split_stats: dict[str, dict[str, int]] = {}
    candidates_all: list[dict[str, Any]] = []

    for split_name in available_splits:
        loader, _dataset = build_split_loader(analysis08, split_name)
        rows = analysis08.collect_prediction_rows(
            model=model,
            loader=loader,
            idx_to_class=idx_to_class_ref,
            device=device,
        )

        rows_with_split = []
        for row in rows:
            updated = dict(row)
            updated['split'] = split_name
            updated['path'] = relativize_to_data_root(str(updated['path']))
            rows_with_split.append(updated)

        selected, stats = build_candidate_rows(rows_with_split, split_name)
        split_stats[split_name] = stats
        candidates_all.extend(selected)

        write_csv(rows_with_split, out_dir / f'all_predictions_{split_name}.csv')
        write_csv(selected, out_dir / f'candidates_{split_name}.csv')

        copied = copy_candidate_images(
            selected,
            out_dir / 'candidate_images' / split_name,
            MAX_COPIED_IMAGES_PER_SPLIT,
        )
        split_stats[split_name]['num_images_copied'] = copied

    # Cross-split ranking to prioritize the very first review pass.
    candidates_all_sorted = sorted(candidates_all, key=lambda r: float(r['suspicion_score']), reverse=True)
    write_csv(candidates_all_sorted, out_dir / 'candidates_all_splits.csv')

    manifest_rows = build_review_manifest_template(candidates_all_sorted)
    write_csv(manifest_rows, out_dir / 'review_manifest_template.csv')

    summary = {
        'timestamp': timestamp,
        'device': str(device),
        'checkpoint_source': str(resolved_source.source_path),
        'state_dict_path': str(resolved_source.state_dict_path),
        'metadata_path': str(resolved_source.metadata_path) if resolved_source.metadata_path else None,
        'available_splits': available_splits,
        'split_stats': split_stats,
        'num_candidates_total': len(candidates_all_sorted),
        'output_dir': str(out_dir),
        'model_info': analysis08.to_json_safe(model_info),
        'hybrid_cfg': analysis08.to_json_safe(hybrid_cfg),
    }

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
