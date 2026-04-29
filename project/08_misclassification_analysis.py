"""Analyzes test-set failure cases for the CNN->ViT hybrid model.

This script loads a trained checkpoint/state dict, runs inference on `data/test`,
and saves artifacts that help inspect poor classifications:
- all predictions CSV
- misclassified-only CSV (sorted by most confidently wrong)
- copied top-K misclassified images
- grid preview of top misclassified samples
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================
# Configuration
# =========================
SEED = 77
IMAGE_SIZE = 224
BATCH_SIZE = 256
NUM_WORKERS = 8
USE_AMP = True

ENFORCE_BINARY_CLASSIFICATION = True
EXPECTED_NUM_CLASSES = 2

TOP_K_MISCLASSIFIED_TO_COPY = 200
TOP_K_MISCLASSIFIED_IN_GRID = 36
TOP_K_LOW_CONFIDENCE_CORRECT = 200

PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_DIR / 'data'
TEST_DIR = DATA_ROOT / 'test'
MODEL_DIR = PROJECT_DIR / 'trained_models'

# Optional explicit source. Supported:
# - best checkpoint: trained_models/cnn_vit_seq_cnn9924_best.pt
# - artifact state_dict: trained_models/<artifact>/model_state_dict.pt
# - artifact directory: trained_models/<artifact>
CHECKPOINT_SOURCE: Path | None = None

DEFAULT_BEST_CHECKPOINT = MODEL_DIR / 'cnn_vit_seq_cnn9924_best.pt'
MODEL_ARTIFACT_PREFIX = 'CNN9924_ViT_Hybrid_Seq_score-'
ANALYSIS_PREFIX = 'misclassification_analysis'


@dataclass
class ResolvedModelSource:
    source_path: Path
    state_dict_path: Path
    metadata_path: Path | None


class PathImageFolder(datasets.ImageFolder):
    """ImageFolder variant that also returns the original sample path."""

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


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


def load_cnn_vit_module(script_path: Path):
    """Dynamically loads project/07_cnn_vit_seq.py as a module."""
    if not script_path.exists():
        raise FileNotFoundError(f'Expected model definition script not found: {script_path}')

    spec = importlib.util.spec_from_file_location('cnn_vit_seq_module', script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not create import spec for: {script_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_checkpoint_source(explicit_source: Path | None) -> ResolvedModelSource:
    """Resolves which trained model file should be analyzed."""
    candidate: Path | None = None

    if explicit_source is not None:
        candidate = explicit_source
    elif DEFAULT_BEST_CHECKPOINT.exists():
        candidate = DEFAULT_BEST_CHECKPOINT
    else:
        artifact_dirs = sorted(
            [p for p in MODEL_DIR.glob(f'{MODEL_ARTIFACT_PREFIX}*') if p.is_dir()],
            key=lambda p: p.name,
        )
        for artifact_dir in reversed(artifact_dirs):
            state_dict = artifact_dir / 'model_state_dict.pt'
            if state_dict.exists():
                candidate = artifact_dir
                break

    if candidate is None:
        raise FileNotFoundError(
            'Could not resolve a model source. Set CHECKPOINT_SOURCE or ensure '
            f'{DEFAULT_BEST_CHECKPOINT} or {MODEL_ARTIFACT_PREFIX}* artifacts exist.'
        )

    if candidate.is_dir():
        state_dict_path = candidate / 'model_state_dict.pt'
        if not state_dict_path.exists():
            raise FileNotFoundError(f'Artifact directory has no model_state_dict.pt: {candidate}')
        metadata_path = candidate / 'metadata.json'
        return ResolvedModelSource(
            source_path=candidate,
            state_dict_path=state_dict_path,
            metadata_path=metadata_path if metadata_path.exists() else None,
        )

    if candidate.is_file():
        metadata_path = candidate.parent / 'metadata.json'
        return ResolvedModelSource(
            source_path=candidate,
            state_dict_path=candidate,
            metadata_path=metadata_path if metadata_path.exists() else None,
        )

    raise FileNotFoundError(f'Invalid model source: {candidate}')


def load_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON in {path}: {exc}') from exc
    if not isinstance(data, dict):
        return {}
    return data


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith('module.') for k in state_dict):
        return {k.removeprefix('module.'): v for k, v in state_dict.items()}
    return state_dict


def extract_state_dict_and_payload(raw_checkpoint: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Supports both full-checkpoint and pure-state_dict formats."""
    if isinstance(raw_checkpoint, dict) and 'model_state_dict' in raw_checkpoint:
        state = raw_checkpoint['model_state_dict']
        if not isinstance(state, dict):
            raise ValueError('Checkpoint key "model_state_dict" is not a dict.')
        return state, raw_checkpoint

    if isinstance(raw_checkpoint, dict):
        if raw_checkpoint and all(isinstance(v, torch.Tensor) for v in raw_checkpoint.values()):
            return raw_checkpoint, {}
        raise ValueError('Unsupported dict checkpoint format.')

    raise ValueError(f'Unsupported checkpoint type: {type(raw_checkpoint)}')


def resolve_reference_path(value: Any, base_dir: Path) -> Path | None:
    """Resolves relative paths found in metadata params."""
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        return None

    p = Path(value)
    if p.is_absolute():
        return p

    candidate_1 = (base_dir / p).resolve()
    if candidate_1.exists():
        return candidate_1

    candidate_2 = (PROJECT_DIR / p).resolve()
    if candidate_2.exists():
        return candidate_2

    return candidate_1


def resolve_cnn_state_dict_from_metadata(metadata_path: Path | None) -> Path | None:
    """Returns sibling model_state_dict.pt for a metadata path when available."""
    if metadata_path is None:
        return None
    if not metadata_path.exists():
        return None

    candidate = metadata_path.parent / 'model_state_dict.pt'
    if candidate.exists():
        return candidate
    return None


def resolve_hybrid_config(
    cnn_module: Any,
    metadata: dict[str, Any],
    checkpoint_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds robust hybrid-model config from metadata with safe defaults."""
    params = metadata.get('params', {}) if isinstance(metadata, dict) else {}
    if not isinstance(params, dict):
        params = {}
    payload = checkpoint_payload if isinstance(checkpoint_payload, dict) else {}
    payload_cnn_info = payload.get('cnn_info', {}) if isinstance(payload.get('cnn_info'), dict) else {}

    source_dir_for_relative_paths = metadata.get('_source_dir', PROJECT_DIR)
    if not isinstance(source_dir_for_relative_paths, Path):
        source_dir_for_relative_paths = PROJECT_DIR

    pretrained_metadata = resolve_reference_path(
        params.get('pretrained_cnn_metadata_path'),
        base_dir=source_dir_for_relative_paths,
    )
    pretrained_source_override = resolve_reference_path(
        params.get('pretrained_cnn_source_override'),
        base_dir=source_dir_for_relative_paths,
    )
    pretrained_source_from_metadata = resolve_cnn_state_dict_from_metadata(pretrained_metadata)
    pretrained_source_from_training = resolve_reference_path(
        params.get('cnn_source'),
        base_dir=source_dir_for_relative_paths,
    )
    pretrained_source_from_payload_state = resolve_reference_path(
        payload_cnn_info.get('state_dict_path'),
        base_dir=PROJECT_DIR,
    )
    pretrained_source_from_payload_source = resolve_reference_path(
        payload_cnn_info.get('source'),
        base_dir=PROJECT_DIR,
    )

    resolved_cnn_hparams = params.get('cnn_hparams')
    if not isinstance(resolved_cnn_hparams, dict):
        payload_resolved = payload_cnn_info.get('resolved_hparams')
        resolved_cnn_hparams = payload_resolved if isinstance(payload_resolved, dict) else None

    # Prefer explicit override first. Otherwise anchor to the artifact metadata
    # (immutable path) to avoid loading a mutable alias like Simple_CNN.pt.
    if pretrained_source_override is not None:
        pretrained_source = pretrained_source_override
        source_resolution = 'pretrained_cnn_source_override'
    elif pretrained_source_from_metadata is not None:
        pretrained_source = pretrained_source_from_metadata
        source_resolution = 'pretrained_cnn_metadata_path:model_state_dict.pt'
    elif isinstance(resolved_cnn_hparams, dict):
        # Best-checkpoint payloads can carry resolved_hparams but no artifact metadata.
        # In that case, rebuild backbone from hparams and load hybrid weights directly.
        pretrained_source = None
        source_resolution = 'checkpoint_payload:resolved_hparams_only'
    elif pretrained_source_from_payload_state is not None:
        pretrained_source = pretrained_source_from_payload_state
        source_resolution = 'checkpoint_payload:cnn_info.state_dict_path'
    elif pretrained_source_from_payload_source is not None:
        pretrained_source = pretrained_source_from_payload_source
        source_resolution = 'checkpoint_payload:cnn_info.source'
    else:
        pretrained_source = pretrained_source_from_training
        source_resolution = 'cnn_source_fallback'

    return {
        'vit_embed_dim': int(params.get('vit_embed_dim', getattr(cnn_module, 'VIT_EMBED_DIM', 256))),
        'vit_num_heads': int(params.get('vit_num_heads', getattr(cnn_module, 'VIT_NUM_HEADS', 8))),
        'vit_depth': int(params.get('vit_depth', getattr(cnn_module, 'VIT_DEPTH', 4))),
        'vit_mlp_ratio': float(params.get('vit_mlp_ratio', getattr(cnn_module, 'VIT_MLP_RATIO', 4.0))),
        'vit_dropout': float(params.get('vit_dropout', getattr(cnn_module, 'VIT_DROPOUT', 0.1))),
        'freeze_backbone': bool(params.get('freeze_backbone', True)),
        'unfreeze_last_conv_blocks': int(params.get('unfreeze_last_conv_blocks', 0)),
        'pretrained_cnn_source': pretrained_source,
        'pretrained_cnn_metadata_path': pretrained_metadata,
        'resolved_cnn_hparams': resolved_cnn_hparams,
        'pretrained_cnn_source_resolution': source_resolution,
    }


def build_simplecnn_from_resolved_hparams(cnn_module: Any, resolved_hparams: dict[str, Any]) -> nn.Module:
    """Builds a SimpleCNN or LegacySimpleCNN directly from resolved hparams."""
    model_variant = str(resolved_hparams.get('model_variant', 'current')).lower()

    if model_variant == 'legacy':
        if not hasattr(cnn_module, 'LegacySimpleCNN'):
            raise AttributeError('07_cnn_vit_seq.py has no LegacySimpleCNN class for legacy checkpoint fallback.')
        return cnn_module.LegacySimpleCNN(
            n_layers=int(resolved_hparams.get('n_layers', 1)),
            layer_sizes=list(resolved_hparams.get('layer_sizes', [128])),
            activation=str(resolved_hparams.get('activation', 'relu')),
            dropout_rate=float(resolved_hparams.get('dropout_rate', 0.0)),
            num_classes=int(resolved_hparams.get('num_classes', 10)),
            conv_channels=tuple(resolved_hparams.get('conv_channels', (32, 64, 128))),
            kernel_size=int(resolved_hparams.get('kernel_size', 3)),
            pool_type=str(resolved_hparams.get('pool_type', 'max')),
            use_batchnorm=bool(resolved_hparams.get('use_batchnorm', False)),
            cnn_dropout_rate=float(resolved_hparams.get('cnn_dropout_rate', 0.0)),
        )

    return cnn_module.SimpleCNN(
        n_fc_layers=int(resolved_hparams.get('n_fc_layers', 3)),
        fc_hidden_size=int(resolved_hparams.get('fc_hidden_size', 128)),
        activation=str(resolved_hparams.get('activation', 'relu')),
        dropout_rate=float(resolved_hparams.get('dropout_rate', 0.0)),
        num_classes=int(resolved_hparams.get('num_classes', 10)),
        conv_channels=tuple(resolved_hparams.get('conv_channels', (32, 64, 128, 256, 256))),
        kernel_size=int(resolved_hparams.get('kernel_size', 3)),
        pool_type=str(resolved_hparams.get('pool_type', 'max')),
        use_batchnorm=bool(resolved_hparams.get('use_batchnorm', False)),
        cnn_dropout_rate=float(resolved_hparams.get('cnn_dropout_rate', 0.0)),
    )


def build_model_for_inference(
    cnn_module: Any,
    num_classes: int,
    hybrid_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Reconstructs the exact CNN->ViT hybrid architecture for loading weights."""
    pretrained_source = hybrid_cfg.get('pretrained_cnn_source')
    pretrained_metadata_path = hybrid_cfg.get('pretrained_cnn_metadata_path')
    resolved_cnn_hparams = hybrid_cfg.get('resolved_cnn_hparams')

    loaded_cnn = None
    if pretrained_source is not None:
        if pretrained_metadata_path is not None:
            # load_pretrained_simplecnn uses this fallback when metadata near checkpoint is absent.
            cnn_module.PRETRAINED_CNN_METADATA_PATH = pretrained_metadata_path
        loaded_cnn = cnn_module.load_pretrained_simplecnn(source=pretrained_source)
        backbone_model = loaded_cnn.model
        source_path_for_info = str(loaded_cnn.source_path)
        state_path_for_info = str(loaded_cnn.state_dict_path)
        resolved_cnn_hparams_for_info = loaded_cnn.resolved_hparams
    elif isinstance(resolved_cnn_hparams, dict):
        # Fallback when external source paths are unavailable (e.g. old best checkpoints
        # without metadata). The hybrid checkpoint still contains backbone weights.
        backbone_model = build_simplecnn_from_resolved_hparams(cnn_module, resolved_cnn_hparams)
        source_path_for_info = None
        state_path_for_info = None
        resolved_cnn_hparams_for_info = resolved_cnn_hparams
    else:
        # Last resort to keep previous behavior.
        loaded_cnn = cnn_module.load_pretrained_simplecnn(source=None)
        backbone_model = loaded_cnn.model
        source_path_for_info = str(loaded_cnn.source_path)
        state_path_for_info = str(loaded_cnn.state_dict_path)
        resolved_cnn_hparams_for_info = loaded_cnn.resolved_hparams

    feature_extractor = cnn_module.extract_feature_extractor(backbone_model)

    if hybrid_cfg['freeze_backbone']:
        cnn_module.freeze_module(feature_extractor)
    if hybrid_cfg['unfreeze_last_conv_blocks'] > 0:
        cnn_module.unfreeze_last_conv_blocks(feature_extractor, hybrid_cfg['unfreeze_last_conv_blocks'])

    feature_channels, feature_grid_size = cnn_module.infer_feature_map_shape(
        feature_extractor,
        IMAGE_SIZE,
    )

    model = cnn_module.CNNViTHybridSequential(
        feature_extractor=feature_extractor,
        feature_channels=feature_channels,
        feature_grid_size=feature_grid_size,
        num_classes=num_classes,
        embed_dim=hybrid_cfg['vit_embed_dim'],
        num_heads=hybrid_cfg['vit_num_heads'],
        depth=hybrid_cfg['vit_depth'],
        mlp_ratio=hybrid_cfg['vit_mlp_ratio'],
        dropout=hybrid_cfg['vit_dropout'],
    ).to(device)

    model_info = {
        'cnn_source': source_path_for_info,
        'state_dict_path': state_path_for_info,
        'feature_channels': int(feature_channels),
        'feature_grid_size': [int(feature_grid_size[0]), int(feature_grid_size[1])],
        'resolved_cnn_hparams': resolved_cnn_hparams_for_info,
        'resolved_hybrid_params': hybrid_cfg,
    }
    return model, model_info


def build_test_loader() -> tuple[DataLoader, PathImageFolder, dict[str, int]]:
    if not TEST_DIR.exists():
        raise FileNotFoundError(f'Expected test directory not found: {TEST_DIR}')

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_ds = PathImageFolder(TEST_DIR, transform=eval_tfms)
    class_to_idx = test_ds.class_to_idx

    num_classes = len(test_ds.classes)
    if ENFORCE_BINARY_CLASSIFICATION and num_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f'Expected binary classification with {EXPECTED_NUM_CLASSES} classes, '
            f'but found {num_classes} classes in {TEST_DIR}.'
        )

    loader_kwargs: dict[str, Any] = {
        'num_workers': NUM_WORKERS,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': NUM_WORKERS > 0,
    }
    if NUM_WORKERS > 0:
        loader_kwargs['prefetch_factor'] = 2

    loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )

    return loader, test_ds, class_to_idx


def collect_prediction_rows(
    model: nn.Module,
    loader: DataLoader,
    idx_to_class: dict[int, str],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []

    amp_enabled = USE_AMP and device.type == 'cuda'
    amp_device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    non_blocking = device.type == 'cuda'

    with torch.no_grad():
        for xb, yb, paths in loader:
            xb = xb.to(device, non_blocking=non_blocking)
            yb = yb.to(device, non_blocking=non_blocking)

            with torch.autocast(device_type=amp_device_type, enabled=amp_enabled):
                logits = model(xb)

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            top2_k = min(2, probs.shape[1])
            top_probs, top_indices = torch.topk(probs, k=top2_k, dim=1)

            for i in range(xb.size(0)):
                true_idx = int(yb[i].item())
                pred_idx = int(preds[i].item())
                pred_prob = float(probs[i, pred_idx].item())
                true_prob = float(probs[i, true_idx].item())

                second_idx = int(top_indices[i, 1].item()) if top2_k > 1 else pred_idx
                second_prob = float(top_probs[i, 1].item()) if top2_k > 1 else pred_prob

                row = {
                    'path': str(paths[i]),
                    'filename': Path(paths[i]).name,
                    'true_idx': true_idx,
                    'true_label': idx_to_class[true_idx],
                    'pred_idx': pred_idx,
                    'pred_label': idx_to_class[pred_idx],
                    'is_misclassified': int(pred_idx != true_idx),
                    'true_prob': true_prob,
                    'pred_prob': pred_prob,
                    'confidence_gap_pred_minus_true': pred_prob - true_prob,
                    'second_best_idx': second_idx,
                    'second_best_label': idx_to_class[second_idx],
                    'second_best_prob': second_prob,
                    'max_prob': float(top_probs[i, 0].item()),
                }
                rows.append(row)

    return rows


def safe_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}:
        return suffix
    return '.jpg'


def copy_top_misclassified_images(rows: list[dict[str, Any]], out_dir: Path, top_k: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for rank, row in enumerate(rows[:top_k], start=1):
        src = Path(row['path'])
        if not src.exists():
            continue

        dst_name = (
            f"{rank:04d}_true-{row['true_label']}_pred-{row['pred_label']}"
            f"_predp-{row['pred_prob']:.4f}_truep-{row['true_prob']:.4f}"
            f"{safe_suffix(src)}"
        )
        dst = out_dir / dst_name
        shutil.copy2(src, dst)
        copied += 1

    return copied


def save_rows_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Keep an empty file with header-less content for explicitness.
        csv_path.write_text('', encoding='utf-8')
        return

    fieldnames = list(rows[0].keys())
    with csv_path.open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_misclassification_grid(rows: list[dict[str, Any]], output_path: Path, top_k: int) -> Path | None:
    selected = rows[:top_k]
    if not selected:
        return None

    n = len(selected)
    ncols = 6
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis('off')

    for i, row in enumerate(selected):
        ax = axes[i]
        src = Path(row['path'])
        if not src.exists():
            ax.set_title('missing file', fontsize=8)
            continue

        try:
            image = Image.open(src).convert('RGB')
        except OSError:
            ax.set_title('unreadable image', fontsize=8)
            continue

        ax.imshow(image)
        ax.set_title(
            f"T:{row['true_label']}  P:{row['pred_label']}\n"
            f"p_pred={row['pred_prob']:.3f}  p_true={row['true_prob']:.3f}",
            fontsize=8,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return output_path


def to_json_safe(value: Any) -> Any:
    """Recursively converts non-JSON-native objects (e.g. Path) to safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    set_seed(SEED)
    device = get_device()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    cnn_vit_module = load_cnn_vit_module(PROJECT_DIR / '07_cnn_vit_seq.py')

    resolved_source = resolve_checkpoint_source(CHECKPOINT_SOURCE)
    metadata = load_json_if_exists(resolved_source.metadata_path)
    metadata['_source_dir'] = (
        resolved_source.metadata_path.parent if resolved_source.metadata_path is not None else PROJECT_DIR
    )

    raw_checkpoint = torch.load(resolved_source.state_dict_path, map_location='cpu')
    state_dict, payload = extract_state_dict_and_payload(raw_checkpoint)
    state_dict = strip_module_prefix(state_dict)

    test_loader, test_ds, class_to_idx = build_test_loader()
    if isinstance(payload.get('class_to_idx'), dict) and payload['class_to_idx'] != class_to_idx:
        raise ValueError(
            'Checkpoint class_to_idx differs from test dataset class_to_idx. '
            'Use matching splits/checkpoint pair.'
        )

    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    num_classes = len(class_to_idx)

    hybrid_cfg = resolve_hybrid_config(cnn_vit_module, metadata, checkpoint_payload=payload)
    model, model_info = build_model_for_inference(
        cnn_module=cnn_vit_module,
        num_classes=num_classes,
        hybrid_cfg=hybrid_cfg,
        device=device,
    )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            'Failed to load hybrid model weights. This usually means architecture mismatch '
            '(e.g., different ViT depth/embed/head settings). '
            f'Hybrid config used: {hybrid_cfg}'
        ) from exc

    rows = collect_prediction_rows(
        model=model,
        loader=test_loader,
        idx_to_class=idx_to_class,
        device=device,
    )
    if not rows:
        raise ValueError('No samples found in test loader.')

    misclassified = [r for r in rows if r['is_misclassified'] == 1]
    misclassified_sorted = sorted(
        misclassified,
        key=lambda r: (r['pred_prob'], r['confidence_gap_pred_minus_true']),
        reverse=True,
    )

    low_confidence_correct = sorted(
        [r for r in rows if r['is_misclassified'] == 0],
        key=lambda r: r['true_prob'],
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_dir = MODEL_DIR / f'{ANALYSIS_PREFIX}_{timestamp}'
    analysis_dir.mkdir(parents=True, exist_ok=False)

    all_predictions_csv = analysis_dir / 'all_test_predictions.csv'
    misclassified_csv = analysis_dir / 'misclassified_sorted.csv'
    low_confidence_correct_csv = analysis_dir / 'low_confidence_correct.csv'

    save_rows_csv(rows, all_predictions_csv)
    save_rows_csv(misclassified_sorted, misclassified_csv)
    save_rows_csv(low_confidence_correct[:TOP_K_LOW_CONFIDENCE_CORRECT], low_confidence_correct_csv)

    copied_dir = analysis_dir / 'top_misclassified_images'
    copied_count = copy_top_misclassified_images(
        rows=misclassified_sorted,
        out_dir=copied_dir,
        top_k=TOP_K_MISCLASSIFIED_TO_COPY,
    )

    grid_path = save_misclassification_grid(
        rows=misclassified_sorted,
        output_path=analysis_dir / 'misclassified_grid.png',
        top_k=TOP_K_MISCLASSIFIED_IN_GRID,
    )

    summary = {
        'device': str(device),
        'test_dir': str(TEST_DIR),
        'model_source': str(resolved_source.source_path),
        'state_dict_path': str(resolved_source.state_dict_path),
        'metadata_path': str(resolved_source.metadata_path) if resolved_source.metadata_path else None,
        'analysis_dir': str(analysis_dir),
        'num_test_samples': int(len(rows)),
        'num_misclassified': int(len(misclassified_sorted)),
        'misclassification_rate': float(len(misclassified_sorted) / len(rows)),
        'all_predictions_csv': str(all_predictions_csv),
        'misclassified_csv': str(misclassified_csv),
        'low_confidence_correct_csv': str(low_confidence_correct_csv),
        'copied_misclassified_images': int(copied_count),
        'copied_images_dir': str(copied_dir),
        'misclassified_grid_path': str(grid_path) if grid_path is not None else None,
        'model_info': model_info,
        'hybrid_cfg': hybrid_cfg,
        'checkpoint_best_val_accuracy': payload.get('best_val_acc'),
        'checkpoint_best_epoch': (int(payload['epoch']) + 1) if isinstance(payload.get('epoch'), int) else None,
    }

    summary_safe = to_json_safe(summary)
    (analysis_dir / 'summary.json').write_text(
        json.dumps(summary_safe, indent=2),
        encoding='utf-8',
    )

    print(json.dumps(summary_safe, indent=2))


if __name__ == '__main__':
    main()
