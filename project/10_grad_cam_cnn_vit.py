"""Generates Grad-CAM explanations for CNN->ViT hybrid checkpoints.

The script reuses the robust checkpoint/model reconstruction helpers from
08_misclassification_analysis.py and applies Grad-CAM to the last convolutional
layer of the CNN feature extractor.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# =========================
# Configuration
# =========================
SEED = 77
IMAGE_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 8
USE_AMP_FOR_PREDICTIONS = True

PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_DIR / 'data'
TEST_DIR = DATA_ROOT / 'test'
MODEL_DIR = PROJECT_DIR / 'trained_models'
ANALYSIS_ROOT = PROJECT_DIR / 'model_analysis'

# Optional explicit model source. Supported:
# - trained_models/<checkpoint>.pt with {'model_state_dict': ...}
# - trained_models/<artifact>/model_state_dict.pt
# - trained_models/<artifact> directory
CHECKPOINT_SOURCE: Path | None = None

DEFAULT_BEST_CHECKPOINTS = [
    MODEL_DIR / 'CNN_ViT_Seq_Optuna_best.pt',
    MODEL_DIR / 'Big_Trans_3_Best.pt',
    MODEL_DIR / 'CNNVIT512V1_best.pt',
]
MODEL_ARTIFACT_PREFIXES = [
    'CNN_ViT_Seq_Optuna_score-',
    'Big_Trans_3_score-',
    'CNNVIT512V1_score-',
    'CNNVIT',
    'CNN_ViT',
]

SAMPLE_MODE = 'true_positive'
MAX_IMAGES = 64
TARGET_MODE = 'true'  # pred | true | both
POSITIVE_CLASS_LABEL = 'y'
FALLBACK_TO_LOW_CONFIDENCE_CORRECT = True

HEATMAP_CMAP = 'jet'
OVERLAY_ALPHA = 0.45
PLOT_DPI = 170
OUTPUT_PREFIX = 'cnn_vit_grad_cam'


@dataclass
class ResolvedModelSource:
    source_path: Path
    state_dict_path: Path
    metadata_path: Path | None


class GradCAM:
    """Minimal Grad-CAM implementation for a selected convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._store_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._store_gradients)

    def _store_activations(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output

    def _store_gradients(
        self,
        _module: nn.Module,
        _grad_inputs: tuple[torch.Tensor, ...],
        grad_outputs: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_outputs[0]

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Returns a normalized CAM and the model logits."""
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(input_tensor)
        score = logits[:, int(target_class)].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                'Grad-CAM hooks did not capture activations/gradients. '
                'Check that the selected layer participates in the forward pass.'
            )

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

        cam = cam[0, 0]
        cam_min = cam.min()
        cam_max = cam.max()
        denom = cam_max - cam_min
        if float(denom.abs().item()) < 1e-12:
            cam = torch.zeros_like(cam)
        else:
            cam = (cam - cam_min) / denom

        return cam.cpu().numpy(), logits.detach()


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


def load_module(module_path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load module spec from {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_analysis_module(analysis: ModuleType) -> None:
    """Keeps imported helper globals aligned with this script."""
    analysis.SEED = SEED
    analysis.IMAGE_SIZE = IMAGE_SIZE
    analysis.BATCH_SIZE = BATCH_SIZE
    analysis.NUM_WORKERS = NUM_WORKERS
    analysis.USE_AMP = USE_AMP_FOR_PREDICTIONS
    analysis.PROJECT_DIR = PROJECT_DIR
    analysis.DATA_ROOT = DATA_ROOT
    analysis.TEST_DIR = TEST_DIR
    analysis.MODEL_DIR = MODEL_DIR


def resolve_checkpoint_source(
    analysis: ModuleType,
    explicit_source: Path | None,
) -> ResolvedModelSource:
    """Finds the checkpoint/artifact that should be explained."""
    candidate: Path | None = explicit_source

    if candidate is None:
        for checkpoint in DEFAULT_BEST_CHECKPOINTS:
            if checkpoint.exists():
                candidate = checkpoint
                break

    if candidate is None:
        artifact_dirs: list[Path] = []
        for prefix in MODEL_ARTIFACT_PREFIXES:
            artifact_dirs.extend([p for p in MODEL_DIR.glob(f'{prefix}*') if p.is_dir()])
        artifact_dirs = sorted(set(artifact_dirs), key=lambda p: p.name)
        for artifact_dir in reversed(artifact_dirs):
            if (artifact_dir / 'model_state_dict.pt').exists():
                candidate = artifact_dir
                break

    if candidate is None:
        raise FileNotFoundError(
            'Could not resolve a CNN->ViT model source. Set CHECKPOINT_SOURCE or ensure '
            f'one of {DEFAULT_BEST_CHECKPOINTS} or a matching artifact directory exists.'
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

    # Keep this compatible with the imported dataclass shape for future reuse.
    return analysis.ResolvedModelSource(candidate, candidate, None)


def normalize_metadata_for_hybrid_config(
    metadata: dict[str, Any],
    checkpoint_payload: dict[str, Any],
) -> dict[str, Any]:
    """Merges 09-style nested best_params and checkpoint params into metadata."""
    normalized = dict(metadata)
    params = normalized.get('params', {})
    if not isinstance(params, dict):
        params = {}
    params = dict(params)

    nested_best_params = params.get('best_params')
    if isinstance(nested_best_params, dict):
        merged = dict(nested_best_params)
        merged.update(params)
        params = merged

    payload_params = checkpoint_payload.get('params')
    if isinstance(payload_params, dict):
        params.update(payload_params)

    if 'freeze_cnn_backbone' in params and 'freeze_backbone' not in params:
        params['freeze_backbone'] = params['freeze_cnn_backbone']

    normalized['params'] = params
    return normalized


def configure_cnn_module(cnn_module: ModuleType) -> None:
    """Makes 07_cnn_vit_seq.py path handling robust from any working dir."""
    cnn_module.SEED = SEED
    cnn_module.IMAGE_SIZE = IMAGE_SIZE
    cnn_module.BATCH_SIZE = BATCH_SIZE
    cnn_module.NUM_WORKERS = NUM_WORKERS
    cnn_module.MODEL_DIR = MODEL_DIR
    cnn_module.DATA_ROOT = DATA_ROOT
    cnn_module.TRAIN_DIR = DATA_ROOT / 'train'
    cnn_module.VAL_DIR = DATA_ROOT / 'val'
    cnn_module.TEST_DIR = TEST_DIR


def find_last_conv_layer(model: nn.Module) -> tuple[str, nn.Conv2d]:
    """Returns the last Conv2d layer, preferring the CNN feature extractor."""
    conv_layers: list[tuple[str, nn.Conv2d]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    if not conv_layers:
        raise ValueError('No Conv2d layer found. Grad-CAM needs a convolutional target layer.')

    feature_extractor_layers = [
        item for item in conv_layers if 'feature_extractor' in item[0]
    ]
    return feature_extractor_layers[-1] if feature_extractor_layers else conv_layers[-1]


def build_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_image_for_gradcam(path: Path, device: torch.device) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(path).convert('RGB')
    tensor = build_image_transform()(image).unsqueeze(0).to(device)
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return resized, tensor


def select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Selects samples for Grad-CAM generation."""
    misclassified = [r for r in rows if int(r['is_misclassified']) == 1]
    correct = [r for r in rows if int(r['is_misclassified']) == 0]
    positives = [
        r for r in rows
        if str(r['true_label']) == POSITIVE_CLASS_LABEL
        or str(r['pred_label']) == POSITIVE_CLASS_LABEL
    ]

    if SAMPLE_MODE == 'true_positive':
        selected = [
            r for r in positives
            if str(r['true_label']) == POSITIVE_CLASS_LABEL
            and str(r['pred_label']) == POSITIVE_CLASS_LABEL
        ]
        return sorted(selected, key=lambda r: float(r['pred_prob']), reverse=True)[:MAX_IMAGES]

    if SAMPLE_MODE == 'false_positive':
        selected = [
            r for r in positives
            if str(r['true_label']) != POSITIVE_CLASS_LABEL
            and str(r['pred_label']) == POSITIVE_CLASS_LABEL
        ]
        return sorted(selected, key=lambda r: float(r['pred_prob']), reverse=True)[:MAX_IMAGES]

    if SAMPLE_MODE == 'false_negative':
        selected = [
            r for r in positives
            if str(r['true_label']) == POSITIVE_CLASS_LABEL
            and str(r['pred_label']) != POSITIVE_CLASS_LABEL
        ]
        return sorted(selected, key=lambda r: float(r['true_prob']))[:MAX_IMAGES]

    if SAMPLE_MODE == 'true_negative':
        selected = [
            r for r in rows
            if str(r['true_label']) != POSITIVE_CLASS_LABEL
            and str(r['pred_label']) != POSITIVE_CLASS_LABEL
        ]
        return sorted(selected, key=lambda r: float(r['pred_prob']), reverse=True)[:MAX_IMAGES]

    if SAMPLE_MODE == 'misclassified':
        selected = sorted(
            misclassified,
            key=lambda r: (float(r['pred_prob']), float(r['confidence_gap_pred_minus_true'])),
            reverse=True,
        )
        if not selected and FALLBACK_TO_LOW_CONFIDENCE_CORRECT:
            selected = sorted(correct, key=lambda r: float(r['true_prob']))
        return selected[:MAX_IMAGES]

    if SAMPLE_MODE == 'low_confidence_correct':
        return sorted(correct, key=lambda r: float(r['true_prob']))[:MAX_IMAGES]

    if SAMPLE_MODE == 'high_confidence':
        return sorted(rows, key=lambda r: float(r['max_prob']), reverse=True)[:MAX_IMAGES]

    if SAMPLE_MODE == 'all':
        return rows[:MAX_IMAGES]

    raise ValueError(
        "SAMPLE_MODE must be one of 'true_positive', 'false_positive', "
        "'false_negative', 'true_negative', 'misclassified', "
        "'low_confidence_correct', 'high_confidence', or 'all'."
    )


def target_classes_for_row(row: dict[str, Any]) -> list[tuple[str, int]]:
    pred = int(row['pred_idx'])
    true = int(row['true_idx'])

    if TARGET_MODE == 'pred':
        return [('pred', pred)]
    if TARGET_MODE == 'true':
        return [('true', true)]
    if TARGET_MODE == 'both':
        targets = [('pred', pred)]
        if true != pred:
            targets.append(('true', true))
        return targets
    raise ValueError("TARGET_MODE must be one of 'pred', 'true', or 'both'.")


def slugify(value: str) -> str:
    value = value.strip().replace(' ', '_')
    value = re.sub(r'[^A-Za-z0-9_.-]+', '-', value)
    return value.strip('-') or 'sample'


def render_gradcam_figure(
    image: Image.Image,
    cam: np.ndarray,
    row: dict[str, Any],
    target_name: str,
    target_label: str,
    output_path: Path,
) -> Path:
    image_arr = np.asarray(image).astype(np.float32) / 255.0
    heatmap_rgb = plt.get_cmap(HEATMAP_CMAP)(cam)[..., :3]
    overlay = np.clip((1.0 - OVERLAY_ALPHA) * image_arr + OVERLAY_ALPHA * heatmap_rgb, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    for ax in axes:
        ax.axis('off')

    axes[0].imshow(image_arr)
    axes[0].set_title('Original', fontsize=10)
    axes[1].imshow(cam, cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0)
    axes[1].set_title('Grad-CAM', fontsize=10)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=10)

    fig.suptitle(
        f"T:{row['true_label']}  P:{row['pred_label']}  "
        f"target:{target_name}/{target_label}  p_pred={float(row['pred_prob']):.3f}",
        fontsize=11,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    return output_path


def save_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text('', encoding='utf-8')
        return

    fieldnames = list(rows[0].keys())
    with output_path.open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_json_safe(value: Any) -> Any:
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

    analysis = load_module(PROJECT_DIR / '08_misclassification_analysis.py', 'misclassification_helpers')
    configure_analysis_module(analysis)

    cnn_module = analysis.load_cnn_vit_module(PROJECT_DIR / '07_cnn_vit_seq.py')
    configure_cnn_module(cnn_module)

    resolved_source = resolve_checkpoint_source(analysis, CHECKPOINT_SOURCE)
    metadata = analysis.load_json_if_exists(resolved_source.metadata_path)
    metadata['_source_dir'] = (
        resolved_source.metadata_path.parent if resolved_source.metadata_path is not None else PROJECT_DIR
    )

    raw_checkpoint = torch.load(resolved_source.state_dict_path, map_location='cpu')
    state_dict, payload = analysis.extract_state_dict_and_payload(raw_checkpoint)
    state_dict = analysis.strip_module_prefix(state_dict)
    metadata = normalize_metadata_for_hybrid_config(metadata, payload)

    test_loader, _test_ds, class_to_idx = analysis.build_test_loader()
    if isinstance(payload.get('class_to_idx'), dict) and payload['class_to_idx'] != class_to_idx:
        raise ValueError(
            'Checkpoint class_to_idx differs from test dataset class_to_idx. '
            'Use matching splits/checkpoint pair.'
        )

    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    num_classes = len(class_to_idx)

    hybrid_cfg = analysis.resolve_hybrid_config(cnn_module, metadata, checkpoint_payload=payload)
    model, model_info = analysis.build_model_for_inference(
        cnn_module=cnn_module,
        num_classes=num_classes,
        hybrid_cfg=hybrid_cfg,
        device=device,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    rows = analysis.collect_prediction_rows(
        model=model,
        loader=test_loader,
        idx_to_class=idx_to_class,
        device=device,
    )
    if not rows:
        raise ValueError('No test samples found.')

    selected_rows = select_rows(rows)
    if not selected_rows:
        raise ValueError(f'No samples selected for SAMPLE_MODE={SAMPLE_MODE!r}.')

    target_layer_name, target_layer = find_last_conv_layer(model)
    grad_cam = GradCAM(model=model, target_layer=target_layer)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_dir = ANALYSIS_ROOT / f'{OUTPUT_PREFIX}_{timestamp}'
    image_dir = analysis_dir / 'images'
    output_rows: list[dict[str, Any]] = []

    try:
        for rank, row in enumerate(selected_rows, start=1):
            src = Path(row['path'])
            if not src.exists():
                continue

            image, input_tensor = load_image_for_gradcam(src, device=device)
            for target_name, target_idx in target_classes_for_row(row):
                target_label = idx_to_class[target_idx]
                cam, logits = grad_cam.generate(input_tensor=input_tensor, target_class=target_idx)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                output_name = (
                    f"{rank:04d}_target-{target_name}-{slugify(target_label)}"
                    f"_true-{slugify(str(row['true_label']))}"
                    f"_pred-{slugify(str(row['pred_label']))}"
                    f"_{slugify(src.stem)}.png"
                )
                output_path = render_gradcam_figure(
                    image=image,
                    cam=cam,
                    row=row,
                    target_name=target_name,
                    target_label=target_label,
                    output_path=image_dir / output_name,
                )

                output_row = dict(row)
                output_row.update(
                    {
                        'rank': rank,
                        'target_mode': target_name,
                        'target_idx': target_idx,
                        'target_label': target_label,
                        'target_prob': float(probs[target_idx]),
                        'grad_cam_path': str(output_path),
                    }
                )
                output_rows.append(output_row)
    finally:
        grad_cam.close()

    selected_csv = analysis_dir / 'selected_predictions.csv'
    gradcam_csv = analysis_dir / 'grad_cam_outputs.csv'
    save_rows_csv(selected_rows, selected_csv)
    save_rows_csv(output_rows, gradcam_csv)

    summary = {
        'device': str(device),
        'test_dir': str(TEST_DIR),
        'model_source': str(resolved_source.source_path),
        'state_dict_path': str(resolved_source.state_dict_path),
        'metadata_path': str(resolved_source.metadata_path) if resolved_source.metadata_path else None,
        'analysis_dir': str(analysis_dir),
        'images_dir': str(image_dir),
        'sample_mode': SAMPLE_MODE,
        'target_mode': TARGET_MODE,
        'max_images': MAX_IMAGES,
        'num_test_samples': int(len(rows)),
        'num_selected_samples': int(len(selected_rows)),
        'num_grad_cam_outputs': int(len(output_rows)),
        'target_layer_name': target_layer_name,
        'selected_predictions_csv': str(selected_csv),
        'grad_cam_outputs_csv': str(gradcam_csv),
        'model_info': model_info,
        'hybrid_cfg': hybrid_cfg,
        'checkpoint_best_val_accuracy': payload.get('best_val_acc'),
        'checkpoint_best_epoch': (int(payload['epoch']) + 1) if isinstance(payload.get('epoch'), int) else None,
    }
    summary_safe = to_json_safe(summary)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / 'summary.json').write_text(
        json.dumps(summary_safe, indent=2),
        encoding='utf-8',
    )

    print(json.dumps(summary_safe, indent=2))


if __name__ == '__main__':
    main()
