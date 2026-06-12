"""Lecturer-friendly runner for the Big_Trans_3 zebra classifier.

The script can:
- download the required public Hugging Face artifacts,
- evaluate Big_Trans_3 on project/data/test,
- classify a single image.

Run from the repository root:
    uv run python project/run_big_trans_3.py --download
    uv run python project/run_big_trans_3.py
    uv run python project/run_big_trans_3.py --image path/to/image.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent

HF_DATA_REPO = 'kamichal/zebra-cv-data'
HF_MODEL_REPO = 'kamichal/zebra-cv-checkpoints'

MODEL_ARTIFACT_NAME = 'Big_Trans_3_score-0.9978_20260516_110849'
MODEL_ARTIFACT_DIR = PROJECT_DIR / 'trained_models' / MODEL_ARTIFACT_NAME
MODEL_STATE_PATH = MODEL_ARTIFACT_DIR / 'model_state_dict.pt'
MODEL_METADATA_PATH = MODEL_ARTIFACT_DIR / 'metadata.json'
MODEL_RESULTS_PATH = MODEL_ARTIFACT_DIR / 'results.json'

DATA_DIR = PROJECT_DIR / 'data'
TEST_DIR = DATA_DIR / 'test'

IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Download and run the Big_Trans_3 zebra classifier.',
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download test data and the Big_Trans_3 model artifact from Hugging Face first.',
    )
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Download artifacts and exit without evaluation.',
    )
    parser.add_argument(
        '--image',
        type=Path,
        help='Optional image path for single-image prediction instead of test-set evaluation.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Batch size for test-set evaluation. Default: {DEFAULT_BATCH_SIZE}.',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f'DataLoader workers for evaluation. Default: {DEFAULT_NUM_WORKERS}.',
    )
    parser.add_argument(
        '--device',
        choices=('auto', 'cpu', 'cuda', 'mps'),
        default='auto',
        help='Execution device. Default: auto.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_DIR / 'big_trans_3_evaluation_results.json',
        help='JSON output path for test-set evaluation results.',
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print(f'$ {" ".join(cmd)}')
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def download_from_hugging_face() -> None:
    """Downloads the public test split and the Big_Trans_3 model artifact."""
    run_command(
        [
            'uvx',
            'hf',
            'download',
            HF_DATA_REPO,
            '--repo-type',
            'dataset',
            '--local-dir',
            str(PROJECT_DIR),
            '--include',
            'data/*',
        ]
    )
    run_command(
        [
            'uvx',
            'hf',
            'download',
            HF_MODEL_REPO,
            '--local-dir',
            str(PROJECT_DIR),
            '--include',
            f'trained_models/{MODEL_ARTIFACT_NAME}/*',
        ]
    )


def require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f'Missing required file: {path}\n{hint}')


def require_dir(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f'Missing required directory: {path}\n{hint}')


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def get_device(requested: str) -> torch.device:
    if requested == 'cpu':
        return torch.device('cpu')
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available.')
        return torch.device('cuda')
    if requested == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS was requested but is not available.')
        return torch.device('mps')

    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, 'Run with --download to fetch the model artifact from Hugging Face.')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected JSON object in {path}')
    return data


def class_labels_from_results() -> list[str]:
    results = load_json(MODEL_RESULTS_PATH)
    labels = results.get('class_labels')
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        return ['n', 'y']
    return labels


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_big_trans_3(device: torch.device, num_classes: int):
    """Reconstructs Big_Trans_3 and loads its state dict."""
    require_file(
        MODEL_STATE_PATH,
        'Run with --download, or download the model with:\n'
        f'uvx hf download {HF_MODEL_REPO} --local-dir project '
        f'--include "trained_models/{MODEL_ARTIFACT_NAME}/*"',
    )
    require_file(MODEL_METADATA_PATH, 'metadata.json is required to reconstruct the architecture.')

    analysis = load_module('zebra_misclassification_analysis', PROJECT_DIR / '08_misclassification_analysis.py')
    cnn_vit_module = analysis.load_cnn_vit_module(PROJECT_DIR / '07_cnn_vit_seq.py')

    metadata = analysis.load_json_if_exists(MODEL_METADATA_PATH)
    metadata['_source_dir'] = MODEL_ARTIFACT_DIR

    raw_checkpoint = torch.load(MODEL_STATE_PATH, map_location='cpu')
    state_dict, payload = analysis.extract_state_dict_and_payload(raw_checkpoint)
    state_dict = analysis.strip_module_prefix(state_dict)

    hybrid_cfg = analysis.resolve_hybrid_config(
        cnn_vit_module,
        metadata,
        checkpoint_payload=payload,
    )
    model, model_info = analysis.build_model_for_inference(
        cnn_module=cnn_vit_module,
        num_classes=num_classes,
        hybrid_cfg=hybrid_cfg,
        device=device,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            'Could not load Big_Trans_3 weights. The model architecture and '
            f'metadata may not match. Used config: {hybrid_cfg}'
        ) from exc

    model.eval()
    return model, model_info


@torch.no_grad()
def predict_image(image_path: Path, device: torch.device) -> dict[str, Any]:
    require_file(image_path, 'Provide an existing image path.')
    class_labels = class_labels_from_results()
    model, model_info = load_big_trans_3(device=device, num_classes=len(class_labels))

    image = Image.open(image_path).convert('RGB')
    tensor = eval_transform()(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))

    return {
        'image': str(image_path),
        'prediction': class_labels[pred_idx],
        'prediction_index': pred_idx,
        'probabilities': {
            class_labels[i]: float(probs[i])
            for i in range(len(class_labels))
        },
        'model_artifact': str(MODEL_ARTIFACT_DIR),
        'model_info': model_info,
    }


@torch.no_grad()
def evaluate_test_set(device: torch.device, batch_size: int, num_workers: int, output_path: Path) -> dict[str, Any]:
    require_dir(
        TEST_DIR,
        'Run with --download, or download the data with:\n'
        f'uvx hf download {HF_DATA_REPO} --repo-type dataset --local-dir project --include "data/*"',
    )

    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_transform())
    class_to_idx = test_ds.class_to_idx
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    model, model_info = load_big_trans_3(device=device, num_classes=len(class_labels))

    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
    )

    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []
    amp_enabled = device.type == 'cuda'

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=device.type == 'cuda')
        with torch.autocast(device_type='cuda', enabled=amp_enabled):
            logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_targets.extend(yb.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.max(dim=1).values.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))

    results = {
        'model': MODEL_ARTIFACT_NAME,
        'device': str(device),
        'test_dir': str(TEST_DIR),
        'num_test_samples': int(len(test_ds)),
        'class_to_idx': class_to_idx,
        'class_labels': class_labels,
        'accuracy': float((y_true == y_pred).mean()),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'confusion_matrix': cm.tolist(),
        'mean_prediction_confidence': float(np.mean(all_probs)),
        'model_artifact_dir': str(MODEL_ARTIFACT_DIR),
        'model_info': model_info,
    }

    output_path = output_path if output_path.is_absolute() else (Path.cwd() / output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    results['output_path'] = str(output_path)
    return results


def print_result(result: dict[str, Any]) -> None:
    printable = dict(result)
    if 'model_info' in printable:
        printable['model_info'] = {
            key: printable['model_info'][key]
            for key in ('feature_channels', 'feature_grid_size', 'state_dict_path')
            if key in printable['model_info']
        }
    print(json.dumps(printable, indent=2))


def main() -> None:
    args = parse_args()

    if args.download or args.download_only:
        download_from_hugging_face()
        if args.download_only:
            return

    device = get_device(args.device)

    if args.image is not None:
        result = predict_image(args.image, device=device)
    else:
        result = evaluate_test_set(
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_path=args.output,
        )

    print_result(result)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise
