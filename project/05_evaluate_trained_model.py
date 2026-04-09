"""Evaluates a trained ResNet18 checkpoint on test data and saves result artifacts."""

import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from training_functions import save_best_model_artifacts



SEED = 77
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 8

DATA_ROOT = Path("data")
TEST_DIR = DATA_ROOT / "test"

MODEL_NAME = "ResNet18_Finetune_Eval"
MODEL_DIR = Path("trained_models")
MODEL_PATH = MODEL_DIR / "resnet18_finetune_best.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_test_loader() -> tuple[DataLoader, int, dict[str, int]]:
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tfms)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return test_loader, len(test_ds.classes), test_ds.class_to_idx


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


@torch.no_grad()
def evaluate_and_collect(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_targets = []
    all_preds = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * xb.size(0)
        correct += (preds == yb).sum().item()
        n += xb.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(yb.cpu().numpy())

    metrics = {
        "loss": total_loss / n,
        "accuracy": correct / n,
    }

    return metrics, np.array(all_targets), np.array(all_preds)


def load_state_dict_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[dict, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint

    if isinstance(checkpoint, dict):
        return checkpoint, {}

    raise ValueError("Unsupported checkpoint format.")


def main() -> None:
    torch.manual_seed(SEED)
    device = get_device()

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Expected test directory not found: {TEST_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    test_loader, num_classes, class_to_idx = build_test_loader()
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = build_model(num_classes=num_classes, device=device)
    model_state_dict, checkpoint_info = load_state_dict_from_checkpoint(MODEL_PATH, device)
    model.load_state_dict(model_state_dict)

    criterion = nn.CrossEntropyLoss()
    test_metrics, y_true, y_pred = evaluate_and_collect(model, test_loader, criterion, device)

    test_precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    test_recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    test_f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    test_precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    test_recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    test_f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))

    final_results = {
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_precision_weighted": float(test_precision_weighted),
        "test_recall_weighted": float(test_recall_weighted),
        "test_f1_weighted": float(test_f1_weighted),
        "test_precision_macro": float(test_precision_macro),
        "test_recall_macro": float(test_recall_macro),
        "test_f1_macro": float(test_f1_macro),
        "class_labels": class_labels,
        "confusion_matrix": cm.tolist(),
        "source_model_path": str(MODEL_PATH),
        "best_val_accuracy": float(checkpoint_info.get("best_val_acc")) if "best_val_acc" in checkpoint_info else None,
        "best_epoch": int(checkpoint_info.get("epoch") + 1) if "epoch" in checkpoint_info else None,
    }

    eval_params = {
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "source_model_path": str(MODEL_PATH),
    }

    artifact_dir = save_best_model_artifacts(
        model=model,
        y_true=y_true,
        y_pred=y_pred,
        model_name=MODEL_NAME,
        score=test_metrics["accuracy"],
        params=eval_params,
        history=None,
        base_dir=str(MODEL_DIR),
        results=final_results,
        class_labels=class_labels,
        save_history=False,
    )

    final_results["artifact_dir"] = str(artifact_dir)
    final_results


if __name__ == "__main__":
    main()
