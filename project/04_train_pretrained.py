"""Fine-tunes a pretrained ResNet18 on local zebra data with on-the-fly augmentation."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from training_functions import save_best_model_artifacts

SEED = 77
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DATA_ROOT = Path("data")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

MODEL_NAME = "ResNet18_Finetune"
MODEL_DIR = Path("trained_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODEL_DIR / "resnet18_finetune_checkpoint.pt"
BEST_MODEL_PATH = MODEL_DIR / "resnet18_finetune_best.pt"

TRAIN_AUGMENTATION_CONFIG = {
    "random_resized_crop_scale": (0.75, 1.0),
    "horizontal_flip_prob": 0.5,
    "rotation_degrees": 15,
    "perspective_prob": 0.25,
    "perspective_distortion_scale": 0.25,
    "color_jitter": {
        "brightness": 0.25,
        "contrast": 0.25,
        "saturation": 0.2,
        "hue": 0.06,
    },
    "autocontrast_prob": 0.1,
    "gaussian_blur_prob": 0.15,
    "gaussian_blur_kernel_size": 5,
    "gaussian_blur_sigma": (0.1, 1.5),
    "random_erasing_prob": 0.15,
    "random_erasing_scale": (0.02, 0.12),
    "random_erasing_ratio": (0.3, 3.3),
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders() -> tuple[dict[str, DataLoader], int, dict[str, int]]:
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=TRAIN_AUGMENTATION_CONFIG["random_resized_crop_scale"],
            ),
            transforms.RandomHorizontalFlip(
                p=TRAIN_AUGMENTATION_CONFIG["horizontal_flip_prob"],
            ),
            transforms.RandomRotation(
                degrees=TRAIN_AUGMENTATION_CONFIG["rotation_degrees"],
            ),
            transforms.RandomPerspective(
                distortion_scale=TRAIN_AUGMENTATION_CONFIG["perspective_distortion_scale"],
                p=TRAIN_AUGMENTATION_CONFIG["perspective_prob"],
            ),
            transforms.ColorJitter(**TRAIN_AUGMENTATION_CONFIG["color_jitter"]),
            transforms.RandomAutocontrast(
                p=TRAIN_AUGMENTATION_CONFIG["autocontrast_prob"],
            ),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=TRAIN_AUGMENTATION_CONFIG["gaussian_blur_kernel_size"],
                        sigma=TRAIN_AUGMENTATION_CONFIG["gaussian_blur_sigma"],
                    ),
                ],
                p=TRAIN_AUGMENTATION_CONFIG["gaussian_blur_prob"],
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=TRAIN_AUGMENTATION_CONFIG["random_erasing_prob"],
                scale=TRAIN_AUGMENTATION_CONFIG["random_erasing_scale"],
                ratio=TRAIN_AUGMENTATION_CONFIG["random_erasing_ratio"],
            ),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tfms)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tfms)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    return loaders, len(train_ds.classes), train_ds.class_to_idx


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.layer4.parameters():
        p.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model.to(device)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            n += xb.size(0)

    return {"loss": total_loss / n, "accuracy": correct / n}


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets = []
    all_preds = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(yb.numpy())

    return np.array(all_targets), np.array(all_preds)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += xb.size(0)

        pbar.set_postfix(loss=total_loss / n, acc=correct / n)

    return {"loss": total_loss / n, "accuracy": correct / n}


def main() -> None:
    torch.manual_seed(SEED)
    device = get_device()

    if not TRAIN_DIR.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Expected directories not found:\n"
            f"{TRAIN_DIR}\n{VAL_DIR}\n{TEST_DIR}"
        )

    loaders, num_classes, class_to_idx = build_dataloaders()
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = build_model(num_classes=num_classes, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    start_epoch = 0
    best_val_acc = 0.0

    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = evaluate(model, loaders["val"], criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        torch.save(
            {
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_idx": class_to_idx,
                "history": history,
                "train_augmentation_config": TRAIN_AUGMENTATION_CONFIG,
            },
            CHECKPOINT_PATH,
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_to_idx": class_to_idx,
                    "history": history,
                    "train_augmentation_config": TRAIN_AUGMENTATION_CONFIG,
                },
                BEST_MODEL_PATH,
            )

        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "best_val_acc": best_val_acc,
        }
        epoch_stats

    best_ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate(model, loaders["test"], criterion, device)
    y_true, y_pred = collect_predictions(model, loaders["test"], device)

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
        "best_val_accuracy": float(best_ckpt["best_val_acc"]),
        "best_epoch": int(best_ckpt["epoch"] + 1),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "best_model_path": str(BEST_MODEL_PATH),
        "class_labels": class_labels,
        "confusion_matrix": cm.tolist(),
    }

    run_params = {
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "train_dir": str(TRAIN_DIR),
        "train_augmentation_config": TRAIN_AUGMENTATION_CONFIG,
    }

    artifact_dir = save_best_model_artifacts(
        model=model,
        y_true=y_true,
        y_pred=y_pred,
        model_name=MODEL_NAME,
        score=test_metrics["accuracy"],
        params=run_params,
        history=history,
        base_dir=str(MODEL_DIR),
        results=final_results,
        class_labels=class_labels,
        save_history=False,
    )

    final_results["artifact_dir"] = str(artifact_dir)
    final_results


if __name__ == "__main__":
    main()
