"""Fine-tunes a pretrained ResNet18 on local zebra data with checkpoint resume."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


SEED = 77
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DATA_ROOT = Path("data")
TRAIN_DIR = DATA_ROOT / "augmented_train_data"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

MODEL_DIR = Path("trained_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODEL_DIR / "resnet18_finetune_checkpoint.pt"
BEST_MODEL_PATH = MODEL_DIR / "resnet18_finetune_best.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders() -> tuple[dict[str, DataLoader], int, dict[str, int]]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

    # Freeze all layers first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last residual block + classifier for fine-tuning
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

    final_results = {
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "best_val_accuracy": best_ckpt["best_val_acc"],
        "best_epoch": best_ckpt["epoch"] + 1,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "best_model_path": str(BEST_MODEL_PATH),
    }
    final_results


if __name__ == "__main__":
    main()