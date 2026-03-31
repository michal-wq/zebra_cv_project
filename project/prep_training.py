import torch
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from typing import Dict, Tuple

def infer_input_config(loader):
    x, y = next(iter(loader))              # x: [B, C, H, W], y: [B]
    input_shape = tuple(x.shape[1:])       # (C, H, W)
    input_size_mlp = int(torch.tensor(input_shape).prod().item())  # C*H*W
    num_classes = int(y.max().item() + 1)  # works for labels 0..K-1
    return {
        "input_shape": input_shape,
        "input_size_mlp": input_size_mlp,
        "num_classes": num_classes,
    }

def make_dataloaders(
    data_root: str = "data",
    image_size: int = 224,
    batch_size: int = 4,
    num_workers: int = 4,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], Dict[int, str]]:
    """
    Expects:
      data/
        train/y, train/n
        val/y,   val/n
        test/y,  test/n
    """

    data_root = Path(data_root)

    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    datasets_by_split = {
        "train": datasets.ImageFolder(data_root / "train", transform=train_tfms),
        "val": datasets.ImageFolder(data_root / "val", transform=eval_tfms),
        "test": datasets.ImageFolder(data_root / "test", transform=eval_tfms),
    }

    dataloaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets_by_split["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    # Same mapping for all splits if folder names are consistent
    class_to_idx = datasets_by_split["train"].class_to_idx   # e.g. {'n': 0, 'y': 1}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    return dataloaders, class_to_idx, idx_to_class

