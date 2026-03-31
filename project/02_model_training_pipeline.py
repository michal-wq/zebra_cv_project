import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from helpers import read_data
import seaborn as sns
from helpers import save_image
# MODEL IMPORT
from models import MLP

# GERÄT Auswählen
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Verwende Gerät: {DEVICE}")

# LOAD DATA
TRAIN_PATH = 'data/train'
VAL_PATH = 'data/val'
TEST_PATH = 'data/test'

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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


def main():
    torch.manual_seed(77)
    dataloaders, class_to_idx, idx_to_class = make_dataloaders(num_workers=0)

    train_loader = dataloaders["train"]
    val_loader = dataloaders['val']
    train_features, train_labels = next(iter(train_loader))
    """
    
    # Schaue ein Bild an
    img = train_features[0].permute(1, 2, 0)
    label_idx = train_labels[0].item()
    label_name = idx_to_class[label_idx]

    plt.imshow(img)
    plt.title(f"Label: {label_name}")
    plt.show()
    """
    cfg = infer_input_config(train_loader)
    num_classes = len(class_to_idx)
    input_size = cfg["input_size_mlp"]
    #N_LAYERS = 4
    LAYER_SIZES = [input_size, 512, 128, num_classes]
    LEARNING_RATE = 1e-3
    EPOCHS = 2
    print(f'input_size: {input_size} | num_classes: {num_classes}')
    model = MLP(layer_sizes=LAYER_SIZES, activation="relu", dropout_rate=0.2).to(DEVICE)
    # %% optimizer and loss function with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    #%% training loop
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        train_loss = 0
        val_loss = 0
        i = 0
        for X_train_batch, y_train_batch in train_loader:
            # move data to device
            X_train_batch = X_train_batch.to(DEVICE)
            y_train_batch = y_train_batch.to(DEVICE).long()
            # forward pass
            y_train_batch_pred = model(X_train_batch)
            # calculate loss
            loss = loss_fn(y_train_batch_pred, y_train_batch)
            # backward pass
            loss.backward()
            # update weights
            optimizer.step()
            # reset gradients
            optimizer.zero_grad()
            # update train loss
            train_loss += loss.item()
            if i > 30:
                break
            i += 1
        # append train loss
        train_losses.append(train_loss/len(train_loader))
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                # move data to device
                X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
                # forward pass
                y_val_batch_pred = model(X_val_batch)
                # calculate loss
                loss = loss_fn(y_val_batch_pred, y_val_batch)
                # update val loss
                val_loss += loss.item()
            # normalize and append val loss
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_losses[-1]:.4f} ,Val Loss: {val_losses[-1]:.4f}")

    plot_name = 'model_data/MLP_training.png'
    plt.figure()
    sns.lineplot(x=list(range(EPOCHS)), y=train_losses)
    sns.lineplot(x=list(range(EPOCHS)), y=val_losses)
    plt.xticks(range(EPOCHS))
    plt.xlabel('Epoche [-]')
    plt.ylabel('Verlust [-]')
    plt.title('Trainingsverlust und Epochen')
    plt.savefig(plot_name)
    print(f'Training finished, plot saved to{plot_name}')


if __name__ == "__main__":
    main()