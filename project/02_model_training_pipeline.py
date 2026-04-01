"""Trainiert ein einfaches MLP-Basismodell und speichert Lernkurven als PNG."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from models import MLP
from prep_training import infer_input_config, make_dataloaders


# Basis-Konfiguration
SEED = 77
EPOCHS = 2
LEARNING_RATE = 1e-3
MAX_TRAIN_BATCHES_PER_EPOCH = 30
PLOT_PATH = Path('model_data/MLP_training.png')


def get_device() -> torch.device:
    """Wählt das beste verfügbare Rechengerät (CUDA, MPS oder CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: int,
) -> float:
    """Trainiert eine Epoche und gibt den mittleren Trainingsverlust zurück."""
    model.train()
    total_loss = 0.0

    for i, (x_batch, y_batch) in enumerate(loader):
        if i >= max_batches:
            break

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).long()

        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, min(len(loader), max_batches))


@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader, loss_fn: nn.Module, device: torch.device) -> float:
    """Evaluiert eine Epoche auf dem Validierungsset und gibt den mittleren Verlust zurück."""
    model.eval()
    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).long()

        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def main():
    """Startet Baseline-Training, speichert Lernkurven und gibt Kennzahlen pro Epoche aus."""
    torch.manual_seed(SEED)
    device = get_device()
    print(f'Verwende Gerät: {device}')

    dataloaders, class_to_idx, _ = make_dataloaders(num_workers=0)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    cfg = infer_input_config(train_loader)
    input_size = cfg['input_size_mlp']
    num_classes = len(class_to_idx)

    # Einfaches Basismodell (2 Hidden-Layer)
    n_layers = 2
    layer_sizes = [512, 128]
    model = MLP(
        input_size=input_size,
        n_layers=n_layers,
        layer_sizes=layer_sizes,
        activation='relu',
        dropout_rate=0.2,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    print(f'input_size: {input_size} | num_classes: {num_classes}')
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            max_batches=MAX_TRAIN_BATCHES_PER_EPOCH,
        )
        val_loss = validate_one_epoch(model=model, loader=val_loader, loss_fn=loss_fn, device=device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f'Epoch {epoch + 1}/{EPOCHS}, '
            f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}'
        )

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    sns.lineplot(x=list(range(EPOCHS)), y=train_losses, label='Train')
    sns.lineplot(x=list(range(EPOCHS)), y=val_losses, label='Val')
    plt.xticks(range(EPOCHS))
    plt.xlabel('Epoche [-]')
    plt.ylabel('Verlust [-]')
    plt.title('Trainings- und Validierungsverlust')
    plt.savefig(PLOT_PATH, dpi=180, bbox_inches='tight')
    plt.close()

    print(f'Training finished, plot saved to {PLOT_PATH}')


if __name__ == '__main__':
    main()
