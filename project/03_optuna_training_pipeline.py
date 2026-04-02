"""Optimiert ein NN mit Optuna, trainiert das beste Modell final und speichert Metriken sowie Plots als Artefakte."""

import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from models import SimpleCNN
from prep_training import make_dataloaders
from training_functions import save_best_model_artifacts


# =========================
# KONFIGURATION
# =========================
SEED = 77

ARTIFACT_BASE_DIR = 'trained_models'
MODEL_NAME = 'Simple_CNN'
BEST_MODEL_CHECKPOINT_PATH = 'trained_models/Simple_CNN.pt'

N_TRIALS = 40
OPTUNA_EPOCHS = 8
OPTUNA_PATIENCE = 3
OPTUNA_PRUNER_STARTUP_TRIALS = 5
OPTUNA_PRUNER_WARMUP_STEPS = 5
STUDY_NAME = 'CNN_zebra_model_optimization'

FINAL_TRAIN_EPOCHS = 100
FINAL_PATIENCE = 10

PLOT_DPI = 180


def save_figure_png(fig: plt.Figure, output_path: Path, dpi: int = PLOT_DPI) -> Path:
    """Speichert eine Matplotlib-Figur als PNG und schließt sie danach."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return output_path


def check_device():
    """Wählt das beste verfügbare Rechengerät (CUDA, MPS oder CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Verwende Gerät: {device}')
    return device


torch.manual_seed(SEED)
DEVICE = check_device()


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    best_params: dict,
) -> None:
    """Speichert den aktuellen besten Trainingszustand als Checkpoint."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'best_val_acc': best_val_acc,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_params': best_params,
        },
        checkpoint_path,
    )


def compute_l1_penalty(model: nn.Module, lambda_: float = 1e-4) -> torch.Tensor:
    """Berechnet die L1-Regularisierungsstrafe über alle Modellparameter."""
    penalty = torch.tensor(0.0, device=DEVICE)
    for p in model.parameters():
        penalty += p.abs().sum()
    return lambda_ * penalty


def compute_l2_penalty(model: nn.Module, lambda_: float = 1e-4) -> torch.Tensor:
    """Berechnet die L2-Regularisierungsstrafe über alle Modellparameter."""
    penalty = torch.tensor(0.0, device=DEVICE)
    for p in model.parameters():
        penalty += p.pow(2).sum()
    return lambda_ * penalty


REGULARIZER_FN = {
    None: lambda m: 0.0,
    'l1': compute_l1_penalty,
    'l2': compute_l2_penalty,
    'l1_l2': lambda m: compute_l1_penalty(m) + compute_l2_penalty(m),
}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    regularizer: str | None,
) -> tuple[float, float]:
    """Trainiert das Modell für eine Epoche und gibt mittleren Loss sowie Accuracy zurück."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    reg_fn = REGULARIZER_FN[regularizer]

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb) + reg_fn(model)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(xb)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(xb)

    return total_loss / n, correct / n


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Sammelt True-Labels und Modellvorhersagen für einen kompletten Loader."""
    model.eval()
    all_preds, all_targets = [], []

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(yb.cpu().numpy())

    return np.array(all_targets), np.array(all_preds)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> dict:
    """Evaluiert Loss, Accuracy, Precision, Recall und F1 für einen Loader."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)

        preds = logits.argmax(1)

        total_loss += loss.item() * len(xb)
        correct += (preds == yb).sum().item()
        n += len(xb)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(yb.cpu().numpy())

    acc = correct / n
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    return {
        'loss': total_loss / n,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# Daten laden.
dataloaders, class_to_idx, idx_to_class = make_dataloaders(num_workers=0)

train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']

print(f'Train Bilder: {len(train_loader)}')



def objective(trial: optuna.Trial) -> float:
    """Definiert einen Optuna-Trial und gibt den besten Validierungs-Loss zurück."""
    n_layers = trial.suggest_int('n_layers', 1, 5)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop', 'adagrad'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
    regularizer = trial.suggest_categorical('regularizer', [None, 'l1', 'l2', 'l1_l2'])

    layer_sizes = [
        trial.suggest_int(f'n_nodes_layer_{i}', 16, 256, step=16)
        for i in range(n_layers)
    ]

    model = SimpleCNN(
        n_layers=n_layers,
        layer_sizes=layer_sizes,
        activation=activation,
        dropout_rate=dropout_rate,
    ).to(DEVICE)

    optimizer_map = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
    }
    optimizer = optimizer_map[optimizer_name](model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = 1e10
    patience_ct = 0

    for epoch in range(OPTUNA_EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, regularizer)
        metrics = evaluate(model, val_loader, criterion)
        val_acc = metrics['accuracy']
        val_loss = metrics['loss']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= OPTUNA_PATIENCE:
                break

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


sampler = TPESampler(seed=SEED)
pruner = MedianPruner(
    n_startup_trials=OPTUNA_PRUNER_STARTUP_TRIALS,
    n_warmup_steps=OPTUNA_PRUNER_WARMUP_STEPS,
)

study = optuna.create_study(
    direction='minimize',
    sampler=sampler,
    pruner=pruner,
    study_name=STUDY_NAME,
)

start_time = time.time()
process = psutil.Process(os.getpid())
ram_start = process.memory_info().rss / 1024**3

print(f'\nStarte Optuna-Optimierung: {N_TRIALS} Trials, Pruning aktiv\n')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

end_time = time.time()
ram_end = process.memory_info().rss / 1024**3

print('\n' + '=' * 60)
print('RESSOURCEN & ZEIT')
print('=' * 60)
print(f'  Gesamtdauer     : {(end_time - start_time) / 60:.2f} Minuten')
print(f'  Ø pro Trial     : {(end_time - start_time) / N_TRIALS:.1f} Sekunden')
print(f'  RAM Start       : {ram_start:.2f} GB')
print(f'  RAM Ende        : {ram_end:.2f} GB')
print(f'  RAM Differenz   : {ram_end - ram_start:.2f} GB')
print(f'  Device          : {DEVICE}')
print(f'  Trials gesamt   : {len(study.trials)}')
print(f'  Trials pruned   : {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}')
print(f'  Trials komplett : {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}')

best = study.best_trial
print('\n' + '=' * 60)
print('BESTER TRIAL')
print('=' * 60)
print(f'  Trial Nummer : {best.number}')
print(f'  Val Loss     : {best.value:.4f}')
print('\n  Hyperparameter:')
for key, value in best.params.items():
    print(f'    {key:30s} = {value}')

print('\n' + '=' * 60)
print(f'Trainiere bestes Modell mit {FINAL_TRAIN_EPOCHS} Epochen...')
print('=' * 60)

p = best.params
layer_sizes = [p[f'n_nodes_layer_{i}'] for i in range(p['n_layers'])]

best_model = SimpleCNN(
    n_layers=p['n_layers'],
    layer_sizes=layer_sizes,
    activation=p['activation'],
    dropout_rate=p['dropout_rate'],
).to(DEVICE)

optimizer_map = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop,
    'adagrad': optim.Adagrad,
}
best_optimizer = optimizer_map[p['optimizer']](best_model.parameters(), lr=p['learning_rate'])
criterion = nn.CrossEntropyLoss()

history = {
    'loss': [],
    'accuracy': [],
    'val_loss': [],
    'val_accuracy': [],
    'val_recall': [],
    'val_precision': [],
    'val_f1': [],
}
best_val_acc = 0.0
patience_ct = 0

for epoch in range(FINAL_TRAIN_EPOCHS):
    tr_loss, tr_acc = train_one_epoch(
        best_model,
        train_loader,
        criterion,
        best_optimizer,
        p['regularizer'],
    )
    metrics = evaluate(best_model, val_loader, criterion)

    val_loss = metrics['loss']
    val_acc = metrics['accuracy']
    val_recall = metrics['recall']
    val_f1 = metrics['f1']
    val_prec = metrics['precision']

    history['loss'].append(tr_loss)
    history['accuracy'].append(tr_acc)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_acc)
    history['val_recall'].append(val_recall)
    history['val_precision'].append(val_prec)
    history['val_f1'].append(val_f1)

    print(
        f'Epoch {epoch:3d}/{FINAL_TRAIN_EPOCHS}  '
        f'loss: {tr_loss:.4f}  acc: {tr_acc:.4f}  '
        f'val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}'
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_ct = 0
        save_training_checkpoint(
            path=BEST_MODEL_CHECKPOINT_PATH,
            model=best_model,
            optimizer=best_optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc,
            best_params=best.params,
        )
    else:
        patience_ct += 1
        if patience_ct >= FINAL_PATIENCE:
            print(f'  → Early Stopping nach Epoch {epoch}.')
            break

best_checkpoint = torch.load(BEST_MODEL_CHECKPOINT_PATH, map_location=DEVICE)
best_model.load_state_dict(best_checkpoint['model_state_dict'])
metrics = evaluate(best_model, test_loader, criterion)

test_loss = metrics['loss']
test_acc = metrics['accuracy']
test_recall = metrics['recall']
test_f1 = metrics['f1']
test_prec = metrics['precision']

y_true, y_pred = collect_predictions(best_model, test_loader)

artifact_dir = save_best_model_artifacts(
    model=best_model,
    y_true=y_true,
    y_pred=y_pred,
    model_name=MODEL_NAME,
    score=test_acc,
    params=best.params,
    history=history,
    base_dir=ARTIFACT_BASE_DIR,
)

print(f'\n  Test-Loss:      {test_loss:.4f}')
print(f'  Test-Accuracy:  {test_acc:.4f}')
print(f'  Test-Recall:    {test_recall:.4f}')
print(f'  Test-F1 Score:  {test_f1:.4f}')
print(f'  Test-Precision: {test_prec:.4f}')
print(f"  → Modell gespeichert: {BEST_MODEL_CHECKPOINT_PATH} (Epoch {best_checkpoint['epoch']}, val_acc={best_checkpoint['best_val_acc']:.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Bestes Modell – Learning Curves')

epochs_range = range(1, len(history['loss']) + 1)
ax1.plot(epochs_range, history['loss'], label='Train Loss')
ax1.plot(epochs_range, history['val_loss'], label='Val Loss', linestyle='--')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(epochs_range, history['accuracy'], label='Train Accuracy')
ax2.plot(epochs_range, history['val_accuracy'], label='Val Accuracy', linestyle='--')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
learning_curve_path = save_figure_png(fig, Path(artifact_dir) / 'learning_curves.png')
print(f'  → Plot gespeichert: {learning_curve_path}')