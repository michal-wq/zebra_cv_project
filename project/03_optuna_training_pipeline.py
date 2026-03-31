import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Dict, Tuple
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from helpers import read_data, save_image
import seaborn as sns
from models import MLP
from training_functions import compute_l1_penalty, compute_l2_penalty, train_one_epoch, evaluate, check_device
from prep_training import infer_input_config, make_dataloaders
import time
import psutil
import os

torch.manual_seed(77)
DEVICE = check_device()
# LOAD DATA
TRAIN_PATH = 'data/train'
VAL_PATH = 'data/val'
TEST_PATH = 'data/test'

dataloaders, class_to_idx, idx_to_class = make_dataloaders(num_workers=0)

train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']
train_features, train_labels = next(iter(train_loader))

cfg = infer_input_config(train_loader)
num_classes = len(class_to_idx)
input_size = cfg['input_size_mlp']

print('good')

N_TRIALS = 50
EPOCHS   = 30   

def objective(trial: optuna.Trial) -> float:
    ## Teil 1: Hyperparameter definieren
    # --- Hyperparameter-Suchraum ---
    #batch_size     = trial.suggest_categorical('batch_size', [16, 32, 64, 112, 128])
    n_layers       = trial.suggest_int('n_layers', 1, 5)
    activation     = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop', 'adagrad'])
    learning_rate  = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    dropout_rate   = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
    regularizer    = trial.suggest_categorical('regularizer', [None, 'l1', 'l2', 'l1_l2'])

    layer_sizes = [
        trial.suggest_int(f'n_nodes_layer_{i}', 16, 256, step=16)
        for i in range(n_layers)
    ]

    ## Teil 2: Modell und Optmierern aufbauen
    # --- DataLoader ---
    train_loader = dataloaders['train']
    val_loader   = dataloaders['val']

    # --- Modell ---
    model = MLP(input_size,n_layers, layer_sizes, activation, dropout_rate).to(DEVICE)

    # --- Optimizer ---
    optimizer_map = {
        'adam':    optim.Adam,
        'sgd':     optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
    }
    optimizer = optimizer_map[optimizer_name](model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    ## Teil 3: Early Stopping & Pruning
    best_val_acc  = 0.0
    patience_ct   = 0
    patience      = 5   # Early Stopping

    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, regularizer, DEVICE)
        _, val_acc = evaluate(model, val_loader, criterion)

        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ct  = 0
        else:
            patience_ct += 1
            if patience_ct >= patience:
                break

        # Pruning: schlechte Trials frühzeitig abbrechen
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc

sampler = TPESampler(seed=77)
pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

study = optuna.create_study(
    direction='maximize',
    sampler=sampler,
    pruner=pruner,
    study_name='zebra_giga_shit_model_optimization',
)

start_time = time.time()
process = psutil.Process(os.getpid())
ram_start = process.memory_info().rss / 1024**3   # GB

print(f'\nStarte Optuna-Optimierung: {N_TRIALS} Trials, Pruning aktiv')
print(f'RAM vor Training:  {ram_start:.2f} GB\n')

print(f'\nStarte Optuna-Optimierung: {N_TRIALS} Trials, Pruning aktiv\n')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# Endwerte erfassen
end_time = time.time()
ram_end = process.memory_info().rss / 1024**3  

# Ausgabe
print('\n' + '='*60)
print('RESSOURCEN & ZEIT')
print('='*60)
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
print('\n' + '='*60)
print('BESTER TRIAL')
print('='*60)
print(f'  Trial Nummer : {best.number}')
print(f'  Val Accuracy : {best.value:.4f}')
print('\n  Hyperparameter:')
for key, value in best.params.items():
    print(f'    {key:30s} = {value}')

'''

Bestes Modell nochmal trainieren

'''


print('\n' + '='*60)
print('Trainiere bestes Modell mit 100 Epochen...')
print('='*60)

p = best.params

layer_sizes = [p[f'n_nodes_layer_{i}'] for i in range(p['n_layers'])]

best_model = MLP(
    input_size,
    n_layers=p['n_layers'],
    layer_sizes=layer_sizes,
    activation=p['activation'],
    dropout_rate=p['dropout_rate'],
).to(DEVICE)

optimizer_map = {
    'adam':    optim.Adam,
    'sgd':     optim.SGD,
    'rmsprop': optim.RMSprop,
    'adagrad': optim.Adagrad,
}
best_optimizer = optimizer_map[p['optimizer']](best_model.parameters(), lr=p['learning_rate'])
criterion      = nn.CrossEntropyLoss()

history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
best_val_acc  = 0.0
patience_ct   = 0
FINAL_PATIENCE = 10

for epoch in range(1, 101):
    tr_loss, tr_acc = train_one_epoch(
        best_model, train_loader, criterion, best_optimizer, p['regularizer']
    )
    val_loss, val_acc = evaluate(best_model, val_loader, criterion)

    history['loss'].append(tr_loss)
    history['accuracy'].append(tr_acc)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_acc)

    print(
        f'Epoch {epoch:3d}/100  '
        f'loss: {tr_loss:.4f}  acc: {tr_acc:.4f}  '
        f'val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}'
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_ct  = 0
        torch.save(best_model.state_dict(), 'best_model_optuna.pt')
    else:
        patience_ct += 1
        if patience_ct >= FINAL_PATIENCE:
            print(f'  → Early Stopping nach Epoch {epoch}.')
            break

# Besten Checkpoint laden & auf Testset evaluieren
best_model.load_state_dict(torch.load('best_model_optuna.pt'))
test_loss, test_acc = evaluate(best_model, test_loader, criterion)
print(f'\n  Test-Loss:     {test_loss:.4f}')
print(f'  Test-Accuracy: {test_acc:.4f}')
print('  → Modell gespeichert: best_model_optuna.pt')

"""

Visualisierungen

"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Bestes Modell – Learning Curves")

epochs_range = range(1, len(history["loss"]) + 1)
ax1.plot(epochs_range, history["loss"],     label="Train Loss")
ax1.plot(epochs_range, history["val_loss"], label="Val Loss", linestyle="--")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()

ax2.plot(epochs_range, history["accuracy"],     label="Train Accuracy")
ax2.plot(epochs_range, history["val_accuracy"], label="Val Accuracy", linestyle="--")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend()

plt.tight_layout()
plt.show()