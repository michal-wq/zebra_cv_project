import os
import random
import time
from collections import Counter
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
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from models import SimpleCNN
from training_functions import save_best_model_artifacts


# =========================
# KONFIGURATION
# =========================
SEED = 77

ARTIFACT_BASE_DIR = 'trained_models'
MODEL_NAME = 'CNN_2"'
BEST_MODEL_CHECKPOINT_PATH = 'trained_models/Simple_CNN_2.pt'

N_TRIALS = 50
OPTUNA_EPOCHS = 10
OPTUNA_PATIENCE = 4
OPTUNA_PRUNER_STARTUP_TRIALS = 5
OPTUNA_PRUNER_WARMUP_STEPS = 5
STUDY_NAME = 'CNN_zebra_model_optimization'

FINAL_TRAIN_EPOCHS = 100
FINAL_PATIENCE = 10

PLOT_DPI = 180

IMAGE_SIZE = 224
BATCH_SIZE = 256
FINAL_BATCH_SIZE_MIN = 16
FINAL_BATCH_SIZE_MAX = 256
FINAL_BATCH_SIZE_STEP = 16
NUM_WORKERS = 16

DATA_ROOT = Path('data')
TRAIN_DIR = DATA_ROOT / 'train'
VAL_DIR = DATA_ROOT / 'val'
TEST_DIR = DATA_ROOT / 'test'

CLASS_REPEAT_FACTORS: dict[str, int] = {
    'y': 36,
    'n': 4,
}

CLASS_AUGMENTATION_CONFIG = {
    'y': {
        'apply_prob': 0.9,
        'hflip_prob': 0.35,
        'rotation_deg': 8,
        'perspective_prob': 0.30,
        'affine_prob': 0.30,
        'affine_deg': 6,
        'affine_translate': (0.08, 0.08),
        'affine_scale': (0.9, 1.1),
        'blur_prob': 0.20,
        'color_jitter': (0.25, 0.25, 0.25, 0.08),
        'grayscale_prob': 0.05,
        'autocontrast_prob': 0.01,
        'equalize_prob': 0.02,
        'sharpness_prob': 0.02,
        'sharpness_factor': 1.8,
        'solarize_prob': 0.01,
        'posterize_prob': 0.02,
        'posterize_bits': 4,
        'randaugment_prob': 0.10,
        'randaugment_num_ops': 2,
        'randaugment_magnitude': 6,
    },
    'n': {
        'apply_prob': 0.8,
        'hflip_prob': 0.25,
        'rotation_deg': 6,
        'perspective_prob': 0.20,
        'affine_prob': 0.20,
        'affine_deg': 4,
        'affine_translate': (0.05, 0.05),
        'affine_scale': (0.95, 1.05),
        'blur_prob': 0.20,
        'color_jitter': (0.25, 0.25, 0.25, 0.08),
        'grayscale_prob': 0.05,
        'autocontrast_prob': 0.01,
        'equalize_prob': 0.02,
        'sharpness_prob': 0.02,
        'sharpness_factor': 1.8,
        'solarize_prob': 0.01,
        'posterize_prob': 0.02,
        'posterize_bits': 4,
        'randaugment_prob': 0.10,
        'randaugment_num_ops': 2,
        'randaugment_magnitude': 6,
    },
}

def get_progressive_batch_size(
    epoch: int,
    total_epochs: int,
    min_bs: int = FINAL_BATCH_SIZE_MIN,
    max_bs: int = FINAL_BATCH_SIZE_MAX,
    step: int = FINAL_BATCH_SIZE_STEP,
) -> int:
    if total_epochs <= 1:
        return max_bs
    t = epoch / (total_epochs - 1)  # 0..1
    raw = min_bs + t * (max_bs - min_bs)  # linear
    bs = int(round(raw / step) * step)     # auf 16er Schritte
    return max(min_bs, min(max_bs, bs))


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
    scheduler=None,
    lr_schedule: str | None = None,
) -> tuple[float, float]:
    """Trainiert das Modell für eine Epoche und gibt mittleren Loss sowie Accuracy zurück."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    reg_fn = REGULARIZER_FN[regularizer]

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb) + reg_fn(model)
        loss.backward()
        optimizer.step()

        if scheduler is not None and lr_schedule == 'onecycle':
            scheduler.step()

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


def build_class_aug_transform(cfg: dict) -> transforms.Compose | None:
    ops: list[nn.Module] = []

    if float(cfg.get('hflip_prob', 0.0)) > 0:
        ops.append(transforms.RandomHorizontalFlip(p=float(cfg['hflip_prob'])))

    if float(cfg.get('vflip_prob', 0.0)) > 0:
        ops.append(transforms.RandomVerticalFlip(p=float(cfg['vflip_prob'])))

    if float(cfg.get('rotation_deg', 0.0)) > 0:
        ops.append(transforms.RandomRotation(degrees=float(cfg['rotation_deg'])))

    if float(cfg.get('perspective_prob', 0.0)) > 0:
        ops.append(
            transforms.RandomPerspective(
                distortion_scale=float(cfg.get('distortion_scale', 0.35)),
                p=float(cfg['perspective_prob']),
            )
        )

    if float(cfg.get('affine_prob', 0.0)) > 0:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomAffine(
                    degrees=float(cfg.get('affine_deg', 0.0)),
                    translate=cfg.get('affine_translate', (0.05, 0.05)),
                    scale=cfg.get('affine_scale', (0.95, 1.05)),
                    shear=cfg.get('affine_shear', 4.0),
                )],
                p=float(cfg['affine_prob']),
            )
        )

    if float(cfg.get('blur_prob', 0.0)) > 0:
        ops.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                p=float(cfg['blur_prob']),
            )
        )

    if cfg.get('color_jitter') is not None:
        b, c, s, h = cfg['color_jitter']
        ops.append(transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h))

    if float(cfg.get('grayscale_prob', 0.0)) > 0:
        ops.append(transforms.RandomGrayscale(p=float(cfg['grayscale_prob'])))

    if float(cfg.get('autocontrast_prob', 0.0)) > 0:
        ops.append(transforms.RandomAutocontrast(p=float(cfg['autocontrast_prob'])))

    if float(cfg.get('equalize_prob', 0.0)) > 0:
        ops.append(transforms.RandomEqualize(p=float(cfg['equalize_prob'])))

    if float(cfg.get('sharpness_prob', 0.0)) > 0:
        ops.append(
            transforms.RandomAdjustSharpness(
                sharpness_factor=float(cfg.get('sharpness_factor', 2.0)),
                p=float(cfg['sharpness_prob']),
            )
        )

    if float(cfg.get('posterize_prob', 0.0)) > 0:
        ops.append(
            transforms.RandomPosterize(
                bits=int(cfg.get('posterize_bits', 4)),
                p=float(cfg['posterize_prob']),
            )
        )

    if float(cfg.get('solarize_prob', 0.0)) > 0:
        ops.append(
            transforms.RandomSolarize(
                threshold=float(cfg.get('solarize_threshold', 128)),
                p=float(cfg['solarize_prob']),
            )
        )

    if float(cfg.get('randaugment_prob', 0.0)) > 0:
        ops.append(
            transforms.RandomApply(
                [transforms.RandAugment(
                    num_ops=int(cfg.get('randaugment_num_ops', 2)),
                    magnitude=int(cfg.get('randaugment_magnitude', 7)),
                )],
                p=float(cfg['randaugment_prob']),
            )
        )

    return transforms.Compose(ops) if ops else None


class ClassAwareAugmentedDataset(Dataset):
    """ImageFolder-Wrapper mit klassenabhängiger On-the-fly-Augmentierung."""

    def __init__(
        self,
        root: Path,
        base_transform: transforms.Compose,
        class_aug_config: dict[str, dict],
        class_repeat_factors: dict[str, int],
    ) -> None:
        self.base_dataset = datasets.ImageFolder(root=root, transform=None)
        self.base_transform = base_transform
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx

        unknown_cfg_classes = set(class_aug_config) - set(self.class_to_idx)
        if unknown_cfg_classes:
            raise ValueError(f'Unknown class in CLASS_AUGMENTATION_CONFIG: {unknown_cfg_classes}')

        unknown_repeat_classes = set(class_repeat_factors) - set(self.class_to_idx)
        if unknown_repeat_classes:
            raise ValueError(f'Unknown class in CLASS_REPEAT_FACTORS: {unknown_repeat_classes}')

        self.class_apply_prob: dict[str, float] = {
            cls_name: float(class_aug_config.get(cls_name, {}).get('apply_prob', 1.0))
            for cls_name in self.classes
        }

        self.class_aug_transforms: dict[str, transforms.Compose | None] = {
            cls_name: build_class_aug_transform(class_aug_config.get(cls_name, {}))
            for cls_name in self.classes
        }

        self.original_class_counts: Counter[int] = Counter(
            target for _, target in self.base_dataset.samples
        )

        self.samples: list[tuple[str, int]] = []
        for path, target in self.base_dataset.samples:
            cls_name = self.classes[target]
            repeat = max(1, int(class_repeat_factors.get(cls_name, 1)))
            self.samples.extend([(path, target)] * repeat)

        self.effective_class_counts: Counter[int] = Counter(target for _, target in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.base_dataset.loader(path)

        class_name = self.classes[target]
        aug_transform = self.class_aug_transforms[class_name]
        apply_prob = self.class_apply_prob[class_name]

        if aug_transform is not None and random.random() < apply_prob:
            image = aug_transform(image)

        image = self.base_transform(image)
        return image, target


def build_dataloaders(num_workers: int = NUM_WORKERS):
    train_base_tfms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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

    train_ds = ClassAwareAugmentedDataset(
        root=TRAIN_DIR,
        base_transform=train_base_tfms,
        class_aug_config=CLASS_AUGMENTATION_CONFIG,
        class_repeat_factors=CLASS_REPEAT_FACTORS,
    )
    val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tfms)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tfms)

    if val_ds.class_to_idx != train_ds.class_to_idx or test_ds.class_to_idx != train_ds.class_to_idx:
        raise ValueError('Train/Val/Test class_to_idx stimmen nicht überein.')

    loaders = {
        'train': DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        'val': DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        'test': DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    return loaders, class_to_idx, idx_to_class



# Daten laden.
dataloaders, class_to_idx, idx_to_class = build_dataloaders(num_workers=NUM_WORKERS)

train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']

print(f'Train Batches: {len(train_loader)}')




def objective(trial: optuna.Trial) -> float:
    """Definiert einen Optuna-Trial und gibt den besten Validierungs-Loss zurück."""
    n_fc_layers = trial.suggest_int("n_fc_layers", 6, 10)
    fc_hidden_size = trial.suggest_int("fc_hidden_size", 128, 256, step=32)

    # Conv fix
    conv_channels = (32, 64, 128, 256, 256)

    activation = trial.suggest_categorical("activation", ["relu", "elu", "swish"])
    optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "nadam"])
    lr_schedule = trial.suggest_categorical("lr_schedule", ["onecycle", "performance"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
    regularizer = trial.suggest_categorical('regularizer', [None, 'l1', 'l2', 'l1_l2'])

    # CNN-spezifische Hyperparameter
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
    pool_type = trial.suggest_categorical('pool_type', ['max', 'avg'])
    use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])
    cnn_dropout_rate = trial.suggest_float('cnn_dropout_rate', 0.0, 0.4, step=0.05)

    model = SimpleCNN(
        n_fc_layers=n_fc_layers,
        fc_hidden_size=fc_hidden_size,
        activation=activation,
        dropout_rate=dropout_rate,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        pool_type=pool_type,
        use_batchnorm=use_batchnorm,
        cnn_dropout_rate=cnn_dropout_rate,
    ).to(DEVICE)

    optimizer_map = {
        'adamw': optim.AdamW,
        'nadam': optim.NAdam,
    }
    optimizer = optimizer_map[optimizer_name](model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # LR-Scheduler
    if lr_schedule == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=OPTUNA_EPOCHS,
            steps_per_epoch=len(train_loader),
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
        )

    best_val_loss = 1e10
    patience_ct = 0


    for epoch in range(OPTUNA_EPOCHS):
        train_one_epoch(
            model,
            train_loader,
            criterion, 
            optimizer, 
            regularizer,
            scheduler = scheduler if lr_schedule== 'onecycle' else None,
            lr_schedule = lr_schedule
            )
        
        metrics = evaluate(model, val_loader, criterion)
        val_acc = metrics['accuracy']
        val_loss = metrics['loss']

        if lr_schedule == 'performance':
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= OPTUNA_PATIENCE:
                break

        trial.report(val_loss, epoch)

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
conv_channels = (32, 64, 128, 256, 256)

best_model = SimpleCNN(
    n_fc_layers=p['n_fc_layers'],
    fc_hidden_size=p['fc_hidden_size'],
    activation=p['activation'],
    dropout_rate=p['dropout_rate'],
    conv_channels=conv_channels,
    kernel_size=p['kernel_size'],
    pool_type=p['pool_type'],
    use_batchnorm=p['use_batchnorm'],
    cnn_dropout_rate=p['cnn_dropout_rate'],
).to(DEVICE)


optimizer_map = {
    'adamw': optim.AdamW,
    'nadam': optim.NAdam,
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
best_val_acc = -1.0
patience_ct = 0

train_dataset = train_loader.dataset

for epoch in range(FINAL_TRAIN_EPOCHS):
    current_bs = get_progressive_batch_size(epoch, FINAL_TRAIN_EPOCHS)

    train_loader_epoch = DataLoader(
        train_dataset,
        batch_size=current_bs,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    tr_loss, tr_acc = train_one_epoch(
        best_model,
        train_loader_epoch,
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
        f'Epoch {epoch:3d}/{FINAL_TRAIN_EPOCHS}  bs: {current_bs:3d}  '
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