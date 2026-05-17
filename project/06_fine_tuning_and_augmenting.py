"""Fine-tuned ein vortrainiertes ResNet mit klassenabhängiger On-the-fly-Augmentierung."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from training_functions import save_best_model_artifacts


# =========================
# Konfiguration
# =========================
SEED = 77
IMAGE_SIZE = 224
BATCH_SIZE = 256
NUM_WORKERS = 24
NUM_EPOCHS = 100
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4

DATA_ROOT = Path('data')
TRAIN_DIR = DATA_ROOT / 'train'
VAL_DIR = DATA_ROOT / 'val'
TEST_DIR = DATA_ROOT / 'test'

MODEL_NAME = 'resnet18_A_Finetune_OnTheFlyAug_class_weights_true'
MODEL_DIR = Path('trained_models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODEL_DIR / 'resnet18_A_finetune_onthefly_checkpoint.pt'
BEST_MODEL_PATH = MODEL_DIR / 'resnet18_A_finetune_onthefly_best.pt'


# =========================
# Klassen-Ungleichgewicht
# =========================
# Oversampling
CLASS_REPEAT_FACTORS: dict[str, int] = {
    'y': 26,
    'n': 8,
}

# Augmentierung
CLASS_AUGMENTATION_CONFIG = {
    'y': {
        'apply_prob': 1.0,
        'hflip_prob': 0.35,
        'rotation_deg': 8,
        'perspective_prob': 0.30,
        'affine_prob': 0.30,
        'affine_deg': 6,
        'affine_translate': (0.08, 0.08),
        'affine_scale': (0.9, 1.1),
        'blur_prob': 0.20,
        'color_jitter': (0.25, 0.25, 0.25, 0.08),
        'grayscale_prob': 0.08,
        'autocontrast_prob': 0.10,
        'equalize_prob': 0.05,
        'sharpness_prob': 0.10,
        'sharpness_factor': 1.8,
        'solarize_prob': 0.04,
        'posterize_prob': 0.04,
        'posterize_bits': 4,
        'randaugment_prob': 0.20,
        'randaugment_num_ops': 2,
        'randaugment_magnitude': 6,
    },
    'n': {
        'apply_prob': 0.9,
        'hflip_prob': 0.25,
        'rotation_deg': 6,
        'perspective_prob': 0.20,
        'affine_prob': 0.20,
        'affine_deg': 4,
        'affine_translate': (0.05, 0.05),
        'affine_scale': (0.95, 1.05),
        'blur_prob': 0.12,
        'color_jitter': (0.18, 0.18, 0.18, 0.06),
        'grayscale_prob': 0.05,
        'autocontrast_prob': 0.06,
        'equalize_prob': 0.04,
        'sharpness_prob': 0.06,
        'sharpness_factor': 1.5,
        'randaugment_prob': 0.10,
        'randaugment_num_ops': 2,
        'randaugment_magnitude': 5,
    },
}


# 3) Optionale klassengewichtete Loss-Funktion.
#    Falls True: Klassen mit weniger ORIGINAL-Samples erhalten höheres Gewicht.
USE_CLASS_WEIGHTS = True


def set_seed(seed: int) -> None:
    """Setzt Zufallsseeds für reproduzierbares Training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Wählt das beste verfügbare Rechengerät (CUDA, MPS oder CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_class_aug_transform(cfg: dict) -> transforms.Compose | None:
    ops: list[nn.Module] = []

    if float(cfg.get('hflip_prob', 0.0)) > 0:
        ops.append(transforms.RandomHorizontalFlip(p=float(cfg['hflip_prob'])))

    if float(cfg.get('vflip_prob', 0.0)) > 0:
        ops.append(transforms.RandomVerticalFlip(p=float(cfg['vflip_prob'])))

    if float(cfg.get('rotation_deg', 0.0)) > 0:
        ops.append(transforms.RandomRotation(degrees=float(cfg['rotation_deg'])))

    if float(cfg.get('perspective_prob', 0.0)) > 0:
        ops.append(transforms.RandomPerspective(
            distortion_scale=float(cfg.get('distortion_scale', 0.35)),
            p=float(cfg['perspective_prob']),
        ))

    if float(cfg.get('affine_prob', 0.0)) > 0:
        ops.append(transforms.RandomApply([transforms.RandomAffine(
            degrees=float(cfg.get('affine_deg', 0.0)),
            translate=cfg.get('affine_translate', (0.05, 0.05)),
            scale=cfg.get('affine_scale', (0.95, 1.05)),
            shear=cfg.get('affine_shear', 4.0),
        )], p=float(cfg['affine_prob'])))

    if float(cfg.get('blur_prob', 0.0)) > 0:
        ops.append(transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
            p=float(cfg['blur_prob']),
        ))

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
        ops.append(transforms.RandomAdjustSharpness(
            sharpness_factor=float(cfg.get('sharpness_factor', 2.0)),
            p=float(cfg['sharpness_prob']),
        ))

    if float(cfg.get('posterize_prob', 0.0)) > 0:
        ops.append(transforms.RandomPosterize(
            bits=int(cfg.get('posterize_bits', 4)),
            p=float(cfg['posterize_prob']),
        ))

    if float(cfg.get('solarize_prob', 0.0)) > 0:
        ops.append(transforms.RandomSolarize(
            threshold=float(cfg.get('solarize_threshold', 128)),
            p=float(cfg['solarize_prob']),
        ))

    if float(cfg.get('randaugment_prob', 0.0)) > 0:
        ops.append(transforms.RandomApply([transforms.RandAugment(
            num_ops=int(cfg.get('randaugment_num_ops', 2)),
            magnitude=int(cfg.get('randaugment_magnitude', 7)),
        )], p=float(cfg['randaugment_prob'])))

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
        """Initialisiert Basisdaten, Wiederholungsfaktoren und Augmentierungsregeln."""
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

        # Virtuelles Oversampling durch Wiederholen von Samples.
        self.samples: list[tuple[str, int]] = []
        for path, target in self.base_dataset.samples:
            cls_name = self.classes[target]
            repeat = max(1, int(class_repeat_factors.get(cls_name, 1)))
            self.samples.extend([(path, target)] * repeat)

        self.effective_class_counts: Counter[int] = Counter(target for _, target in self.samples)

    def __len__(self) -> int:
        """Liefert die effektive Sample-Anzahl nach Wiederholungen."""
        return len(self.samples)

    def __getitem__(self, index: int):
        """Lädt ein Bild, führt ggf. klassenabhängige Augmentierung aus und normalisiert."""
        path, target = self.samples[index]
        image = self.base_dataset.loader(path)

        class_name = self.classes[target]
        aug_transform = self.class_aug_transforms[class_name]
        apply_prob = self.class_apply_prob[class_name]

        if aug_transform is not None and random.random() < apply_prob:
            image = aug_transform(image)

        image = self.base_transform(image)
        return image, target


def build_dataloaders() -> tuple[dict[str, DataLoader], int, dict[str, int], ClassAwareAugmentedDataset]:
    """Erzeugt DataLoader für Train/Val/Test inklusive On-the-fly-Augmentierung für Train."""
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

    loaders = {
        'train': DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        ),
        'val': DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        ),
        'test': DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    return loaders, len(train_ds.classes), train_ds.class_to_idx, train_ds


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



def build_class_weights(dataset: ClassAwareAugmentedDataset, device: torch.device) -> torch.Tensor:
    """Berechnet inverse Klassengewichte basierend auf den ORIGINAL-Trainingsdaten."""
    n_classes = len(dataset.classes)
    counts = np.array([dataset.original_class_counts.get(i, 1) for i in range(n_classes)], dtype=np.float32)
    inv = 1.0 / np.clip(counts, 1.0, None)
    weights = inv / inv.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    """Evaluiert Loss und Accuracy auf einem DataLoader."""
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

    return {'loss': total_loss / n, 'accuracy': correct / n}


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Sammelt True-Labels und Vorhersagen für den gesamten Loader."""
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
    """Trainiert das Modell für eine Epoche und gibt mittleren Loss/Accuracy zurück."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    pbar = tqdm(loader, desc='Train', leave=False)
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

    return {'loss': total_loss / n, 'accuracy': correct / n}


def main() -> None:
    """Führt Training, Auswahl des besten Checkpoints und abschliessende Test-Evaluation aus."""
    set_seed(SEED)
    device = get_device()

    if not TRAIN_DIR.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            'Expected directories not found:\n'
            f'{TRAIN_DIR}\n{VAL_DIR}\n{TEST_DIR}'
        )

    loaders, num_classes, class_to_idx, train_ds = build_dataloaders()
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = build_model(num_classes=num_classes, device=device)

    if USE_CLASS_WEIGHTS:
        class_weights = build_class_weights(train_ds, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    start_epoch = 0
    best_val_acc = -float('inf')

    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = int(ckpt['epoch']) + 1
        best_val_acc = float(ckpt['best_val_acc'])

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        if 'history' in ckpt and isinstance(ckpt['history'], dict):
            history = ckpt['history']

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_metrics = train_one_epoch(model, loaders['train'], criterion, optimizer, device)
        val_metrics = evaluate(model, loaders['val'], criterion, device)

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        if val_metrics['accuracy'] >= best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(
                {
                    'epoch': epoch,
                    'best_val_acc': best_val_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'class_to_idx': class_to_idx,
                    'history': history,
                    'class_repeat_factors': CLASS_REPEAT_FACTORS,
                    'class_augmentation_config': CLASS_AUGMENTATION_CONFIG,
                },
                BEST_MODEL_PATH,
            )

        torch.save(
            {
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_to_idx': class_to_idx,
                'history': history,
                'class_repeat_factors': CLASS_REPEAT_FACTORS,
                'class_augmentation_config': CLASS_AUGMENTATION_CONFIG,
            },
            CHECKPOINT_PATH,
        )

        print(
            f'Epoch {epoch + 1}/{NUM_EPOCHS} | '
            f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['accuracy']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}, "
            f'best_val_acc={best_val_acc:.4f}'
        )

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f'Best checkpoint missing after training: {BEST_MODEL_PATH}. '
            'Check whether training loop ran at least one epoch.'
        )

    best_ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_metrics = evaluate(model, loaders['test'], criterion, device)
    y_true, y_pred = collect_predictions(model, loaders['test'], device)

    test_precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    test_recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    test_f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    test_precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    test_recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    test_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))

    original_counts = {
        train_ds.classes[i]: int(train_ds.original_class_counts.get(i, 0))
        for i in range(len(train_ds.classes))
    }
    effective_counts = {
        train_ds.classes[i]: int(train_ds.effective_class_counts.get(i, 0))
        for i in range(len(train_ds.classes))
    }

    final_results = {
        'test_loss': float(test_metrics['loss']),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_precision_weighted': float(test_precision_weighted),
        'test_recall_weighted': float(test_recall_weighted),
        'test_f1_weighted': float(test_f1_weighted),
        'test_precision_macro': float(test_precision_macro),
        'test_recall_macro': float(test_recall_macro),
        'test_f1_macro': float(test_f1_macro),
        'best_val_accuracy': float(best_ckpt['best_val_acc']),
        'best_epoch': int(best_ckpt['epoch'] + 1),
        'checkpoint_path': str(CHECKPOINT_PATH),
        'best_model_path': str(BEST_MODEL_PATH),
        'class_labels': class_labels,
        'confusion_matrix': cm.tolist(),
        'train_original_class_counts': original_counts,
        'train_effective_class_counts': effective_counts,
    }

    run_params = {
        'seed': SEED,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'train_dir': str(TRAIN_DIR),
        'class_repeat_factors': CLASS_REPEAT_FACTORS,
        'class_augmentation_config': CLASS_AUGMENTATION_CONFIG,
        'use_class_weights': USE_CLASS_WEIGHTS,
    }

    artifact_dir = save_best_model_artifacts(
        model=model,
        y_true=y_true,
        y_pred=y_pred,
        model_name=MODEL_NAME,
        score=test_metrics['accuracy'],
        params=run_params,
        history=history,
        base_dir=str(MODEL_DIR),
        results=final_results,
        class_labels=class_labels,
        save_history=True,
    )

    final_results['artifact_dir'] = str(artifact_dir)
    print(final_results)


if __name__ == '__main__':
    main()
