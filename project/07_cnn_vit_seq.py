"""Trainiert ein sequentielles CNN->ViT-Hybridmodell für binäre Bildklassifikation.

Die Pipeline lädt ein bereits optimiertes SimpleCNN-Checkpoint, friert dessen
Convolution-Feature-Extractor ein und trainiert darauf einen Transformer-Head.
"""

from __future__ import annotations

import json
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from models import SimpleCNN
from training_functions import save_best_model_artifacts


# =========================
# Konfiguration
# =========================
SEED = 77
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 16
NUM_EPOCHS = 200
LEARNING_RATE = 0.0002443855945462861
WEIGHT_DECAY = 0.0009819663619102651
USE_AMP = True
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 55

ENFORCE_BINARY_CLASSIFICATION = True
EXPECTED_NUM_CLASSES = 2

DATA_ROOT = Path('data')
TRAIN_DIR = DATA_ROOT / 'train'
VAL_DIR = DATA_ROOT / 'val'
TEST_DIR = DATA_ROOT / 'test'

MODEL_NAME = 'Last_model_VITCNN512'
MODEL_DIR = Path('trained_models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODEL_DIR / 'Last_model_VITCNN512_checkpoint.pt'
BEST_MODEL_PATH = MODEL_DIR / 'Last_model_VITCNN512_best.pt'

# Optional explizit setzen. Mögliche Werte:
# 1) Datei mit {'model_state_dict': ...} oder direktem state_dict
# 2) Artefakt-Ordner mit model_state_dict.pt + metadata.json
PRETRAINED_CNN_SOURCE: Path | None = (
    MODEL_DIR / 'CNN_512"_score-0.9942_20260517_100438'  # Ordner mit model_state_dict.pt
)

PRETRAINED_CNN_METADATA_PATH: Path | None = (
    MODEL_DIR / 'CNN_512"_score-0.9942_20260517_100438' / 'metadata.json'
)

# Falls Metadata/Checkpoint unvollständig sind, können Architekturwerte hier
# explizit überschrieben werden.
CNN_HPARAMS_OVERRIDE: dict[str, Any] = {}

# CNN-Backbone standardmäßig komplett einfrieren. Optional können letzte
# Convolution-Blöcke wieder freigegeben werden.
FREEZE_CNN_BACKBONE = True
UNFREEZE_LAST_CONV_BLOCKS = 1

# Transformer-Konfiguration
VIT_EMBED_DIM = 512
VIT_NUM_HEADS = 16
VIT_DEPTH = 2
VIT_MLP_RATIO = 3.0
VIT_DROPOUT = 0.0

# Klassen-Ungleichgewicht und On-the-fly-Augs analog zur Optuna-CNN-Pipeline,
# aus der CNN_score-0.9924_20260426_081737 stammt.
CLASS_REPEAT_FACTORS: dict[str, int] = {
    'y': 26,
    'n': 8,
}

CLASS_AUGMENTATION_CONFIG = {
    "y": {
        "apply_prob": 1.0,

        # Horizontal-Flip (p=0.5)
        "hflip_prob": 0.5,

        # Vertical-Flip (p=0.5)
        "vflip_prob": 0.5,

        # Rotation (limit=30, p=0.5)
        "rotation_deg": 30,
        "rotation_prob": 0.5,

        # Median-Blur (limit=7, p=0.3)
        "median_blur_limit": 7,
        "median_blur_prob": 0.3,

        # Gaussian-Noise (var_limit=0.38, p=0.5)
        "gaussian_noise_var_limit": 0.38,
        "gaussian_noise_prob": 0.5,

        # Hue-Saturation-Value (h/s/v shift=10, p=0.3)
        "hue_shift_limit": 10,
        "sat_shift_limit": 10,
        "val_shift_limit": 10,
        "hsv_prob": 0.3,

        # Random-Brightness-Contrast (0.2 / 0.2, p=0.3)
        "brightness_limit": (0.2, 0.2),
        "contrast_limit": (0.2, 0.2),
        "brightness_contrast_prob": 0.3,

        # Cutout (max_h=20, max_w=20, holes=5, p=0.5)
        "cutout_max_height": 20,
        "cutout_max_width": 20,
        "cutout_num_holes": 5,
        "cutout_prob": 0.5,
    },
    "n": {
        "apply_prob": 1.0,
        "hflip_prob": 0.5,
        "vflip_prob": 0.5,
        "rotation_deg": 30,
        "rotation_prob": 0.5,
        "median_blur_limit": 7,
        "median_blur_prob": 0.3,
        "gaussian_noise_var_limit": 0.38,
        "gaussian_noise_prob": 0.5,
        "hue_shift_limit": 10,
        "sat_shift_limit": 10,
        "val_shift_limit": 10,
        "hsv_prob": 0.3,
        "brightness_limit": (0.2, 0.2),
        "contrast_limit": (0.2, 0.2),
        "brightness_contrast_prob": 0.3,
        "cutout_max_height": 20,
        "cutout_max_width": 20,
        "cutout_num_holes": 5,
        "cutout_prob": 0.5,
    },
}



class LoadedSimpleCNN(NamedTuple):
    """Container für geladenes CNN samt aufgelöster Konfiguration."""

    model: nn.Module
    source_path: Path
    state_dict_path: Path
    resolved_hparams: dict[str, Any]


class LegacySimpleCNN(nn.Module):
    """Kompatibilitätsklasse für ältere 3-Conv-SimpleCNN-Checkpoints."""

    def __init__(
        self,
        input_size=None,
        n_layers: int = 1,
        layer_sizes: list[int] | None = None,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        num_classes: int = 10,
        conv_channels: tuple[int, int, int] = (32, 64, 128),
        kernel_size: int = 3,
        pool_type: str = 'max',
        use_batchnorm: bool = False,
        cnn_dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
        }
        if activation not in activation_map:
            raise ValueError(f'Unknown legacy activation: {activation}')
        if len(conv_channels) != 3:
            raise ValueError('Legacy conv_channels must contain exactly 3 values.')
        if kernel_size not in (3, 5):
            raise ValueError('Legacy kernel_size must be 3 or 5.')
        if pool_type not in ('max', 'avg'):
            raise ValueError("Legacy pool_type must be 'max' or 'avg'.")

        _ = input_size
        if layer_sizes is None:
            layer_sizes = [128] * max(0, n_layers)
        if len(layer_sizes) != n_layers:
            raise ValueError('Legacy n_layers must match len(layer_sizes).')

        activation_cls = activation_map[activation]
        padding = kernel_size // 2

        def make_activation(inplace: bool = False) -> nn.Module:
            if activation == 'relu':
                return activation_cls(inplace=inplace)
            return activation_cls()

        def pool_layer() -> nn.Module:
            if pool_type == 'max':
                return nn.MaxPool2d(kernel_size=2)
            return nn.AvgPool2d(kernel_size=2)

        feature_layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in conv_channels:
            feature_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            if use_batchnorm:
                feature_layers.append(nn.BatchNorm2d(out_channels))
            feature_layers.append(make_activation(inplace=True))
            feature_layers.append(pool_layer())
            if cnn_dropout_rate > 0.0:
                feature_layers.append(nn.Dropout2d(cnn_dropout_rate))
            in_channels = out_channels

        feature_layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ])
        self.features = nn.Sequential(*feature_layers)

        head_layers: list[nn.Module] = []
        in_features = conv_channels[-1]
        for hidden_size in layer_sizes:
            head_layers.append(nn.Linear(in_features, hidden_size))
            head_layers.append(make_activation())
            if dropout_rate > 0.0:
                head_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size

        head_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNViTHybridSequential(nn.Module):
    """Verknüpft gefrorene CNN-Feature-Maps sequentiell mit einem Transformer-Encoder."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_channels: int,
        feature_grid_size: tuple[int, int],
        num_classes: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_grid_size = feature_grid_size
        self.embed_dim = embed_dim

        num_patches = int(feature_grid_size[0] * feature_grid_size[1])
        if num_patches <= 0:
            raise ValueError(f'Invalid feature grid size: {feature_grid_size}')

        self.token_projection = nn.Linear(feature_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.positional_dropout = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _interpolate_positional_embedding(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Interpoliert Positions-Embeddings, falls Eingabegröße variiert."""
        base_h, base_w = self.feature_grid_size
        if grid_h == base_h and grid_w == base_w:
            return self.positional_embedding

        cls_pos = self.positional_embedding[:, :1, :]
        patch_pos = self.positional_embedding[:, 1:, :]
        patch_pos = patch_pos.transpose(1, 2).reshape(1, self.embed_dim, base_h, base_w)
        patch_pos = F.interpolate(
            patch_pos,
            size=(grid_h, grid_w),
            mode='bicubic',
            align_corners=False,
        )
        patch_pos = patch_pos.flatten(2).transpose(1, 2)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = self.feature_extractor(x)
        if feature_maps.ndim != 4:
            raise RuntimeError(
                f'Expected 4D feature maps from CNN backbone, got shape {tuple(feature_maps.shape)}'
            )

        batch_size, _, grid_h, grid_w = feature_maps.shape

        # [B, C, H, W] -> [B, H*W, C] -> Projektion in Transformer-Embedding-Dim.
        tokens = feature_maps.flatten(2).transpose(1, 2)
        tokens = self.token_projection(tokens)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)

        pos = self._interpolate_positional_embedding(grid_h, grid_w)
        tokens = self.positional_dropout(tokens + pos)

        encoded = self.transformer(tokens)
        cls_out = self.norm(encoded[:, 0])
        return self.head(cls_out)


def set_seed(seed: int) -> None:
    """Setzt Seeds für reproduzierbares Verhalten."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Wählt das beste verfügbare Rechengerät."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def resolve_train_dir() -> Path:
    """Validiert den festen Trainingsordner auf der Serverstruktur."""
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f'Training directory not found: {TRAIN_DIR}')
    return TRAIN_DIR


def make_loader_kwargs() -> dict[str, Any]:
    """Erzeugt DataLoader-Argumente mit sinnvollen Defaults für CPU/GPU."""
    kwargs: dict[str, Any] = {
        'num_workers': NUM_WORKERS,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': NUM_WORKERS > 0,
    }
    if NUM_WORKERS > 0:
        kwargs['prefetch_factor'] = 2
    return kwargs


def build_class_aug_transform(cfg: dict[str, Any]) -> transforms.Compose | None:
    """Erzeugt aus Klassenkonfiguration die On-the-fly-Augmentierungs-Pipeline."""
    ops: list[nn.Module] = []

    hflip_prob = float(cfg.get('hflip_prob', 0.0))
    if hflip_prob > 0.0:
        ops.append(transforms.RandomHorizontalFlip(p=hflip_prob))

    vflip_prob = float(cfg.get('vflip_prob', 0.0))
    if vflip_prob > 0.0:
        ops.append(transforms.RandomVerticalFlip(p=vflip_prob))

    rotation_deg = float(cfg.get('rotation_deg', 0.0))
    if rotation_deg > 0.0:
        ops.append(transforms.RandomRotation(degrees=rotation_deg))

    perspective_prob = float(cfg.get('perspective_prob', 0.0))
    if perspective_prob > 0.0:
        ops.append(
            transforms.RandomPerspective(
                distortion_scale=0.35,
                p=perspective_prob,
            )
        )

    affine_prob = float(cfg.get('affine_prob', 0.0))
    if affine_prob > 0.0:
        ops.append(
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=float(cfg.get('affine_deg', 0.0)),
                        translate=cfg.get('affine_translate', (0.05, 0.05)),
                        scale=cfg.get('affine_scale', (0.95, 1.05)),
                        shear=cfg.get('affine_shear', 4.0),
                    )
                ],
                p=affine_prob,
            )
        )

    blur_prob = float(cfg.get('blur_prob', 0.0))
    if blur_prob > 0.0:
        ops.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                p=blur_prob,
            )
        )

    color_jitter = cfg.get('color_jitter')
    if color_jitter is not None:
        brightness, contrast, saturation, hue = color_jitter
        ops.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )

    grayscale_prob = float(cfg.get('grayscale_prob', 0.0))
    if grayscale_prob > 0.0:
        ops.append(transforms.RandomGrayscale(p=grayscale_prob))

    autocontrast_prob = float(cfg.get('autocontrast_prob', 0.0))
    if autocontrast_prob > 0.0:
        ops.append(transforms.RandomAutocontrast(p=autocontrast_prob))

    equalize_prob = float(cfg.get('equalize_prob', 0.0))
    if equalize_prob > 0.0:
        ops.append(transforms.RandomEqualize(p=equalize_prob))

    sharpness_prob = float(cfg.get('sharpness_prob', 0.0))
    if sharpness_prob > 0.0:
        ops.append(
            transforms.RandomAdjustSharpness(
                sharpness_factor=float(cfg.get('sharpness_factor', 2.0)),
                p=sharpness_prob,
            )
        )

    posterize_prob = float(cfg.get('posterize_prob', 0.0))
    if posterize_prob > 0.0:
        ops.append(
            transforms.RandomPosterize(
                bits=int(cfg.get('posterize_bits', 4)),
                p=posterize_prob,
            )
        )

    solarize_prob = float(cfg.get('solarize_prob', 0.0))
    if solarize_prob > 0.0:
        ops.append(
            transforms.RandomSolarize(
                threshold=float(cfg.get('solarize_threshold', 128.0)),
                p=solarize_prob,
            )
        )

    randaugment_prob = float(cfg.get('randaugment_prob', 0.0))
    if randaugment_prob > 0.0:
        ops.append(
            transforms.RandomApply(
                [
                    transforms.RandAugment(
                        num_ops=int(cfg.get('randaugment_num_ops', 2)),
                        magnitude=int(cfg.get('randaugment_magnitude', 7)),
                    )
                ],
                p=randaugment_prob,
            )
        )

    if not ops:
        return None
    return transforms.Compose(ops)


class ClassAwareAugmentedDataset(Dataset):
    """ImageFolder-Wrapper mit klassenabhängiger On-the-fly-Augmentierung."""

    def __init__(
        self,
        root: Path,
        base_transform: transforms.Compose,
        class_aug_config: dict[str, dict[str, Any]],
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

        # Virtuelles Oversampling via Wiederholung von Sample-Referenzen.
        self.samples: list[tuple[str, int]] = []
        for path, target in self.base_dataset.samples:
            cls_name = self.classes[target]
            repeat = max(1, int(class_repeat_factors.get(cls_name, 1)))
            self.samples.extend([(path, target)] * repeat)

        self.effective_class_counts: Counter[int] = Counter(target for _, target in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        image = self.base_dataset.loader(path)

        class_name = self.classes[target]
        aug_transform = self.class_aug_transforms[class_name]
        apply_prob = self.class_apply_prob[class_name]

        if aug_transform is not None and random.random() < apply_prob:
            image = aug_transform(image)

        image = self.base_transform(image)
        return image, target


def build_dataloaders(train_dir: Path) -> tuple[dict[str, DataLoader], int, dict[str, int]]:
    """Erzeugt DataLoader für Train/Val/Test."""
    if not train_dir.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            'Expected directories not found:\n'
            f'{train_dir}\n{VAL_DIR}\n{TEST_DIR}'
        )

    # Gleiches Normalisierungs-Schema wie in der Optuna-CNN-Pipeline.
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
        root=train_dir,
        base_transform=train_base_tfms,
        class_aug_config=CLASS_AUGMENTATION_CONFIG,
        class_repeat_factors=CLASS_REPEAT_FACTORS,
    )
    val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tfms)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tfms)
    if val_ds.class_to_idx != train_ds.class_to_idx or test_ds.class_to_idx != train_ds.class_to_idx:
        raise ValueError('Train/Val/Test class_to_idx mismatch.')

    loader_kwargs = make_loader_kwargs()
    loaders = {
        'train': DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs),
        'val': DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs),
        'test': DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs),
    }
    return loaders, len(train_ds.classes), train_ds.class_to_idx


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Entfernt optionales 'module.' Präfix (z. B. aus DataParallel-Checkpoints)."""
    if any(k.startswith('module.') for k in state_dict):
        return {k.removeprefix('module.'): v for k, v in state_dict.items()}
    return state_dict


def extract_state_dict_and_payload(raw_checkpoint: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Extrahiert state_dict aus unterschiedlichen Checkpoint-Formaten."""
    if isinstance(raw_checkpoint, dict) and 'model_state_dict' in raw_checkpoint:
        state = raw_checkpoint['model_state_dict']
        if not isinstance(state, dict):
            raise ValueError('Checkpoint key "model_state_dict" is not a dict.')
        return state, raw_checkpoint

    if isinstance(raw_checkpoint, dict):
        # Direkt gespeichertes state_dict: alle Werte sind Tensoren.
        if raw_checkpoint and all(isinstance(v, torch.Tensor) for v in raw_checkpoint.values()):
            return raw_checkpoint, {}
        raise ValueError('Unsupported dict checkpoint format.')

    raise ValueError(f'Unsupported checkpoint type: {type(raw_checkpoint)}')


def load_metadata_params(metadata_path: Path) -> dict[str, Any]:
    """Lädt optionale Hyperparameter aus metadata.json."""
    if not metadata_path.exists():
        return {}

    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        warnings.warn(f'Could not parse metadata JSON at {metadata_path}: {exc}')
        return {}

    params = metadata.get('params', {})
    if isinstance(params, dict):
        return params
    return {}


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'true', '1', 'yes', 'y'}:
            return True
        if lowered in {'false', '0', 'no', 'n'}:
            return False
    return default


def infer_simplecnn_hparams_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Leitet CNN-Hyperparameter robust aus state_dict ab (aktuell + Legacy)."""
    conv_entries: list[tuple[int, tuple[int, ...]]] = []
    linear_entries: list[tuple[int, tuple[int, ...]]] = []
    has_batchnorm = False

    for key, tensor in state_dict.items():
        parts = key.split('.')
        if len(parts) == 3 and parts[0] == 'features' and parts[2] == 'weight' and tensor.ndim == 4:
            conv_entries.append((int(parts[1]), tuple(tensor.shape)))
            continue
        if len(parts) == 3 and parts[0] == 'classifier' and parts[2] == 'weight' and tensor.ndim == 2:
            linear_entries.append((int(parts[1]), tuple(tensor.shape)))
            continue
        if len(parts) == 3 and parts[0] == 'features' and parts[2] == 'running_mean':
            has_batchnorm = True

    if not conv_entries:
        raise ValueError('Could not infer convolutional layers from state_dict.')
    if not linear_entries:
        raise ValueError('Could not infer classifier layers from state_dict.')

    conv_entries.sort(key=lambda x: x[0])
    linear_entries.sort(key=lambda x: x[0])

    conv_indices = [idx for idx, _ in conv_entries]
    linear_indices = [idx for idx, _ in linear_entries]
    conv_channels = tuple(int(shape[0]) for _, shape in conv_entries)
    kernel_size = int(conv_entries[0][1][-1])
    n_linear_layers = len(linear_entries)
    num_classes = int(linear_entries[-1][1][0])

    if len(conv_channels) == 5:
        n_fc_layers = max(1, n_linear_layers - 1)
        fc_hidden_size = int(linear_entries[0][1][0])
        return {
            'model_variant': 'current',
            'n_fc_layers': n_fc_layers,
            'fc_hidden_size': fc_hidden_size,
            'activation': 'relu',
            'dropout_rate': 0.0,
            'conv_channels': conv_channels,
            'kernel_size': kernel_size,
            'pool_type': 'max',
            'use_batchnorm': has_batchnorm,
            'cnn_dropout_rate': 0.0,
            'num_classes': num_classes,
        }

    if len(conv_channels) == 3:
        n_layers = max(0, n_linear_layers - 1)
        layer_sizes = [int(shape[0]) for _, shape in linear_entries[:-1]]

        # Dropoutwerte sind nicht direkt im state_dict enthalten;
        # die Schrittweite der Layer-Indizes liefert eine robuste Annäherung.
        cnn_dropout_rate = 0.0
        if len(conv_indices) >= 2:
            observed_step = conv_indices[1] - conv_indices[0]
            expected_step_without_dropout = 4 if has_batchnorm else 3
            if observed_step > expected_step_without_dropout:
                cnn_dropout_rate = 0.1

        dropout_rate = 0.0
        if n_layers >= 1 and len(linear_indices) >= 2:
            observed_step = linear_indices[1] - linear_indices[0]
            if observed_step > 2:
                dropout_rate = 0.1

        return {
            'model_variant': 'legacy',
            'n_layers': n_layers,
            'layer_sizes': layer_sizes,
            'activation': 'relu',
            'dropout_rate': dropout_rate,
            'conv_channels': conv_channels,
            'kernel_size': kernel_size,
            'pool_type': 'max',
            'use_batchnorm': has_batchnorm,
            'cnn_dropout_rate': cnn_dropout_rate,
            'num_classes': num_classes,
        }

    raise ValueError(
        'Unsupported SimpleCNN variant inferred from checkpoint: '
        f'found {len(conv_channels)} convolutional blocks, expected 3 or 5.'
    )


def resolve_simplecnn_init_hparams(
    inferred: dict[str, Any],
    hints: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Kombiniert inferierte Werte, Checkpoint-Hints und Overrides (aktuell + Legacy)."""
    merged = dict(inferred)
    merged.update(hints)
    merged.update(overrides)

    model_variant = str(merged.get('model_variant', inferred.get('model_variant', 'current'))).lower()
    if model_variant not in {'current', 'legacy'}:
        warnings.warn(f'Unsupported model_variant "{model_variant}"; fallback to inferred value.')
        model_variant = str(inferred.get('model_variant', 'current')).lower()

    pool_type = str(merged.get('pool_type', inferred.get('pool_type', 'max'))).lower()
    if pool_type not in {'max', 'avg'}:
        warnings.warn(f'Unsupported pool_type "{pool_type}" in hints; fallback to max.')
        pool_type = 'max'

    if model_variant == 'current':
        if all(
            k in merged
            for k in ('conv_channels_1', 'conv_channels_2', 'conv_channels_3', 'conv_channels_4', 'conv_channels_5')
        ):
            conv_channels = (
                int(merged['conv_channels_1']),
                int(merged['conv_channels_2']),
                int(merged['conv_channels_3']),
                int(merged['conv_channels_4']),
                int(merged['conv_channels_5']),
            )
        else:
            raw_conv_channels = merged.get('conv_channels', inferred.get('conv_channels'))
            if not isinstance(raw_conv_channels, (list, tuple)) or len(raw_conv_channels) != 5:
                raise ValueError(f'Invalid conv_channels for current model: {raw_conv_channels}')
            conv_channels = tuple(int(v) for v in raw_conv_channels)

        activation = str(merged.get('activation', inferred.get('activation', 'relu'))).lower()
        if activation not in {'relu', 'elu', 'swish'}:
            warnings.warn(f'Unsupported activation "{activation}" in hints; fallback to relu.')
            activation = 'relu'

        return {
            'model_variant': 'current',
            'n_fc_layers': int(merged.get('n_fc_layers', inferred.get('n_fc_layers', 3))),
            'fc_hidden_size': int(merged.get('fc_hidden_size', inferred.get('fc_hidden_size', 128))),
            'activation': activation,
            'dropout_rate': safe_float(merged.get('dropout_rate', inferred.get('dropout_rate', 0.0)), 0.0),
            'num_classes': int(inferred['num_classes']),
            'conv_channels': conv_channels,
            'kernel_size': int(merged.get('kernel_size', inferred.get('kernel_size', 3))),
            'pool_type': pool_type,
            'use_batchnorm': safe_bool(
                merged.get('use_batchnorm', inferred.get('use_batchnorm', False)),
                bool(inferred.get('use_batchnorm', False)),
            ),
            'cnn_dropout_rate': safe_float(
                merged.get('cnn_dropout_rate', inferred.get('cnn_dropout_rate', 0.0)),
                0.0,
            ),
        }

    # Legacy (3-Conv) fallback.
    if all(k in merged for k in ('conv_channels_1', 'conv_channels_2', 'conv_channels_3')):
        conv_channels = (
            int(merged['conv_channels_1']),
            int(merged['conv_channels_2']),
            int(merged['conv_channels_3']),
        )
    else:
        raw_conv_channels = merged.get('conv_channels', inferred.get('conv_channels'))
        if not isinstance(raw_conv_channels, (list, tuple)) or len(raw_conv_channels) != 3:
            raise ValueError(f'Invalid conv_channels for legacy model: {raw_conv_channels}')
        conv_channels = tuple(int(v) for v in raw_conv_channels)

    n_layers = int(merged.get('n_layers', inferred.get('n_layers', 1)))
    layer_sizes: list[int] = []
    if isinstance(merged.get('layer_sizes'), (list, tuple)):
        layer_sizes = [int(v) for v in merged['layer_sizes']]

    if not layer_sizes:
        nodes_from_optuna = []
        for i in range(max(0, n_layers)):
            key = f'n_nodes_layer_{i}'
            if key in merged:
                nodes_from_optuna.append(int(merged[key]))
        layer_sizes = nodes_from_optuna

    if len(layer_sizes) != n_layers:
        inferred_layer_sizes = inferred.get('layer_sizes', [])
        if inferred_layer_sizes is None:
            inferred_layer_sizes = []
        elif not isinstance(inferred_layer_sizes, list):
            inferred_layer_sizes = list(inferred_layer_sizes)
        layer_sizes = [int(v) for v in inferred_layer_sizes]
        n_layers = len(layer_sizes)

    activation = str(merged.get('activation', inferred.get('activation', 'relu'))).lower()
    if activation not in {'relu', 'tanh', 'sigmoid'}:
        warnings.warn(f'Unsupported legacy activation "{activation}" in hints; fallback to relu.')
        activation = 'relu'

    return {
        'model_variant': 'legacy',
        'n_layers': n_layers,
        'layer_sizes': layer_sizes,
        'activation': activation,
        'dropout_rate': safe_float(merged.get('dropout_rate', inferred.get('dropout_rate', 0.0)), 0.0),
        'num_classes': int(inferred['num_classes']),
        'conv_channels': conv_channels,
        'kernel_size': int(merged.get('kernel_size', inferred.get('kernel_size', 3))),
        'pool_type': pool_type,
        'use_batchnorm': safe_bool(
            merged.get('use_batchnorm', inferred.get('use_batchnorm', False)),
            bool(inferred.get('use_batchnorm', False)),
        ),
        'cnn_dropout_rate': safe_float(
            merged.get('cnn_dropout_rate', inferred.get('cnn_dropout_rate', 0.0)),
            0.0,
        ),
    }


def resolve_pretrained_cnn_source(explicit_source: Path | None) -> Path:
    """Ermittelt den besten verfügbaren Pfad für das vortrainierte SimpleCNN."""
    if explicit_source is not None:
        if explicit_source.exists():
            return explicit_source
        raise FileNotFoundError(f'Configured PRETRAINED_CNN_SOURCE does not exist: {explicit_source}')

    preferred_candidates = [
        #MODEL_DIR / 'Simple_CNN_2.pt',
        #MODEL_DIR / 'Simple_CNN.pt',
        #MODEL_DIR / 'CNN_score-0.9924_20260426_081737' / 'model_state_dict.pt',
        #MODEL_DIR / 'CNN_score-0.9924_20260426_081737',
        MODEL_DIR / 'CNN_4_Best_of_two"_score-0.9898_20260516_174615',

    ]
    for candidate in preferred_candidates:
        if candidate.is_file():
            return candidate
        if candidate.is_dir() and (candidate / 'model_state_dict.pt').exists():
            return candidate

    artifact_dirs = sorted(MODEL_DIR.glob('CNN_score-*'))
    for artifact_dir in reversed(artifact_dirs):
        if (artifact_dir / 'model_state_dict.pt').exists():
            return artifact_dir

    artifact_dirs = sorted(MODEL_DIR.glob('Simple_CNN_score-*'))
    for artifact_dir in reversed(artifact_dirs):
        if (artifact_dir / 'model_state_dict.pt').exists():
            return artifact_dir

    raise FileNotFoundError(
        'No pretrained SimpleCNN source found. '
        'Set PRETRAINED_CNN_SOURCE to a checkpoint file or artifact directory '
        '(e.g. trained_models/Simple_CNN_2.pt).'
    )


def load_pretrained_simplecnn(source: Path | None = None) -> LoadedSimpleCNN:
    """Lädt ein vortrainiertes SimpleCNN inkl. robuster Hyperparameter-Rekonstruktion."""
    resolved_source = resolve_pretrained_cnn_source(source)

    if resolved_source.is_dir():
        state_dict_path = resolved_source / 'model_state_dict.pt'
        metadata_path = resolved_source / 'metadata.json'
    elif resolved_source.is_file():
        state_dict_path = resolved_source
        metadata_path = resolved_source.parent / 'metadata.json'
    else:
        raise FileNotFoundError(f'Invalid SimpleCNN source path: {resolved_source}')

    if not state_dict_path.exists():
        raise FileNotFoundError(f'State dict file not found: {state_dict_path}')
    if not metadata_path.exists() and PRETRAINED_CNN_METADATA_PATH is not None:
        if PRETRAINED_CNN_METADATA_PATH.exists():
            metadata_path = PRETRAINED_CNN_METADATA_PATH

    raw_checkpoint = torch.load(state_dict_path, map_location='cpu')
    state_dict, payload = extract_state_dict_and_payload(raw_checkpoint)
    state_dict = strip_module_prefix(state_dict)

    hints: dict[str, Any] = {}
    if isinstance(payload.get('best_params'), dict):
        hints.update(payload['best_params'])
    if isinstance(payload.get('params'), dict):
        hints.update(payload['params'])

    # Metadata ist optional; falls vorhanden ergänzt es fehlende Felder.
    metadata_hints = load_metadata_params(metadata_path)
    for key, value in metadata_hints.items():
        hints.setdefault(key, value)

    inferred = infer_simplecnn_hparams_from_state_dict(state_dict)
    resolved_hparams = resolve_simplecnn_init_hparams(
        inferred=inferred,
        hints=hints,
        overrides=CNN_HPARAMS_OVERRIDE,
    )

    if resolved_hparams['model_variant'] == 'current':
        model = SimpleCNN(
            n_fc_layers=resolved_hparams['n_fc_layers'],
            fc_hidden_size=resolved_hparams['fc_hidden_size'],
            activation=resolved_hparams['activation'],
            dropout_rate=resolved_hparams['dropout_rate'],
            num_classes=resolved_hparams['num_classes'],
            conv_channels=resolved_hparams['conv_channels'],
            kernel_size=resolved_hparams['kernel_size'],
            pool_type=resolved_hparams['pool_type'],
            use_batchnorm=resolved_hparams['use_batchnorm'],
            cnn_dropout_rate=resolved_hparams['cnn_dropout_rate'],
        )
    else:
        model = LegacySimpleCNN(
            n_layers=resolved_hparams['n_layers'],
            layer_sizes=resolved_hparams['layer_sizes'],
            activation=resolved_hparams['activation'],
            dropout_rate=resolved_hparams['dropout_rate'],
            num_classes=resolved_hparams['num_classes'],
            conv_channels=resolved_hparams['conv_channels'],
            kernel_size=resolved_hparams['kernel_size'],
            pool_type=resolved_hparams['pool_type'],
            use_batchnorm=resolved_hparams['use_batchnorm'],
            cnn_dropout_rate=resolved_hparams['cnn_dropout_rate'],
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            'Failed to load SimpleCNN state_dict with inferred parameters. '
            f'Resolved hyperparameters: {resolved_hparams}'
        ) from exc

    return LoadedSimpleCNN(
        model=model,
        source_path=resolved_source,
        state_dict_path=state_dict_path,
        resolved_hparams=resolved_hparams,
    )


def extract_feature_extractor(cnn_model: nn.Module) -> nn.Sequential:
    """Extrahiert alle CNN-Feature-Layer bis vor AdaptiveAvgPool/Flatten."""
    modules = list(cnn_model.features.children())
    while modules and isinstance(modules[-1], (nn.Flatten, nn.AdaptiveAvgPool2d)):
        modules.pop()

    if not modules:
        raise ValueError('Could not extract convolutional feature extractor from SimpleCNN.')
    return nn.Sequential(*modules)


def freeze_module(module: nn.Module) -> None:
    """Setzt requires_grad=False für alle Parameter im Modul."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_last_conv_blocks(feature_extractor: nn.Sequential, n_blocks: int) -> None:
    """Optionales Feintuning: letzte n Conv-Blöcke im Backbone freigeben."""
    if n_blocks <= 0:
        return

    conv_indices = [i for i, m in enumerate(feature_extractor) if isinstance(m, nn.Conv2d)]
    if not conv_indices:
        return

    selected = conv_indices[-n_blocks:]
    for start_idx in selected:
        # Entsperrt Layer vom aktuellen Conv bis zum nächsten Conv-Layer.
        next_conv_candidates = [idx for idx in conv_indices if idx > start_idx]
        end_idx = next_conv_candidates[0] if next_conv_candidates else len(feature_extractor)
        for layer in feature_extractor[start_idx:end_idx]:
            for param in layer.parameters():
                param.requires_grad = True


@torch.no_grad()
def infer_feature_map_shape(feature_extractor: nn.Module, image_size: int) -> tuple[int, tuple[int, int]]:
    """Berechnet Kanäle und räumliche Größe der CNN-Feature-Maps per Dummy-Forward."""
    was_training = feature_extractor.training
    feature_extractor.eval()
    dummy = torch.zeros(1, 3, image_size, image_size)
    fmap = feature_extractor(dummy)
    if was_training:
        feature_extractor.train()

    if fmap.ndim != 4:
        raise ValueError(f'Expected 4D feature maps, got shape {tuple(fmap.shape)}')
    return int(fmap.shape[1]), (int(fmap.shape[2]), int(fmap.shape[3]))


def unwrap_model(model: nn.Module) -> nn.Module:
    """Gibt das Basismodell zurück (ohne DataParallel-Hülle)."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def maybe_wrap_data_parallel(model: nn.Module, device: torch.device) -> nn.Module:
    """Aktiviert Multi-GPU-Training via DataParallel, wenn mehrere CUDA-GPUs vorhanden sind."""
    if device.type == 'cuda':
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f'Using DataParallel on {n_gpus} GPUs.')
            return nn.DataParallel(model)
    return model


def move_optimizer_state_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    """Verschiebt Optimizer-State-Tensoren auf das gewünschte Gerät."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    """Evaluiert Loss und Accuracy auf einem Loader."""
    model.eval()
    total_loss, correct, n_samples = 0.0, 0, 0
    amp_enabled = USE_AMP and device.type == 'cuda'
    amp_device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    non_blocking = device.type == 'cuda'

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=non_blocking)
            yb = yb.to(device, non_blocking=non_blocking)

            with torch.autocast(device_type=amp_device_type, enabled=amp_enabled):
                logits = model(xb)
                loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            n_samples += xb.size(0)

    if n_samples == 0:
        raise ValueError('Received empty dataloader during evaluation.')

    return {
        'loss': total_loss / n_samples,
        'accuracy': correct / n_samples,
    }


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Sammelt True-Labels und Vorhersagen für einen kompletten Loader."""
    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []

    non_blocking = device.type == 'cuda'
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=non_blocking)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(yb.numpy().tolist())

    return np.array(all_targets), np.array(all_preds)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    """Trainiert das Modell für eine Epoche und liefert mittleren Loss/Accuracy."""
    model.train()
    total_loss, correct, n_samples = 0.0, 0, 0

    amp_enabled = USE_AMP and device.type == 'cuda'
    amp_device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    non_blocking = device.type == 'cuda'

    pbar = tqdm(loader, desc='Train', leave=False)
    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=non_blocking)
        yb = yb.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=amp_device_type, enabled=amp_enabled):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n_samples += xb.size(0)

        running_loss = total_loss / max(1, n_samples)
        running_acc = correct / max(1, n_samples)
        pbar.set_postfix(loss=running_loss, acc=running_acc)

    if n_samples == 0:
        raise ValueError('Received empty dataloader during training.')

    return {
        'loss': total_loss / n_samples,
        'accuracy': correct / n_samples,
    }


def build_hybrid_model(num_classes: int, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Lädt SimpleCNN-Backbone, extrahiert Conv-Feature-Maps und baut Hybridmodell."""
    loaded_cnn = load_pretrained_simplecnn(source=PRETRAINED_CNN_SOURCE)
    feature_extractor = extract_feature_extractor(loaded_cnn.model)

    if FREEZE_CNN_BACKBONE:
        freeze_module(feature_extractor)
    if UNFREEZE_LAST_CONV_BLOCKS > 0:
        unfreeze_last_conv_blocks(feature_extractor, UNFREEZE_LAST_CONV_BLOCKS)

    feature_channels, feature_grid_size = infer_feature_map_shape(feature_extractor, IMAGE_SIZE)

    hybrid = CNNViTHybridSequential(
        feature_extractor=feature_extractor,
        feature_channels=feature_channels,
        feature_grid_size=feature_grid_size,
        num_classes=num_classes,
        embed_dim=VIT_EMBED_DIM,
        num_heads=VIT_NUM_HEADS,
        depth=VIT_DEPTH,
        mlp_ratio=VIT_MLP_RATIO,
        dropout=VIT_DROPOUT,
    )
    hybrid = hybrid.to(device)
    hybrid = maybe_wrap_data_parallel(hybrid, device=device)

    cnn_info = {
        'source': str(loaded_cnn.source_path),
        'state_dict_path': str(loaded_cnn.state_dict_path),
        'resolved_hparams': loaded_cnn.resolved_hparams,
        'feature_channels': feature_channels,
        'feature_grid_size': feature_grid_size,
    }
    return hybrid, cnn_info


def load_training_checkpoint_if_available(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    class_to_idx: dict[str, int],
    device: torch.device,
) -> tuple[int, float, dict[str, list[float]]]:
    """Lädt optional einen Trainingscheckpoint für Resume."""
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    start_epoch = 0
    best_val_acc = -float('inf')

    if not CHECKPOINT_PATH.exists():
        return start_epoch, best_val_acc, history

    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
    state = ckpt.get('model_state_dict')
    if not isinstance(state, dict):
        raise ValueError(f'Invalid checkpoint format in {CHECKPOINT_PATH}')

    state = strip_module_prefix(state)
    current_state = unwrap_model(model).state_dict()
    missing_keys = [k for k in current_state.keys() if k not in state]
    unexpected_keys = [k for k in state.keys() if k not in current_state]
    shape_mismatches = [
        k for k in current_state.keys() & state.keys()
        if current_state[k].shape != state[k].shape
    ]
    if missing_keys or unexpected_keys or shape_mismatches:
        warnings.warn(
            'Checkpoint/model architecture mismatch detected. '
            f'Skipping resume from {CHECKPOINT_PATH} and starting fresh. '
            f'missing={len(missing_keys)}, unexpected={len(unexpected_keys)}, '
            f'shape_mismatches={len(shape_mismatches)}'
        )
        return start_epoch, best_val_acc, history

    unwrap_model(model).load_state_dict(state, strict=True)

    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        move_optimizer_state_to_device(optimizer, device)

    if 'scaler_state_dict' in ckpt and isinstance(ckpt['scaler_state_dict'], dict):
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    saved_class_to_idx = ckpt.get('class_to_idx')
    if isinstance(saved_class_to_idx, dict) and saved_class_to_idx != class_to_idx:
        raise ValueError(
            'class_to_idx in checkpoint differs from current dataset. '
            'Please clear checkpoint or use matching data split.'
        )

    if isinstance(ckpt.get('history'), dict):
        ckpt_history = ckpt['history']
        if all(k in ckpt_history for k in history):
            history = ckpt_history

    start_epoch = int(ckpt.get('epoch', -1)) + 1
    best_val_acc = float(ckpt.get('best_val_acc', -float('inf')))
    return start_epoch, best_val_acc, history


def main() -> None:
    """Startet Training, speichert Bestmodell und evaluiert auf dem Testset."""
    set_seed(SEED)
    device = get_device()
    print(f'Using device: {device}')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    train_dir = resolve_train_dir()
    loaders, num_classes, class_to_idx = build_dataloaders(train_dir=train_dir)

    if ENFORCE_BINARY_CLASSIFICATION and num_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f'Expected binary classification with {EXPECTED_NUM_CLASSES} classes, '
            f'but found {num_classes} classes in {train_dir}.'
        )

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    model, cnn_info = build_hybrid_model(num_classes=num_classes, device=device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError('No trainable parameters found. Check backbone freeze settings.')

    criterion = nn.CrossEntropyLoss()
    base_lr = LEARNING_RATE
    optimizer = optim.AdamW(trainable_params, lr=base_lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=USE_AMP and device.type == 'cuda')

    start_epoch, best_val_acc, history = load_training_checkpoint_if_available(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        class_to_idx=class_to_idx,
        device=device,
    )

    print(f'Start epoch: {start_epoch} | Best val acc so far: {best_val_acc:.4f}')

    epochs_without_improvement = 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        # LR-Schedule ohne globales LEARNING_RATE zu überschreiben
        if epoch >= 150:
            current_lr = 1e-7
        elif epoch >= 75:
            current_lr = 1e-6
        elif epoch >= 3:
            current_lr = 1e-5
        else:
            current_lr = base_lr

        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        train_metrics = train_one_epoch(
            model=model,
            loader=loaders['train'],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip_norm=GRAD_CLIP_NORM,
        )
        val_metrics = evaluate(model=model, loader=loaders['val'], criterion=criterion, device=device)

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        improved = val_metrics['accuracy'] >= best_val_acc
        if improved:
            best_val_acc = val_metrics['accuracy']
            epochs_without_improvement = 0
            torch.save(
                {
                    'epoch': epoch,
                    'best_val_acc': best_val_acc,
                    'model_state_dict': unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'class_to_idx': class_to_idx,
                    'history': history,
                    'cnn_info': cnn_info,
                },
                BEST_MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'model_state_dict': unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'class_to_idx': class_to_idx,
                'history': history,
                'cnn_info': cnn_info,
            },
            CHECKPOINT_PATH,
        )

        print(
            f'Epoch {epoch + 1}/{NUM_EPOCHS} | '
            f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['accuracy']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}, "
            f'best_val_acc={best_val_acc:.4f}, lr={current_lr:.1e}'
        )

        if EARLY_STOPPING_PATIENCE > 0 and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f'Early stopping triggered after {EARLY_STOPPING_PATIENCE} '
                f'epochs without improvement.'
            )
            break

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f'Best checkpoint missing after training: {BEST_MODEL_PATH}. '
            'Check whether training loop ran at least one epoch.'
        )

    best_ckpt = torch.load(BEST_MODEL_PATH, map_location='cpu')
    best_state = best_ckpt.get('model_state_dict')
    if not isinstance(best_state, dict):
        raise ValueError(f'Invalid best checkpoint format in {BEST_MODEL_PATH}')
    unwrap_model(model).load_state_dict(strip_module_prefix(best_state))

    test_metrics = evaluate(model=model, loader=loaders['test'], criterion=criterion, device=device)
    y_true, y_pred = collect_predictions(model=model, loader=loaders['test'], device=device)

    test_precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    test_recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    test_f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    test_precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    test_recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    test_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))
    train_dataset = loaders['train'].dataset
    train_original_class_counts: dict[str, int] | None = None
    train_effective_class_counts: dict[str, int] | None = None
    if isinstance(train_dataset, ClassAwareAugmentedDataset):
        train_original_class_counts = {
            train_dataset.classes[i]: int(train_dataset.original_class_counts.get(i, 0))
            for i in range(len(train_dataset.classes))
        }
        train_effective_class_counts = {
            train_dataset.classes[i]: int(train_dataset.effective_class_counts.get(i, 0))
            for i in range(len(train_dataset.classes))
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
        'best_val_accuracy': float(best_ckpt.get('best_val_acc', float('nan'))),
        'best_epoch': int(best_ckpt.get('epoch', -1) + 1),
        'checkpoint_path': str(CHECKPOINT_PATH),
        'best_model_path': str(BEST_MODEL_PATH),
        'class_labels': class_labels,
        'confusion_matrix': cm.tolist(),
        'train_dir': str(train_dir),
        'cnn_source': cnn_info['source'],
        'cnn_hparams': cnn_info['resolved_hparams'],
        'feature_channels': int(cnn_info['feature_channels']),
        'feature_grid_size': list(cnn_info['feature_grid_size']),
        'train_original_class_counts': train_original_class_counts,
        'train_effective_class_counts': train_effective_class_counts,
    }

    run_params = {
        'seed': SEED,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': base_lr,
        'weight_decay': WEIGHT_DECAY,
        'use_amp': USE_AMP,
        'grad_clip_norm': GRAD_CLIP_NORM,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'freeze_backbone': FREEZE_CNN_BACKBONE,
        'unfreeze_last_conv_blocks': UNFREEZE_LAST_CONV_BLOCKS,
        'vit_embed_dim': VIT_EMBED_DIM,
        'vit_num_heads': VIT_NUM_HEADS,
        'vit_depth': VIT_DEPTH,
        'vit_mlp_ratio': VIT_MLP_RATIO,
        'vit_dropout': VIT_DROPOUT,
        'class_repeat_factors': CLASS_REPEAT_FACTORS,
        'class_augmentation_config': CLASS_AUGMENTATION_CONFIG,
        'pretrained_cnn_source_override': (
            str(PRETRAINED_CNN_SOURCE) if PRETRAINED_CNN_SOURCE is not None else None
        ),
        'pretrained_cnn_metadata_path': (
            str(PRETRAINED_CNN_METADATA_PATH)
            if PRETRAINED_CNN_METADATA_PATH is not None
            else None
        ),
        'cnn_source': cnn_info['source'],
        'cnn_hparams': cnn_info['resolved_hparams'],
    }

    artifact_dir = save_best_model_artifacts(
        model=unwrap_model(model),
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
