"""Trainiert ein sequentielles CNN->ViT-Hybridmodell für binäre Bildklassifikation.

Die Pipeline lädt ein bereits optimiertes SimpleCNN-Checkpoint, friert dessen
Convolution-Feature-Extractor ein und trainiert darauf einen Transformer-Head.
"""

from __future__ import annotations

import json
import random
import re
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
BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_EPOCHS = 60
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
USE_AMP = True
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 12

ENFORCE_BINARY_CLASSIFICATION = True
EXPECTED_NUM_CLASSES = 2

DATA_ROOT = Path('data')
TRAIN_DIR = DATA_ROOT / 'train'
VAL_DIR = DATA_ROOT / 'val'
TEST_DIR = DATA_ROOT / 'test'

MODEL_NAME = 'SimpleCNN_ViT_Hybrid_Seq'
MODEL_DIR = Path('trained_models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODEL_DIR / 'cnn_vit_seq_checkpoint.pt'
BEST_MODEL_PATH = MODEL_DIR / 'cnn_vit_seq_best.pt'

# Optional explizit setzen. Mögliche Werte:
# 1) Datei mit {'model_state_dict': ...} oder direktem state_dict
# 2) Artefakt-Ordner mit model_state_dict.pt + metadata.json
PRETRAINED_CNN_SOURCE: Path | None = None

# Falls Metadata/Checkpoint unvollständig sind, können Architekturwerte hier
# explizit überschrieben werden.
CNN_HPARAMS_OVERRIDE: dict[str, Any] = {}

# CNN-Backbone standardmäßig komplett einfrieren. Optional können letzte
# Convolution-Blöcke wieder freigegeben werden.
FREEZE_CNN_BACKBONE = True
UNFREEZE_LAST_CONV_BLOCKS = 0

# Transformer-Konfiguration
VIT_EMBED_DIM = 256
VIT_NUM_HEADS = 8
VIT_DEPTH = 4
VIT_MLP_RATIO = 4.0
VIT_DROPOUT = 0.1

# Klassen-Ungleichgewicht und On-the-fly-Augs (analog zu 06).
CLASS_REPEAT_FACTORS: dict[str, int] = {
    'y': 12,
    'n': 3,
}

CLASS_AUGMENTATION_CONFIG: dict[str, dict[str, Any]] = {
    'y': {
        'apply_prob': 1.0,
        'hflip_prob': 0.5,
        'rotation_deg': 6,
        'perspective_prob': 0.2,
        'blur_prob': 0.20,
        'color_jitter': (0.25, 0.25, 0.25, 0.10),
    },
    'n': {
        'apply_prob': 1.0,
        'hflip_prob': 0.5,
        'rotation_deg': 6,
        'perspective_prob': 0.2,
        'blur_prob': 0.20,
        'color_jitter': (0.25, 0.25, 0.25, 0.10),
    },
}


class LoadedSimpleCNN(NamedTuple):
    """Container für geladenes CNN samt aufgelöster Konfiguration."""

    model: SimpleCNN
    source_path: Path
    state_dict_path: Path
    resolved_hparams: dict[str, Any]


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

    # Keine Normalisierung: konsistent mit SimpleCNN-Backbone-Training.
    train_base_tfms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
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
    """Leitet die notwendige SimpleCNN-Architektur robust aus einem state_dict ab."""
    conv_pattern = re.compile(r'^features\.(\d+)\.weight$')
    linear_pattern = re.compile(r'^classifier\.(\d+)\.weight$')
    bn_pattern = re.compile(r'^features\.\d+\.running_mean$')

    conv_entries: list[tuple[int, tuple[int, ...]]] = []
    linear_entries: list[tuple[int, tuple[int, ...]]] = []
    has_batchnorm = any(bn_pattern.match(k) for k in state_dict)

    for key, tensor in state_dict.items():
        conv_match = conv_pattern.match(key)
        if conv_match and tensor.ndim == 4:
            conv_entries.append((int(conv_match.group(1)), tuple(tensor.shape)))
            continue

        linear_match = linear_pattern.match(key)
        if linear_match and tensor.ndim == 2:
            linear_entries.append((int(linear_match.group(1)), tuple(tensor.shape)))

    if not conv_entries:
        raise ValueError('Could not infer convolutional layers from state_dict.')
    if not linear_entries:
        raise ValueError('Could not infer classifier layers from state_dict.')

    conv_entries.sort(key=lambda x: x[0])
    linear_entries.sort(key=lambda x: x[0])

    conv_indices = [idx for idx, _ in conv_entries]
    conv_channels = tuple(int(shape[0]) for _, shape in conv_entries)
    kernel_size = int(conv_entries[0][1][-1])

    # Dropout-Rate selbst ist im state_dict nicht enthalten; wir schätzen nur,
    # ob ein Dropout-Layer strukturell vorhanden war (für kompatibles Rebuild).
    cnn_dropout_rate = 0.0
    if len(conv_indices) >= 2:
        observed_step = conv_indices[1] - conv_indices[0]
        expected_step_without_dropout = 4 if has_batchnorm else 3
        if observed_step > expected_step_without_dropout:
            cnn_dropout_rate = 0.1

    linear_indices = [idx for idx, _ in linear_entries]
    n_linear_layers = len(linear_entries)
    n_hidden_layers = max(0, n_linear_layers - 1)
    hidden_layer_sizes = [int(shape[0]) for _, shape in linear_entries[:-1]]
    num_classes = int(linear_entries[-1][1][0])

    dropout_rate = 0.0
    if n_hidden_layers >= 1 and len(linear_indices) >= 2:
        observed_step = linear_indices[1] - linear_indices[0]
        if observed_step > 2:
            dropout_rate = 0.1

    return {
        'n_layers': n_hidden_layers,
        'layer_sizes': hidden_layer_sizes,
        'activation': 'relu',
        'dropout_rate': dropout_rate,
        'conv_channels': conv_channels,
        'kernel_size': kernel_size,
        'pool_type': 'max',
        'use_batchnorm': has_batchnorm,
        'cnn_dropout_rate': cnn_dropout_rate,
        'num_classes': num_classes,
    }


def resolve_simplecnn_init_hparams(
    inferred: dict[str, Any],
    hints: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Kombiniert inferierte Werte, Checkpoint-Hints und explizite Overrides."""
    merged = dict(inferred)
    merged.update(hints)
    merged.update(overrides)

    if all(k in merged for k in ('conv_channels_1', 'conv_channels_2', 'conv_channels_3')):
        conv_channels = (
            int(merged['conv_channels_1']),
            int(merged['conv_channels_2']),
            int(merged['conv_channels_3']),
        )
    else:
        raw_conv_channels = merged.get('conv_channels', inferred['conv_channels'])
        if not isinstance(raw_conv_channels, (list, tuple)) or len(raw_conv_channels) != 3:
            raise ValueError(f'Invalid conv_channels: {raw_conv_channels}')
        conv_channels = tuple(int(v) for v in raw_conv_channels)

    n_layers = int(merged.get('n_layers', inferred['n_layers']))
    hidden_sizes: list[int] = []

    if isinstance(merged.get('layer_sizes'), (list, tuple)):
        hidden_sizes = [int(v) for v in merged['layer_sizes']]

    if not hidden_sizes:
        nodes_from_optuna = []
        for i in range(max(0, n_layers)):
            key = f'n_nodes_layer_{i}'
            if key in merged:
                nodes_from_optuna.append(int(merged[key]))
        hidden_sizes = nodes_from_optuna

    if len(hidden_sizes) != n_layers:
        hidden_sizes = [int(v) for v in inferred['layer_sizes']]
        n_layers = len(hidden_sizes)

    activation = str(merged.get('activation', inferred['activation'])).lower()
    if activation not in {'relu', 'tanh', 'sigmoid'}:
        warnings.warn(f'Unsupported activation "{activation}" in hints; fallback to relu.')
        activation = 'relu'

    pool_type = str(merged.get('pool_type', inferred['pool_type'])).lower()
    if pool_type not in {'max', 'avg'}:
        warnings.warn(f'Unsupported pool_type "{pool_type}" in hints; fallback to max.')
        pool_type = 'max'

    return {
        'n_layers': n_layers,
        'layer_sizes': hidden_sizes,
        'activation': activation,
        'dropout_rate': safe_float(merged.get('dropout_rate', inferred['dropout_rate']), 0.0),
        'num_classes': int(inferred['num_classes']),
        'conv_channels': conv_channels,
        'kernel_size': int(merged.get('kernel_size', inferred['kernel_size'])),
        'pool_type': pool_type,
        'use_batchnorm': safe_bool(
            merged.get('use_batchnorm', inferred['use_batchnorm']),
            bool(inferred['use_batchnorm']),
        ),
        'cnn_dropout_rate': safe_float(
            merged.get('cnn_dropout_rate', inferred['cnn_dropout_rate']),
            0.0,
        ),
    }


def resolve_pretrained_cnn_source(explicit_source: Path | None) -> Path:
    """Ermittelt den besten verfügbaren Pfad für das vortrainierte SimpleCNN."""
    if explicit_source is not None:
        if explicit_source.exists():
            return explicit_source
        raise FileNotFoundError(f'Configured PRETRAINED_CNN_SOURCE does not exist: {explicit_source}')

    direct_ckpt = MODEL_DIR / 'Simple_CNN.pt'
    if direct_ckpt.exists():
        return direct_ckpt

    artifact_dirs = sorted(MODEL_DIR.glob('Simple_CNN_score-*'))
    for artifact_dir in reversed(artifact_dirs):
        if (artifact_dir / 'model_state_dict.pt').exists():
            return artifact_dir

    raise FileNotFoundError(
        'No pretrained SimpleCNN source found. '
        'Set PRETRAINED_CNN_SOURCE to a checkpoint file or artifact directory.'
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

    model = SimpleCNN(
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


def extract_feature_extractor(cnn_model: SimpleCNN) -> nn.Sequential:
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
    unwrap_model(model).load_state_dict(state)

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
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
            f'best_val_acc={best_val_acc:.4f}'
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
    }

    run_params = {
        'seed': SEED,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
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
