"""Optuna hyperparameter optimization for the CNN->ViT sequential model.

This script combines the Optuna flow from 03_optuna_training_pipeline.py with
the robust CNN backbone loading and CNNViTHybridSequential architecture from
07_cnn_vit_seq.py.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import random
import time
from pathlib import Path
from types import ModuleType
from typing import Any

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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler


# =========================
# Configuration
# =========================
SEED = 77

MODEL_NAME = 'CNN_ViT_Seq_Optuna'
STUDY_NAME = 'CNN_ViT_Seq_HPO'
ARTIFACT_BASE_DIR = Path('trained_models')
BEST_MODEL_CHECKPOINT_PATH = ARTIFACT_BASE_DIR / 'CNN_ViT_Seq_Optuna_best.pt'
STUDY_SUMMARY_PATH = ARTIFACT_BASE_DIR / 'CNN_ViT_Seq_Optuna_study_summary.json'

N_TRIALS = 25
OPTUNA_EPOCHS = 12
OPTUNA_PATIENCE = 4
OPTUNA_PRUNER_STARTUP_TRIALS = 5
OPTUNA_PRUNER_WARMUP_STEPS = 3

FINAL_TRAIN_EPOCHS = 100
FINAL_PATIENCE = 15

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 16
USE_AMP = True
PLOT_DPI = 180

# Optional override. Leave as None to use the same resolution logic as 07.
PRETRAINED_CNN_SOURCE: Path | None = None
PRETRAINED_CNN_METADATA_PATH: Path | None = None
CNN_HPARAMS_OVERRIDE: dict[str, Any] = {}


def load_base_module() -> ModuleType:
    """Loads 07_cnn_vit_seq.py even though the filename is not importable."""
    base_path = Path(__file__).with_name('07_cnn_vit_seq.py')
    spec = importlib.util.spec_from_file_location('cnn_vit_seq_base', base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load module spec from {base_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def configure_base_module() -> None:
    """Applies shared runtime settings to the imported 07 module."""
    base.SEED = SEED
    base.IMAGE_SIZE = IMAGE_SIZE
    base.BATCH_SIZE = BATCH_SIZE
    base.NUM_WORKERS = NUM_WORKERS
    base.USE_AMP = USE_AMP
    base.PRETRAINED_CNN_SOURCE = PRETRAINED_CNN_SOURCE
    if PRETRAINED_CNN_METADATA_PATH is not None:
        base.PRETRAINED_CNN_METADATA_PATH = PRETRAINED_CNN_METADATA_PATH
    base.CNN_HPARAMS_OVERRIDE = CNN_HPARAMS_OVERRIDE


def set_seed(seed: int) -> None:
    """Sets deterministic-ish seeds for Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_figure_png(fig: plt.Figure, output_path: Path, dpi: int = PLOT_DPI) -> Path:
    """Saves a Matplotlib figure and closes it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return output_path


def valid_head_choices(embed_dim: int) -> list[int]:
    """Returns transformer head counts that divide the embedding dimension."""
    candidates = [2, 4, 8, 16]
    return [heads for heads in candidates if embed_dim % heads == 0]


def suggest_trial_params(trial: optuna.Trial) -> dict[str, Any]:
    """Defines the CNN_ViT_Seq hyperparameter search space."""
    embed_dim = trial.suggest_categorical('vit_embed_dim', [128, 256, 384, 512])
    num_heads = trial.suggest_categorical('vit_num_heads', valid_head_choices(embed_dim))
    freeze_backbone = trial.suggest_categorical('freeze_cnn_backbone', [True, False])

    if freeze_backbone:
        unfreeze_last_conv_blocks = trial.suggest_int('unfreeze_last_conv_blocks', 0, 2)
    else:
        unfreeze_last_conv_blocks = 0

    return {
        'vit_embed_dim': embed_dim,
        'vit_num_heads': num_heads,
        'vit_depth': trial.suggest_int('vit_depth', 2, 8),
        'vit_mlp_ratio': trial.suggest_float('vit_mlp_ratio', 2.0, 5.0, step=0.5),
        'vit_dropout': trial.suggest_float('vit_dropout', 0.0, 0.4, step=0.05),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ['adamw', 'nadam']),
        'lr_schedule': trial.suggest_categorical('lr_schedule', ['none', 'cosine', 'plateau']),
        'grad_clip_norm': trial.suggest_categorical('grad_clip_norm', [0.0, 0.5, 1.0, 2.0]),
        'weighted_loss': trial.suggest_categorical('weighted_loss', [False, True]),
        'freeze_cnn_backbone': freeze_backbone,
        'unfreeze_last_conv_blocks': unfreeze_last_conv_blocks,
    }


def build_hybrid_model_from_params(
    params: dict[str, Any],
    num_classes: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Builds a fresh CNN->ViT model for a trial or final training."""
    loaded_cnn = base.load_pretrained_simplecnn(source=PRETRAINED_CNN_SOURCE)
    feature_extractor = base.extract_feature_extractor(loaded_cnn.model)

    if params['freeze_cnn_backbone']:
        base.freeze_module(feature_extractor)
        base.unfreeze_last_conv_blocks(
            feature_extractor,
            int(params['unfreeze_last_conv_blocks']),
        )

    feature_channels, feature_grid_size = base.infer_feature_map_shape(feature_extractor, IMAGE_SIZE)
    model = base.CNNViTHybridSequential(
        feature_extractor=feature_extractor,
        feature_channels=feature_channels,
        feature_grid_size=feature_grid_size,
        num_classes=num_classes,
        embed_dim=int(params['vit_embed_dim']),
        num_heads=int(params['vit_num_heads']),
        depth=int(params['vit_depth']),
        mlp_ratio=float(params['vit_mlp_ratio']),
        dropout=float(params['vit_dropout']),
    )
    model = model.to(device)
    model = base.maybe_wrap_data_parallel(model, device=device)

    cnn_info = {
        'source': str(loaded_cnn.source_path),
        'state_dict_path': str(loaded_cnn.state_dict_path),
        'resolved_hparams': loaded_cnn.resolved_hparams,
        'feature_channels': int(feature_channels),
        'feature_grid_size': list(feature_grid_size),
    }
    return model, cnn_info


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Returns parameters that should be optimized."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError('No trainable parameters found.')
    return params


def make_optimizer(params: list[nn.Parameter], trial_params: dict[str, Any]) -> optim.Optimizer:
    """Constructs the optimizer selected by Optuna."""
    optimizer_map = {
        'adamw': optim.AdamW,
        'nadam': optim.NAdam,
    }
    optimizer_cls = optimizer_map[trial_params['optimizer']]
    return optimizer_cls(
        params,
        lr=float(trial_params['learning_rate']),
        weight_decay=float(trial_params['weight_decay']),
    )


def make_scheduler(
    optimizer: optim.Optimizer,
    trial_params: dict[str, Any],
    epochs: int,
):
    """Constructs an epoch-level scheduler or returns None."""
    schedule = trial_params['lr_schedule']
    if schedule == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=float(trial_params['learning_rate']) * 0.01,
        )
    if schedule == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
        )
    return None


def make_criterion(loaders: dict[str, Any], num_classes: int, device: torch.device, weighted: bool) -> nn.Module:
    """Creates CrossEntropyLoss with optional inverse-frequency class weights."""
    if not weighted:
        return nn.CrossEntropyLoss()

    train_dataset = loaders['train'].dataset
    class_counts = getattr(train_dataset, 'effective_class_counts', None)
    if class_counts is None:
        return nn.CrossEntropyLoss()

    counts = torch.tensor(
        [max(1, int(class_counts.get(i, 0))) for i in range(num_classes)],
        dtype=torch.float32,
        device=device,
    )
    weights = counts.sum() / (num_classes * counts)
    return nn.CrossEntropyLoss(weight=weights)


def step_scheduler(scheduler: Any, val_loss: float) -> None:
    """Steps either a normal epoch scheduler or ReduceLROnPlateau."""
    if scheduler is None:
        return
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


def train_for_epochs(
    model: nn.Module,
    loaders: dict[str, Any],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    trial_params: dict[str, Any],
    epochs: int,
    patience: int,
    trial: optuna.Trial | None = None,
    checkpoint_path: Path | None = None,
    class_to_idx: dict[str, int] | None = None,
    cnn_info: dict[str, Any] | None = None,
) -> tuple[float, float, dict[str, list[float]], int]:
    """Trains a model and optionally reports/prunes Optuna trials."""
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    best_val_acc = -float('inf')
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_without_improvement = 0
    scheduler = make_scheduler(optimizer, trial_params, epochs=epochs)
    grad_clip_norm = float(trial_params['grad_clip_norm']) or None

    for epoch in range(epochs):
        train_metrics = base.train_one_epoch(
            model=model,
            loader=loaders['train'],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = base.evaluate(
            model=model,
            loader=loaders['val'],
            criterion=criterion,
            device=device,
        )
        val_loss = float(val_metrics['loss'])
        val_acc = float(val_metrics['accuracy'])
        step_scheduler(scheduler, val_loss)

        history['train_loss'].append(float(train_metrics['loss']))
        history['train_acc'].append(float(train_metrics['accuracy']))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        'epoch': epoch,
                        'best_val_acc': best_val_acc,
                        'best_val_loss': best_val_loss,
                        'model_state_dict': base.unwrap_model(model).state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'class_to_idx': class_to_idx,
                        'history': history,
                        'params': trial_params,
                        'cnn_info': cnn_info,
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1

        print(
            f'Epoch {epoch + 1:03d}/{epochs} | '
            f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['accuracy']:.4f}, "
            f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, '
            f'best_val_acc={best_val_acc:.4f}'
        )

        if trial is not None:
            trial.report(best_val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if patience > 0 and epochs_without_improvement >= patience:
            print(f'Early stopping after {patience} epochs without val_acc improvement.')
            break

    return best_val_acc, best_val_loss, history, best_epoch


def objective_factory(
    loaders: dict[str, Any],
    num_classes: int,
    device: torch.device,
) -> Any:
    """Creates an Optuna objective with the prepared data loaders."""
    def objective(trial: optuna.Trial) -> float:
        params = suggest_trial_params(trial)
        set_seed(SEED + trial.number)

        model, _cnn_info = build_hybrid_model_from_params(
            params=params,
            num_classes=num_classes,
            device=device,
        )
        optimizer = make_optimizer(trainable_parameters(model), params)
        criterion = make_criterion(
            loaders=loaders,
            num_classes=num_classes,
            device=device,
            weighted=bool(params['weighted_loss']),
        )
        scaler = GradScaler(enabled=USE_AMP and device.type == 'cuda')

        try:
            best_val_acc, best_val_loss, _history, best_epoch = train_for_epochs(
                model=model,
                loaders=loaders,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                trial_params=params,
                epochs=OPTUNA_EPOCHS,
                patience=OPTUNA_PATIENCE,
                trial=trial,
            )
            trial.set_user_attr('best_val_loss', best_val_loss)
            trial.set_user_attr('best_epoch', best_epoch + 1)
            return best_val_acc
        finally:
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    return objective


def save_study_summary(study: optuna.Study, path: Path) -> None:
    """Writes a compact JSON summary of all trials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in study.trials:
        rows.append(
            {
                'number': trial.number,
                'state': trial.state.name,
                'value': trial.value,
                'params': trial.params,
                'user_attrs': trial.user_attrs,
            }
        )
    path.write_text(json.dumps(rows, indent=2), encoding='utf-8')


def class_count_summary(loaders: dict[str, Any]) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    """Returns original and oversampled train class counts when available."""
    train_dataset = loaders['train'].dataset
    if not isinstance(train_dataset, base.ClassAwareAugmentedDataset):
        return None, None

    original = {
        train_dataset.classes[i]: int(train_dataset.original_class_counts.get(i, 0))
        for i in range(len(train_dataset.classes))
    }
    effective = {
        train_dataset.classes[i]: int(train_dataset.effective_class_counts.get(i, 0))
        for i in range(len(train_dataset.classes))
    }
    return original, effective


def plot_learning_curves(history: dict[str, list[float]], artifact_dir: Path) -> Path:
    """Stores train/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('CNN_ViT_Seq Optuna Best Model - Learning Curves')

    epochs_range = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs_range, history['train_acc'], label='Train Accuracy')
    ax2.plot(epochs_range, history['val_acc'], label='Val Accuracy', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    fig.tight_layout()
    return save_figure_png(fig, artifact_dir / 'learning_curves.png')


def run_final_training(
    best_params: dict[str, Any],
    loaders: dict[str, Any],
    num_classes: int,
    class_to_idx: dict[str, int],
    device: torch.device,
) -> None:
    """Retrains the best Optuna configuration and exports final artifacts."""
    print('\n' + '=' * 60)
    print(f'Final training for {FINAL_TRAIN_EPOCHS} epochs')
    print('=' * 60)

    set_seed(SEED)
    model, cnn_info = build_hybrid_model_from_params(
        params=best_params,
        num_classes=num_classes,
        device=device,
    )
    optimizer = make_optimizer(trainable_parameters(model), best_params)
    criterion = make_criterion(
        loaders=loaders,
        num_classes=num_classes,
        device=device,
        weighted=bool(best_params['weighted_loss']),
    )
    scaler = GradScaler(enabled=USE_AMP and device.type == 'cuda')

    best_val_acc, best_val_loss, history, best_epoch = train_for_epochs(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        trial_params=best_params,
        epochs=FINAL_TRAIN_EPOCHS,
        patience=FINAL_PATIENCE,
        checkpoint_path=BEST_MODEL_CHECKPOINT_PATH,
        class_to_idx=class_to_idx,
        cnn_info=cnn_info,
    )

    checkpoint = torch.load(BEST_MODEL_CHECKPOINT_PATH, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict')
    if not isinstance(state_dict, dict):
        raise ValueError(f'Invalid checkpoint format in {BEST_MODEL_CHECKPOINT_PATH}')
    base.unwrap_model(model).load_state_dict(base.strip_module_prefix(state_dict))

    test_metrics = base.evaluate(model=model, loader=loaders['test'], criterion=criterion, device=device)
    y_true, y_pred = base.collect_predictions(model=model, loader=loaders['test'], device=device)

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))
    original_counts, effective_counts = class_count_summary(loaders)

    test_precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    test_recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    test_f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    test_precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    test_recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    test_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    final_results = {
        'test_loss': float(test_metrics['loss']),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_precision_weighted': float(test_precision_weighted),
        'test_recall_weighted': float(test_recall_weighted),
        'test_f1_weighted': float(test_f1_weighted),
        'test_precision_macro': float(test_precision_macro),
        'test_recall_macro': float(test_recall_macro),
        'test_f1_macro': float(test_f1_macro),
        'best_val_accuracy': float(best_val_acc),
        'best_val_loss': float(best_val_loss),
        'best_epoch': int(best_epoch + 1),
        'checkpoint_path': str(BEST_MODEL_CHECKPOINT_PATH),
        'class_labels': class_labels,
        'confusion_matrix': cm.tolist(),
        'cnn_source': cnn_info['source'],
        'cnn_hparams': cnn_info['resolved_hparams'],
        'feature_channels': int(cnn_info['feature_channels']),
        'feature_grid_size': list(cnn_info['feature_grid_size']),
        'train_original_class_counts': original_counts,
        'train_effective_class_counts': effective_counts,
    }

    run_params = {
        'seed': SEED,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'use_amp': USE_AMP,
        'n_trials': N_TRIALS,
        'optuna_epochs': OPTUNA_EPOCHS,
        'final_train_epochs': FINAL_TRAIN_EPOCHS,
        'final_patience': FINAL_PATIENCE,
        'best_params': best_params,
        'class_repeat_factors': base.CLASS_REPEAT_FACTORS,
        'class_augmentation_config': base.CLASS_AUGMENTATION_CONFIG,
        'pretrained_cnn_source_override': (
            str(PRETRAINED_CNN_SOURCE) if PRETRAINED_CNN_SOURCE is not None else None
        ),
        'pretrained_cnn_metadata_path': (
            str(base.PRETRAINED_CNN_METADATA_PATH)
            if base.PRETRAINED_CNN_METADATA_PATH is not None
            else None
        ),
        'cnn_source': cnn_info['source'],
        'cnn_hparams': cnn_info['resolved_hparams'],
    }

    artifact_dir = base.save_best_model_artifacts(
        model=base.unwrap_model(model),
        y_true=y_true,
        y_pred=y_pred,
        model_name=MODEL_NAME,
        score=float(test_metrics['accuracy']),
        params=run_params,
        history=history,
        base_dir=str(ARTIFACT_BASE_DIR),
        results=final_results,
        class_labels=class_labels,
        save_history=True,
    )
    learning_curve_path = plot_learning_curves(history, Path(artifact_dir))

    final_results['artifact_dir'] = str(artifact_dir)
    final_results['learning_curve_path'] = str(learning_curve_path)

    print('\n' + '=' * 60)
    print('Final test metrics')
    print('=' * 60)
    print(json.dumps(final_results, indent=2))


def main() -> None:
    """Runs Optuna search followed by final training of the best configuration."""
    configure_base_module()
    ARTIFACT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = base.get_device()
    print(f'Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    train_dir = base.resolve_train_dir()
    loaders, num_classes, class_to_idx = base.build_dataloaders(train_dir=train_dir)

    if base.ENFORCE_BINARY_CLASSIFICATION and num_classes != base.EXPECTED_NUM_CLASSES:
        raise ValueError(
            f'Expected {base.EXPECTED_NUM_CLASSES} classes, found {num_classes} in {train_dir}.'
        )

    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner(
        n_startup_trials=OPTUNA_PRUNER_STARTUP_TRIALS,
        n_warmup_steps=OPTUNA_PRUNER_WARMUP_STEPS,
    )
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=STUDY_NAME,
    )

    start_time = time.time()
    process = psutil.Process()
    ram_start = process.memory_info().rss / 1024**3

    print(f'\nStarting Optuna search: {N_TRIALS} trials, {OPTUNA_EPOCHS} epochs each')
    study.optimize(
        objective_factory(loaders=loaders, num_classes=num_classes, device=device),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    elapsed = time.time() - start_time
    ram_end = process.memory_info().rss / 1024**3
    save_study_summary(study, STUDY_SUMMARY_PATH)

    print('\n' + '=' * 60)
    print('Resources and best trial')
    print('=' * 60)
    print(f'Total duration      : {elapsed / 60:.2f} minutes')
    print(f'Average per trial   : {elapsed / max(1, N_TRIALS):.1f} seconds')
    print(f'RAM start           : {ram_start:.2f} GB')
    print(f'RAM end             : {ram_end:.2f} GB')
    print(f'RAM delta           : {ram_end - ram_start:.2f} GB')
    print(f'Device              : {device}')
    print(f'Trials total        : {len(study.trials)}')
    print(f'Trials pruned       : {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}')
    print(f'Trials complete     : {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}')
    print(f'Study summary       : {STUDY_SUMMARY_PATH}')

    best = study.best_trial
    print('\nBest trial')
    print(f'  Trial number      : {best.number}')
    print(f'  Best val accuracy : {best.value:.4f}')
    print('  Hyperparameters:')
    for key, value in best.params.items():
        print(f'    {key:28s} = {value}')
    for key, value in best.user_attrs.items():
        print(f'    {key:28s} = {value}')

    best_params = dict(best.params)
    if not best_params.get('freeze_cnn_backbone', False):
        best_params['unfreeze_last_conv_blocks'] = 0

    run_final_training(
        best_params=best_params,
        loaders=loaders,
        num_classes=num_classes,
        class_to_idx=class_to_idx,
        device=device,
    )


if __name__ == '__main__':
    main()
