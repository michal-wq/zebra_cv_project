"""Stellt Speicherfunktionen für Modellartefakte wie Gewichte, Confusion Matrix und Metadaten bereit."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def create_model_artifact_dir(base_dir: str, model_name: str, score: float) -> Path:
    """Erzeugt einen eindeutigen Artefaktordner für einen Modelllauf."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f'{model_name}_score-{score:.4f}_{timestamp}'
    model_dir = Path(base_dir) / folder_name
    model_dir.mkdir(parents=True, exist_ok=False)
    return model_dir


def save_best_model_artifacts(
    model: torch.nn.Module,
    y_true,
    y_pred,
    model_name: str,
    score: float,
    params: dict | None = None,
    history: dict | None = None,
    base_dir: str = 'trained_models',
    results: dict | None = None,
    class_labels: list[str] | None = None,
    save_history: bool = True,
):
    """Speichert Gewichte, Confusion-Matrix-Plot, optionale Verläufe, Ergebnisse und Metadaten."""
    model_dir = create_model_artifact_dir(base_dir, model_name, score)

    torch.save(model.state_dict(), model_dir / 'model_state_dict.pt')

    if class_labels is not None:
        labels = list(range(len(class_labels)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    else:
        cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    tick_marks = np.arange(cm.shape[0])
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    if class_labels is not None and len(class_labels) == len(tick_marks):
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.set_yticklabels(class_labels)

    thresh = cm.max() / 2 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], 'd'),
                ha='center',
                va='center',
                color='white' if cm[i, j] > thresh else 'black',
            )

    fig.tight_layout()
    fig.savefig(model_dir / 'confusion_matrix.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    if save_history and history is not None:
        (model_dir / 'history.json').write_text(
            json.dumps(history, indent=2),
            encoding='utf-8',
        )

    if results is not None:
        (model_dir / 'results.json').write_text(
            json.dumps(results, indent=2),
            encoding='utf-8',
        )

    metadata = {
        'model_name': model_name,
        'score': float(score),
        'params': params if params is not None else {},
        'created_at': datetime.now().isoformat(),
    }
    (model_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2),
        encoding='utf-8',
    )

    return model_dir