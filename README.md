# zebra_cv_project

Bildklassifikation (Zebra/kein Zebra) mit PyTorch und Optuna.

## Projektüberblick

Dieses Repository enthält den kompletten Trainings-Workflow:

1. **Daten aufteilen** in `train/val/test`
2. **(Optional) Datenaugmentierung**
3. **Baseline-Training**
4. **Optuna-Hyperparameteroptimierung + finales Training**
5. **Speichern von Modellartefakten** (Gewichte, Plots, Metriken)

> Rohdaten sind **nicht** im Repository und sollen separat lokal liegen.

## Repository-Struktur

- `project/00_target_data_aug.py` – Augmentierung einzelner Bilder
- `project/01_split_data.py` – Split der Klassendaten in Train/Val/Test
- `project/02_model_training_pipeline.py` – einfache Baseline-Pipeline
- `project/03_optuna_training_pipeline.py` – Optuna + finales Training
- `project/prep_training.py` – DataLoader/Preprocessing-Helfer
- `project/models.py` – Modellarchitektur (MLP)
- `project/training_functions.py` – Speichern von Artefakten
- `pyproject.toml` + `uv.lock` – reproduzierbare Abhängigkeiten

## Voraussetzungen

- Python `3.12`
- [uv](https://docs.astral.sh/uv/)

## Setup

Im Repository-Root ausführen:

```bash
uv sync
```

Damit wird die Umgebung aus `pyproject.toml` und `uv.lock` erstellt.

## Datenstruktur (lokal, nicht in Git)

Erwartete Struktur für Rohdaten:

```text
raw_data/
  data/
    luzern/
      y/
      n/
    st gallen/
      y/
      n/
```

Erzeugte Splits liegen danach unter:

```text
project/data/
  train/{y,n}
  val/{y,n}
  test/{y,n}
```

## Typischer Workflow

In den Projektordner wechseln:

```bash
cd project
```

1) Daten splitten:

```bash
uv run 01_split_data.py
```

2) (Optional) Baseline trainieren:

```bash
uv run 02_model_training_pipeline.py
```

3) Optuna + finales Modell trainieren:

```bash
uv run 03_optuna_training_pipeline.py
```

## Ergebnisse / Artefakte

Für jeden besten Lauf wird ein eigener Ordner in `project/trained_models/` erzeugt, z. B.:

```text
project/trained_models/MLP_optuna_best_score-0.9231_YYYYMMDD_HHMMSS/
```

Darin u. a.:

- `model_state_dict.pt`
- `confusion_matrix.png`
- `learning_curves.png`
- `history.json`
- `metadata.json`

## Hinweis zu `requirements.txt`

Für dieses Projekt reicht **uv** mit:

- `pyproject.toml`
- `uv.lock`

Eine `requirements.txt` ist optional und nur für Nutzer ohne `uv` relevant.

## GitHub-Hinweise

- Trainingsdaten, Rohdaten und Modellartefakte sind absichtlich nicht versioniert.
- Logs/temporäre Dateien werden über `.gitignore` ausgeschlossen.
# zebra_cv_project

PyTorch-based image classification pipeline (Zebra vs. Non-Zebra) with Optuna hyperparameter optimization.

## 1. Scope

This repository contains the full training workflow for a binary image classification task:

1. dataset split into `train/val/test`
2. optional data augmentation utilities
3. baseline MLP training
4. Optuna optimization and final model training
5. artifact export (weights, plots, metrics)

Raw data is intentionally excluded from version control.

## 2. Repository Structure

- `project/00_target_data_aug.py` – batch image augmentation utility
- `project/01_split_data.py` – class-wise train/val/test split generation
- `project/02_model_training_pipeline.py` – baseline MLP training pipeline
- `project/03_optuna_training_pipeline.py` – Optuna search + final training + artifact export
- `project/prep_training.py` – dataset loading and dataloader helpers
- `project/models.py` – model definitions (MLP)
- `project/training_functions.py` – artifact persistence helpers
- `pyproject.toml` + `uv.lock` – environment and dependency lock

## 3. Requirements

- Python `3.12`
- [`uv`](https://docs.astral.sh/uv/)

## 4. Environment Setup

Run in repository root:

```bash
uv sync
```

## 5. Data Layout

### 5.1 Input Data (local only)

Expected raw data directory structure:

```text
raw_data/
  data/
    luzern/
      y/
      n/
    st gallen/
      y/
      n/
```

### 5.2 Generated Split

After running the split script, training data is written to:

```text
project/data/
  train/{y,n}
  val/{y,n}
  test/{y,n}
```

## 6. Execution

Change into the project directory:

```bash
cd project
```

### 6.1 Create train/val/test split

```bash
uv run 01_split_data.py
```

### 6.2 Run baseline training (optional)

```bash
uv run 02_model_training_pipeline.py
```

### 6.3 Run Optuna + final training

```bash
uv run 03_optuna_training_pipeline.py
```

## 7. Output Artifacts

For each best run, an artifact directory is created under:

```text
project/trained_models/MLP_optuna_best_score-<metric>_<timestamp>/
```

Generated files include:

- `model_state_dict.pt` – trained model weights
- `confusion_matrix.png` – confusion matrix visualization
- `learning_curves.png` – train/validation curves
- `history.json` – epoch-wise metrics
- `metadata.json` – run metadata and hyperparameters

## 8. Version Control Notes

The following content is intentionally not tracked:

- raw input datasets
- generated split datasets
- trained model artifacts
- temporary/cache files

Exclusions are handled via `.gitignore`.