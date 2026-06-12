# Zebra CV Project

Bildklassifikation fuer Zebra / kein Zebra mit PyTorch. Dieses Repository enthaelt den Code, die Trainings- und Evaluationsskripte sowie kleine Ergebnisartefakte. Grosse Trainingsdaten und Modellgewichte werden separat ueber Hugging Face bereitgestellt.

## Kurzstart

Diese Befehle laden das GitHub-Repository, installieren die Python-Abhaengigkeiten und holen Daten sowie Modelle von Hugging Face.

```bash
git clone https://github.com/michal-wq/zebra_cv_project.git
cd zebra_cv_project

uv sync

uvx hf download kamichal/zebra-cv-data \
  --repo-type dataset \
  --local-dir project \
  --include "data/*"

uvx hf download kamichal/zebra-cv-checkpoints \
  --local-dir project \
  --include "trained_models/*"

cd project
uv run python 05_evaluate_trained_model.py
```

Die Hugging-Face-Repositories sind public gedacht. Dafuer wird kein Hugging-Face-Account und kein Token benoetigt.

## Voraussetzungen

- Git
- Python 3.12
- uv: <https://docs.astral.sh/uv/>

Falls `uv` noch nicht installiert ist:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Danach gegebenenfalls ein neues Terminal oeffnen oder die angezeigte PATH-Anweisung ausfuehren.

## Projektstruktur

```text
zebra_cv_project/
  pyproject.toml
  uv.lock
  README.md
  project/
    data/                # wird von Hugging Face geladen
      train/
      val/
      test/
    trained_models/      # wird von Hugging Face geladen
    01_split_data.py
    04_train_pretrained.py
    05_evaluate_trained_model.py
    07_cnn_vit_seq.py
    08_misclassification_analysis.py
    10_grad_cam_cnn_vit.py
    prep_training.py
    training_functions.py
```

## Daten und Modelle herunterladen

Alle Befehle in diesem Abschnitt werden aus dem Repository-Root ausgefuehrt, also aus `zebra_cv_project/`.

Trainings-, Validierungs- und Testdaten:

```bash
uvx hf download kamichal/zebra-cv-data \
  --repo-type dataset \
  --local-dir project \
  --include "data/*"
```

Trainierte Modelle und Ergebnisdateien:

```bash
uvx hf download kamichal/zebra-cv-checkpoints \
  --local-dir project \
  --include "trained_models/*"
```

Danach sollte die lokale Struktur so aussehen:

```text
project/data/
  train/n
  train/y
  val/n
  val/y
  test/n
  test/y

project/trained_models/
  <model_name>/
    model_state_dict.pt
    metadata.json
    results.json
    confusion_matrix.png
```

## Evaluation ausfuehren

Die meisten Skripte erwarten, dass sie aus dem Ordner `project/` gestartet werden.

```bash
cd project
uv run python 05_evaluate_trained_model.py
```

Weitere Analyse-Skripte:

```bash
uv run python 08_misclassification_analysis.py
uv run python 10_grad_cam_cnn_vit.py
```

## Training erneut ausfuehren

Wenn die Daten bereits unter `project/data/` liegen, koennen Trainingsskripte ebenfalls aus `project/` gestartet werden.

```bash
cd project
uv run python 04_train_pretrained.py
uv run python 07_cnn_vit_seq.py
```

Neue Modellartefakte werden unter `project/trained_models/` gespeichert.

## Daten neu splitten

Die Rohdaten sind nicht im GitHub-Repository enthalten. Falls Rohdaten lokal vorhanden sind, erwartet das Split-Skript diese Struktur:

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

Aus `project/`:

```bash
uv run python 01_split_data.py
```

Das erzeugt:

```text
project/data/train
project/data/val
project/data/test
```

## Artefakte

Ein trainierter Lauf liegt typischerweise in einem eigenen Ordner:

```text
project/trained_models/<model_name>/
```

Wichtige Dateien:

- `model_state_dict.pt`: trainierte Modellgewichte fuer Evaluation / Inference
- `metadata.json`: Hyperparameter und Lauf-Metadaten
- `results.json`: Metriken
- `confusion_matrix.png`: Confusion Matrix
- `history.json`: Trainingsverlauf, falls vorhanden
- `learning_curves.png`: Lernkurven, falls vorhanden

Grosse vollstaendige Training-Checkpoints werden nicht im GitHub-Repository versioniert. Fuer die Nutzung der Modelle reichen die `model_state_dict.pt` Dateien zusammen mit Code und `metadata.json`.

## Troubleshooting

Wenn Hugging Face beim Download meldet, dass Dateien bereits existieren, ist das normal. Der Befehl kann erneut ausgefuehrt werden.

Wenn ein Skript `data/train`, `data/val` oder `data/test` nicht findet, wurde es wahrscheinlich aus dem falschen Ordner gestartet. In diesem Projekt die Python-Skripte aus `project/` ausfuehren:

```bash
cd project
uv run python 05_evaluate_trained_model.py
```

Wenn `uvx hf download` nicht funktioniert, zuerst pruefen:

```bash
uv --version
uvx hf --help
```

## Version-Control-Hinweise

Nicht in GitHub versioniert werden:

- Rohdaten
- generierte Split-Daten
- grosse Modellgewichte und Checkpoints
- Logs, Caches und temporaere Dateien

Diese Dateien werden separat ueber Hugging Face bereitgestellt oder lokal erzeugt.
