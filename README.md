# Zebra CV Project

Bildklassifikation fuer Zebra / kein Zebra mit PyTorch. Dieses Repository enthaelt den Code, die Trainings- und Evaluationsskripte sowie kleine Ergebnisartefakte. Grosse Trainingsdaten und Modellgewichte werden separat ueber Hugging Face bereitgestellt.

Data: https://huggingface.co/datasets/kamichal/zebra-cv-data

Models: https://huggingface.co/kamichal/zebra-cv-checkpoints

## Kurzstart

Diese Befehle laden das GitHub-Repository, installieren die Python-Abhaengigkeiten und holen Daten sowie Modelle von Hugging Face.

```bash
git clone https://github.com/michal-wq/zebra_cv_project.git
cd zebra_cv_project

uv sync

uvx hf download kamichal/zebra-cv-data \
  data_archive/zebra-cv-data-train-val-test.tar.gz \
  --repo-type dataset \
  --local-dir .

tar -xzf data_archive/zebra-cv-data-train-val-test.tar.gz -C project

uvx hf download kamichal/zebra-cv-checkpoints \
  --local-dir project \
  --include "trained_models/*"

cd project
uv run python 05_evaluate_trained_model.py
```

Die Hugging-Face-Repositories sind public gedacht. Dafuer wird kein Hugging-Face-Account und kein Token benoetigt.

## Big_Trans_3 Ausfuehren

Fuer eine einfache Pruefung des besten Modells gibt es ein eigenes Skript fuer `Big_Trans_3`. Es laedt bei Bedarf den Testsplit und das Modell von Hugging Face, baut die passende Architektur aus `metadata.json` nach und evaluiert den Testsplit.

Aus dem Repository-Root:

```bash
uv run python project/run_big_trans_3.py --download
```

Der Download wird uebersprungen, wenn `project/data/test` und das Big-Trans-3-Modell bereits lokal vorhanden sind.

Wenn Daten und Modell bereits heruntergeladen wurden:

```bash
uv run python project/run_big_trans_3.py
```

Ein einzelnes Bild klassifizieren:

```bash
uv run python project/run_big_trans_3.py --image path/to/image.png
```

Das Skript schreibt die Testergebnisse nach:

```text
project/big_trans_3_evaluation_results.json
```

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
    run_big_trans_3.py
    prep_training.py
    training_functions.py
```

## Daten und Modelle herunterladen

Alle Befehle in diesem Abschnitt werden aus dem Repository-Root ausgefuehrt, also aus `zebra_cv_project/`.

Trainings-, Validierungs- und Testdaten:

```bash
uvx hf download kamichal/zebra-cv-data \
  data_archive/zebra-cv-data-train-val-test.tar.gz \
  --repo-type dataset \
  --local-dir .

tar -xzf data_archive/zebra-cv-data-train-val-test.tar.gz -C project
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
