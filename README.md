# FSDS MLE Assignment

This project packages a small binary-classification workflow as a reusable Python
library. It separates synthetic data generation, model training, and model scoring,
and provides command-line scripts, tests, logging, and a GitHub Actions workflow.

## Project structure

```text
.
├── .github/workflows/ci.yml       # Style, test, build, and install workflow
├── scripts/                       # Command-line workflow entrypoints
├── src/fsds_mle_assignment/       # Installable Python package
├── tests/                         # Unit and package import tests
├── env.yml                        # Conda environment used by CI
├── pyproject.toml                 # Package metadata and build configuration
└── setup.cfg                      # Flake8 and pytest configuration
```

## Workflow overview

The workflow has three stages:

1. **Ingest:** generate a reproducible synthetic binary-classification dataset.
2. **Train:** fit and save a preprocessing-plus-classifier pipeline.
3. **Score:** evaluate the saved model against the validation CSV and save metrics.

The output folders are created automatically when they do not already exist.

## Data generation and schema

[`generate_dataset`](src/fsds_mle_assignment/data_ingestion.py) uses
`sklearn.datasets.make_classification` to create synthetic data. By default it
generates 1,000 rows in one dataset and writes the first 800 rows to `train.csv`
and the remaining 200 rows to `validation.csv`. This is a contiguous slice of the
generated rows, not a separate stratified split.

The generated classification problem uses 12 numeric features:

| Column | Description |
| --- | --- |
| `feature_0` through `feature_11` | Generated numeric input features |
| `target` | Binary class label (`0` or `1`) |

The generator uses 8 informative features, 2 redundant features, 2 classes, one
cluster per class, a class separation of 1.8, and a 3% label-flip rate. Its default
random seed is `42`, so the generated files are reproducible for the same arguments
and dependency versions.

You can change the number of train and validation rows and the random seed with
the ingestion script options described below.

## Model training

[`train_model`](src/fsds_mle_assignment/training.py) reads `train.csv`, treats all
columns except `target` as features, and uses `target` as the label. It trains a
scikit-learn `Pipeline` consisting of:

1. `StandardScaler`, to standardize the feature values.
2. `LogisticRegression`, configured with `max_iter=1000`.

The trained pipeline is saved as `model.joblib`. Training also writes
`training_metrics.json`, which contains the model path and training-set accuracy.
That training accuracy is a quick check on the training data; use the validation
metrics from the scoring step to assess held-out performance.

## Scoring and output metrics

[`score_model`](src/fsds_mle_assignment/scoring.py) loads `model.joblib`, predicts
labels for the rows in `validation.csv`, and writes `scores.json` containing:

| Metric | Meaning |
| --- | --- |
| `accuracy` | Fraction of validation labels predicted correctly |
| `precision` | Fraction of positive predictions that are correct |
| `recall` | Fraction of actual positive labels identified |
| `f1` | Harmonic mean of precision and recall |
| `samples` | Number of validation rows scored |

Precision, recall, and F1 use scikit-learn's binary classification defaults
(positive label `1`); `zero_division=0` makes the metrics return zero rather than
raising a warning if a denominator is zero.

## Install and run locally

Run these commands from the project root. Use the same Python interpreter for
installation and test execution:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

`python -m pytest` runs pytest using the selected `python` interpreter. This avoids
confusion if a standalone `pytest` command on `PATH` belongs to another Python
environment.

To create and activate the Conda environment defined for this project instead:

```powershell
conda env create -f env.yml
conda activate fsds-mle-assignment
python -m pip install -e ".[dev]"
python -m pytest -q
```

### Run the complete workflow

Run each script from the project root, in order:

```powershell
python scripts/ingest_data.py --output-dir .\data
python scripts/train.py --dataset-dir .\data --output-dir .\models
python scripts/score.py --model-dir .\models --dataset-dir .\data --output-dir .\scores
```

The workflow creates:

```text
data/
├── train.csv
└── validation.csv
models/
├── model.joblib
└── training_metrics.json
scores/
└── scores.json
```

### Script options

Each script accepts `--log-level`, `--log-path`, and `--no-console-log`:

| Option | Behavior |
| --- | --- |
| `--log-level LEVEL` | Set the minimum log level, for example `DEBUG`, `INFO`, or `WARNING`. Default: `INFO`. |
| `--log-path PATH` | Also write logs to the specified file. By default no log file is written. Parent directories are created if necessary. |
| `--no-console-log` | Disable console logs. By default logs are written to the console. |

The ingestion script also accepts:

| Option | Default | Purpose |
| --- | --- | --- |
| `--output-dir PATH` | Required | Directory for `train.csv` and `validation.csv` |
| `--train-size N` | `800` | Number of training rows |
| `--validation-size N` | `200` | Number of validation rows |
| `--random-state N` | `42` | Dataset generation seed |

The training script accepts:

| Option | Default | Purpose |
| --- | --- | --- |
| `--dataset-dir PATH` | Required | Directory containing `train.csv` |
| `--output-dir PATH` | Required | Directory for the model and training metrics |
| `--random-state N` | `42` | Random state passed to logistic regression |

The scoring script accepts:

| Option | Default | Purpose |
| --- | --- | --- |
| `--model-dir PATH` | Required | Directory containing `model.joblib` |
| `--dataset-dir PATH` | Required | Directory containing `validation.csv` |
| `--output-dir PATH` | Required | Directory for `scores.json` |

For example, to write training logs to a file while keeping console output:

```powershell
python scripts/train.py `
  --dataset-dir .\data `
  --output-dir .\models `
  --log-level DEBUG `
  --log-path .\logs\train.log
```

To disable console output and log only to a file, add `--no-console-log`.

## Logging

Each script configures the **root logger**, so messages from both the script and
package modules are handled consistently. Log records include a timestamp,
severity level, logger name, and message, for example:

```text
2026-10-13 12:00:00 INFO fsds_mle_assignment.data_ingestion - Created dataset CSVs ...
```

The scripts log workflow start/completion. Package functions log dataset file
creation and row counts, successful model training and training accuracy, and
validation accuracy, precision, recall, and F1. Logs go to the console by default;
file logging is opt-in with `--log-path`.

## Tests and quality checks

Run the tests with:

```powershell
python -m pytest -q
```

The test suite checks:

- Dataset generation creates both CSV files with a `target` column, non-empty
  train/validation data, and two classes in the training labels.
- Training saves the model and training-metrics JSON and achieves the test's
  expected minimum training accuracy.
- Scoring saves validation metrics and returns accuracy, F1, and a positive sample
  count.
- The installed package can be imported and exposes its version.

Formatting and lint checks:

```powershell
python -m isort .
python -m black .
python -m flake8 .
```

The CI workflow runs style checks, tests, package build, and an installation
workflow on GitHub Actions.
