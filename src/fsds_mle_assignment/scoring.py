import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def score_model(
    model_dir: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
):
    """Load a trained model and score it against the validation dataset."""
    model_path = Path(model_dir) / "model.joblib"
    dataset_path = Path(dataset_dir)
    validation_file = dataset_path / "validation.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"The trained model file was not found: {model_path}")
    if not validation_file.exists():
        raise FileNotFoundError(
            f"The validation dataset file was not found: {validation_file}"
        )

    model = joblib.load(model_path)
    validation_df = pd.read_csv(validation_file)
    feature_columns = [column for column in validation_df.columns if column != "target"]
    features = validation_df[feature_columns]
    labels = validation_df["target"]
    predictions = model.predict(features)

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "samples": int(len(labels)),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    score_file = output_path / "scores.json"
    score_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info(
        "Validation metrics: accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f.",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    return metrics
