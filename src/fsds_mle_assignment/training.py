import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def train_model(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    random_state: int = 42,
):
    """Train a logistic-regression model and save it to disk."""
    dataset_path = Path(dataset_dir)
    train_file = dataset_path / "train.csv"
    if not train_file.exists():
        raise FileNotFoundError(f"The training data file was not found: {train_file}")

    train_df = pd.read_csv(train_file)
    features = train_df.drop(columns=["target"])
    labels = train_df["target"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )

    model.fit(features, labels)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_file = output_path / "model.joblib"
    joblib.dump(model, model_file)

    train_accuracy = float(model.score(features, labels))
    metrics = {
        "model_path": str(model_file),
        "train_accuracy": train_accuracy,
    }

    metrics_file = output_path / "training_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info(
        "Model trained successfully with training accuracy %.4f.", train_accuracy
    )
    return metrics
