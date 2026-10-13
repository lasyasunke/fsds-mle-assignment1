import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

logger = logging.getLogger(__name__)


def generate_dataset(
    output_dir: str | Path,
    *,
    train_size: int = 800,
    validation_size: int = 200,
    random_state: int = 42,
):
    """Generate a small synthetic classification dataset and save it to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_samples = train_size + validation_size
    features, labels = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.8,
        flip_y=0.03,
        random_state=random_state,
    )

    feature_names = [f"feature_{index}" for index in range(features.shape[1])]
    train_df = pd.DataFrame(features[:train_size], columns=feature_names)
    train_df["target"] = labels[:train_size]

    validation_df = pd.DataFrame(features[train_size:], columns=feature_names)
    validation_df["target"] = labels[train_size:]

    train_path = output_path / "train.csv"
    validation_path = output_path / "validation.csv"

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)

    logger.info(
        "Created dataset CSVs at %s and %s with %s train rows and %s validation rows.",
        train_path,
        validation_path,
        train_size,
        validation_size,
    )

    return train_path, validation_path
