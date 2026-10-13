import pandas as pd

from fsds_mle_assignment.data_ingestion import generate_dataset


def test_generate_dataset_creates_train_and_validation_csv(tmp_path):
    train_path, validation_path = generate_dataset(tmp_path)

    assert train_path.exists()
    assert validation_path.exists()

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    assert "target" in train_df.columns
    assert "target" in validation_df.columns
    assert len(train_df) > 0
    assert len(validation_df) > 0
    assert train_df["target"].nunique() == 2
