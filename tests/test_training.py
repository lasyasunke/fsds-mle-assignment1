from fsds_mle_assignment.data_ingestion import generate_dataset
from fsds_mle_assignment.training import train_model


def test_train_model_saves_model_and_reports_accuracy(tmp_path):
    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"

    generate_dataset(dataset_dir)
    metrics = train_model(dataset_dir, model_dir)

    assert (model_dir / "model.joblib").exists()
    assert (model_dir / "training_metrics.json").exists()
    assert metrics["train_accuracy"] > 0.8
