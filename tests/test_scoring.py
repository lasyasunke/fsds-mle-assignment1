from fsds_mle_assignment.data_ingestion import generate_dataset
from fsds_mle_assignment.scoring import score_model
from fsds_mle_assignment.training import train_model


def test_score_model_outputs_validation_metrics(tmp_path):
    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "scores"

    generate_dataset(dataset_dir)
    train_model(dataset_dir, model_dir)
    metrics = score_model(model_dir, dataset_dir, output_dir)

    assert (output_dir / "scores.json").exists()
    assert metrics["accuracy"] > 0.8
    assert "f1" in metrics
    assert metrics["samples"] > 0
