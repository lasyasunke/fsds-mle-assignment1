import argparse
import logging
from pathlib import Path

import mlflow

from fsds_mle_assignment.logging_utils import configure_root_logger
from fsds_mle_assignment.training import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the ML model on a generated dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing the train.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where model artifacts are saved.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the model training process.",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level, e.g. DEBUG, INFO, WARNING."
    )
    parser.add_argument(
        "--log-path", type=str, default=None, help="Optional log file path."
    )
    parser.add_argument(
        "--no-console-log", action="store_true", help="Disable console logging."
    )
    return parser


def run_train(dataset_dir: Path, output_dir: Path, random_state: int) -> dict:
    mlflow.log_params(
        {
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "random_state": random_state,
        }
    )

    metrics = train_model(dataset_dir, output_dir, random_state=random_state)

    if isinstance(metrics, dict):
        for metric_name, val in metrics.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"train_{metric_name}", float(val))

    return metrics


def main() -> None:
    args = build_parser().parse_args()
    configure_root_logger(
        args.log_level, args.log_path, console_log=not args.no_console_log
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting the training workflow.")

    if mlflow.active_run() is None:
        with mlflow.start_run(run_name="model_training"):
            metrics = run_train(args.dataset_dir, args.output_dir, args.random_state)
    else:
        metrics = run_train(args.dataset_dir, args.output_dir, args.random_state)

    logger.info("Training workflow completed successfully. %s", metrics)


if __name__ == "__main__":
    main()
