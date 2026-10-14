import argparse
import logging
from pathlib import Path

import mlflow

from fsds_mle_assignment.logging_utils import configure_root_logger

# Import execution modules from scripts
from scripts.ingest_data import run_ingest
from scripts.score import run_score
from scripts.train import run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run end-to-end ML workflow under a parent MLflow run."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory where CSV files are written.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./models"),
        help="Directory where model artifacts are saved.",
    )
    parser.add_argument(
        "--score-dir",
        type=Path,
        default=Path("./scores"),
        help="Directory for output scoring JSONs.",
    )
    parser.add_argument(
        "--train-size", type=int, default=800, help="Total number of training samples."
    )
    parser.add_argument(
        "--validation-size", type=int, default=200, help="Number of validation samples."
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Seed for reproducibility."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="FSDS_MLE_Assignment",
        help="MLflow Experiment Name.",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level, e.g. DEBUG, INFO, WARNING."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_root_logger(args.log_level)
    logger = logging.getLogger(__name__)

    # Ensure output folders exist
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.score_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(args.experiment_name)

    logger.info("Starting Master Pipeline MLflow Run...")

    with mlflow.start_run(run_name="Master_Pipeline_Workflow") as parent_run:
        parent_id = parent_run.info.run_id
        logger.info("Parent MLflow Run ID: %s", parent_id)

        # Log Top-Level Pipeline Parameters
        mlflow.log_params(
            {
                "pipeline_random_state": args.random_state,
                "train_size": args.train_size,
                "validation_size": args.validation_size,
            }
        )

        # Step 1: Ingest (Child Run 1)
        logger.info("--- Step 1: Data Ingestion ---")
        with mlflow.start_run(run_name="01_ingest_data", nested=True) as child1:
            run_ingest(
                output_dir=args.data_dir,
                train_size=args.train_size,
                validation_size=args.validation_size,
                random_state=args.random_state,
            )
            logger.info("Completed Step 1 (Child Run ID: %s)", child1.info.run_id)

        # Step 2: Train (Child Run 2)
        logger.info("--- Step 2: Model Training ---")
        with mlflow.start_run(run_name="02_train_model", nested=True) as child2:
            run_train(
                dataset_dir=args.data_dir,
                output_dir=args.model_dir,
                random_state=args.random_state,
            )
            logger.info("Completed Step 2 (Child Run ID: %s)", child2.info.run_id)

        # Step 3: Score (Child Run 3)
        logger.info("--- Step 3: Model Scoring ---")
        with mlflow.start_run(run_name="03_score_model", nested=True) as child3:
            run_score(
                model_dir=args.model_dir,
                dataset_dir=args.data_dir,
                output_dir=args.score_dir,
            )
            logger.info("Completed Step 3 (Child Run ID: %s)", child3.info.run_id)

    logger.info("Master Pipeline Run completed successfully!")


if __name__ == "__main__":
    main()
