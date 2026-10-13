import argparse
import logging
from pathlib import Path

from fsds_mle_assignment.data_ingestion import generate_dataset
from fsds_mle_assignment.logging_utils import configure_root_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate training and validation datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where CSV files are written.",
    )
    parser.add_argument(
        "--train-size", type=int, default=800, help="Total number of training samples."
    )
    parser.add_argument(
        "--validation-size", type=int, default=200, help="Number of validation samples."
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Seed for dataset generation."
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


def main() -> None:
    args = build_parser().parse_args()
    configure_root_logger(
        args.log_level, args.log_path, console_log=not args.no_console_log
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting the data ingestion workflow.")
    train_path, validation_path = generate_dataset(
        args.output_dir,
        train_size=args.train_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
    logger.info(
        "Ingestion complete. Training data: %s; validation data: %s",
        train_path,
        validation_path,
    )


if __name__ == "__main__":
    main()
