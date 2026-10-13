import argparse
import logging
from pathlib import Path

from fsds_mle_assignment.logging_utils import configure_root_logger
from fsds_mle_assignment.scoring import score_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a trained model against validation data."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing the trained model joblib artifact.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing the validation.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for JSON scoring output.",
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

    logger.info("Starting the scoring workflow.")
    metrics = score_model(args.model_dir, args.dataset_dir, args.output_dir)
    logger.info("Scoring workflow completed successfully. %s", metrics)


if __name__ == "__main__":
    main()
