import logging
from pathlib import Path


def configure_root_logger(
    log_level: str = "INFO", log_path: str | None = None, console_log: bool = True
):
    """Configure the application root logger for scripts in the ML workflow."""
    root_logger = logging.getLogger()
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console_log:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger
