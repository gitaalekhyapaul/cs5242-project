from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path


LOGGER_NAME = "steam_crawler"


def setup_logger(log_dir: Path) -> logging.Logger:
    """Create the shared run logger once per process."""

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


ERROR_LOG_FIELDS = [
    "stage",
    "appid",
    "url",
    "params_json",
    "attempt",
    "status_code",
    "response_headers_json",
    "response_body",
    "exception_type",
    "exception_message",
    "retry_after_seconds",
    "logged_at",
]


@dataclass(slots=True)
class CsvErrorLogger:
    """Append structured API failure rows to a CSV debugging log."""

    log_path: Path

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=ERROR_LOG_FIELDS)
                writer.writeheader()

    def log(self, row: dict[str, object]) -> None:
        """Write one error record with the declared schema and blank-fill missing fields."""

        safe_row = {field: row.get(field, "") for field in ERROR_LOG_FIELDS}
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ERROR_LOG_FIELDS)
            writer.writerow(safe_row)
