from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steam_crawler.logging_utils import LOGGER_NAME, setup_logger


def clear_logger_handlers() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


class SetupLoggerTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_logger_handlers()

    def test_setup_logger_retargets_file_handler_for_new_log_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            first_dir = root / "logs-one"
            second_dir = root / "logs-two"

            logger = setup_logger(first_dir)
            logger = setup_logger(second_dir)

            file_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]
            self.assertEqual(len(file_handlers), 1)
            self.assertEqual(Path(file_handlers[0].baseFilename), (second_dir / "run.log").resolve())


if __name__ == "__main__":
    unittest.main()
