from __future__ import annotations

import logging
import random
import sys
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steam_crawler.config import Config
from steam_crawler.http_client import HttpClient, compute_backoff_delay, parse_retry_after


class RecordingErrorLogger:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def log(self, row: dict[str, object]) -> None:
        self.rows.append(row)


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        json_payload: dict[str, object] | None = None,
        text: str = "",
        headers: dict[str, str] | None = None,
        url: str = "https://example.test/api",
        reason: str = "error",
    ) -> None:
        self.status_code = status_code
        self._json_payload = json_payload or {}
        self.text = text
        self.headers = headers or {}
        self.url = url
        self.reason = reason

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} {self.reason} for url: {self.url}",
                response=self,
            )

    def json(self) -> dict[str, object]:
        return self._json_payload


def build_config(root: Path, **overrides: object) -> Config:
    defaults = {
        "root_dir": root,
        "steam_api_key": "test-key",
        "data_dir": root / "data",
        "log_dir": root / "logs",
        "request_timeout_sec": 0.1,
        "max_retries": 0,
        "base_backoff_sec": 0.0,
        "max_backoff_sec": 0.0,
        "api_host_delay_sec": 0.0,
        "store_host_delay_sec": 0.0,
        "default_host_delay_sec": 0.0,
    }
    defaults.update(overrides)
    return Config(**defaults)


class RetryDelayTests(unittest.TestCase):
    def test_parse_numeric_retry_after(self) -> None:
        delay = parse_retry_after({"Retry-After": "7"})
        self.assertEqual(delay, 7.0)

    def test_parse_http_date_retry_after(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        retry_at = now + timedelta(seconds=90)
        delay = parse_retry_after({"Retry-After": format_datetime(retry_at, usegmt=True)}, now=now)
        self.assertEqual(delay, 90.0)

    def test_parse_reset_epoch_header(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        future_epoch = now.timestamp() + 45
        delay = parse_retry_after({"X-RateLimit-Reset": str(future_epoch)}, now=now)
        self.assertEqual(delay, 45.0)

    def test_compute_backoff_uses_jitter_without_headers(self) -> None:
        delay = compute_backoff_delay(
            attempt=3,
            base_delay=2.0,
            max_delay=60.0,
            headers={},
            rng=random.Random(0),
        )
        self.assertGreaterEqual(delay, 0.0)
        self.assertLessEqual(delay, 8.0)


class HttpClientErrorHandlingTests(unittest.TestCase):
    def test_gpaul_proxy_urls_map_to_distinct_delay_buckets(self) -> None:
        with TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            client = HttpClient(
                build_config(
                    root,
                    api_host_delay_sec=1.5,
                    store_host_delay_sec=2.5,
                    default_host_delay_sec=3.5,
                ),
                logger=Mock(spec=logging.Logger),
                error_logger=RecordingErrorLogger(),
                session=Mock(),
            )

            self.assertEqual(
                client._host_delay("https://gpaul.cc/steamapi/IStoreService/GetAppList/v1/"),
                1.5,
            )
            self.assertEqual(
                client._host_delay("https://gpaul.cc/steamstore/api/appdetails"),
                2.5,
            )
            self.assertEqual(
                client._request_bucket("https://gpaul.cc/steamapi/IStoreService/GetAppList/v1/"),
                "steam_api",
            )
            self.assertEqual(
                client._request_bucket("https://gpaul.cc/steamstore/api/appdetails"),
                "steam_store",
            )

    def test_final_retryable_http_error_is_logged_once(self) -> None:
        with TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            logger = Mock(spec=logging.Logger)
            error_logger = RecordingErrorLogger()
            session = Mock()
            session.get.return_value = FakeResponse(
                status_code=429,
                text="rate limited",
                headers={"Retry-After": "1"},
                reason="Too Many Requests",
            )
            client = HttpClient(
                build_config(root, max_retries=0),
                logger=logger,
                error_logger=error_logger,
                session=session,
            )

            with self.assertRaises(RuntimeError):
                client.get_json("https://example.test/api", stage="stage_02", appid=10)

            self.assertEqual(session.get.call_count, 1)
            self.assertEqual(len(error_logger.rows), 1)
            self.assertEqual(logger.warning.call_count, 1)

    def test_non_retryable_http_error_does_not_retry(self) -> None:
        with TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            logger = Mock(spec=logging.Logger)
            error_logger = RecordingErrorLogger()
            session = Mock()
            session.get.return_value = FakeResponse(
                status_code=404,
                text="not found",
                reason="Not Found",
            )
            client = HttpClient(
                build_config(root, max_retries=3),
                logger=logger,
                error_logger=error_logger,
                session=session,
            )

            with self.assertRaises(RuntimeError):
                client.get_json("https://example.test/api", stage="stage_02", appid=10)

            self.assertEqual(session.get.call_count, 1)
            self.assertEqual(len(error_logger.rows), 1)

    def test_warning_log_includes_exception_summary_for_timeouts(self) -> None:
        with TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            logger = Mock(spec=logging.Logger)
            error_logger = RecordingErrorLogger()
            session = Mock()
            session.get.side_effect = requests.Timeout("socket timed out")
            client = HttpClient(
                build_config(root, max_retries=0),
                logger=logger,
                error_logger=error_logger,
                session=session,
            )

            with self.assertRaises(RuntimeError):
                client.get_json("https://example.test/api", stage="stage_02", appid=10)

            warning_args = logger.warning.call_args.args
            self.assertIn("exception=%s", warning_args[0])
            self.assertEqual(warning_args[6], "Timeout: socket timed out")


if __name__ == "__main__":
    unittest.main()
