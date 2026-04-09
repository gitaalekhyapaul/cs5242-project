from __future__ import annotations

import random
import sys
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steam_crawler.http_client import compute_backoff_delay, parse_retry_after


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


if __name__ == "__main__":
    unittest.main()

