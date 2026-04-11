from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import requests

from .config import Config
from .logging_utils import CsvErrorLogger
from .transforms import minified_json, utc_timestamp


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def parse_retry_after(
    headers: dict[str, str], now: datetime | None = None
) -> float | None:
    if not headers:
        return None

    current_time = now or datetime.now(timezone.utc)
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - current_time).total_seconds())
            except (TypeError, ValueError):
                pass

    for key, value in headers.items():
        lowered = key.lower()
        if "reset" not in lowered and "retry" not in lowered:
            continue
        try:
            numeric_value = float(value)
        except ValueError:
            continue
        if numeric_value > 1_000_000:
            return max(0.0, numeric_value - current_time.timestamp())
        return max(0.0, numeric_value)
    return None


def compute_backoff_delay(
    *,
    attempt: int,
    base_delay: float,
    max_delay: float,
    headers: dict[str, str] | None = None,
    rng: random.Random | None = None,
    now: datetime | None = None,
) -> float:
    hinted_delay = parse_retry_after(headers or {}, now=now)
    if hinted_delay is not None:
        return min(max_delay, hinted_delay)

    random_source = rng or random.Random()
    raw_delay = min(max_delay, base_delay * (2 ** max(0, attempt - 1)))
    return random_source.uniform(0.0, raw_delay)


class HttpClient:
    """Shared Steam HTTP client with per-host throttling and retry/backoff handling."""

    def __init__(
        self,
        config: Config,
        *,
        logger: Any,
        error_logger: CsvErrorLogger,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.error_logger = error_logger
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "steam-crawler/0.1",
                "Accept": "application/json",
            }
        )
        self.retry_count = 0
        self.error_count = 0
        self._last_request_at: dict[str, float] = {}
        self._rng = random.Random(config.random_seed)

    def _request_bucket(self, url: str) -> str:
        parsed = urlparse(url)
        host = parsed.netloc
        path = parsed.path or ""
        if host == "api.steampowered.com" or (
            host == "gpaul.cc" and path.startswith("/steamapi/")
        ):
            return "steam_api"
        if host == "store.steampowered.com" or (
            host == "gpaul.cc" and path.startswith("/steamstore/")
        ):
            return "steam_store"
        return host

    def _host_delay(self, url: str) -> float:
        bucket = self._request_bucket(url)
        if bucket == "steam_api":
            return self.config.api_host_delay_sec
        if bucket == "steam_store":
            return self.config.store_host_delay_sec
        return self.config.default_host_delay_sec

    def _throttle(self, url: str) -> None:
        bucket = self._request_bucket(url)
        delay = self._host_delay(url)
        if delay <= 0:
            return
        now = time.monotonic()
        last_request_at = self._last_request_at.get(bucket)
        if last_request_at is not None:
            elapsed = now - last_request_at
            if elapsed < delay:
                time.sleep(delay - elapsed)
        self._last_request_at[bucket] = time.monotonic()

    def _record_error(
        self,
        *,
        stage: str,
        appid: int | None,
        url: str,
        params: dict[str, object] | None,
        attempt: int,
        status_code: int | None,
        headers: dict[str, str] | None,
        body: str,
        exception: Exception | None,
        retry_after_seconds: float | None,
    ) -> None:
        exception_summary = ""
        if exception is not None:
            exception_summary = f"{type(exception).__name__}: {exception}"
        self.error_count += 1
        self.error_logger.log(
            {
                "stage": stage,
                "appid": appid if appid is not None else "",
                "url": url,
                "params_json": minified_json(params or {}),
                "attempt": attempt,
                "status_code": status_code or "",
                "response_headers_json": minified_json(headers or {}),
                "response_body": body,
                "exception_type": type(exception).__name__ if exception else "",
                "exception_message": str(exception) if exception else "",
                "retry_after_seconds": (
                    retry_after_seconds if retry_after_seconds is not None else ""
                ),
                "logged_at": utc_timestamp(),
            }
        )
        self.logger.warning(
            "HTTP error | stage=%s | appid=%s | attempt=%s | status=%s | retry_after=%s | exception=%s | headers=%s | body=%s",
            stage,
            appid,
            attempt,
            status_code,
            retry_after_seconds,
            exception_summary,
            headers or {},
            body,
        )

    def get_json(
        self,
        url: str,
        *,
        stage: str,
        appid: int | None = None,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        last_exception: Exception | None = None
        for attempt in range(1, self.config.max_retries + 2):
            headers: dict[str, str] | None = None
            body = ""
            status_code: int | None = None
            try:
                # Keep request pacing predictable per host before hitting the Steam endpoints.
                self._throttle(url)
                response = self.session.get(
                    url, params=params, timeout=self.config.request_timeout_sec
                )
                headers = dict(response.headers)
                status_code = response.status_code
                body = response.text
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as exc:
                last_exception = exc
                should_retry = (
                    status_code in RETRYABLE_STATUS_CODES
                    and attempt <= self.config.max_retries
                )
                retry_after_seconds = None
                if should_retry:
                    retry_after_seconds = compute_backoff_delay(
                        attempt=attempt,
                        base_delay=self.config.base_backoff_sec,
                        max_delay=self.config.max_backoff_sec,
                        headers=headers,
                        rng=self._rng,
                    )
                self._record_error(
                    stage=stage,
                    appid=appid,
                    url=url,
                    params=params,
                    attempt=attempt,
                    status_code=status_code,
                    headers=headers,
                    body=body,
                    exception=exc,
                    retry_after_seconds=retry_after_seconds,
                )
                if not should_retry:
                    break
                self.retry_count += 1
                time.sleep(retry_after_seconds or 0)
            except (requests.RequestException, ValueError) as exc:
                last_exception = exc
                retry_after_seconds = None
                if attempt <= self.config.max_retries:
                    retry_after_seconds = compute_backoff_delay(
                        attempt=attempt,
                        base_delay=self.config.base_backoff_sec,
                        max_delay=self.config.max_backoff_sec,
                        headers=headers,
                        rng=self._rng,
                    )
                self._record_error(
                    stage=stage,
                    appid=appid,
                    url=url,
                    params=params,
                    attempt=attempt,
                    status_code=status_code,
                    headers=headers,
                    body=body,
                    exception=exc,
                    retry_after_seconds=retry_after_seconds,
                )
                if attempt > self.config.max_retries:
                    break
                self.retry_count += 1
                time.sleep(retry_after_seconds or 0)

        raise RuntimeError(
            f"Failed request for stage={stage}, appid={appid}, url={url}"
        ) from last_exception
