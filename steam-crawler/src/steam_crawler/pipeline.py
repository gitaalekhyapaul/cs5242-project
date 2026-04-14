from __future__ import annotations

import argparse
import csv
import gzip
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator

from tqdm.auto import tqdm as auto_tqdm

from .config import Config
from .http_client import HttpClient
from .logging_utils import CsvErrorLogger, setup_logger
from .transforms import (
    flatten_app_catalog_row,
    flatten_app_details,
    flatten_review_row,
    merge_catalog_and_details,
    sample_rows,
    utc_timestamp,
)

STAGE_01_FIELDS = ["appid", "name", "last_modified", "price_change_number", "raw_json"]
STAGE_02_FIELDS = [
    "appid",
    "success",
    "type",
    "category_ids",
    "category_descriptions",
    "recommendations_total",
    "raw_json",
]
STAGE_03_FIELDS = [
    "appid",
    "name",
    "last_modified",
    "price_change_number",
    "raw_app_json",
    "details_success",
    "type",
    "category_ids",
    "category_descriptions",
    "recommendations_total",
    "raw_details_json",
    "eligible_for_sampling",
]
STAGE_04_FIELDS = STAGE_03_FIELDS + ["sample_rank", "random_seed", "sampled_at"]
STAGE_05_FIELDS = [
    "appid",
    "recommendationid",
    "author_steamid",
    "timestamp_created",
    "review_text",
    "source_stream",
    "raw_json",
]
STAGE_05_PROGRESS_FIELDS = [
    "appid",
    "status",
    "phase",
    "recent_cursor",
    "helpful_cursor",
    "recent_count",
    "helpful_count",
    "total_unique",
    "recent_exhausted",
    "helpful_exhausted",
    "started_at",
    "finished_at",
    "error",
]


def _is_notebook_runtime() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def _progress_bar(*args, **kwargs):
    kwargs.setdefault("file", sys.stdout)
    # Prefer notebook-native progress bars when the pipeline is invoked from Jupyter.
    if _is_notebook_runtime():
        from tqdm.notebook import tqdm as notebook_tqdm

        return notebook_tqdm(*args, **kwargs)
    return auto_tqdm(*args, **kwargs)


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


@dataclass(slots=True)
class StageResult:
    stage_name: str
    output_path: Path
    rows_written: int
    skipped: bool
    elapsed_seconds: float
    retry_count: int
    error_count: int


@dataclass(slots=True)
class StagePaths:
    """Filesystem layout for all cacheable stage outputs."""

    data_dir: Path

    @property
    def stage_01(self) -> Path:
        return self.data_dir / "stage_01_apps_catalog.csv"

    @property
    def stage_02(self) -> Path:
        return self.data_dir / "stage_02_app_details.csv.gz"

    @property
    def stage_03(self) -> Path:
        return self.data_dir / "stage_03_apps_with_metadata.csv.gz"

    @property
    def stage_04(self) -> Path:
        return self.data_dir / "stage_04_selected_games.csv"

    @property
    def stage_05(self) -> Path:
        return self.data_dir / "stage_05_reviews_dataset.csv.gz"

    @property
    def stage_05_progress(self) -> Path:
        return self.data_dir / "stage_05_progress.csv"


@dataclass(slots=True)
class ReviewCollectionState:
    appid: int
    status: str = "in_progress"
    phase: str = "recent_quota"
    recent_cursor: str = "*"
    helpful_cursor: str = "*"
    recent_count: int = 0
    helpful_count: int = 0
    total_unique: int = 0
    recent_exhausted: bool = False
    helpful_exhausted: bool = False
    started_at: str = ""
    finished_at: str = ""
    error: str = ""

    @classmethod
    def from_progress_row(cls, appid: int, row: dict[str, str] | None) -> "ReviewCollectionState":
        if not row:
            return cls(appid=appid, started_at=utc_timestamp())
        return cls(
            appid=appid,
            status=row.get("status", "in_progress") or "in_progress",
            phase=row.get("phase", "recent_quota") or "recent_quota",
            recent_cursor=row.get("recent_cursor", "*") or "*",
            helpful_cursor=row.get("helpful_cursor", "*") or "*",
            recent_count=int(row.get("recent_count", "0") or 0),
            helpful_count=int(row.get("helpful_count", "0") or 0),
            total_unique=int(row.get("total_unique", "0") or 0),
            recent_exhausted=_parse_bool(row.get("recent_exhausted", "")),
            helpful_exhausted=_parse_bool(row.get("helpful_exhausted", "")),
            started_at=row.get("started_at", "") or utc_timestamp(),
            finished_at=row.get("finished_at", ""),
            error=row.get("error", ""),
        )

    def to_progress_row(self) -> dict[str, object]:
        return {
            "appid": self.appid,
            "status": self.status,
            "phase": self.phase,
            "recent_cursor": self.recent_cursor,
            "helpful_cursor": self.helpful_cursor,
            "recent_count": self.recent_count,
            "helpful_count": self.helpful_count,
            "total_unique": self.total_unique,
            "recent_exhausted": self.recent_exhausted,
            "helpful_exhausted": self.helpful_exhausted,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def _iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with _open_text(path, "rt") as handle:
        reader = csv.DictReader(handle)
        yield from reader


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in _iter_csv_rows(path))


def _read_completed_ids(path: Path, key: str = "appid") -> set[int]:
    completed: set[int] = set()
    if not path.exists():
        return completed
    for row in _iter_csv_rows(path):
        value = row.get(key)
        if not value:
            continue
        completed.add(int(value))
    return completed


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]], append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "at" if append else "wt"
    write_header = not append or not path.exists()
    with _open_text(path, mode) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


class ReviewCollector:
    """Encapsulates the two-stream review collection strategy for a single app."""

    def __init__(self, *, config: Config, http_client: HttpClient, logger: logging.Logger) -> None:
        self.config = config
        self.http_client = http_client
        self.logger = logger

    def page_reviews(
        self,
        *,
        appid: int,
        review_filter: str,
        cursor: str,
        day_range: int | None = None,
    ) -> dict[str, object]:
        params: dict[str, object] = {
            "json": 1,
            "language": self.config.reviews_language,
            "filter": review_filter,
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": self.config.reviews_page_size,
            "cursor": cursor,
        }
        if day_range is not None:
            params["day_range"] = day_range
        return self.http_client.get_json(
            self.config.app_reviews_url(appid),
            stage="stage_05",
            appid=appid,
            params=params,
        )

    def _consume_page(
        self,
        *,
        appid: int,
        seen_ids: set[str],
        state: ReviewCollectionState,
        review_filter: str,
        source_stream: str,
        cursor: str,
        day_range: int | None,
        inner_progress,
        seen_stream_cursors: dict[str, set[str]],
        repeated_cursor_counts: dict[str, int],
    ) -> list[dict[str, object]]:
        payload = self.page_reviews(
            appid=appid,
            review_filter=review_filter,
            cursor=cursor,
            day_range=day_range,
        )
        reviews = payload.get("reviews", [])
        next_cursor = payload.get("cursor", cursor)
        collected: list[dict[str, object]] = []

        for review in reviews:
            recommendation_id = str(review.get("recommendationid", ""))
            if not recommendation_id or recommendation_id in seen_ids:
                continue
            seen_ids.add(recommendation_id)
            collected.append(flatten_review_row(appid, review, source_stream))
            if source_stream == "recent":
                state.recent_count += 1
            else:
                state.helpful_count += 1
            inner_progress.update(1)
            if source_stream == "recent" and state.phase == "recent_quota" and state.recent_count >= self.config.recent_quota:
                break
            if source_stream == "helpful" and state.phase == "helpful_fill" and state.helpful_count >= self.config.helpful_quota:
                break
            if state.total_unique + len(collected) >= self.config.reviews_per_game:
                break

        no_new_unique_reviews = not collected
        if next_cursor in seen_stream_cursors[source_stream] and no_new_unique_reviews:
            repeated_cursor_counts[source_stream] += 1
        else:
            repeated_cursor_counts[source_stream] = 0
        seen_stream_cursors[source_stream].add(next_cursor)

        exhausted = (
            not reviews
            or next_cursor == cursor
            or repeated_cursor_counts[source_stream] >= self.config.review_cursor_loop_limit
        )
        if repeated_cursor_counts[source_stream] >= self.config.review_cursor_loop_limit:
            self.logger.warning(
                "Stopping stage_05 %s pagination for app %s after %s repeated cursors without new reviews.",
                source_stream,
                appid,
                self.config.review_cursor_loop_limit,
            )
        if source_stream == "recent":
            state.recent_cursor = next_cursor
            state.recent_exhausted = exhausted
        else:
            state.helpful_cursor = next_cursor
            state.helpful_exhausted = exhausted
        state.total_unique = len(seen_ids)
        return collected

    def collect_for_app(
        self,
        *,
        appid: int,
        seen_ids: set[str],
        state: ReviewCollectionState,
        checkpoint: Callable[[list[dict[str, object]], ReviewCollectionState], None],
    ) -> ReviewCollectionState:
        inner_progress = _progress_bar(
            total=self.config.reviews_per_game,
            initial=state.total_unique,
            desc=f"App {appid} reviews",
            unit="reviews",
            leave=False,
        )
        seen_stream_cursors = {
            "recent": {state.recent_cursor},
            "helpful": {state.helpful_cursor},
        }
        repeated_cursor_counts = {"recent": 0, "helpful": 0}
        try:
            # Phase 1: take a recent-review slice first to preserve a time-oriented sample.
            while (
                state.recent_count < self.config.recent_quota
                and state.total_unique < self.config.reviews_per_game
                and not state.recent_exhausted
            ):
                state.phase = "recent_quota"
                page_rows = self._consume_page(
                    appid=appid,
                    seen_ids=seen_ids,
                    state=state,
                    review_filter="recent",
                    source_stream="recent",
                    cursor=state.recent_cursor,
                    day_range=None,
                    inner_progress=inner_progress,
                    seen_stream_cursors=seen_stream_cursors,
                    repeated_cursor_counts=repeated_cursor_counts,
                )
                checkpoint(page_rows, state)

            # Phase 2: take a helpful-ranking slice and deduplicate overlaps against the recent quota.
            while (
                state.helpful_count < self.config.helpful_quota
                and state.total_unique < self.config.reviews_per_game
                and not state.helpful_exhausted
            ):
                state.phase = "helpful_fill"
                page_rows = self._consume_page(
                    appid=appid,
                    seen_ids=seen_ids,
                    state=state,
                    review_filter="all",
                    source_stream="helpful",
                    cursor=state.helpful_cursor,
                    day_range=365,
                    inner_progress=inner_progress,
                    seen_stream_cursors=seen_stream_cursors,
                    repeated_cursor_counts=repeated_cursor_counts,
                )
                checkpoint(page_rows, state)

            # Phase 3: if the total is still low, continue paging recent reviews as backfill.
            while state.total_unique < self.config.reviews_per_game and not state.recent_exhausted:
                state.phase = "recent_backfill"
                page_rows = self._consume_page(
                    appid=appid,
                    seen_ids=seen_ids,
                    state=state,
                    review_filter="recent",
                    source_stream="recent",
                    cursor=state.recent_cursor,
                    day_range=None,
                    inner_progress=inner_progress,
                    seen_stream_cursors=seen_stream_cursors,
                    repeated_cursor_counts=repeated_cursor_counts,
                )
                checkpoint(page_rows, state)
        finally:
            inner_progress.close()

        return state


class Pipeline:
    """Notebook-friendly orchestrator for the staged Steam crawl."""

    def __init__(
        self,
        config: Config,
        *,
        logger: logging.Logger | None = None,
        error_logger: CsvErrorLogger | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        self.config = config
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logger(config.log_dir)
        self.error_logger = error_logger or CsvErrorLogger(config.log_dir / "errors.csv")
        self.paths = StagePaths(config.data_dir)
        self.http_client = http_client or HttpClient(
            config,
            logger=self.logger,
            error_logger=self.error_logger,
        )
        self.review_collector = ReviewCollector(
            config=config,
            http_client=self.http_client,
            logger=self.logger,
        )

    @property
    def stage_01_path(self) -> Path:
        return self.paths.stage_01

    @property
    def stage_02_path(self) -> Path:
        return self.paths.stage_02

    @property
    def stage_03_path(self) -> Path:
        return self.paths.stage_03

    @property
    def stage_04_path(self) -> Path:
        return self.paths.stage_04

    @property
    def stage_05_path(self) -> Path:
        return self.paths.stage_05

    @property
    def stage_05_progress_path(self) -> Path:
        return self.paths.stage_05_progress

    def _append_review_progress(self, state: ReviewCollectionState) -> None:
        _write_rows(
            self.stage_05_progress_path,
            STAGE_05_PROGRESS_FIELDS,
            [state.to_progress_row()],
            append=self.stage_05_progress_path.exists(),
        )

    def _load_existing_review_rows(self, appid: int) -> tuple[set[str], dict[str, int]]:
        seen_ids: set[str] = set()
        stats = {"recent_count": 0, "helpful_count": 0}
        if not self.stage_05_path.exists():
            return seen_ids, stats
        for row in _iter_csv_rows(self.stage_05_path):
            if int(row.get("appid", "0") or 0) != appid:
                continue
            recommendation_id = str(row.get("recommendationid", "") or "")
            if not recommendation_id or recommendation_id in seen_ids:
                continue
            seen_ids.add(recommendation_id)
            if row.get("source_stream") == "recent":
                stats["recent_count"] += 1
            else:
                stats["helpful_count"] += 1
        return seen_ids, stats

    def _restore_review_state(self, appid: int, row: dict[str, str] | None) -> tuple[ReviewCollectionState, set[str]]:
        state = ReviewCollectionState.from_progress_row(appid, row)
        seen_ids, stats = self._load_existing_review_rows(appid)
        state.recent_count = max(state.recent_count, stats["recent_count"])
        state.helpful_count = max(state.helpful_count, stats["helpful_count"])
        state.total_unique = len(seen_ids)
        if state.total_unique >= self.config.reviews_per_game:
            state.status = "completed"
        elif state.helpful_exhausted and state.recent_exhausted:
            state.status = "exhausted"
        else:
            state.status = "in_progress"
        return state, seen_ids

    def _result(
        self,
        stage_name: str,
        output_path: Path,
        rows_written: int,
        skipped: bool,
        start: float,
        *,
        retry_count_start: int,
        error_count_start: int,
    ) -> StageResult:
        result = StageResult(
            stage_name=stage_name,
            output_path=output_path,
            rows_written=rows_written,
            skipped=skipped,
            elapsed_seconds=perf_counter() - start,
            retry_count=self.http_client.retry_count - retry_count_start,
            error_count=self.http_client.error_count - error_count_start,
        )
        self.logger.info(
            "%s summary | skipped=%s | rows=%s | elapsed=%.2fs | retries=%s | errors=%s | output=%s",
            result.stage_name,
            result.skipped,
            result.rows_written,
            result.elapsed_seconds,
            result.retry_count,
            result.error_count,
            result.output_path,
        )
        return result

    def run_stage_01(self, *, force_refresh: bool = False, max_pages: int | None = None) -> StageResult:
        start = perf_counter()
        retry_count_start = self.http_client.retry_count
        error_count_start = self.http_client.error_count
        if self.stage_01_path.exists() and not force_refresh:
            self.logger.info("Reusing cached stage 01 output: %s", self.stage_01_path)
            return self._result(
                "stage_01",
                self.stage_01_path,
                _count_csv_rows(self.stage_01_path),
                True,
                start,
                retry_count_start=retry_count_start,
                error_count_start=error_count_start,
            )

        if force_refresh and self.stage_01_path.exists():
            self.stage_01_path.unlink()

        rows_written = 0
        last_appid: int | None = None
        page_count = 0
        progress = _progress_bar(desc="Stage 1 app list", unit="apps")
        try:
            while True:
                # Page through the Steam app list using the official cursor-ish last_appid contract.
                params: dict[str, object] = {
                    "key": self.config.steam_api_key,
                    "max_results": self.config.app_list_page_size,
                }
                if last_appid is not None:
                    params["last_appid"] = last_appid
                response = self.http_client.get_json(self.config.app_list_url, stage="stage_01", params=params)
                payload = response.get("response", {})
                apps = payload.get("apps", [])
                rows = [flatten_app_catalog_row(app) for app in apps]
                rows_written += _write_rows(self.stage_01_path, STAGE_01_FIELDS, rows, append=rows_written > 0)
                progress.update(len(rows))
                page_count += 1
                if max_pages is not None and page_count >= max_pages:
                    break
                if not payload.get("have_more_results"):
                    break
                last_appid = payload.get("last_appid")
        finally:
            progress.close()
        return self._result(
            "stage_01",
            self.stage_01_path,
            rows_written,
            False,
            start,
            retry_count_start=retry_count_start,
            error_count_start=error_count_start,
        )

    def run_stage_02(self, *, force_refresh: bool = False, max_apps: int | None = None) -> StageResult:
        start = perf_counter()
        retry_count_start = self.http_client.retry_count
        error_count_start = self.http_client.error_count
        if not self.stage_01_path.exists():
            raise FileNotFoundError("Stage 01 output is required before running stage 02.")

        if force_refresh:
            for path in [self.stage_02_path]:
                if path.exists():
                    path.unlink()

        completed_ids = _read_completed_ids(self.stage_02_path)
        source_rows = list(_iter_csv_rows(self.stage_01_path))
        if max_apps is not None:
            source_rows = source_rows[:max_apps]
        total_apps = len(source_rows)
        scoped_appids = {int(row["appid"]) for row in source_rows}
        completed_scoped_ids = completed_ids & scoped_appids
        if total_apps > 0 and len(completed_scoped_ids) >= total_apps and not force_refresh:
            self.logger.info("Reusing cached stage 02 output: %s", self.stage_02_path)
            return self._result(
                "stage_02",
                self.stage_02_path,
                _count_csv_rows(self.stage_02_path),
                True,
                start,
                retry_count_start=retry_count_start,
                error_count_start=error_count_start,
            )
        progress = _progress_bar(total=total_apps, initial=len(completed_scoped_ids), desc="Stage 2 appdetails", unit="apps")
        try:
            for row in source_rows:
                appid = int(row["appid"])
                if appid in completed_ids:
                    continue
                params = {
                    "appids": appid,
                    "cc": self.config.appdetails_country_code,
                    "l": self.config.appdetails_language,
                    # `type` only comes through when `basic` is included in the filter list.
                    "filters": "basic,categories,recommendations",
                }
                try:
                    # Persist rows incrementally so reruns only retry the missing appids.
                    payload = self.http_client.get_json(
                        self.config.app_details_url,
                        stage="stage_02",
                        appid=appid,
                        params=params,
                    )
                except RuntimeError as exc:
                    self.logger.error("Skipping appdetails for %s after retries: %s", appid, exc)
                    progress.update(1)
                    continue
                # Persist every successful appdetails row so reruns only revisit missing appids.
                _write_rows(
                    self.stage_02_path,
                    STAGE_02_FIELDS,
                    [flatten_app_details(appid, payload)],
                    append=self.stage_02_path.exists(),
                )
                progress.update(1)
        finally:
            progress.close()

        return self._result(
            "stage_02",
            self.stage_02_path,
            _count_csv_rows(self.stage_02_path),
            False,
            start,
            retry_count_start=retry_count_start,
            error_count_start=error_count_start,
        )

    def run_stage_03(self, *, force_refresh: bool = False, max_apps: int | None = None) -> StageResult:
        start = perf_counter()
        retry_count_start = self.http_client.retry_count
        error_count_start = self.http_client.error_count
        if not self.stage_01_path.exists() or not self.stage_02_path.exists():
            raise FileNotFoundError("Stages 01 and 02 outputs are required before running stage 03.")

        if self.stage_03_path.exists() and not force_refresh:
            self.logger.info("Reusing cached stage 03 output: %s", self.stage_03_path)
            return self._result(
                "stage_03",
                self.stage_03_path,
                _count_csv_rows(self.stage_03_path),
                True,
                start,
                retry_count_start=retry_count_start,
                error_count_start=error_count_start,
            )
        if force_refresh and self.stage_03_path.exists():
            self.stage_03_path.unlink()

        details_by_appid = {row["appid"]: row for row in _iter_csv_rows(self.stage_02_path)}
        rows_written = 0
        batch: list[dict[str, object]] = []
        source_rows = list(_iter_csv_rows(self.stage_01_path))
        if max_apps is not None:
            source_rows = source_rows[:max_apps]
        for app_row in _progress_bar(source_rows, desc="Stage 3 merge", unit="apps"):
            # Merge raw catalog data with flattened details so sampling only touches one CSV.
            merged = merge_catalog_and_details(
                app_row,
                details_by_appid.get(app_row["appid"]),
                self.config.min_recommendations,
            )
            batch.append(merged)
            if len(batch) >= 1000:
                rows_written += _write_rows(self.stage_03_path, STAGE_03_FIELDS, batch, append=self.stage_03_path.exists())
                batch.clear()
        if batch:
            rows_written += _write_rows(self.stage_03_path, STAGE_03_FIELDS, batch, append=self.stage_03_path.exists())

        return self._result(
            "stage_03",
            self.stage_03_path,
            rows_written,
            False,
            start,
            retry_count_start=retry_count_start,
            error_count_start=error_count_start,
        )

    def run_stage_04(self, *, force_refresh: bool = False, sample_size: int | None = None) -> StageResult:
        start = perf_counter()
        retry_count_start = self.http_client.retry_count
        error_count_start = self.http_client.error_count
        if not self.stage_03_path.exists():
            raise FileNotFoundError("Stage 03 output is required before running stage 04.")

        if self.stage_04_path.exists() and not force_refresh:
            self.logger.info("Reusing cached stage 04 output: %s", self.stage_04_path)
            return self._result(
                "stage_04",
                self.stage_04_path,
                _count_csv_rows(self.stage_04_path),
                True,
                start,
                retry_count_start=retry_count_start,
                error_count_start=error_count_start,
            )
        if force_refresh and self.stage_04_path.exists():
            self.stage_04_path.unlink()

        eligible_rows = [
            row for row in _iter_csv_rows(self.stage_03_path) if str(row.get("eligible_for_sampling", "")).lower() == "true"
        ]
        # Sampling happens after all metadata is cached, so reruns are deterministic and cheap.
        selected_rows = sample_rows(
            eligible_rows,
            sample_size=sample_size or self.config.sample_size,
            seed=self.config.random_seed,
        )
        sampled_at = utc_timestamp()
        enriched_rows: list[dict[str, object]] = []
        for index, row in enumerate(selected_rows, start=1):
            enriched = dict(row)
            enriched["sample_rank"] = index
            enriched["random_seed"] = self.config.random_seed
            enriched["sampled_at"] = sampled_at
            enriched_rows.append(enriched)

        rows_written = _write_rows(self.stage_04_path, STAGE_04_FIELDS, enriched_rows)
        return self._result(
            "stage_04",
            self.stage_04_path,
            rows_written,
            False,
            start,
            retry_count_start=retry_count_start,
            error_count_start=error_count_start,
        )

    def run_stage_05(self, *, force_refresh: bool = False, max_games: int | None = None) -> StageResult:
        start = perf_counter()
        retry_count_start = self.http_client.retry_count
        error_count_start = self.http_client.error_count
        if not self.stage_04_path.exists():
            raise FileNotFoundError("Stage 04 output is required before running stage 05.")

        if force_refresh:
            for path in [self.stage_05_path, self.stage_05_progress_path]:
                if path.exists():
                    path.unlink()

        progress_rows = {}
        if self.stage_05_progress_path.exists():
            for row in _iter_csv_rows(self.stage_05_progress_path):
                progress_rows[row["appid"]] = row
        completed_ids = {
            int(appid)
            for appid, row in progress_rows.items()
            if row.get("status") in {"completed", "exhausted"}
        }
        initial_completed_count = len(completed_ids)

        selected_games = list(_iter_csv_rows(self.stage_04_path))
        if max_games is not None:
            selected_games = selected_games[:max_games]
        if selected_games and initial_completed_count >= len(selected_games) and not force_refresh:
            self.logger.info("Reusing cached stage 05 output: %s", self.stage_05_path)
            return self._result(
                "stage_05",
                self.stage_05_path,
                _count_csv_rows(self.stage_05_path),
                True,
                start,
                retry_count_start=retry_count_start,
                error_count_start=error_count_start,
            )
        outer_progress = _progress_bar(
            total=len(selected_games),
            initial=len(completed_ids),
            desc="Stage 5 selected games",
            unit="games",
        )
        rows_written = _count_csv_rows(self.stage_05_path)
        try:
            for row in selected_games:
                appid = int(row["appid"])
                if appid in completed_ids:
                    continue
                state, seen_ids = self._restore_review_state(appid, progress_rows.get(str(appid)))
                if state.status in {"completed", "exhausted"}:
                    state.finished_at = state.finished_at or utc_timestamp()
                    self._append_review_progress(state)
                    completed_ids.add(appid)
                    outer_progress.update(1)
                    continue
                state.status = "in_progress"
                state.error = ""
                state.finished_at = ""
                self._append_review_progress(state)

                def checkpoint(page_rows: list[dict[str, object]], checkpoint_state: ReviewCollectionState) -> None:
                    nonlocal rows_written
                    if page_rows:
                        rows_written += _write_rows(
                            self.stage_05_path,
                            STAGE_05_FIELDS,
                            page_rows,
                            append=self.stage_05_path.exists(),
                        )
                    checkpoint_state.status = "in_progress"
                    checkpoint_state.error = ""
                    checkpoint_state.finished_at = ""
                    self._append_review_progress(checkpoint_state)

                try:
                    # Flush both rows and cursor checkpoints after every fetched page.
                    state = self.review_collector.collect_for_app(
                        appid=appid,
                        seen_ids=seen_ids,
                        state=state,
                        checkpoint=checkpoint,
                    )
                    state.status = "completed" if state.total_unique >= self.config.reviews_per_game else "exhausted"
                    state.finished_at = utc_timestamp()
                    self._append_review_progress(state)
                    completed_ids.add(appid)
                    outer_progress.update(1)
                except Exception as exc:
                    state.status = "failed"
                    state.finished_at = utc_timestamp()
                    state.error = str(exc)
                    self._append_review_progress(state)
                    outer_progress.update(1)
                    if isinstance(exc, RuntimeError):
                        self.logger.error("Failed to collect reviews for app %s: %s", appid, exc)
                        continue
                    self.logger.exception("Unexpected failure while collecting reviews for app %s", appid)
                    raise
        finally:
            outer_progress.close()

        return self._result(
            "stage_05",
            self.stage_05_path,
            _count_csv_rows(self.stage_05_path),
            False,
            start,
            retry_count_start=retry_count_start,
            error_count_start=error_count_start,
        )

    def run_all_missing(
        self,
        *,
        force_refresh: bool = False,
        max_pages: int | None = None,
        max_apps: int | None = None,
        sample_size: int | None = None,
        max_games: int | None = None,
    ) -> list[StageResult]:
        results = [self.run_stage_01(force_refresh=force_refresh, max_pages=max_pages)]
        results.append(self.run_stage_02(force_refresh=force_refresh, max_apps=max_apps))
        results.append(self.run_stage_03(force_refresh=force_refresh, max_apps=max_apps))
        results.append(self.run_stage_04(force_refresh=force_refresh, sample_size=sample_size))
        results.append(self.run_stage_05(force_refresh=force_refresh, max_games=max_games))
        return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the staged Steam dataset crawler.")
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parents[2],
        type=Path,
        help="Root directory for the steam-crawler workspace.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        type=Path,
        help="Optional override for stage output storage. Overrides STEAM_DATA_DIR from the environment or .env.",
    )
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2", "stage3", "stage4", "stage5", "all"],
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--endpoint-mode",
        choices=["proxy", "direct"],
        default=None,
        help="Endpoint mode. Overrides STEAM_ENDPOINT_MODE from the environment or .env.",
    )
    parser.add_argument("--max-pages", type=int, default=None, help="Optional smoke-test limit for stage 1 page count.")
    parser.add_argument(
        "--max-apps",
        type=int,
        default=None,
        help="Optional total cap for the first N Stage 1 app ids considered by stages 2 and 3 across reruns.",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample-size override for stage 4.")
    parser.add_argument("--max-games", type=int, default=None, help="Optional smoke-test limit for stage 5 game count.")
    parser.add_argument(
        "--gap-delay",
        type=float,
        default=None,
        help="Optional override for the 429 rate-limit cooling-off gap, in seconds.",
    )
    parser.add_argument(
        "--loop-limit",
        type=int,
        default=None,
        help="Optional override for the repeated-cursor stop-gap in stage 5. Overrides STEAM_CURSOR_LOOP_LIMIT from the environment or .env.",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached outputs for the selected stage.")
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    config_overrides: dict[str, object] = {"endpoint_mode": args.endpoint_mode}
    if args.gap_delay is not None:
        config_overrides["rate_limit_gap_delay_sec"] = args.gap_delay
    if args.loop_limit is not None:
        config_overrides["review_cursor_loop_limit"] = args.loop_limit
    if args.data_dir is not None:
        config_overrides["data_dir"] = args.data_dir
    config = Config.from_env(args.root, **config_overrides)
    pipeline = Pipeline(config)
    dispatch = {
        "stage1": lambda: pipeline.run_stage_01(force_refresh=args.force_refresh, max_pages=args.max_pages),
        "stage2": lambda: pipeline.run_stage_02(force_refresh=args.force_refresh, max_apps=args.max_apps),
        "stage3": lambda: pipeline.run_stage_03(force_refresh=args.force_refresh, max_apps=args.max_apps),
        "stage4": lambda: pipeline.run_stage_04(force_refresh=args.force_refresh, sample_size=args.sample_size),
        "stage5": lambda: pipeline.run_stage_05(force_refresh=args.force_refresh, max_games=args.max_games),
        "all": lambda: pipeline.run_all_missing(
            force_refresh=args.force_refresh,
            max_pages=args.max_pages,
            max_apps=args.max_apps,
            sample_size=args.sample_size,
            max_games=args.max_games,
        ),
    }
    dispatch[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
