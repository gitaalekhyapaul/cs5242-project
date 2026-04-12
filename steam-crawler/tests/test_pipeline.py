from __future__ import annotations

import csv
import gzip
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steam_crawler.config import Config
from steam_crawler.pipeline import Pipeline


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class FakeHttpClient:
    def __init__(self, handler):
        self.handler = handler
        self.retry_count = 0
        self.error_count = 0

    def get_json(self, url: str, *, stage: str, appid: int | None = None, params: dict[str, object] | None = None):
        return self.handler(url=url, stage=stage, appid=appid, params=params or {})


class PipelineResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def build_config(self, **overrides: object) -> Config:
        defaults = {
            "root_dir": self.root,
            "steam_api_key": "test-key",
            "data_dir": self.root / "data",
            "log_dir": self.root / "logs",
            "sample_size": 2,
            "min_recommendations": 5_000,
            "reviews_per_game": 4,
            "recent_quota": 2,
            "helpful_quota": 2,
            "reviews_page_size": 2,
            "api_host_delay_sec": 0.0,
            "store_host_delay_sec": 0.0,
            "default_host_delay_sec": 0.0,
        }
        defaults.update(overrides)
        return Config(**defaults)

    def test_stage_02_persists_each_successful_app_before_crash(self) -> None:
        write_csv(
            self.root / "data" / "stage_01_apps_catalog.csv",
            ["appid", "name", "last_modified", "price_change_number", "raw_json"],
            [
                {"appid": 1, "name": "One", "last_modified": "", "price_change_number": "", "raw_json": "{}"},
                {"appid": 2, "name": "Two", "last_modified": "", "price_change_number": "", "raw_json": "{}"},
                {"appid": 3, "name": "Three", "last_modified": "", "price_change_number": "", "raw_json": "{}"},
            ],
        )

        def crashing_handler(*, url: str, stage: str, appid: int | None, params: dict[str, object]):
            if stage != "stage_02":
                raise AssertionError(stage)
            if appid == 3:
                raise ValueError("synthetic crash")
            return {
                str(appid): {
                    "success": True,
                    "data": {"type": "game", "categories": [], "recommendations": {"total": 10_000}},
                }
            }

        pipeline = Pipeline(self.build_config(), http_client=FakeHttpClient(crashing_handler))
        with self.assertRaises(ValueError):
            pipeline.run_stage_02()

        partial_rows = read_csv(self.root / "data" / "stage_02_app_details.csv.gz")
        self.assertEqual([row["appid"] for row in partial_rows], ["1", "2"])

        def recovery_handler(*, url: str, stage: str, appid: int | None, params: dict[str, object]):
            return {
                str(appid): {
                    "success": True,
                    "data": {"type": "game", "categories": [], "recommendations": {"total": 10_000}},
                }
            }

        pipeline = Pipeline(self.build_config(), http_client=FakeHttpClient(recovery_handler))
        pipeline.run_stage_02()
        resumed_rows = read_csv(self.root / "data" / "stage_02_app_details.csv.gz")
        self.assertEqual([row["appid"] for row in resumed_rows], ["1", "2", "3"])

    def test_stage_02_max_apps_caps_total_scope_across_reruns(self) -> None:
        write_csv(
            self.root / "data" / "stage_01_apps_catalog.csv",
            ["appid", "name", "last_modified", "price_change_number", "raw_json"],
            [
                {"appid": 1, "name": "One", "last_modified": "", "price_change_number": "", "raw_json": "{}"},
                {"appid": 2, "name": "Two", "last_modified": "", "price_change_number": "", "raw_json": "{}"},
                {"appid": 3, "name": "Three", "last_modified": "", "price_change_number": "", "raw_json": "{}"},
            ],
        )
        stage_02_path = self.root / "data" / "stage_02_app_details.csv.gz"
        stage_02_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(stage_02_path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "appid",
                    "success",
                    "type",
                    "category_ids",
                    "category_descriptions",
                    "recommendations_total",
                    "raw_json",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "appid": 1,
                    "success": "True",
                    "type": "game",
                    "category_ids": "",
                    "category_descriptions": "",
                    "recommendations_total": "10000",
                    "raw_json": "{}",
                }
            )

        seen_appids: list[int] = []

        def handler(*, url: str, stage: str, appid: int | None, params: dict[str, object]):
            if stage != "stage_02":
                raise AssertionError(stage)
            assert appid is not None
            seen_appids.append(appid)
            return {
                str(appid): {
                    "success": True,
                    "data": {"type": "game", "categories": [], "recommendations": {"total": 10_000}},
                }
            }

        pipeline = Pipeline(self.build_config(), http_client=FakeHttpClient(handler))
        pipeline.run_stage_02(max_apps=2)

        rows = read_csv(stage_02_path)
        self.assertEqual([row["appid"] for row in rows], ["1", "2"])
        self.assertEqual(seen_appids, [2])

    def test_config_from_env_prefers_endpoint_mode_env_over_override(self) -> None:
        with patch.dict(os.environ, {"STEAM_API_KEY": "test-key", "STEAM_ENDPOINT_MODE": "direct"}):
            config = Config.from_env(self.root, endpoint_mode="proxy")
        self.assertEqual(config.endpoint_mode, "direct")
        self.assertEqual(config.app_list_url, "https://api.steampowered.com/IStoreService/GetAppList/v1/")
        self.assertEqual(config.app_details_url, "https://store.steampowered.com/api/appdetails")

    def test_config_from_env_prefers_cursor_loop_limit_env_over_override(self) -> None:
        with patch.dict(os.environ, {"STEAM_API_KEY": "test-key", "STEAM_CURSOR_LOOP_LIMIT": "12"}):
            config = Config.from_env(self.root, review_cursor_loop_limit=4)
        self.assertEqual(config.review_cursor_loop_limit, 12)

    def test_config_from_env_prefers_data_dir_env_over_override(self) -> None:
        with patch.dict(os.environ, {"STEAM_API_KEY": "test-key", "STEAM_DATA_DIR": "cluster-data"}):
            config = Config.from_env(self.root, data_dir=self.root / "custom-data")
        self.assertEqual(config.data_dir, (self.root / "cluster-data").resolve())

    def test_stage_01_uses_direct_endpoint_mode_urls(self) -> None:
        seen_urls: list[str] = []

        def stage_01_handler(*, url: str, stage: str, appid: int | None, params: dict[str, object]):
            seen_urls.append(url)
            return {
                "response": {
                    "apps": [],
                    "have_more_results": False,
                }
            }

        pipeline = Pipeline(
            self.build_config(endpoint_mode="direct"),
            http_client=FakeHttpClient(stage_01_handler),
        )
        pipeline.run_stage_01(force_refresh=True, max_pages=1)
        self.assertEqual(seen_urls, ["https://api.steampowered.com/IStoreService/GetAppList/v1/"])

    def test_stage_results_report_per_stage_retry_and_error_counts(self) -> None:
        class CountingHttpClient(FakeHttpClient):
            def get_json(self, url: str, *, stage: str, appid: int | None = None, params: dict[str, object] | None = None):
                self.retry_count += 2
                self.error_count += 3
                return {
                    "response": {
                        "apps": [
                            {
                                "appid": 1,
                                "name": "One",
                                "last_modified": 0,
                                "price_change_number": 0,
                            }
                        ],
                        "have_more_results": False,
                    }
                }

        counting_client = CountingHttpClient(lambda **_: {})
        pipeline = Pipeline(self.build_config(), http_client=counting_client)
        stage_01_result = pipeline.run_stage_01()
        self.assertEqual(stage_01_result.retry_count, 2)
        self.assertEqual(stage_01_result.error_count, 3)

        write_csv(
            self.root / "data" / "stage_01_apps_catalog.csv",
            ["appid", "name", "last_modified", "price_change_number", "raw_json"],
            [{"appid": 1, "name": "One", "last_modified": 0, "price_change_number": 0, "raw_json": "{}"}],
        )
        stage_02_path = self.root / "data" / "stage_02_app_details.csv.gz"
        stage_02_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(stage_02_path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "appid",
                    "success",
                    "type",
                    "category_ids",
                    "category_descriptions",
                    "recommendations_total",
                    "raw_json",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "appid": 1,
                    "success": "True",
                    "type": "game",
                    "category_ids": "",
                    "category_descriptions": "",
                    "recommendations_total": "10000",
                    "raw_json": "{}",
                }
            )

        stage_03_result = pipeline.run_stage_03()
        self.assertEqual(stage_03_result.retry_count, 0)
        self.assertEqual(stage_03_result.error_count, 0)

    def test_stage_05_resumes_mid_game_without_duplicate_reviews(self) -> None:
        write_csv(
            self.root / "data" / "stage_04_selected_games.csv",
            ["appid"],
            [{"appid": 10}],
        )

        call_log: list[tuple[str, str]] = []
        fail_helpful_once = {"value": True}

        def review_handler(*, url: str, stage: str, appid: int | None, params: dict[str, object]):
            if stage != "stage_05":
                raise AssertionError(stage)
            review_filter = str(params["filter"])
            cursor = str(params["cursor"])
            call_log.append((review_filter, cursor))
            if review_filter == "recent" and cursor == "*":
                return {
                    "reviews": [
                        {
                            "recommendationid": "r1",
                            "author": {"steamid": "100"},
                            "timestamp_created": 1,
                            "review": "recent one",
                        },
                        {
                            "recommendationid": "r2",
                            "author": {"steamid": "101"},
                            "timestamp_created": 2,
                            "review": "recent two",
                        },
                    ],
                    "cursor": "recent-2",
                }
            if review_filter == "all" and cursor == "*":
                if fail_helpful_once["value"]:
                    fail_helpful_once["value"] = False
                    raise RuntimeError("synthetic helpful failure")
                return {
                    "reviews": [
                        {
                            "recommendationid": "h1",
                            "author": {"steamid": "200"},
                            "timestamp_created": 3,
                            "review": "helpful one",
                        },
                        {
                            "recommendationid": "h2",
                            "author": {"steamid": "201"},
                            "timestamp_created": 4,
                            "review": "helpful two",
                        },
                    ],
                    "cursor": "helpful-2",
                }
            raise AssertionError((review_filter, cursor))

        config = self.build_config()
        pipeline = Pipeline(config, http_client=FakeHttpClient(review_handler))
        first_result = pipeline.run_stage_05()
        self.assertEqual(first_result.rows_written, 2)

        first_rows = read_csv(self.root / "data" / "stage_05_reviews_dataset.csv.gz")
        self.assertEqual([row["recommendationid"] for row in first_rows], ["r1", "r2"])

        progress_rows = read_csv(self.root / "data" / "stage_05_progress.csv")
        self.assertEqual(progress_rows[-1]["status"], "failed")
        self.assertEqual(progress_rows[-1]["recent_cursor"], "recent-2")
        self.assertEqual(progress_rows[-1]["helpful_cursor"], "*")

        pipeline = Pipeline(config, http_client=FakeHttpClient(review_handler))
        second_result = pipeline.run_stage_05()
        self.assertEqual(second_result.rows_written, 4)

        final_rows = read_csv(self.root / "data" / "stage_05_reviews_dataset.csv.gz")
        self.assertEqual([row["recommendationid"] for row in final_rows], ["r1", "r2", "h1", "h2"])

        final_progress = read_csv(self.root / "data" / "stage_05_progress.csv")
        self.assertEqual(final_progress[-1]["status"], "completed")
        self.assertEqual(call_log.count(("recent", "*")), 1)
        self.assertEqual(call_log.count(("all", "*")), 2)

    def test_stage_05_day_range_is_only_sent_on_helpful_requests(self) -> None:
        write_csv(
            self.root / "data" / "stage_04_selected_games.csv",
            ["appid"],
            [{"appid": 10}],
        )

        seen_params: list[dict[str, object]] = []

        def review_handler(*, url: str, stage: str, appid: int | None, params: dict[str, object]):
            if stage != "stage_05":
                raise AssertionError(stage)
            seen_params.append(dict(params))
            if str(params["filter"]) == "recent":
                return {
                    "reviews": [
                        {
                            "recommendationid": "r1",
                            "author": {"steamid": "100"},
                            "timestamp_created": 1,
                            "review": "recent one",
                        },
                        {
                            "recommendationid": "r2",
                            "author": {"steamid": "101"},
                            "timestamp_created": 2,
                            "review": "recent two",
                        },
                    ],
                    "cursor": "recent-done",
                }
            return {
                "reviews": [
                    {
                        "recommendationid": "h1",
                        "author": {"steamid": "200"},
                        "timestamp_created": 3,
                        "review": "helpful one",
                    },
                    {
                        "recommendationid": "h2",
                        "author": {"steamid": "201"},
                        "timestamp_created": 4,
                        "review": "helpful two",
                    },
                ],
                "cursor": "helpful-done",
            }

        pipeline = Pipeline(self.build_config(), http_client=FakeHttpClient(review_handler))
        pipeline.run_stage_05()

        helpful_params = [params for params in seen_params if str(params.get("filter")) == "all"]
        recent_params = [params for params in seen_params if str(params.get("filter")) == "recent"]
        self.assertTrue(helpful_params)
        self.assertTrue(recent_params)
        self.assertTrue(all(params.get("day_range") == 365 for params in helpful_params))
        self.assertTrue(all("day_range" not in params for params in recent_params))

    def test_stage_05_stops_after_repeated_helpful_cursor_loop(self) -> None:
        write_csv(
            self.root / "data" / "stage_04_selected_games.csv",
            ["appid"],
            [{"appid": 20}],
        )

        helpful_cursors = ["loop-a", "loop-b"]
        helpful_index = {"value": 0}
        helpful_calls = {"value": 0}

        def review_handler(*, url: str, stage: str, appid: int | None, params: dict[str, object]):
            if stage != "stage_05":
                raise AssertionError(stage)
            if str(params["filter"]) == "recent":
                return {
                    "reviews": [
                        {
                            "recommendationid": "r1",
                            "author": {"steamid": "100"},
                            "timestamp_created": 1,
                            "review": "recent one",
                        },
                        {
                            "recommendationid": "r2",
                            "author": {"steamid": "101"},
                            "timestamp_created": 2,
                            "review": "recent two",
                        },
                    ],
                    "cursor": "recent-done",
                }

            if str(params["filter"]) == "all":
                helpful_calls["value"] += 1
                cursor = helpful_cursors[helpful_index["value"] % len(helpful_cursors)]
                helpful_index["value"] += 1
                return {
                    "reviews": [
                        {
                            "author": {"steamid": "100"},
                            "timestamp_created": 1,
                            "review": "helpful duplicate one",
                        },
                        {
                            "author": {"steamid": "101"},
                            "timestamp_created": 2,
                            "review": "helpful duplicate two",
                        },
                    ],
                    "cursor": cursor,
                }

            raise AssertionError(params)

        pipeline = Pipeline(self.build_config(review_cursor_loop_limit=3), http_client=FakeHttpClient(review_handler))
        result = pipeline.run_stage_05()

        self.assertEqual(result.rows_written, 2)
        progress_rows = read_csv(self.root / "data" / "stage_05_progress.csv")
        self.assertEqual(progress_rows[-1]["status"], "exhausted")
        self.assertEqual(progress_rows[-1]["total_unique"], "2")
        self.assertEqual(helpful_calls["value"], 5)

    def test_stage_05_records_failed_progress_for_unexpected_exception_then_reraises(self) -> None:
        write_csv(
            self.root / "data" / "stage_04_selected_games.csv",
            ["appid"],
            [{"appid": 10}],
        )

        def review_handler(*, url: str, stage: str, appid: int | None, params: dict[str, object]):
            if stage != "stage_05":
                raise AssertionError(stage)
            if str(params["filter"]) == "recent":
                return {
                    "reviews": [
                        {
                            "recommendationid": "r1",
                            "author": {"steamid": "100"},
                            "timestamp_created": 1,
                            "review": "recent one",
                        }
                    ],
                    "cursor": "recent-1",
                }
            raise OSError("disk write failed")

        pipeline = Pipeline(self.build_config(), http_client=FakeHttpClient(review_handler))
        with self.assertRaises(OSError):
            pipeline.run_stage_05()

        review_rows = read_csv(self.root / "data" / "stage_05_reviews_dataset.csv.gz")
        self.assertEqual([row["recommendationid"] for row in review_rows], ["r1"])

        progress_rows = read_csv(self.root / "data" / "stage_05_progress.csv")
        self.assertEqual(progress_rows[-1]["status"], "failed")
        self.assertEqual(progress_rows[-1]["recent_cursor"], "recent-1")
        self.assertEqual(progress_rows[-1]["helpful_cursor"], "*")
        self.assertIn("disk write failed", progress_rows[-1]["error"])


@unittest.skipUnless(os.getenv("RUN_LIVE_STEAM_TESTS") == "1", "Set RUN_LIVE_STEAM_TESTS=1 to run live Steam tests.")
class PipelineLiveIntegrationTests(unittest.TestCase):
    VALID_APP_IDS = (10, 300)
    INVALID_APP_ID = 999_999_999

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_stage_02_live_appdetails_handles_valid_and_invalid_appids(self) -> None:
        steam_api_key = os.getenv("STEAM_API_KEY")
        if not steam_api_key:
            self.skipTest("STEAM_API_KEY is required for live Steam integration tests.")

        stage_01_rows = [
            {
                "appid": appid,
                "name": f"app-{appid}",
                "last_modified": "",
                "price_change_number": "",
                "raw_json": "{}",
            }
            for appid in (*self.VALID_APP_IDS, self.INVALID_APP_ID)
        ]
        write_csv(
            self.root / "data" / "stage_01_apps_catalog.csv",
            ["appid", "name", "last_modified", "price_change_number", "raw_json"],
            stage_01_rows,
        )

        pipeline = Pipeline(
            Config(
                root_dir=self.root,
                steam_api_key=steam_api_key,
                data_dir=self.root / "data",
                log_dir=self.root / "logs",
                request_timeout_sec=15.0,
                max_retries=1,
                api_host_delay_sec=0.0,
                store_host_delay_sec=0.0,
                default_host_delay_sec=0.0,
            )
        )
        result = pipeline.run_stage_02()
        self.assertEqual(result.rows_written, 3)

        rows = read_csv(self.root / "data" / "stage_02_app_details.csv.gz")
        self.assertEqual(len(rows), 3)
        rows_by_appid = {int(row["appid"]): row for row in rows}

        for appid in self.VALID_APP_IDS:
            row = rows_by_appid[appid]
            self.assertEqual(row["success"].lower(), "true")
            self.assertTrue(row["type"])
            self.assertIn(f'"{appid}"', row["raw_json"])

        invalid_row = rows_by_appid[self.INVALID_APP_ID]
        self.assertEqual(invalid_row["success"].lower(), "false")
        self.assertEqual(invalid_row["type"], "")
        self.assertEqual(invalid_row["category_ids"], "")
        self.assertEqual(invalid_row["recommendations_total"], "")
        self.assertIn(f'"{self.INVALID_APP_ID}"', invalid_row["raw_json"])


if __name__ == "__main__":
    unittest.main()
