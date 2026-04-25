from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import steam_crawler.stage4a as stage4a


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class FakeStage4aHttpClient:
    calls: list[tuple[str, int | None, dict[str, object]]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.retry_count = 0
        self.error_count = 0

    def get_json(
        self,
        url: str,
        *,
        stage: str,
        appid: int | None = None,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del url
        self.calls.append((stage, appid, dict(params or {})))
        if stage == "stage_04a_metadata":
            assert appid is not None
            return {
                str(appid): {
                    "success": True,
                    "data": {
                        "is_free": False,
                        "price_overview": {"final": 999},
                        "genres": [
                            {"id": "1", "description": "Action"},
                            {"id": "23", "description": "Indie"},
                        ],
                    },
                }
            }
        if stage == "stage_04a_review_summary":
            return {
                "query_summary": {
                    "total_reviews": 4,
                    "total_positive": 3,
                }
            }
        raise AssertionError(f"Unexpected stage: {stage}")


class Stage4aGenrePatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        FakeStage4aHttpClient.calls = []

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_stage_04a_rebuilds_legacy_cache_with_genres(self) -> None:
        data_dir = self.root / "data"
        write_csv(
            data_dir / "stage_04_selected_games.csv",
            ["appid", "recommendations_total", "category_ids"],
            [
                {
                    "appid": 10,
                    "recommendations_total": 100,
                    "category_ids": "66|68",
                }
            ],
        )
        write_csv(
            data_dir / "stage_04a_selected_games.csv",
            ["id", "num_reviews", "%positive_reviews", "price", "app_category"],
            [
                {
                    "id": 10,
                    "num_reviews": 100,
                    "%positive_reviews": 0,
                    "price": 0,
                    "app_category": "66|68",
                }
            ],
        )

        with patch.dict(os.environ, {"STEAM_API_KEY": "test-key"}, clear=True):
            with patch.object(stage4a, "HttpClient", FakeStage4aHttpClient):
                stage_04a_df = stage4a.build_stage_04a(self.root)

        metadata_calls = [
            call
            for call in FakeStage4aHttpClient.calls
            if call[0] == "stage_04a_metadata"
        ]
        self.assertEqual(len(metadata_calls), 1)
        self.assertEqual(
            metadata_calls[0][2]["filters"],
            "basic,price_overview,genres",
        )

        self.assertEqual(stage_04a_df.loc[0, "app_category"], "1|23")
        stage_04a_rows = read_csv(data_dir / "stage_04a_selected_games.csv")
        self.assertEqual(stage_04a_rows[0]["app_category"], "1|23")
        self.assertEqual(stage_04a_rows[0]["price"], "9.99")
        self.assertEqual(stage_04a_rows[0]["%positive_reviews"], "75.0")

        genre_rows = read_csv(data_dir / "stage_04a_genre_mapping.csv")
        self.assertEqual(
            genre_rows,
            [
                {"app_category": "1", "category_description": "Action"},
                {"app_category": "23", "category_description": "Indie"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
