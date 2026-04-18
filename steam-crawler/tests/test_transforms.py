from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steam_crawler.transforms import (
    flatten_app_catalog_row,
    flatten_app_details,
    flatten_review_row,
    sample_rows,
)


class TransformTests(unittest.TestCase):
    def test_flatten_app_catalog_row(self) -> None:
        row = flatten_app_catalog_row(
            {
                "appid": 10,
                "name": "Counter-Strike",
                "last_modified": 1,
                "price_change_number": 2,
            }
        )
        self.assertEqual(row["appid"], 10)
        self.assertEqual(row["name"], "Counter-Strike")
        self.assertIn('"appid":10', row["raw_json"])

    def test_flatten_app_details(self) -> None:
        payload = {
            "10": {
                "success": True,
                "data": {
                    "type": "game",
                    "categories": [{"id": 1, "description": "Multi-player"}],
                    "recommendations": {"total": 1234},
                },
            }
        }
        row = flatten_app_details(10, payload)
        self.assertTrue(row["success"])
        self.assertEqual(row["type"], "game")
        self.assertEqual(row["category_ids"], "1")
        self.assertEqual(row["recommendations_total"], 1234)

    def test_flatten_review_row(self) -> None:
        row = flatten_review_row(
            10,
            {
                "recommendationid": "abc",
                "author": {"steamid": "7656"},
                "timestamp_created": 111,
                "review": "good game",
            },
            "recent",
        )
        self.assertEqual(row["appid"], 10)
        self.assertEqual(row["author_steamid"], "7656")
        self.assertEqual(row["source_stream"], "recent")

    def test_sample_rows_is_deterministic(self) -> None:
        rows = [{"appid": str(index)} for index in range(20)]
        sample_a = sample_rows(rows, sample_size=5, seed=42)
        sample_b = sample_rows(rows, sample_size=5, seed=42)
        self.assertEqual(sample_a, sample_b)


if __name__ == "__main__":
    unittest.main()
