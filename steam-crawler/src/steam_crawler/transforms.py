from __future__ import annotations

import json
import random
from datetime import datetime, timezone


def minified_json(payload: object) -> str:
    """Serialize raw API payloads compactly so they remain CSV-friendly."""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def flatten_app_catalog_row(app: dict[str, object]) -> dict[str, object]:
    """Keep the catalog row narrow while preserving the original record."""

    return {
        "appid": app.get("appid"),
        "name": app.get("name", ""),
        "last_modified": app.get("last_modified", ""),
        "price_change_number": app.get("price_change_number", ""),
        "raw_json": minified_json(app),
    }


def flatten_app_details(appid: int, payload: dict[str, object]) -> dict[str, object]:
    """Extract only the appdetails fields needed for later filtering and joins."""

    keyed_payload = payload.get(str(appid), {}) if isinstance(payload, dict) else {}
    data = keyed_payload.get("data", {}) if isinstance(keyed_payload, dict) else {}
    categories = data.get("categories", []) if isinstance(data, dict) else []
    category_ids = [str(category.get("id", "")) for category in categories if isinstance(category, dict)]
    category_descriptions = [
        str(category.get("description", "")) for category in categories if isinstance(category, dict)
    ]
    recommendations = data.get("recommendations", {}) if isinstance(data, dict) else {}
    return {
        "appid": appid,
        "success": bool(keyed_payload.get("success", False)),
        "type": data.get("type", "") if isinstance(data, dict) else "",
        "category_ids": "|".join(filter(None, category_ids)),
        "category_descriptions": "|".join(filter(None, category_descriptions)),
        "recommendations_total": recommendations.get("total", "") if isinstance(recommendations, dict) else "",
        "raw_json": minified_json(payload),
    }


def flatten_review_row(appid: int, review: dict[str, object], source_stream: str) -> dict[str, object]:
    """Normalize one Steam review row for the final dataset CSV."""

    author = review.get("author", {}) if isinstance(review, dict) else {}
    return {
        "appid": appid,
        "recommendationid": review.get("recommendationid", ""),
        "author_steamid": author.get("steamid", "") if isinstance(author, dict) else "",
        "timestamp_created": review.get("timestamp_created", ""),
        "review_text": review.get("review", ""),
        "source_stream": source_stream,
        "raw_json": minified_json(review),
    }


def merge_catalog_and_details(
    app_row: dict[str, str], detail_row: dict[str, str] | None, min_recommendations: int
) -> dict[str, object]:
    """Join catalog rows with appdetails and derive the sampling eligibility flag."""

    detail_row = detail_row or {}
    recommendations_total = detail_row.get("recommendations_total") or ""
    try:
        numeric_recommendations = int(recommendations_total)
    except (TypeError, ValueError):
        numeric_recommendations = 0

    detail_success = str(detail_row.get("success", "")).lower() == "true"
    app_type = detail_row.get("type", "")
    eligible = detail_success and app_type == "game" and numeric_recommendations > min_recommendations
    return {
        "appid": app_row.get("appid", ""),
        "name": app_row.get("name", ""),
        "last_modified": app_row.get("last_modified", ""),
        "price_change_number": app_row.get("price_change_number", ""),
        "raw_app_json": app_row.get("raw_json", ""),
        "details_success": detail_row.get("success", ""),
        "type": app_type,
        "category_ids": detail_row.get("category_ids", ""),
        "category_descriptions": detail_row.get("category_descriptions", ""),
        "recommendations_total": recommendations_total,
        "raw_details_json": detail_row.get("raw_json", ""),
        "eligible_for_sampling": eligible,
    }


def sample_rows(rows: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    """Sample rows deterministically so reruns reproduce the same selected games."""

    if len(rows) <= sample_size:
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for logs and checkpoint files."""

    return datetime.now(timezone.utc).isoformat()
