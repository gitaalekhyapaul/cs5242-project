from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .config import Config, load_project_env
from .http_client import HttpClient
from .logging_utils import CsvErrorLogger, setup_logger

STAGE_04A_FIELDS = ["id", "num_reviews", "%positive_reviews", "price", "app_category"]


def _stage_04_path(data_dir: Path) -> Path:
    return data_dir / "stage_04_selected_games.csv"


def _stage_04a_csv_path(data_dir: Path) -> Path:
    return data_dir / "stage_04a_selected_games.csv"


def _stage_04a_parquet_path(data_dir: Path) -> Path:
    return data_dir / "stage_04a_selected_games.parquet"


def _load_stage_04_df(path: Path) -> pd.DataFrame:
    stage_04_df = pd.read_csv(
        path, usecols=["appid", "recommendations_total", "category_ids"]
    )
    return pd.DataFrame(
        {
            "id": pd.to_numeric(stage_04_df["appid"], errors="coerce").astype("Int64"),
            "num_reviews": pd.to_numeric(
                stage_04_df["recommendations_total"], errors="coerce"
            ).astype("Int64"),
            "app_category": stage_04_df["category_ids"].astype("string"),
        }
    )


def _load_stage_04a_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=STAGE_04A_FIELDS)
    stage_04a_df = pd.read_csv(path)
    if stage_04a_df.empty:
        return pd.DataFrame(columns=STAGE_04A_FIELDS)
    stage_04a_df["id"] = pd.to_numeric(stage_04a_df["id"], errors="coerce").astype(
        "Int64"
    )
    stage_04a_df["num_reviews"] = pd.to_numeric(
        stage_04a_df["num_reviews"], errors="coerce"
    ).astype("Int64")
    stage_04a_df["%positive_reviews"] = pd.to_numeric(
        stage_04a_df["%positive_reviews"], errors="coerce"
    ).astype("Float64")
    stage_04a_df["price"] = pd.to_numeric(
        stage_04a_df["price"], errors="coerce"
    ).astype("Float64")
    stage_04a_df["app_category"] = stage_04a_df["app_category"].astype("string")
    return stage_04a_df[STAGE_04A_FIELDS]


def _append_stage_04a_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAGE_04A_FIELDS)
        if should_write_header:
            writer.writeheader()
        serialized_row: dict[str, object] = {}
        for field in STAGE_04A_FIELDS:
            value = row.get(field, "")
            serialized_row[field] = "" if value is pd.NA or pd.isna(value) else value
        writer.writerow(serialized_row)


def _write_stage_04a_parquet(csv_path: Path, parquet_path: Path) -> pd.DataFrame:
    stage_04a_df = _load_stage_04a_df(csv_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    stage_04a_df.to_parquet(parquet_path, index=False)
    return stage_04a_df


def _extract_price(
    detail_payload: dict[str, object], appid: int
) -> float | pd._libs.missing.NAType:
    keyed_payload = (
        detail_payload.get(str(appid), {}) if isinstance(detail_payload, dict) else {}
    )
    data = keyed_payload.get("data", {}) if isinstance(keyed_payload, dict) else {}
    if not isinstance(data, dict):
        return pd.NA
    if data.get("is_free") is True:
        return 0.0
    price_overview = data.get("price_overview", {})
    if not isinstance(price_overview, dict):
        return pd.NA
    final_price = price_overview.get("final")
    if final_price is None:
        final_price = price_overview.get("initial")
    try:
        return int(final_price) / 100
    except (TypeError, ValueError):
        return pd.NA


def _rewrite_stage_04a_csv(path: Path, stage_04a_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAGE_04A_FIELDS)
        writer.writeheader()
        for row in stage_04a_df.to_dict(orient="records"):
            serialized_row: dict[str, object] = {}
            for field in STAGE_04A_FIELDS:
                value = row.get(field, "")
                serialized_row[field] = (
                    "" if value is pd.NA or pd.isna(value) else value
                )
            writer.writerow(serialized_row)


def _extract_positive_review_pct(
    review_payload: dict[str, object],
) -> float | pd._libs.missing.NAType:
    query_summary = (
        review_payload.get("query_summary", {})
        if isinstance(review_payload, dict)
        else {}
    )
    if not isinstance(query_summary, dict):
        return pd.NA
    total_reviews = query_summary.get("total_reviews")
    total_positive = query_summary.get("total_positive")
    try:
        total_reviews_value = int(total_reviews)
        total_positive_value = int(total_positive)
    except (TypeError, ValueError):
        return pd.NA
    if total_reviews_value <= 0:
        return pd.NA
    return (total_positive_value / total_reviews_value) * 100


def build_stage_04a(
    root_dir: str | Path,
    *,
    force_refresh: bool = False,
    endpoint_mode: str | None = None,
) -> pd.DataFrame:
    resolved_root = Path(root_dir).resolve()
    load_project_env(resolved_root, override=True)

    config_overrides: dict[str, object] = {}
    if endpoint_mode:
        config_overrides["endpoint_mode"] = endpoint_mode
    config = Config.from_env(resolved_root, **config_overrides)

    stage_04_path = _stage_04_path(config.data_dir)
    if not stage_04_path.exists():
        raise FileNotFoundError(f"Stage 4 output not found: {stage_04_path}")

    stage_04a_csv_path = _stage_04a_csv_path(config.data_dir)
    stage_04a_parquet_path = _stage_04a_parquet_path(config.data_dir)
    if force_refresh:
        for path in [stage_04a_csv_path, stage_04a_parquet_path]:
            if path.exists():
                path.unlink()

    logger = setup_logger(config.log_dir)
    error_logger = CsvErrorLogger(config.log_dir / "errors_stage_04a.csv")
    http_client = HttpClient(config, logger=logger, error_logger=error_logger)

    stage_04_df = _load_stage_04_df(stage_04_path)
    existing_df = _load_stage_04a_df(stage_04a_csv_path)
    if stage_04a_csv_path.exists() and not existing_df.empty:
        _rewrite_stage_04a_csv(stage_04a_csv_path, existing_df)
    completed_ids = (
        set(existing_df["id"].dropna().astype(int).tolist())
        if not existing_df.empty
        else set()
    )

    pending_df = stage_04_df[~stage_04_df["id"].isin(completed_ids)]
    progress = tqdm(
        total=len(stage_04_df),
        initial=len(completed_ids),
        desc="Stage 4a patch",
        unit="apps",
    )

    try:
        for row in pending_df.itertuples(index=False):
            appid = int(row.id)
            try:
                detail_payload = http_client.get_json(
                    config.app_details_url,
                    stage="stage_04a_metadata",
                    appid=appid,
                    params={
                        "appids": appid,
                        "cc": config.appdetails_country_code,
                        "l": config.appdetails_language,
                        "filters": "basic,price_overview",
                    },
                )
                review_payload = http_client.get_json(
                    config.app_reviews_url(appid),
                    stage="stage_04a_review_summary",
                    appid=appid,
                    params={
                        "json": 1,
                        "language": config.reviews_language,
                        "filter": "recent",
                        "review_type": "all",
                        "purchase_type": "all",
                        "num_per_page": 1,
                        "cursor": "*",
                    },
                )
            except RuntimeError as exc:
                logger.error(
                    "Skipping stage_04a patch for %s after retries: %s", appid, exc
                )
                progress.update(1)
                continue

            output_row = {
                "id": appid,
                "num_reviews": row.num_reviews,
                "%positive_reviews": _extract_positive_review_pct(review_payload),
                "price": _extract_price(detail_payload, appid),
                "app_category": row.app_category,
            }
            _append_stage_04a_row(stage_04a_csv_path, output_row)
            progress.update(1)
    finally:
        progress.close()

    stage_04a_df = _load_stage_04a_df(stage_04a_csv_path)
    logger.info(
        "stage_04a csv summary | rows=%s | csv=%s | retries=%s | errors=%s",
        len(stage_04a_df),
        stage_04a_csv_path,
        http_client.retry_count,
        http_client.error_count,
    )
    return stage_04a_df


def write_stage_04a_parquet(
    root_dir: str | Path,
    *,
    endpoint_mode: str | None = None,
) -> pd.DataFrame:
    resolved_root = Path(root_dir).resolve()
    load_project_env(resolved_root, override=True)

    config_overrides: dict[str, object] = {}
    if endpoint_mode:
        config_overrides["endpoint_mode"] = endpoint_mode
    config = Config.from_env(resolved_root, **config_overrides)

    stage_04a_csv_path = _stage_04a_csv_path(config.data_dir)
    if not stage_04a_csv_path.exists():
        raise FileNotFoundError(f"Stage 4a CSV output not found: {stage_04a_csv_path}")

    stage_04a_parquet_path = _stage_04a_parquet_path(config.data_dir)
    return _write_stage_04a_parquet(stage_04a_csv_path, stage_04a_parquet_path)
