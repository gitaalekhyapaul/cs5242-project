from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .config import load_project_env, resolve_data_dir

STAGE_05A_FIELDS = [
    "timestamp",
    "user_id",
    "app_id",
    "review_id",
    "review_score",
    "review_rating",
]

CSV_FIELD_SIZE_LIMIT_READY = False


def _configure_csv_field_size_limit() -> None:
    global CSV_FIELD_SIZE_LIMIT_READY
    if CSV_FIELD_SIZE_LIMIT_READY:
        return
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            CSV_FIELD_SIZE_LIMIT_READY = True
            return
        except OverflowError:
            limit //= 10


def _stage_05_path(data_dir: Path) -> Path:
    return data_dir / "stage_05_reviews_dataset.csv.gz"


def _stage_05a_csv_path(data_dir: Path) -> Path:
    return data_dir / "stage_05a_reviews_dataset.csv.gz"


def _stage_05a_parquet_path(data_dir: Path) -> Path:
    return data_dir / "stage_05a_reviews_dataset.parquet"


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    _configure_csv_field_size_limit()
    with _open_text(path, "rt") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _parquet_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return pq.ParquetFile(path).metadata.num_rows


def _parquet_has_expected_schema(path: Path) -> bool:
    if not path.exists():
        return False
    parquet_file = pq.ParquetFile(path)
    return parquet_file.schema_arrow.names == STAGE_05A_FIELDS


def _preview_csv(path: Path, limit: int = 5) -> pd.DataFrame:
    _configure_csv_field_size_limit()
    with _open_text(path, "rt") as handle:
        preview_df = pd.read_csv(handle, nrows=limit)
    if preview_df.empty:
        return pd.DataFrame(columns=STAGE_05A_FIELDS)
    preview_df["timestamp"] = pd.to_numeric(
        preview_df["timestamp"], errors="coerce"
    ).astype("Int64")
    preview_df["user_id"] = pd.to_numeric(
        preview_df["user_id"], errors="coerce"
    ).astype("Int64")
    preview_df["app_id"] = pd.to_numeric(
        preview_df["app_id"], errors="coerce"
    ).astype("Int64")
    preview_df["review_id"] = pd.to_numeric(
        preview_df["review_id"], errors="coerce"
    ).astype("Int64")
    preview_df["review_score"] = pd.to_numeric(
        preview_df["review_score"], errors="coerce"
    ).astype("Int64")
    preview_df["review_rating"] = pd.to_numeric(
        preview_df["review_rating"], errors="coerce"
    ).astype("Int64")
    return preview_df[STAGE_05A_FIELDS]


def _empty_stage_05a_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="Int64"),
            "user_id": pd.Series(dtype="Int64"),
            "app_id": pd.Series(dtype="Int64"),
            "review_id": pd.Series(dtype="Int64"),
            "review_score": pd.Series(dtype="Int64"),
            "review_rating": pd.Series(dtype="Int64"),
        }
    )


def _coerce_int(value: object) -> int | pd._libs.missing.NAType:
    try:
        return int(value)
    except (TypeError, ValueError):
        return pd.NA


def _serialize_row(row: dict[str, object]) -> dict[str, object]:
    serialized_row: dict[str, object] = {}
    for field in STAGE_05A_FIELDS:
        value = row.get(field, "")
        serialized_row[field] = "" if value is pd.NA or pd.isna(value) else value
    return serialized_row


def _extract_stage_05a_row(stage_05_row: dict[str, str]) -> dict[str, object]:
    raw_json = stage_05_row.get("raw_json", "")
    try:
        payload = json.loads(raw_json) if raw_json else {}
    except json.JSONDecodeError:
        payload = {}

    author = payload.get("author", {}) if isinstance(payload, dict) else {}
    user_id = author.get("steamid", "") if isinstance(author, dict) else ""
    if not user_id:
        user_id = stage_05_row.get("author_steamid", "")

    voted_up = payload.get("voted_up") if isinstance(payload, dict) else None
    review_score: int | pd._libs.missing.NAType
    if voted_up is True:
        review_score = 1
    elif voted_up is False:
        review_score = 0
    else:
        review_score = pd.NA

    return {
        "timestamp": _coerce_int(
            payload.get("timestamp_created", stage_05_row.get("timestamp_created", ""))
        ),
        "user_id": _coerce_int(user_id),
        "app_id": _coerce_int(stage_05_row.get("appid", "")),
        "review_id": _coerce_int(
            payload.get("recommendationid", stage_05_row.get("recommendationid", ""))
        ),
        "review_score": review_score,
        "review_rating": _coerce_int(payload.get("votes_up", "")),
    }


def _csv_has_expected_schema(path: Path) -> bool:
    _configure_csv_field_size_limit()
    with _open_text(path, "rt") as handle:
        reader = csv.DictReader(handle)
        return (reader.fieldnames or []) == STAGE_05A_FIELDS


def build_stage_05a_csv(
    root_dir: str | Path,
    *,
    force_refresh: bool = False,
) -> dict[str, object]:
    resolved_root = Path(root_dir).resolve()
    load_project_env(resolved_root, override=True)
    data_dir = resolve_data_dir(resolved_root)

    stage_05_path = _stage_05_path(data_dir)
    if not stage_05_path.exists():
        raise FileNotFoundError(f"Stage 5 output not found: {stage_05_path}")
    source_row_count = _count_csv_rows(stage_05_path)

    stage_05a_csv_path = _stage_05a_csv_path(data_dir)
    stage_05a_parquet_path = _stage_05a_parquet_path(data_dir)
    if force_refresh:
        for output_path in [stage_05a_csv_path, stage_05a_parquet_path]:
            if output_path.exists():
                output_path.unlink()

    if stage_05a_csv_path.exists():
        if not _csv_has_expected_schema(stage_05a_csv_path):
            stage_05a_csv_path.unlink()
        else:
            existing_row_count = _count_csv_rows(stage_05a_csv_path)
            if existing_row_count != source_row_count:
                stage_05a_csv_path.unlink()
            else:
                return {
                    "path": stage_05a_csv_path,
                    "rows": existing_row_count,
                    "skipped": True,
                }

    stage_05a_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    _configure_csv_field_size_limit()
    with _open_text(stage_05_path, "rt") as input_handle:
        reader = csv.DictReader(input_handle)
        with _open_text(stage_05a_csv_path, "wt") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=STAGE_05A_FIELDS)
            writer.writeheader()
            for stage_05_row in tqdm(reader, desc="Stage 5a csv", unit="reviews"):
                writer.writerow(_serialize_row(_extract_stage_05a_row(stage_05_row)))
                rows_written += 1

    return {"path": stage_05a_csv_path, "rows": rows_written, "skipped": False}


def preview_stage_05a(root_dir: str | Path, *, limit: int = 5) -> pd.DataFrame:
    resolved_root = Path(root_dir).resolve()
    load_project_env(resolved_root, override=True)
    data_dir = resolve_data_dir(resolved_root)
    stage_05a_csv_path = _stage_05a_csv_path(data_dir)
    if not stage_05a_csv_path.exists():
        raise FileNotFoundError(f"Stage 5a CSV output not found: {stage_05a_csv_path}")
    return _preview_csv(stage_05a_csv_path, limit=limit)


def write_stage_05a_parquet(
    root_dir: str | Path,
    *,
    force_refresh: bool = False,
    chunk_size: int = 100_000,
) -> dict[str, object]:
    resolved_root = Path(root_dir).resolve()
    load_project_env(resolved_root, override=True)
    data_dir = resolve_data_dir(resolved_root)

    stage_05a_csv_path = _stage_05a_csv_path(data_dir)
    if not stage_05a_csv_path.exists():
        raise FileNotFoundError(f"Stage 5a CSV output not found: {stage_05a_csv_path}")
    source_row_count = _count_csv_rows(stage_05a_csv_path)

    stage_05a_parquet_path = _stage_05a_parquet_path(data_dir)
    if force_refresh and stage_05a_parquet_path.exists():
        stage_05a_parquet_path.unlink()

    if stage_05a_parquet_path.exists():
        existing_row_count = _parquet_row_count(stage_05a_parquet_path)
        if (
            not _parquet_has_expected_schema(stage_05a_parquet_path)
            or existing_row_count != source_row_count
        ):
            stage_05a_parquet_path.unlink()
        else:
            return {
                "path": stage_05a_parquet_path,
                "rows": source_row_count,
                "skipped": True,
            }

    stage_05a_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_writer: pq.ParquetWriter | None = None
    rows_written = 0
    try:
        chunk_reader = pd.read_csv(
            stage_05a_csv_path,
            compression="gzip",
            chunksize=chunk_size,
        )
        for chunk_df in tqdm(chunk_reader, desc="Stage 5a parquet", unit="rows"):
            if chunk_df.empty:
                continue
            chunk_df["timestamp"] = pd.to_numeric(
                chunk_df["timestamp"], errors="coerce"
            ).astype("Int64")
            chunk_df["user_id"] = pd.to_numeric(
                chunk_df["user_id"], errors="coerce"
            ).astype("Int64")
            chunk_df["app_id"] = pd.to_numeric(
                chunk_df["app_id"], errors="coerce"
            ).astype("Int64")
            chunk_df["review_id"] = pd.to_numeric(
                chunk_df["review_id"], errors="coerce"
            ).astype("Int64")
            chunk_df["review_score"] = pd.to_numeric(
                chunk_df["review_score"], errors="coerce"
            ).astype("Int64")
            chunk_df["review_rating"] = pd.to_numeric(
                chunk_df["review_rating"], errors="coerce"
            ).astype("Int64")
            chunk_df = chunk_df[STAGE_05A_FIELDS]
            table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(stage_05a_parquet_path, table.schema)
            parquet_writer.write_table(table)
            rows_written += len(chunk_df)
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if parquet_writer is None:
        _empty_stage_05a_df().to_parquet(stage_05a_parquet_path, index=False)

    return {"path": stage_05a_parquet_path, "rows": rows_written, "skipped": False}
