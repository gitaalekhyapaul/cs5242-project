from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd


INTERACTIONS_FILENAME = "mobilerec_final.csv"
APP_META_FILENAME = "app_meta.csv"


@dataclass
class PreparationSummary:
    clean_rows: int
    filtered_rows: int
    user_count: int
    item_count: int
    min_sequence_length: int
    validation_users: int
    test_users: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess MobileRec for SASRec using disk-backed DuckDB transforms."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/mobilerec"),
        help="Directory containing the Hugging Face CLI download output.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/mobilerec"),
        help="Directory where processed parquet artifacts are written.",
    )
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=5,
        help="Keep only users with at least this many interactions after cleaning.",
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        default=None,
        help="Optional deterministic cap on the number of users for quick experiments.",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default="2GB",
        help="DuckDB memory limit. Lower this if the VM is under pressure.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="DuckDB worker thread count.",
    )
    return parser.parse_args()


def resolve_raw_file(raw_dir: Path, subdir: str, filename: str) -> Path:
    candidates = [
        raw_dir / subdir / filename,
        raw_dir / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing expected file. Checked: {', '.join(str(path) for path in candidates)}"
    )


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def load_app_meta(path: Path, item_mapping: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "app_package" not in df.columns:
        raise ValueError(f"Expected app_package column in {path}")
    df = df.drop_duplicates(subset=["app_package"]).reset_index(drop=True)
    return df[df["app_package"].isin(item_mapping["app_package"])].reset_index(drop=True)


def build_sequences_from_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = interactions.groupby("user_id", sort=True)
    for user_id, frame in grouped:
        frame = frame.sort_values("position", kind="stable")
        item_ids = frame["item_id"].astype(int).tolist()
        timestamps = frame["timestamp"].astype(int).tolist()
        app_packages = frame["app_package"].astype(str).tolist()
        ratings = frame["rating"].tolist()
        rows.append(
            {
                "user_id": int(user_id),
                "sequence_length": len(item_ids),
                "train_sequence": item_ids[:-2],
                "validation_sequence": item_ids[:-1],
                "test_sequence": item_ids,
                "validation_target": item_ids[-2],
                "test_target": item_ids[-1],
                "timestamps": timestamps,
                "app_packages": app_packages,
                "ratings": ratings,
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    processed_dir: Path,
    clean_rows: int,
    filtered_rows: int,
    interactions: pd.DataFrame,
    sequences: pd.DataFrame,
    min_user_interactions: int,
) -> None:
    summary = PreparationSummary(
        clean_rows=clean_rows,
        filtered_rows=filtered_rows,
        user_count=int(interactions["user_id"].nunique()),
        item_count=int(interactions["item_id"].nunique()),
        min_sequence_length=min_user_interactions,
        validation_users=int(sequences["validation_target"].notna().sum()),
        test_users=int(sequences["test_target"].notna().sum()),
    )
    (processed_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))


def main() -> None:
    args = parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    interactions_path = resolve_raw_file(args.raw_dir, "interactions", INTERACTIONS_FILENAME)
    app_meta_path = resolve_raw_file(args.raw_dir, "app_meta", APP_META_FILENAME)
    db_path = args.processed_dir / "mobilerec_prepare.duckdb"
    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
        con.execute(f"PRAGMA threads={args.threads}")
        con.execute("PRAGMA preserve_insertion_order=false")
        con.execute("PRAGMA temp_directory='data'")

        interactions_sql_path = sql_path(interactions_path)
        con.execute(
            f"""
            CREATE OR REPLACE TABLE interactions_clean AS
            WITH source AS (
              SELECT
                app_package,
                review,
                rating,
                votes,
                date,
                uid,
                formated_date,
                unix_timestamp,
                app_category
              FROM read_csv(
                '{interactions_sql_path}',
                columns={{
                  'app_package': 'VARCHAR',
                  'review': 'VARCHAR',
                  'rating': 'VARCHAR',
                  'votes': 'VARCHAR',
                  'date': 'VARCHAR',
                  'uid': 'VARCHAR',
                  'formated_date': 'VARCHAR',
                  'unix_timestamp': 'VARCHAR',
                  'app_category': 'VARCHAR'
                }},
                header=true,
                auto_detect=false,
                delim=',',
                quote='\"',
                escape='\"',
                ignore_errors=false
              )
            )
            SELECT
              uid,
              app_package,
              app_category,
              TRY_CAST(rating AS DOUBLE) AS rating,
              COALESCE(
                TRY_CAST(unix_timestamp AS BIGINT),
                CAST(epoch(TRY_CAST(formated_date AS TIMESTAMP)) AS BIGINT)
              ) AS timestamp
            FROM source
            WHERE uid IS NOT NULL
              AND app_package IS NOT NULL
              AND COALESCE(
                TRY_CAST(unix_timestamp AS BIGINT),
                CAST(epoch(TRY_CAST(formated_date AS TIMESTAMP)) AS BIGINT)
              ) IS NOT NULL
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TABLE keep_users AS
            SELECT uid
            FROM interactions_clean
            GROUP BY uid
            HAVING COUNT(*) >= {args.min_user_interactions}
            """
        )
        if args.sample_users is not None:
            con.execute(
                f"""
                CREATE OR REPLACE TABLE selected_users AS
                SELECT uid
                FROM keep_users
                ORDER BY uid
                LIMIT {args.sample_users}
                """
            )
        else:
            con.execute("CREATE OR REPLACE TABLE selected_users AS SELECT uid FROM keep_users")

        con.execute(
            """
            CREATE OR REPLACE TABLE user_mapping AS
            SELECT
              uid,
              ROW_NUMBER() OVER (ORDER BY uid) AS user_id
            FROM selected_users
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE item_mapping AS
            SELECT
              app_package,
              ROW_NUMBER() OVER (ORDER BY app_package) AS item_id
            FROM (
              SELECT DISTINCT c.app_package
              FROM interactions_clean c
              INNER JOIN selected_users u ON c.uid = u.uid
            )
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE encoded_interactions AS
            SELECT
              u.user_id,
              i.item_id,
              c.uid,
              c.app_package,
              c.app_category,
              c.rating,
              c.timestamp,
              ROW_NUMBER() OVER (
                PARTITION BY u.user_id
                ORDER BY c.timestamp, c.app_package
              ) AS position
            FROM interactions_clean c
            INNER JOIN user_mapping u ON c.uid = u.uid
            INNER JOIN item_mapping i ON c.app_package = i.app_package
            """
        )
        con.execute(
            """
            COPY (
              SELECT
                user_id,
                item_id,
                uid,
                app_package,
                app_category,
                rating,
                timestamp,
                position
              FROM encoded_interactions
              ORDER BY user_id, position
            )
            TO ?
            (FORMAT PARQUET)
            """,
            [str(args.processed_dir / "interactions.parquet")],
        )
        con.execute(
            """
            COPY (
              SELECT uid, user_id
              FROM user_mapping
              ORDER BY user_id
            )
            TO ?
            (FORMAT PARQUET)
            """,
            [str(args.processed_dir / "user_mapping.parquet")],
        )
        con.execute(
            """
            COPY (
              SELECT app_package, item_id
              FROM item_mapping
              ORDER BY item_id
            )
            TO ?
            (FORMAT PARQUET)
            """,
            [str(args.processed_dir / "item_mapping.parquet")],
        )
        clean_rows = int(con.execute("SELECT COUNT(*) FROM interactions_clean").fetchone()[0])
        filtered_rows = int(con.execute("SELECT COUNT(*) FROM encoded_interactions").fetchone()[0])
        item_mapping = con.execute("SELECT * FROM item_mapping ORDER BY item_id").df()
    finally:
        con.close()

    interactions = pd.read_parquet(args.processed_dir / "interactions.parquet")
    sequences = build_sequences_from_interactions(interactions)
    app_meta = load_app_meta(app_meta_path, item_mapping)

    sequences.to_parquet(args.processed_dir / "sequences.parquet", index=False)
    app_meta.to_parquet(args.processed_dir / "app_metadata.parquet", index=False)
    write_summary(
        processed_dir=args.processed_dir,
        clean_rows=clean_rows,
        filtered_rows=filtered_rows,
        interactions=interactions,
        sequences=sequences,
        min_user_interactions=args.min_user_interactions,
    )

    if db_path.exists():
        db_path.unlink()

    print(f"Prepared MobileRec data in {args.processed_dir}")
    print(
        "users="
        f"{interactions['user_id'].nunique()} "
        "items="
        f"{interactions['item_id'].nunique()} "
        "rows="
        f"{filtered_rows}"
    )


if __name__ == "__main__":
    main()
