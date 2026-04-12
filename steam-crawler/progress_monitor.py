from __future__ import annotations

import argparse
import csv
import gzip
import os
from collections import Counter
from pathlib import Path


def resolve_root(root: Path) -> Path:
    resolved = root.resolve()
    if resolved.name == "notebooks":
        return resolved.parent
    return resolved


def resolve_data_dir(root: Path, data_dir: Path | None) -> Path:
    raw_value = os.getenv("STEAM_DATA_DIR")
    resolved = Path(raw_value) if raw_value is not None else data_dir
    if resolved is None:
        return root / "data"
    if not resolved.is_absolute():
        resolved = root / resolved
    return resolved.resolve()


def build_stage_paths(root: Path, data_dir: Path | None = None) -> dict[str, Path]:
    resolved_data_dir = resolve_data_dir(root, data_dir)
    log_dir = root / "logs"
    return {
        "stage_01": resolved_data_dir / "stage_01_apps_catalog.csv",
        "stage_02": resolved_data_dir / "stage_02_app_details.csv.gz",
        "stage_03": resolved_data_dir / "stage_03_apps_with_metadata.csv.gz",
        "stage_04": resolved_data_dir / "stage_04_selected_games.csv",
        "stage_05": resolved_data_dir / "stage_05_reviews_dataset.csv.gz",
        "stage_05_progress": resolved_data_dir / "stage_05_progress.csv",
        "errors": log_dir / "errors.csv",
        "run_log": log_dir / "run.log",
    }


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="")


def iter_csv_rows(path: Path):
    with open_text(path) as handle:
        yield from csv.DictReader(handle)


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in iter_csv_rows(path))


def tail_rows(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = list(iter_csv_rows(path))
    return rows[-limit:]


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{(numerator / denominator) * 100:.2f}%"


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def summarize_stage_outputs(stage_paths: dict[str, Path]) -> dict[str, int]:
    return {
        "stage_01": count_csv_rows(stage_paths["stage_01"]),
        "stage_02": count_csv_rows(stage_paths["stage_02"]),
        "stage_03": count_csv_rows(stage_paths["stage_03"]),
        "stage_04": count_csv_rows(stage_paths["stage_04"]),
        "stage_05": count_csv_rows(stage_paths["stage_05"]),
        "stage_05_progress": count_csv_rows(stage_paths["stage_05_progress"]),
        "errors": count_csv_rows(stage_paths["errors"]),
    }


def eligible_summary(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    eligible = 0
    total = 0
    for row in iter_csv_rows(path):
        total += 1
        if str(row.get("eligible_for_sampling", "")).lower() == "true":
            eligible += 1
    return eligible, total


def stage_05_status_counts(path: Path) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not path.exists():
        return counter
    latest_by_appid: dict[str, dict[str, str]] = {}
    for row in iter_csv_rows(path):
        latest_by_appid[row.get("appid", "")] = row
    for row in latest_by_appid.values():
        counter[row.get("status", "unknown") or "unknown"] += 1
    return counter


def review_counts_by_app(path: Path, top_n: int) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    if not path.exists():
        return []
    for row in iter_csv_rows(path):
        counter[row.get("appid", "")] += 1
    return counter.most_common(top_n)


def recent_error_counts(path: Path) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not path.exists():
        return counter
    for row in iter_csv_rows(path):
        stage = row.get("stage", "unknown") or "unknown"
        status = row.get("status_code", "") or "exc"
        counter[f"{stage}:{status}"] += 1
    return counter


def print_paths(root: Path, stage_paths: dict[str, Path]) -> None:
    print(f"ROOT_DIR: {root}")
    for name, path in stage_paths.items():
        print(f"{name:>16}: {path}")


def print_summary(stage_paths: dict[str, Path], *, top_n: int, error_tail: int) -> None:
    counts = summarize_stage_outputs(stage_paths)
    eligible_count, eligible_total = eligible_summary(stage_paths["stage_03"])
    status_counts = stage_05_status_counts(stage_paths["stage_05_progress"])

    print_section("Stage Output Counts")
    for name in ["stage_01", "stage_02", "stage_03", "stage_04", "stage_05", "stage_05_progress", "errors"]:
        print(f"{name:>16}: {counts[name]:>12,}")

    print_section("Coverage")
    print(
        f"Stage 2 coverage vs Stage 1: {counts['stage_02']:,} / {counts['stage_01']:,} "
        f"({format_ratio(counts['stage_02'], counts['stage_01'])})"
    )
    print(
        f"Stage 3 coverage vs Stage 1: {counts['stage_03']:,} / {counts['stage_01']:,} "
        f"({format_ratio(counts['stage_03'], counts['stage_01'])})"
    )
    print(
        f"Eligible-for-sampling rows in Stage 3: {eligible_count:,} / {eligible_total:,} "
        f"({format_ratio(eligible_count, eligible_total)})"
    )
    print(f"Selected games in Stage 4: {counts['stage_04']:,}")
    print(f"Review rows in Stage 5: {counts['stage_05']:,}")

    print_section("Stage 5 Status Counts")
    if status_counts:
        for status, count in sorted(status_counts.items()):
            print(f"{status:>16}: {count:>12,}")
    else:
        print("stage_05_progress.csv not present yet")

    print_section("Top Review Counts by App")
    top_review_counts = review_counts_by_app(stage_paths["stage_05"], top_n=top_n)
    if not top_review_counts:
        print("stage_05_reviews_dataset.csv.gz not present yet")
    else:
        for appid, count in top_review_counts:
            print(f"appid={appid:>10} | reviews={count:>6,}")

    print_section("Recent Error Buckets")
    error_counts = recent_error_counts(stage_paths["errors"])
    if not error_counts:
        print("logs/errors.csv not present yet")
    else:
        for label, count in error_counts.most_common(12):
            print(f"{label:>20}: {count:>8,}")

    print_section(f"Last {error_tail} Error Rows")
    for row in tail_rows(stage_paths["errors"], limit=error_tail):
        stage = row.get("stage", "")
        appid = row.get("appid", "")
        status = row.get("status_code", "") or "exc"
        retry_after = row.get("retry_after_seconds", "") or "-"
        exc_type = row.get("exception_type", "")
        exc_message = row.get("exception_message", "")
        print(f"[{stage}] appid={appid or '-'} status={status} retry_after={retry_after} {exc_type}: {exc_message}")


def print_app_inspection(stage_paths: dict[str, Path], appid: int | None) -> None:
    print_section("Optional App Inspection")
    if appid is None:
        print("Pass --appid to inspect one app across Stage 2, Stage 4, and Stage 5 outputs.")
        return

    appid_text = str(appid)
    stage_02_rows = [row for row in iter_csv_rows(stage_paths["stage_02"]) if row.get("appid") == appid_text] if stage_paths["stage_02"].exists() else []
    stage_04_rows = [row for row in iter_csv_rows(stage_paths["stage_04"]) if row.get("appid") == appid_text] if stage_paths["stage_04"].exists() else []
    stage_05_progress_rows = [row for row in iter_csv_rows(stage_paths["stage_05_progress"]) if row.get("appid") == appid_text] if stage_paths["stage_05_progress"].exists() else []
    stage_05_review_count = sum(
        1 for row in iter_csv_rows(stage_paths["stage_05"]) if row.get("appid") == appid_text
    ) if stage_paths["stage_05"].exists() else 0

    print(f"appid={appid_text}")
    print(f"  stage_02 rows: {len(stage_02_rows)}")
    if stage_02_rows:
        row = stage_02_rows[-1]
        print(f"  stage_02 success: {row.get('success')}")
        print(f"  stage_02 type: {row.get('type')}")
        print(f"  recommendations_total: {row.get('recommendations_total')}")
    print(f"  selected in stage_04: {'yes' if stage_04_rows else 'no'}")
    print(f"  review rows in stage_05: {stage_05_review_count}")
    if stage_05_progress_rows:
        latest = stage_05_progress_rows[-1]
        print(f"  latest stage_05 status: {latest.get('status')}")
        print(f"  latest phase: {latest.get('phase')}")
        print(f"  total_unique: {latest.get('total_unique')}")
        print(f"  error: {latest.get('error')}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only progress monitor for staged Steam crawler outputs."
    )
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Root directory for the steam-crawler workspace.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        type=Path,
        help="Optional override for stage output storage. Ignored when STEAM_DATA_DIR is set.",
    )
    parser.add_argument(
        "--appid",
        type=int,
        default=None,
        help="Optional app id to inspect across Stage 2, Stage 4, and Stage 5 outputs.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="How many apps to show in the top review-count table.",
    )
    parser.add_argument(
        "--error-tail",
        type=int,
        default=10,
        help="How many trailing error rows to print from logs/errors.csv.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    root_dir = resolve_root(args.root)
    stage_paths = build_stage_paths(root_dir, args.data_dir)
    print_paths(root_dir, stage_paths)
    print_summary(stage_paths, top_n=args.top_n, error_tail=args.error_tail)
    print_app_inspection(stage_paths, args.appid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
