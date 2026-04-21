from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import kagglehub
from dotenv import load_dotenv


DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a processed dataset directory to Kaggle using the same "
            "KAGGLE_USERNAME / KAGGLE_API_TOKEN env contract as steam-crawler/notebooks/eda.ipynb."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the processed data files to upload.",
    )
    parser.add_argument(
        "--dataset-handle",
        required=True,
        help="Kaggle dataset handle in the format '<KAGGLE_USERNAME>/<DATASET_SLUG>'.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Env file that provides KAGGLE_USERNAME and KAGGLE_API_TOKEN.",
    )
    parser.add_argument(
        "--kaggle-username",
        default=None,
        help="Optional CLI override for KAGGLE_USERNAME.",
    )
    parser.add_argument(
        "--kaggle-api-token",
        default=None,
        help="Optional CLI override for KAGGLE_API_TOKEN.",
    )
    parser.add_argument(
        "--version-notes",
        default=None,
        help="Optional Kaggle version notes. Defaults to a timestamped refresh message.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (Path.cwd() / expanded).resolve()


def load_kaggle_credentials(
    env_file: Path,
    *,
    kaggle_username: str | None = None,
    kaggle_api_token: str | None = None,
) -> tuple[Path, str, str]:
    env_path = resolve_path(env_file)
    load_dotenv(env_path, override=True)

    resolved_username = kaggle_username or os.getenv("KAGGLE_USERNAME")
    resolved_token = kaggle_api_token or os.getenv("KAGGLE_API_TOKEN")

    required_env = {
        "KAGGLE_USERNAME": resolved_username,
        "KAGGLE_API_TOKEN": resolved_token,
    }
    missing_env = [name for name, value in required_env.items() if not value]
    if missing_env:
        raise RuntimeError(
            "Missing required Kaggle credentials: "
            + ", ".join(missing_env)
            + f". Add them to the environment or to {env_path} before running this script."
        )

    assert resolved_username is not None
    assert resolved_token is not None
    os.environ["KAGGLE_USERNAME"] = resolved_username
    os.environ["KAGGLE_API_TOKEN"] = resolved_token
    os.environ["KAGGLE_KEY"] = resolved_token
    return env_path, resolved_username, resolved_token


def collect_dataset_files(input_dir: Path) -> dict[str, Path]:
    resolved_input_dir = resolve_path(input_dir)
    if not resolved_input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {resolved_input_dir}")
    if not resolved_input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {resolved_input_dir}")

    dataset_files: dict[str, Path] = {}
    for source_path in sorted(resolved_input_dir.rglob("*")):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(resolved_input_dir)
        if any(part.startswith(".") for part in relative_path.parts):
            continue
        dataset_files[relative_path.as_posix()] = source_path.resolve()

    if not dataset_files:
        raise FileNotFoundError(
            f"No uploadable files were found under {resolved_input_dir}."
        )
    return dataset_files


def upload_kaggle_dataset_snapshot(
    dataset_handle: str,
    dataset_files: dict[str, Path],
    *,
    version_notes: str = "",
) -> dict[str, object]:
    if dataset_handle.count("/") != 1:
        raise ValueError(
            "Kaggle dataset handle must look like '<KAGGLE_USERNAME>/<DATASET_SLUG>'."
        )

    staged_files: dict[str, str] = {}
    with TemporaryDirectory(prefix="kagglehub_dataset_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for target_name, source_path in sorted(dataset_files.items()):
            source_path = Path(source_path)
            if not source_path.exists():
                continue
            staged_path = tmp_dir / target_name
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, staged_path)
            staged_files[target_name] = str(source_path)

        if not staged_files:
            raise FileNotFoundError(
                "No source files were found for the Kaggle dataset snapshot."
            )

        kagglehub.dataset_upload(
            dataset_handle,
            str(tmp_dir),
            version_notes=version_notes,
        )

    dataset_url = f"https://www.kaggle.com/datasets/{dataset_handle}"
    print(f"Uploaded {len(staged_files)} file(s) to {dataset_url}")
    for target_name, source_path in staged_files.items():
        print(f"- {target_name}: {source_path}")
    return {
        "handle": dataset_handle,
        "url": dataset_url,
        "files": staged_files,
        "version_notes": version_notes,
    }


def main() -> None:
    args = parse_args()

    input_dir = resolve_path(args.input_dir)
    env_path, kaggle_username, _ = load_kaggle_credentials(
        args.env_file,
        kaggle_username=args.kaggle_username,
        kaggle_api_token=args.kaggle_api_token,
    )
    dataset_files = collect_dataset_files(input_dir)
    version_notes = args.version_notes or (
        "Refresh processed dataset snapshot after local preprocessing update "
        f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(f"ENV_PATH: {env_path}")
    print(f"INPUT_DIR: {input_dir}")
    print(f"Kaggle credentials: ready for kagglehub as {kaggle_username}")
    print(f"Dataset handle: {args.dataset_handle}")

    upload_kaggle_dataset_snapshot(
        args.dataset_handle,
        dataset_files,
        version_notes=version_notes,
    )


if __name__ == "__main__":
    main()
