from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

try:
    import psutil
except ModuleNotFoundError:
    psutil = None


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from steam_crawler import Config, Pipeline
from steam_crawler.config import resolve_endpoint_mode


PREFLIGHT_TIMEOUT_SEC = 30

FULL_CONFIG = {
    "sample_size": 10_000,
    "min_recommendations": 5_000,
    "reviews_per_game": 1_000,
    "recent_quota": 500,
    "helpful_quota": 500,
    "random_seed": 5242,
    "request_timeout_sec": 30.0,
    "max_retries": 10,
    "base_backoff_sec": 1.0,
    "max_backoff_sec": 120.0,
    "rate_limit_gap_delay_sec": 300.0,
    "app_list_page_size": 5_000,
}

SMOKE_CONFIG = {
    "sample_size": 5,
    "min_recommendations": 5_000,
    "reviews_per_game": 20,
    "recent_quota": 10,
    "helpful_quota": 10,
    "random_seed": 5242,
    "request_timeout_sec": 30.0,
    "max_retries": 10,
    "base_backoff_sec": 1.0,
    "max_backoff_sec": 120.0,
    "rate_limit_gap_delay_sec": 300.0,
    "app_list_page_size": 25,
}

FULL_LIMITS = {
    "max_pages": None,
    "max_apps": None,
    "sample_size": None,
    "max_games": None,
}
SMOKE_LIMITS = {"max_pages": 1, "max_apps": 25, "sample_size": 5, "max_games": 2}


def load_project_env(root_dir: Path) -> None:
    load_dotenv(root_dir / ".env", override=False)


def resolve_run_mode(cli_value: str | None) -> str:
    run_mode = (cli_value or os.getenv("STEAM_RUN_MODE", "smoke")).strip().lower()
    if run_mode not in {"smoke", "full"}:
        raise ValueError(
            f"Invalid STEAM_RUN_MODE: {run_mode!r}. Expected 'smoke' or 'full'."
        )
    return run_mode


def fetch_preflight_json(label: str, url: str, *, params: dict[str, object]) -> object:
    try:
        response = requests.get(url, params=params, timeout=PREFLIGHT_TIMEOUT_SEC)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"{label} timed out after {PREFLIGHT_TIMEOUT_SEC} seconds. "
            "Check outbound network access and Steam endpoint availability, then rerun."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"{label} failed: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        response_preview = response.text.strip()[:200]
        raise RuntimeError(
            f"{label} returned invalid JSON. Response preview: {response_preview!r}"
        ) from exc


def print_system_preflight() -> None:
    hostname = socket.gethostname()
    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    print(f"Hostname: {hostname}")
    print(f"Available CPUs: {available_cpus}")
    if psutil is None:
        print(
            "RAM check: psutil not installed in this Python environment; skipping memory probe."
        )
    else:
        memory = psutil.virtual_memory()
        print(f"Total RAM (GiB): {memory.total / (1024 ** 3):.2f}")
        print(f"Available RAM (GiB): {memory.available / (1024 ** 3):.2f}")

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        print("NVIDIA GPU check: nvidia-smi not found on this system.")
        return

    gpu_command = [
        nvidia_smi,
        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader",
    ]
    gpu_result = subprocess.run(
        gpu_command, capture_output=True, text=True, check=False
    )
    if gpu_result.returncode == 0 and gpu_result.stdout.strip():
        print("NVIDIA GPUs:")
        for line in gpu_result.stdout.strip().splitlines():
            print(f"  {line}")
        return

    print("NVIDIA GPU check failed or no GPUs were reported by nvidia-smi.")
    if gpu_result.stderr.strip():
        print(gpu_result.stderr.strip())


def run_preflight(root_dir: Path, *, endpoint_mode: str) -> None:
    load_project_env(root_dir)
    print_system_preflight()

    steam_api_key = os.getenv("STEAM_API_KEY")
    if not steam_api_key:
        raise RuntimeError(
            "STEAM_API_KEY is missing. Set it in the environment or in steam-crawler/.env before running."
        )
    preflight_config = Config.from_env(root_dir, steam_api_key=steam_api_key, endpoint_mode=endpoint_mode)

    preflight_payload = fetch_preflight_json(
        "Steam API preflight",
        preflight_config.app_list_url,
        params={"key": steam_api_key, "max_results": 1},
    )
    apps = preflight_payload.get("response", {}).get("apps", [])
    if not isinstance(apps, list):
        raise RuntimeError(
            f"Steam API preflight returned an unexpected payload: {preflight_payload}"
        )
    print(f"Steam API preflight succeeded with {len(apps)} sample app(s).")

    appdetails_payload = fetch_preflight_json(
        "Steam appdetails preflight",
        preflight_config.app_details_url,
        params={"appids": 10, "cc": "us", "l": "english", "filters": "basic"},
    )
    appdetails_entry = (
        appdetails_payload.get("10", {}) if isinstance(appdetails_payload, dict) else {}
    )
    if not isinstance(appdetails_entry, dict) or not appdetails_entry.get("success"):
        raise RuntimeError(
            f"Steam appdetails preflight returned an unexpected payload: {appdetails_payload}"
        )
    print("Steam appdetails preflight succeeded for app 10.")


def build_active_config(
    run_mode: str,
) -> tuple[dict[str, object], dict[str, int | None]]:
    if run_mode == "smoke":
        return dict(SMOKE_CONFIG), dict(SMOKE_LIMITS)
    return dict(FULL_CONFIG), dict(FULL_LIMITS)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the notebook-equivalent Steam crawler flow without Jupyter."
    )
    parser.add_argument(
        "--root",
        default=ROOT_DIR,
        type=Path,
        help="Root directory for the steam-crawler workspace.",
    )
    parser.add_argument(
        "--run-mode",
        choices=["smoke", "full"],
        default=None,
        help="Execution profile. Overrides STEAM_RUN_MODE from the environment or steam-crawler/.env.",
    )
    parser.add_argument(
        "--endpoint-mode",
        choices=["proxy", "direct"],
        default=None,
        help="Endpoint mode. Ignored when STEAM_ENDPOINT_MODE is set in the environment or .env.",
    )
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2", "stage3", "stage4", "stage5", "all"],
        default="all",
        help="Stage to run. The default mirrors running the notebook end-to-end.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional override for the stage 1 page limit.",
    )
    parser.add_argument(
        "--max-apps",
        type=int,
        default=None,
        help="Optional override for the stage 2/3 app limit.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional override for the stage 4 sample size.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional override for the stage 5 game limit.",
    )
    parser.add_argument(
        "--gap-delay",
        type=float,
        default=None,
        help="Optional override for the 429 rate-limit cooling-off gap, in seconds.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached outputs for the selected stage.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the Steam/system preflight checks.",
    )
    return parser


def apply_limit_overrides(
    limits: dict[str, int | None],
    *,
    max_pages: int | None,
    max_apps: int | None,
    sample_size: int | None,
    max_games: int | None,
) -> dict[str, int | None]:
    resolved = dict(limits)
    if max_pages is not None:
        resolved["max_pages"] = max_pages
    if max_apps is not None:
        resolved["max_apps"] = max_apps
    if sample_size is not None:
        resolved["sample_size"] = sample_size
    if max_games is not None:
        resolved["max_games"] = max_games
    return resolved


def run_selected_stage(
    pipeline: Pipeline,
    *,
    stage: str,
    force_refresh: bool,
    limits: dict[str, int | None],
) -> object:
    dispatch = {
        "stage1": lambda: pipeline.run_stage_01(
            force_refresh=force_refresh, max_pages=limits["max_pages"]
        ),
        "stage2": lambda: pipeline.run_stage_02(
            force_refresh=force_refresh, max_apps=limits["max_apps"]
        ),
        "stage3": lambda: pipeline.run_stage_03(
            force_refresh=force_refresh, max_apps=limits["max_apps"]
        ),
        "stage4": lambda: pipeline.run_stage_04(
            force_refresh=force_refresh, sample_size=limits["sample_size"]
        ),
        "stage5": lambda: pipeline.run_stage_05(
            force_refresh=force_refresh, max_games=limits["max_games"]
        ),
        "all": lambda: pipeline.run_all_missing(
            force_refresh=force_refresh,
            max_pages=limits["max_pages"],
            max_apps=limits["max_apps"],
            sample_size=limits["sample_size"],
            max_games=limits["max_games"],
        ),
    }
    return dispatch[stage]()


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    root_dir = args.root.resolve()
    load_project_env(root_dir)
    run_mode = resolve_run_mode(args.run_mode)
    endpoint_mode = resolve_endpoint_mode(args.endpoint_mode)

    if not args.skip_preflight:
        run_preflight(root_dir, endpoint_mode=endpoint_mode)

    active_config, active_limits = build_active_config(run_mode)
    active_limits = apply_limit_overrides(
        active_limits,
        max_pages=args.max_pages,
        max_apps=args.max_apps,
        sample_size=args.sample_size,
        max_games=args.max_games,
    )
    if args.gap_delay is not None:
        active_config["rate_limit_gap_delay_sec"] = args.gap_delay

    settings = Config.from_env(root_dir, endpoint_mode=endpoint_mode, **active_config)
    pipeline = Pipeline(settings)

    print(f"Run mode: {run_mode}")
    print(f"Endpoint mode: {settings.endpoint_mode}")
    print(f"Stage limits: {active_limits}")

    result = run_selected_stage(
        pipeline,
        stage=args.stage,
        force_refresh=args.force_refresh,
        limits=active_limits,
    )
    if isinstance(result, list):
        for item in result:
            print(item)
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
