from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv


ENDPOINT_MODES = {"proxy", "direct"}
PROXY_API_BASE_URL = "https://gpaul.cc/steamapi"
PROXY_STORE_BASE_URL = "https://gpaul.cc/steamstore"
DIRECT_API_BASE_URL = "https://api.steampowered.com"
DIRECT_STORE_BASE_URL = "https://store.steampowered.com"


def resolve_endpoint_mode(cli_value: str | None = None) -> str:
    endpoint_mode = (os.getenv("STEAM_ENDPOINT_MODE") or cli_value or "proxy").strip().lower()
    if endpoint_mode not in ENDPOINT_MODES:
        raise ValueError(f"Invalid STEAM_ENDPOINT_MODE: {endpoint_mode!r}. Expected one of {sorted(ENDPOINT_MODES)}.")
    return endpoint_mode


def resolve_review_cursor_loop_limit(cli_value: int | None = None) -> int:
    raw_value = os.getenv("STEAM_CURSOR_LOOP_LIMIT")
    resolved = raw_value if raw_value is not None else cli_value
    if resolved is None:
        return 10
    try:
        loop_limit = int(resolved)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid STEAM_CURSOR_LOOP_LIMIT: {resolved!r}. Expected a positive integer."
        ) from exc
    if loop_limit <= 0:
        raise ValueError(
            f"Invalid STEAM_CURSOR_LOOP_LIMIT: {loop_limit!r}. Expected a positive integer."
        )
    return loop_limit


def resolve_data_dir(root_dir: Path, cli_value: str | Path | None = None) -> Path:
    raw_value = os.getenv("STEAM_DATA_DIR")
    resolved = raw_value if raw_value is not None else cli_value
    if resolved is None:
        return root_dir / "data"
    path = Path(resolved)
    if not path.is_absolute():
        path = root_dir / path
    return path.resolve()


@dataclass(slots=True)
class Config:
    """Central runtime configuration for both notebook and CLI execution."""

    root_dir: Path
    steam_api_key: str
    data_dir: Path
    log_dir: Path
    sample_size: int = 10_000
    min_recommendations: int = 5_000
    reviews_per_game: int = 1_000
    recent_quota: int = 500
    helpful_quota: int = 500
    random_seed: int = 5242
    request_timeout_sec: float = 30.0
    max_retries: int = 5
    base_backoff_sec: float = 1.0
    max_backoff_sec: float = 60.0
    rate_limit_gap_delay_sec: float = 300.0
    review_cursor_loop_limit: int = 10
    app_list_page_size: int = 5_000
    appdetails_country_code: str = "us"
    appdetails_language: str = "english"
    reviews_language: str = "all"
    reviews_page_size: int = 100
    api_host_delay_sec: float = 0.05
    store_host_delay_sec: float = 0.20
    default_host_delay_sec: float = 0.10
    endpoint_mode: str = "proxy"

    @property
    def api_base_url(self) -> str:
        return DIRECT_API_BASE_URL if self.endpoint_mode == "direct" else PROXY_API_BASE_URL

    @property
    def store_base_url(self) -> str:
        return DIRECT_STORE_BASE_URL if self.endpoint_mode == "direct" else PROXY_STORE_BASE_URL

    @property
    def app_list_url(self) -> str:
        return f"{self.api_base_url}/IStoreService/GetAppList/v1/"

    @property
    def app_details_url(self) -> str:
        return f"{self.store_base_url}/api/appdetails"

    def app_reviews_url(self, appid: int) -> str:
        return f"{self.store_base_url}/appreviews/{appid}"

    @classmethod
    def from_env(
        cls,
        root_dir: str | Path,
        *,
        dotenv_path: str | Path | None = None,
        steam_api_key: str | None = None,
        **overrides: object,
    ) -> "Config":
        """Build config from a project root plus optional environment overrides."""

        resolved_root = Path(root_dir).resolve()
        env_path = Path(dotenv_path).resolve() if dotenv_path is not None else resolved_root / ".env"
        load_dotenv(env_path, override=False)
        api_key = steam_api_key or os.getenv("STEAM_API_KEY")
        if not api_key:
            raise ValueError("STEAM_API_KEY is required. Set it in the environment or in steam-crawler/.env.")
        endpoint_mode = resolve_endpoint_mode(str(overrides.get("endpoint_mode")) if "endpoint_mode" in overrides else None)
        loop_limit_override = overrides.get("review_cursor_loop_limit")
        review_cursor_loop_limit = resolve_review_cursor_loop_limit(
            int(loop_limit_override) if loop_limit_override is not None else None
        )
        data_dir_override = overrides.get("data_dir")
        data_dir = resolve_data_dir(resolved_root, data_dir_override)

        settings = cls(
            root_dir=resolved_root,
            steam_api_key=api_key,
            data_dir=data_dir,
            log_dir=resolved_root / "logs",
            endpoint_mode=endpoint_mode,
            review_cursor_loop_limit=review_cursor_loop_limit,
        )
        if overrides:
            merged_overrides = dict(overrides)
            merged_overrides["endpoint_mode"] = endpoint_mode
            merged_overrides["review_cursor_loop_limit"] = review_cursor_loop_limit
            merged_overrides["data_dir"] = data_dir
            settings = replace(settings, **merged_overrides)
        return settings
