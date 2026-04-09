from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv


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
    app_list_page_size: int = 50_000
    appdetails_country_code: str = "us"
    appdetails_language: str = "english"
    reviews_language: str = "all"
    reviews_page_size: int = 100
    api_host_delay_sec: float = 0.05
    store_host_delay_sec: float = 0.20
    default_host_delay_sec: float = 0.10

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

        settings = cls(
            root_dir=resolved_root,
            steam_api_key=api_key,
            data_dir=resolved_root / "data",
            log_dir=resolved_root / "logs",
        )
        if overrides:
            settings = replace(settings, **overrides)
        return settings
