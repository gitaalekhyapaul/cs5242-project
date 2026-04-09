# Steam Dataset Crawler

This folder contains a stage-based Steam crawler designed for two modes of operation:

1. Small-batch smoke tests that validate the flow cheaply.
2. Full dataset runs intended for long-running execution on a SLURM cluster.

The implementation is intentionally split between reusable Python classes in `src/steam_crawler` and a single orchestration notebook in `notebooks/steam_crawler.ipynb`.

## Approach

The crawler is built around cacheable CSV stages so each expensive step can be resumed without recomputing earlier work.
Stage 2 now checkpoints after every successful appdetails response, and Stage 5 checkpoints after every fetched review page.

### Stage flow

1. Stage 1 fetches the Steam app list from `IStoreService/GetAppList/v1` and writes `data/stage_01_apps_catalog.csv`.
2. Stage 2 fetches `appdetails` per app and writes `data/stage_02_app_details.csv.gz`.
3. Stage 3 merges the catalog and metadata into `data/stage_03_apps_with_metadata.csv.gz`.
4. Stage 4 samples eligible games into `data/stage_04_selected_games.csv`.
5. Stage 5 fetches reviews for the sampled games into `data/stage_05_reviews_dataset.csv.gz`.

Implementation note: Stage 2 uses the `basic,categories,recommendations` filter set because Steam omits the `type` field if `basic` is not included, and `type == "game"` is required for Stage 4 eligibility.

### Why the code is class-based

The main runtime logic is encapsulated in classes for readability and reuse:

- `Config`: runtime configuration and path layout.
- `HttpClient`: shared retry/backoff HTTP client using `requests`.
- `StagePaths`: centralizes output locations for all stages.
- `ReviewCollectionState`: serializable per-game cursor and counter state for review resume.
- `ReviewCollector`: encapsulates the `recent + helpful + backfill` review strategy.
- `Pipeline`: notebook-friendly orchestrator for all stages.

### Retry and error handling

- Requests retry on `429`, `500`, `502`, `503`, and `504`.
- `Retry-After` is honored first when present.
- Similar reset/retry headers are inspected if `Retry-After` is absent.
- Otherwise the client falls back to capped exponential backoff with jitter.
- Every API error is logged to both:
  - `logs/run.log`
  - `logs/errors.csv`

The error CSV includes the stage name, app id, request params, status code, headers, response body, exception type, exception message, and computed retry delay.

Progress bars use notebook-native `tqdm.notebook` widgets when the code is running inside Jupyter, and fall back to terminal-safe `tqdm.auto` elsewhere.

## Project layout

```text
steam-crawler/
├── .env.example
├── PLAN.md
├── README.md
├── SteamAPI.postman_collection.json
├── requirements.txt
├── pyproject.toml
├── notebooks/
│   └── steam_crawler.ipynb
├── src/steam_crawler/
│   ├── __init__.py
│   ├── config.py
│   ├── http_client.py
│   ├── logging_utils.py
│   ├── pipeline.py
│   └── transforms.py
└── tests/
    ├── test_http_client.py
    ├── test_pipeline.py
    └── test_transforms.py
```

## Installation

Create a local virtual environment and install the runtime dependencies:

```bash
cd steam-crawler
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If you want to execute the notebook non-interactively on a cluster, also install Jupyter tooling:

```bash
.venv/bin/pip install notebook nbconvert
```

## Configuration

The crawler expects a Steam Web API key.

Priority order:

1. `STEAM_API_KEY` environment variable
2. `steam-crawler/.env`

Create the local env file before running anything:

```bash
cd steam-crawler
cp .env.example .env
```

The notebook is profile-driven:

- `STEAM_RUN_MODE=smoke`: small-batch validation
- `STEAM_RUN_MODE=full`: full crawl for cluster execution

The notebook uses two profiles:

- Smoke profile:
  - `app_list_page_size=25`
  - `sample_size=5`
  - `reviews_per_game=20`
  - `recent_quota=10`
  - `helpful_quota=10`
  - stage limits: `max_pages=1`, `max_apps=25`, `max_games=2`
- Full profile:
  - `app_list_page_size=50_000`
  - `sample_size=10_000`
  - `reviews_per_game=1_000`
  - `recent_quota=500`
  - `helpful_quota=500`
  - no stage limits

## How to test locally in small batches

The intended validation path is:

1. Install dependencies in `.venv`.
2. Keep `STEAM_RUN_MODE=smoke`.
3. Run unit tests.
4. Run only small-batch stage slices.
5. Inspect generated CSVs and logs.

### Unit tests

Run:

```bash
cd steam-crawler
.venv/bin/python -m unittest discover -s tests
```

These tests currently cover:

- `Retry-After` parsing
- exponential backoff behavior
- app/review flattening
- deterministic sampling
- Stage 2 crash-safe checkpointing
- Stage 5 mid-game resume without duplicate reviews

### Smoke test through the notebook

Open the single notebook:

```bash
cd steam-crawler
.venv/bin/python -m notebook notebooks/steam_crawler.ipynb
```

Keep `STEAM_RUN_MODE=smoke` and run the cells in order.
The first code cells resolve the project root, install the packages from [`requirements.txt`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/requirements.txt) into the current notebook kernel via `sys.executable -m pip`, and then run a preflight check.
The preflight prints the hostname, available CPU count, total and available RAM, and visible NVIDIA GPU information when `nvidia-smi` exists, then validates the Steam API key, the app-list endpoint, and a lightweight `appdetails` request.
If the API key is missing or either preflight request fails, the preflight cell raises immediately so the later stage cells do not start.

Because the smoke profile has stage limits, it will:

- fetch only one app-list page
- fetch appdetails for only a small number of apps
- sample only a handful of games
- fetch reviews for only a couple of games

### Smoke test through the CLI

The pipeline module also supports bounded runs from the terminal:

```bash
cd steam-crawler
STEAM_API_KEY=... .venv/bin/python -m steam_crawler.pipeline --stage stage1 --max-pages 1
STEAM_API_KEY=... .venv/bin/python -m steam_crawler.pipeline --stage stage2 --max-apps 25
STEAM_API_KEY=... .venv/bin/python -m steam_crawler.pipeline --stage stage3 --max-apps 25
STEAM_API_KEY=... .venv/bin/python -m steam_crawler.pipeline --stage stage4 --sample-size 5
STEAM_API_KEY=... .venv/bin/python -m steam_crawler.pipeline --stage stage5 --max-games 2
```

Or run all bounded smoke stages at once:

```bash
cd steam-crawler
STEAM_API_KEY=... .venv/bin/python -m steam_crawler.pipeline --stage all --max-pages 1 --max-apps 25 --sample-size 5 --max-games 2
```

## Running the full crawl

### Full run through the notebook

For a real dataset build:

1. Ensure earlier smoke outputs are either removed or run with `force_refresh=True` in the notebook cells when needed.
2. Set `STEAM_RUN_MODE=full`.
3. Execute the notebook end-to-end.

The notebook remains the single submission artifact; the mode flips through the environment variable rather than by editing the notebook.
The Postman collection is kept only as an API reference artifact and request example bundle.

### Full run on SLURM

Below is a simple SLURM script that executes the notebook headlessly.

```bash
#!/bin/bash
#SBATCH --job-name=steam-crawler
#SBATCH --output=steam-crawler-%j.out
#SBATCH --error=steam-crawler-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /path/to/cs5242-project/steam-crawler

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install notebook nbconvert

export STEAM_API_KEY="your-steam-api-key"
export STEAM_RUN_MODE="full"

.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute notebooks/steam_crawler.ipynb \
  --output steam_crawler.executed.ipynb
```

Submit it with:

```bash
sbatch run_steam_crawler.slurm
```

## Output artifacts

Generated outputs are written under `data/`:

- `stage_01_apps_catalog.csv`
- `stage_02_app_details.csv.gz`
- `stage_03_apps_with_metadata.csv.gz`
- `stage_04_selected_games.csv`
- `stage_05_reviews_dataset.csv.gz`
- `stage_05_progress.csv`

Generated logs are written under `logs/`:

- `run.log`
- `errors.csv`

These directories are ignored by git.

## Notes on resume behavior

- Stage 1 reuses the full cached app catalog if it already exists.
- Stage 2 skips app ids already present in the metadata CSV and writes each success immediately.
- Stage 3 and Stage 4 are cheap enough to rerun from their prior CSVs.
- Stage 5 skips games already marked `completed` or `exhausted` in `stage_05_progress.csv`.
- Stage 5 appends review rows after each fetched page and appends cursor checkpoints after each page, so interrupted runs resume inside the current game instead of restarting it from scratch.
- If a crash happens after rows are flushed but before the latest cursor checkpoint is appended, the rerun reconstructs the existing review ids from the cached dataset and deduplicates them before continuing.

## Updating this README

This README should be kept in sync with any major flow changes, refactors, or new runtime knobs.
If you change:

- stage boundaries
- output schemas
- smoke/full configuration
- cluster execution flow
- required packages

update this file in the same change set.
