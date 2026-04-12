# Steam Dataset Crawler

This folder contains a stage-based Steam crawler designed for two modes of operation:

1. Small-batch smoke tests that validate the flow cheaply.
2. Full dataset runs intended for long-running execution on a SLURM cluster.

The implementation is intentionally split between reusable Python classes in `src/steam_crawler` and two thin operator surfaces:

- the submission notebook at `notebooks/steam_crawler.ipynb`
- the notebook-equivalent runner at `run_notebook.py`

## Status Against PLAN.md

The current implementation matches the staged crawler design in [`PLAN.md`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/PLAN.md) for the runtime pipeline, cacheable CSV outputs, retry/backoff handling, notebook orchestration, the notebook-equivalent terminal runner, resume behavior, and test coverage.

By default, crawler traffic is routed through the configured proxy bases:

- `https://gpaul.cc/steamapi`
- `https://gpaul.cc/steamstore`

You can switch back to the original Steam hosts with endpoint mode:

- `STEAM_ENDPOINT_MODE=proxy` keeps the default proxy routing
- `STEAM_ENDPOINT_MODE=direct` uses `https://api.steampowered.com` and `https://store.steampowered.com`
- `--endpoint-mode {proxy,direct}` is available on both terminal runners, but `STEAM_ENDPOINT_MODE` wins if both are set

The test suite now has two layers:

- deterministic local tests that run quickly and do not depend on Steam availability
- an opt-in live integration test that exercises the real `appdetails` endpoint with stable app ids plus one intentionally invalid app id

One audit follow-up is still open: Stage 2 currently logs and skips `appdetails` requests that exhaust retries, so Stage 3 can only infer the missing metadata by absence rather than from an explicit failed detail row.

This keeps the default development flow stable while still covering the live API path when you explicitly request it.

## Approach

The crawler is built around cacheable CSV stages so each expensive step can be resumed without recomputing earlier work.
Stage 2 now checkpoints after every successful appdetails response, and Stage 5 checkpoints after every fetched review page.

### Stage flow

1. Stage 1 fetches the Steam app list from the configured app-list base (`gpaul.cc` proxy by default, original Steam API in `direct` mode) and writes `data/stage_01_apps_catalog.csv`.
2. Stage 2 fetches `appdetails` per app from the configured store base (`gpaul.cc` proxy by default, original Steam store host in `direct` mode) and writes `data/stage_02_app_details.csv.gz`.
3. Stage 3 merges the catalog and metadata into `data/stage_03_apps_with_metadata.csv.gz`.
4. Stage 4 samples eligible games into `data/stage_04_selected_games.csv`.
5. Stage 5 fetches reviews for the sampled games from the configured reviews base into `data/stage_05_reviews_dataset.csv.gz`.

Implementation note: Stage 2 uses the `basic,categories,recommendations` filter set because Steam omits the `type` field if `basic` is not included, and `type == "game"` is required for Stage 4 eligibility.

### Stage schemas

The pipeline intentionally keeps a narrow flattened schema for downstream analysis while retaining a raw JSON column for recovery and debugging.

- Stage 1 fields:
  - `appid`
  - `name`
  - `last_modified`
  - `price_change_number`
  - `raw_json`
- Stage 2 fields:
  - `appid`
  - `success`
  - `type`
  - `category_ids`
  - `category_descriptions`
  - `recommendations_total`
  - `raw_json`
- Stage 3 fields:
  - Stage 1 catalog columns
  - Stage 2 detail columns
  - `eligible_for_sampling`
- Stage 4 fields:
  - Stage 3 fields
  - `sample_rank`
  - `random_seed`
  - `sampled_at`
- Stage 5 fields:
  - `appid`
  - `recommendationid`
  - `author_steamid`
  - `timestamp_created`
  - `review_text`
  - `source_stream`
  - `raw_json`

### Why the code is class-based

The main runtime logic is encapsulated in classes for readability and reuse:

- `Config`: runtime configuration and path layout.
- `HttpClient`: shared retry/backoff HTTP client using `requests`.
- `StagePaths`: centralizes output locations for all stages.
- `ReviewCollectionState`: serializable per-game cursor and counter state for review resume.
- `ReviewCollector`: encapsulates the `recent + helpful + backfill` review strategy.
- `Pipeline`: notebook-friendly orchestrator for all stages.

### Module map

The package is intentionally split by responsibility instead of putting all crawler logic in the notebook:

- [`config.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/src/steam_crawler/config.py): runtime configuration, path layout, and `.env` loading
- [`http_client.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/src/steam_crawler/http_client.py): shared `requests` session, host throttling, retry, backoff, and error capture
- [`transforms.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/src/steam_crawler/transforms.py): CSV flatteners, JSON serialization, sampling, and merge helpers
- [`logging_utils.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/src/steam_crawler/logging_utils.py): run logger and structured CSV error logger
- [`pipeline.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/src/steam_crawler/pipeline.py): staged orchestration, checkpointing, progress bars, and CLI entrypoint
- [`run_notebook.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/run_notebook.py): notebook-equivalent terminal runner with the same smoke/full profiles and preflight checks
- [`steam_crawler.ipynb`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/notebooks/steam_crawler.ipynb): submission notebook for smoke validation and full cluster execution

### Retry and error handling

- Requests retry on network errors plus `429`, `500`, `502`, `503`, and `504`.
- Non-retryable HTTP statuses fail immediately after the first logged attempt.
- `Retry-After` is honored first when present.
- Similar reset/retry headers are inspected if `Retry-After` is absent.
- Otherwise the client falls back to deterministic exponential backoff starting at `2^0 = 1` second and doubling per retry, capped by `max_backoff_sec`.
- For `429` responses without a usable server retry hint, the client now adds a fixed `rate_limit_gap_delay_sec` before the exponential component. With the current default, that means a 5-minute cooling-off gap plus `1, 2, 4, ...` seconds across retries.
- Warning logs include the exception summary, so timeout/socket failures are visible in `run.log` without opening `errors.csv`.
- `run.log` is rebound to the active `log_dir` when a run uses a different workspace.
- Stage summaries report stage-local retry and error counts instead of cumulative lifetime totals.
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
├── run_notebook.py
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
    ├── test_logging_utils.py
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

If you want to execute the notebook non-interactively on a cluster, install Jupyter tooling into whichever Python environment the cluster job will activate:

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

The notebook and `run_notebook.py` are profile-driven:

- `STEAM_RUN_MODE=smoke`: small-batch validation
- `STEAM_RUN_MODE=full`: full crawl for cluster execution

Endpoint selection is also environment-driven:

- `STEAM_ENDPOINT_MODE=proxy`: default, uses `https://gpaul.cc/steamapi` and `https://gpaul.cc/steamstore`
- `STEAM_ENDPOINT_MODE=direct`: uses `https://api.steampowered.com` and `https://store.steampowered.com`
- `STEAM_CURSOR_LOOP_LIMIT=10`: default stop-gap for repeated no-yield review cursors in Stage 5

For terminal runs, both entrypoints also accept `--endpoint-mode proxy` or `--endpoint-mode direct`. If the env var is set, it takes priority over the CLI flag.
Both terminal entrypoints also accept `--gap-delay <seconds>` to override the `429` cooling-off gap without editing the profile config.
Both terminal entrypoints also accept `--loop-limit <count>` to override the Stage 5 repeated-cursor stop-gap, but `STEAM_CURSOR_LOOP_LIMIT` wins if both are set.
`progress_monitor.py` mirrors `notebooks/progress_monitor.ipynb` as a plain terminal script so you can inspect crawl progress on a compute node without Jupyter.
When you pass `--max-apps N`, stages 2 and 3 operate on the first `N` app ids from Stage 1 as a total scope across reruns, not `N` additional appdetails fetches per invocation.

The notebook uses two profiles:

- Smoke profile:
  - `app_list_page_size=25`
  - `sample_size=5`
  - `reviews_per_game=20`
  - `recent_quota=10`
  - `helpful_quota=10`
  - stage limits: `max_pages=1`, `max_apps=25`, `max_games=2`
- Full profile:
  - `app_list_page_size=5_000`
  - `sample_size=10_000`
  - `reviews_per_game=1_000`
  - `recent_quota=500`
  - `helpful_quota=500`
  - no stage limits

Additional configuration knobs exposed through the package config include:

- `request_timeout_sec`
- `max_retries`
- `base_backoff_sec`
- `max_backoff_sec`
- `rate_limit_gap_delay_sec`
- `appdetails_country_code`
- `appdetails_language`
- `reviews_language`
- `reviews_page_size`
- `api_host_delay_sec`
- `store_host_delay_sec`
- `default_host_delay_sec`

The runtime resolves endpoints from `endpoint_mode`:

- `proxy` mode:
  - app list: `https://gpaul.cc/steamapi/IStoreService/GetAppList/v1/`
  - app details: `https://gpaul.cc/steamstore/api/appdetails`
  - app reviews: `https://gpaul.cc/steamstore/appreviews/{appid}`
- `direct` mode:
  - app list: `https://api.steampowered.com/IStoreService/GetAppList/v1/`
  - app details: `https://store.steampowered.com/api/appdetails`
  - app reviews: `https://store.steampowered.com/appreviews/{appid}`

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
- an opt-in live Stage 2 integration test against the real `appdetails` endpoint

### Opt-in live integration test

The live integration test is intentionally skipped by default so it does not disrupt normal development, notebook execution, or commit flow.

It runs only when both of these are present:

- `RUN_LIVE_STEAM_TESTS=1`
- `STEAM_API_KEY`

Run it with:

```bash
cd steam-crawler
RUN_LIVE_STEAM_TESTS=1 STEAM_API_KEY=... .venv/bin/python -m unittest discover -s tests
```

What it validates:

- real `appdetails` responses for a couple of stable app ids
- correct CSV persistence through Stage 2
- graceful handling of one intentionally invalid app id without breaking the stage

If the env flag is not set, the test is reported as skipped rather than silently ignored.

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

If you want a heavier but still local smoke run, raise the smoke profile limits in the notebook or run the CLI with larger bounds, for example `max_apps=200`, `sample_size=2`, `max_games=2`, and `reviews_per_game=200`.

### Smoke test through the notebook-equivalent runner

The quickest local path without Jupyter is:

```bash
cd steam-crawler
STEAM_API_KEY=... .venv/bin/python run_notebook.py --run-mode smoke --stage all
```

Switch to the original Steam hosts for that run with either:

```bash
STEAM_API_KEY=... STEAM_ENDPOINT_MODE=direct .venv/bin/python run_notebook.py --run-mode smoke --stage all
```

or, if `STEAM_ENDPOINT_MODE` is unset:

```bash
STEAM_API_KEY=... .venv/bin/python run_notebook.py --run-mode smoke --endpoint-mode direct --stage all
```

You can also run a single stage with the same smoke defaults:

```bash
cd steam-crawler
STEAM_API_KEY=... .venv/bin/python run_notebook.py --run-mode smoke --stage stage2
```

### Smoke test through the stage CLI

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

You can add `--endpoint-mode direct` to either CLI, but the env var still has higher priority when both are set.

## Running the full crawl

### Full run through the notebook

For a real dataset build:

1. Ensure earlier smoke outputs are either removed or run with `force_refresh=True` in the notebook cells when needed.
2. Set `STEAM_RUN_MODE=full`.
3. Execute the notebook end-to-end, or run `run_notebook.py --run-mode full --stage all`.

The notebook remains the single submission artifact; `run_notebook.py` exists as an execution-equivalent terminal wrapper around the same profiles and stage calls.
The Postman collection is kept only as an API reference artifact and request example bundle.

### Full run on SLURM

The tracked cluster script is [`sbatch.sh`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/sbatch.sh).
It reflects your current NUS SoC workflow:

- activates the shared `~/env`
- runs from the `steam-crawler` folder itself
- redirects runtime stdout/stderr to `$HOME/logs` and `$HOME/errors`
- keeps the interactive Jupyter Lab command available but commented
- executes the notebook headlessly via `jupyter nbconvert`
- disables the notebook cell timeout for long-running full crawls

Before using it, make sure the shared environment already has the required packages:

```bash
source ~/env/bin/activate
pip install -r requirements.txt
pip install notebook nbconvert
```

Then submit the tracked script from the project folder:

```bash
cd steam-crawler
sbatch sbatch.sh
```

The batch path in [`sbatch.sh`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/sbatch.sh) runs:

```bash
jupyter nbconvert \
  --to notebook \
  --execute notebooks/steam_crawler.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --output steam_crawler.executed.ipynb
```

For the full cluster run, make sure your cluster-side [`steam-crawler/.env`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/.env) or exported environment includes both:

- `STEAM_API_KEY=...`
- `STEAM_RUN_MODE=full`

Optionally add one of:

- `STEAM_ENDPOINT_MODE=proxy`
- `STEAM_ENDPOINT_MODE=direct`

## Architecture and execution model

The notebook and `run_notebook.py` are deliberately thin. Their job is:

1. Locate the project root.
2. Install the declared dependencies into the current kernel.
3. Run a preflight check for node resources and endpoint availability.
4. Select smoke or full limits.
5. Call the staged Python package methods.

The package owns the expensive or stateful logic:

1. HTTP retries and throttling happen in `HttpClient`.
2. CSV normalization happens in `transforms.py`.
3. Resume-aware stage orchestration happens in `Pipeline`.
4. Per-game review cursors are represented by `ReviewCollectionState`.

This split keeps the notebook submission-friendly while ensuring the actual crawler logic is testable outside Jupyter.

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

## Security and operational notes

- Secrets are loaded from `STEAM_API_KEY` or `steam-crawler/.env`; the `.env` file is gitignored and is not part of the tracked project.
- The notebook preflight checks resource visibility and endpoint reachability before any expensive crawl starts.
- Every network request uses an explicit timeout.
- Retry behavior is bounded by `max_retries` and `max_backoff_sec`; the crawler does not spin indefinitely on throttling or transient failures.
- Error logs intentionally include headers and bodies for debugging. Treat `logs/errors.csv` as a debugging artifact and avoid sharing it blindly if upstream services ever return sensitive values in headers or response bodies.

## Notes on resume behavior

- Stage 1 reuses the full cached app catalog if it already exists.
- Stage 2 skips app ids already present in the metadata CSV and writes each success immediately.
- Stage 3 and Stage 4 are cheap enough to rerun from their prior CSVs.
- Stage 5 skips games already marked `completed` or `exhausted` in `stage_05_progress.csv`.
- Stage 5 appends review rows after each fetched page and appends cursor checkpoints after each page, so interrupted runs resume inside the current game instead of restarting it from scratch.
- Stage 5 no longer sends `day_range=365` on the helpful stream, and it now stops a review stream after the configured repeated-cursor limit without any new unique reviews so one game cannot spin forever in a cursor cycle.
- Stage 5 now records a terminal `failed` progress row even when an unexpected exception aborts the run, so resume state stays explicit.
- If a crash happens after rows are flushed but before the latest cursor checkpoint is appended, the rerun reconstructs the existing review ids from the cached dataset and deduplicates them before continuing.

## Known audit gap

- Stage 2 still logs and skips `appdetails` rows that exhaust retries instead of emitting an explicit failed metadata row or failing the stage. That means Stage 3 currently treats those missing rows as absent metadata rather than as a first-class fetch failure.

## Updating this README

This README should be kept in sync with any major flow changes, refactors, or new runtime knobs.
If you change:

- stage boundaries
- output schemas
- smoke/full configuration
- cluster execution flow
- required packages

update this file in the same change set.
