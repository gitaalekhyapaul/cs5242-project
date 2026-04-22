# Steam Dataset Crawler

This folder contains a stage-based Steam crawler designed for two modes of operation:

1. Small-batch smoke tests that validate the flow cheaply.
2. Full dataset runs intended for long-running execution on a SLURM cluster.

The implementation is intentionally split between reusable Python classes in `src/steam_crawler` and a small set of operator surfaces:

- the submission notebook at `notebooks/steam_crawler.ipynb`
- the analysis / ETL notebook at `notebooks/eda_etl.ipynb`
- the notebook-equivalent runner at `run_notebook.py`

## Status Against PLAN.md

The current implementation matches the staged crawler design in [`PLAN.md`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/PLAN.md) for the runtime pipeline, cacheable CSV outputs, retry/backoff handling, notebook orchestration, the notebook-equivalent terminal runner, resume behavior, and test coverage.

By default, crawler traffic is routed through the configured proxy bases:

- `https://gpaul.cc/steamapi`
- `https://gpaul.cc/steamstore`

You can switch back to the original Steam hosts with endpoint mode:

- `STEAM_ENDPOINT_MODE=proxy` keeps the default proxy routing
- `STEAM_ENDPOINT_MODE=direct` uses `https://api.steampowered.com` and `https://store.steampowered.com`
- `--endpoint-mode {proxy,direct}` is available on both terminal runners and overrides `STEAM_ENDPOINT_MODE` from the environment or `.env`

The test suite now has two layers:

- deterministic local tests that run quickly and do not depend on Steam availability
- an opt-in live integration test that exercises the real `appdetails` endpoint with stable app ids plus one intentionally invalid app id

One audit follow-up is still open: Stage 2 currently logs and skips `appdetails` requests that exhaust retries, so Stage 3 can only infer the missing metadata by absence rather than from an explicit failed detail row.

This keeps the default development flow stable while still covering the live API path when you explicitly request it.

For downstream analysis, the repo also includes two optional follow-up transforms that do not modify the main 5-stage crawler:

- Stage 4a starts from the sampled Stage 4 games and re-fetches only the missing store metadata needed for `price` plus one review page per game to recover `query_summary` and derive `%positive_reviews`.
- Stage 5a starts from the Stage 5 review dataset and derives a narrow review table for downstream modeling and analytics.

## Approach

The crawler is built around cacheable CSV stages so each expensive step can be resumed without recomputing earlier work.
Stage 2 now checkpoints after every successful appdetails response, and Stage 5 checkpoints after every fetched review page.

### Stage flow

1. Stage 1 fetches the Steam app list from the configured app-list base (`gpaul.cc` proxy by default, original Steam API in `direct` mode) and writes `data/stage_01_apps_catalog.csv`.
2. Stage 2 fetches `appdetails` per app from the configured store base (`gpaul.cc` proxy by default, original Steam store host in `direct` mode) and writes `data/stage_02_app_details.csv.gz`.
3. Stage 3 merges the catalog and metadata into `data/stage_03_apps_with_metadata.csv.gz`.
4. Stage 4 samples eligible games into `data/stage_04_selected_games.csv`.
5. Stage 5 fetches reviews for the sampled games from the configured reviews base into `data/stage_05_reviews_dataset.csv.gz`.
6. Optional Stage 4a patch enriches the sampled Stage 4 games into `data/stage_04a_selected_games.csv`, then materializes `data/raw_selected_games.parquet` from the completed CSV.
7. Optional Stage 5a transform derives `data/stage_05a_reviews_dataset.csv.gz`, then materializes `data/raw_reviews_dataset.parquet` from that gzipped CSV.

Implementation note: Stage 2 uses the `basic,categories,recommendations` filter set because Steam omits the `type` field if `basic` is not included, and `type == "game"` is required for Stage 4 eligibility.
That means `price_overview` is not present in the cached Stage 2 and Stage 3 payloads. Stage 4a exists specifically to recover price and review-summary fields for the sampled games without rerunning the entire crawler.

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
- Stage 4a fields:
  - `id`
  - `num_reviews`
  - `%positive_reviews`
  - `price`
  - `app_category`
- Stage 5a fields:
  - `timestamp`
  - `user_id`
  - `app_id`
  - `review_id`
  - `review_score`
  - `review_rating`

### Why the code is class-based

The main runtime logic is encapsulated in classes for readability and reuse:

- `Config`: runtime configuration and path layout.
- `HttpClient`: shared retry/backoff HTTP client using `requests`.
- `StagePaths`: centralizes output locations for all stages.
- `ReviewCollectionState`: serializable per-game cursor and counter state for review resume.
- `ReviewCollector`: encapsulates the `recent + helpful + recent-backfill` review strategy.
- `Pipeline`: notebook-friendly orchestrator for all stages.

### Module map

The package is intentionally split by responsibility instead of putting all crawler logic in the notebook:

- [`config.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/src/steam_crawler/config.py): runtime configuration, path layout, and `.env` loading
- [`http_client.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/src/steam_crawler/http_client.py): shared `requests` session, host throttling, retry, backoff, and error capture
- [`transforms.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/src/steam_crawler/transforms.py): CSV flatteners, JSON serialization, sampling, and merge helpers
- [`logging_utils.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/src/steam_crawler/logging_utils.py): run logger and structured CSV error logger
- [`pipeline.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/src/steam_crawler/pipeline.py): staged orchestration, checkpointing, progress bars, and CLI entrypoint
- [`stage4a.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/src/steam_crawler/stage4a.py): Stage 4a sample-only enrichment for price and review-summary patching
- [`stage5a.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/src/steam_crawler/stage5a.py): Stage 5a streaming transform from the Stage 5 review dataset into a narrow review table
- [`run_notebook.py`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/run_notebook.py): notebook-equivalent terminal runner with the same smoke/full profiles and preflight checks
- [`steam_crawler.ipynb`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/notebooks/steam_crawler.ipynb): submission notebook for smoke validation and full cluster execution
- [`eda_etl.ipynb`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/notebooks/eda_etl.ipynb): generic EDA + ETL notebook for Stage 4 sample inspection, Stage 4a / Stage 5a transforms, downstream SteamRec artifacts, and Kaggle publishing

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
│   ├── eda_etl.ipynb
│   └── steam_crawler.ipynb
├── src/steam_crawler/
│   ├── __init__.py
│   ├── config.py
│   ├── http_client.py
│   ├── logging_utils.py
│   ├── pipeline.py
│   ├── stage4a.py
│   ├── stage5a.py
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
- `STEAM_DATA_DIR=/path/to/stage-data`: optional override for stage CSV / gzip outputs; when relative, it is resolved under the crawler root
- `STEAM_SAMPLE_SIZE=10000`: optional Stage 4 sample-size override
- `STEAM_REVIEWS_PER_GAME=1000`: optional Stage 5 per-game review target override
- `STEAM_MIN_RECOMMENDATIONS=5000`: optional Stage 3 eligibility threshold override
- `STEAM_MAX_PAGES=1`: optional Stage 1 page cap
- `STEAM_MAX_APPS=25`: optional Stage 2 and Stage 3 app cap
- `STEAM_MAX_GAMES=2`: optional Stage 5 game cap
- `STEAM_GAP_DELAY=300`: optional `429` cooling-off gap in seconds
- `KAGGLE_USERNAME`: required by `notebooks/eda_etl.ipynb` when using Kaggle-backed notebook workflows and when setting the shared Kaggle dataset handle for notebook uploads
- `KAGGLE_API_TOKEN`: required by `notebooks/eda_etl.ipynb`; the notebook validates it for `kaggle` and `kagglehub` use and maps it to the Kaggle client's expected key env var internally

For terminal runs, both entrypoints also accept `--endpoint-mode proxy` or `--endpoint-mode direct`. The CLI flag overrides `STEAM_ENDPOINT_MODE` from the environment or `.env`.
Both terminal entrypoints also accept `--max-pages <count>` to override `STEAM_MAX_PAGES` from the environment or `.env`.
Both terminal entrypoints also accept `--max-apps <count>` to override `STEAM_MAX_APPS` from the environment or `.env`.
Both terminal entrypoints also accept `--sample-size <count>` to override `STEAM_SAMPLE_SIZE` from the environment or `.env`.
Both terminal entrypoints also accept `--reviews-per-game <count>` to override `STEAM_REVIEWS_PER_GAME` from the environment or `.env`.
Both terminal entrypoints also accept `--min-recommendations <count>` to override `STEAM_MIN_RECOMMENDATIONS` from the environment or `.env`.
Both terminal entrypoints also accept `--max-games <count>` to override `STEAM_MAX_GAMES` from the environment or `.env`.
Both terminal entrypoints also accept `--gap-delay <seconds>` to override `STEAM_GAP_DELAY` from the environment or `.env`.
Both terminal entrypoints also accept `--loop-limit <count>` to override the Stage 5 repeated-cursor stop-gap, and the CLI flag overrides `STEAM_CURSOR_LOOP_LIMIT` from the environment or `.env`.
Both terminal entrypoints also accept `--data-dir <path>` to override stage output storage, and the CLI flag overrides `STEAM_DATA_DIR` from the environment or `.env`.
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

The EDA notebook uses the same env-loading flow as the crawler notebook:

- resolve `ROOT_DIR`
- install from `requirements.txt` into the active kernel
- call `load_project_env(ROOT_DIR, override=True)`
- resolve the data directory through `STEAM_DATA_DIR`

This keeps exploratory work and patch jobs pointed at the same staged data location as the main crawler.

## EDA + ETL notebook

`notebooks/eda_etl.ipynb` is intentionally generic so later EDA and ETL work can live in one place. The current sections cover:

- Stage 4 sample inspection from the configured data directory
- env and dependency checks for notebook execution
- a Stage 4a CSV patch cell
- a separate Stage 4a parquet materialization cell
- a Stage 4a parquet upload cell that pushes the current parquet snapshot into one shared Kaggle dataset through `kagglehub`
- Stage 5 review dataset inspection from the configured data directory
- a Stage 5a gzipped CSV transform cell
- a separate Stage 5a parquet materialization cell
- a SteamRec app metadata ETL cell that writes `steamrec_app_metadata.parquet` and `steamrec_app_metadata.csv`
- a mapping-tables cell that writes `steamrec_app_category_mapping.parquet`, `steamrec_app_category_mapping.csv`, `steamrec_item_mapping.parquet`, and `steamrec_item_mapping.csv`
- a Stage 5a user-diagnostics cell that reports reviews per user, plots the review-count distribution with a threshold reference line, and assigns chronological review positions per user
- a SteamRec interactions ETL cell that reshapes `user_review_positions_df` into `steamrec_interactions.parquet` and `steamrec_interactions.csv`
- a final shared Kaggle upload cell that publishes the locally available raw and ETL parquet files into one dataset through `kagglehub`
- a final Kaggle sanity-check cell that downloads the shared dataset back from Kaggle, verifies the locally available parquet resources exist there too, and previews whichever files the local parquet stack can decode

Stage 4a is sample-only and resumable:

- source input: `data/stage_04_selected_games.csv`
- output CSV: `data/stage_04a_selected_games.csv`
- output parquet: `data/raw_selected_games.parquet`
- log file: `logs/errors_stage_04a.csv`

The Stage 4a CSV patch derives the final schema as follows:

- `id`: from Stage 4 `appid`
- `num_reviews`: from Stage 4 `recommendations_total`
- `%positive_reviews`: from `appreviews.query_summary.total_positive / total_reviews * 100`
- `price`: from `appdetails.data.price_overview.final / 100`, falling back to `price_overview.initial / 100` if `final` is absent, with free games forced to `0.0`
- `app_category`: from Stage 4 `category_ids`

Missing Stage 4a values are written as empty CSV cells rather than the literal string `<NA>`.

The downstream SteamRec app metadata ETL keeps the Stage 4a columns, rescales `%positive_reviews` from `0-100` into `0-1` with two decimal places, converts `app_category` from the pipe-separated Stage 4a string into an array of the original integer category ids, fills missing `price` values with `0.0`, and writes both:

- `data/steamrec_app_metadata.parquet`
- `data/steamrec_app_metadata.csv`

The SteamRec app-category mapping then sorts the original category ids in increasing order, assigns dense ids from `1..n`, keeps the original id in a separate column, and writes both:

- `data/steamrec_app_category_mapping.parquet`
- `data/steamrec_app_category_mapping.csv`

Its columns are:

- `app_category_id`: the dense mapped category id from `1..n`
- `app_category`: the original Steam category id from Stage 4a
- `count`: the number of apps whose category arrays contain that original category id

The SteamRec item mapping sorts the original app ids in increasing order, assigns dense `item_id` values from `1..n`, and writes both:

- `data/steamrec_item_mapping.parquet`
- `data/steamrec_item_mapping.csv`

Its columns are:

- `app_id`: the original Steam app id
- `item_id`: the dense mapped item id from `1..n`

The Stage 5a transform derives the final schema as follows:

- `timestamp`: from `raw_json.timestamp_created`
- `user_id`: from `raw_json.author.steamid`
- `app_id`: from the Stage 5 `appid`
- `review_id`: from `recommendationid`, which is the unique Steam review id
- `review_score`: `1` when `raw_json.voted_up` is true, otherwise `0` when false
- `review_rating`: from `raw_json.votes_up`

The Stage 5a user-diagnostics section then:

- counts reviews per `user_id`
- exposes `USER_REVIEW_COUNT_THRESHOLD` so you can ask how many users exceed a given review count
- plots the review-count histogram on logarithmic axes, using `0-1`, `1-2`, `2-3`, `3-4`, and `4-5` buckets first and then widening buckets after `5`, with a red threshold reference line so you can tune that cutoff visually
- sorts each user's reviews by timestamp and assigns `position = 1, 2, 3, ...` in chronological order
- remaps `app_id` in the position output to the dense `item_id` while keeping the column name `app_id`
- adds `app_category` to the position output as the array of mapped SteamRec category ids for that app

The SteamRec interactions ETL section then reshapes `user_review_positions_df` for downstream use by:

- renaming `app_id` to `item_id`
- renaming `review_rating` to `review_upvotes`
- keeping `app_category` as the mapped SteamRec category-id array for each interaction
- writing both `data/steamrec_interactions.parquet` and `data/steamrec_interactions.csv`

If the local parquet stack cannot decode `raw_selected_games.parquet` or `raw_reviews_dataset.parquet`, the notebook falls back to `stage_04a_selected_games.csv` or `stage_05a_reviews_dataset.csv.gz` for the downstream ETL and diagnostics cells.

Run order in the notebook:

1. Execute the Stage 4a CSV cell until `stage_04a_selected_games.csv` reaches the same row count as `stage_04_selected_games.csv`.
2. Execute the next parquet cell to write `raw_selected_games.parquet` from the completed CSV.
3. Optionally edit `KAGGLE_SHARED_DATASET_HANDLE` in the next cell, then run the upload cell to publish the current Stage 4a parquet snapshot into one Kaggle dataset.
4. Execute the Stage 5a CSV cell to derive `stage_05a_reviews_dataset.csv.gz` from `stage_05_reviews_dataset.csv.gz`.
5. Execute the next Stage 5a parquet cell to write `raw_reviews_dataset.parquet` from the completed Stage 5a gzipped CSV.
6. Execute the SteamRec app metadata ETL cell to write `steamrec_app_metadata.parquet` and `steamrec_app_metadata.csv`.
7. Execute the mapping-tables cell to write `steamrec_app_category_mapping.parquet`, `steamrec_app_category_mapping.csv`, `steamrec_item_mapping.parquet`, and `steamrec_item_mapping.csv`.
8. Inspect the Stage 5a user-diagnostics cell and adjust `USER_REVIEW_COUNT_THRESHOLD` if you want a different review-count cutoff.
9. Execute the SteamRec interactions ETL cell to write `steamrec_interactions.parquet` and `steamrec_interactions.csv` from `user_review_positions_df`.
10. Optionally edit `KAGGLE_SHARED_DATASET_HANDLE` in the final upload cell, then publish whichever raw and ETL parquet files are currently present locally into the shared Kaggle dataset, including `steamrec_interactions.parquet`.
11. Run the final Kaggle sanity-check cell to download the shared dataset from Kaggle, confirm that those parquet resources are present, and inspect the previews that the local parquet stack can decode.

Do not run either parquet cell while its corresponding CSV build step is still incomplete.

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
- Stage 5 now pages the recent stream first up to the recent quota, then uses the helpful stream up to the helpful quota, and finally falls back to recent backfill if the unique review target is still not met.
- Stage 5 sends `day_range=365` only on the helpful stream (`filter=all`), never on recent requests, and it stops a review stream after the configured repeated-cursor limit without any new unique reviews so one game cannot spin forever in a cursor cycle.
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
