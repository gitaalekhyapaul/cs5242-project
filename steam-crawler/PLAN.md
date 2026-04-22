# Steam Dataset Crawler Plan

## Summary
- Bootstrap a new Python crawler workspace inside `steam-crawler` because the folder currently only contains the Postman collection: [SteamAPI.postman_collection.json](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/SteamAPI.postman_collection.json).
- Build the crawler as reusable Python modules under `steam-crawler/src/steam_crawler` with thin operator surfaces:
  `steam-crawler/notebooks/steam_crawler.ipynb`, `steam-crawler/notebooks/eda_etl.ipynb`, and `steam-crawler/run_notebook.py`. The modules own HTTP, retry, flattening, resume, and CSV writing.
- Use 5 cacheable stages, each producing a CSV-format artifact and reusing the previous stage if present:
  1. `stage_01_apps_catalog.csv`
  2. `stage_02_app_details.csv.gz`
  3. `stage_03_apps_with_metadata.csv.gz`
  4. `stage_04_selected_games.csv`
  5. `stage_05_reviews_dataset.csv.gz`
- Add an optional Stage 4a patch artifact for analysis only, built from the Stage 4 sample rather than the full Stage 3 population:
  `stage_04a_selected_games.csv` and `raw_selected_games.parquet`
- Add an optional Stage 5a analysis artifact derived from the Stage 5 review dataset:
  `stage_05a_reviews_dataset.csv.gz` and `raw_reviews_dataset.parquet`
- Stage 4 will sample up to 10,000 games where `recommendations_total > 5000`, using a fixed seed for reproducibility.
- Stage 5 will pull 1,000 reviews per selected game as `500 recent + 500 helpful`, deduplicated by `recommendationid`, with recent backfill if the unique target is still not reached.
- Stage 4a will enrich the sampled Stage 4 games only by re-fetching `appdetails` for `price_overview` and one `appreviews` page for `query_summary`, so downstream analysis can recover `price` and `%positive_reviews` without rerunning the main crawl.
- The production runtime defaults to the proxy bases `https://gpaul.cc/steamapi` and `https://gpaul.cc/steamstore`, but can switch back to the original Steam hosts when `STEAM_ENDPOINT_MODE=direct` is set. If both env and CLI endpoint mode are set, the CLI flag wins.
- Stage 5's repeated-cursor stop-gap defaults to `STEAM_CURSOR_LOOP_LIMIT=10`; `--loop-limit` is available on both terminal runners, and the CLI flag wins if both are set.
- Stage outputs default to `<root>/data`, but `STEAM_DATA_DIR` or `--data-dir` can relocate the staged CSV / gzip outputs. If both are set, the CLI flag wins.
- Stage limits can also be driven by `STEAM_MAX_PAGES`, `STEAM_MAX_APPS`, `STEAM_SAMPLE_SIZE`, `STEAM_MIN_RECOMMENDATIONS`, `STEAM_MAX_GAMES`, and `STEAM_GAP_DELAY`; the corresponding CLI flags win if both are set.

## Interfaces
- Config can come from notebook cells, `run_notebook.py`, or env vars:
  `STEAM_API_KEY`, `STEAM_RUN_MODE`, `STEAM_ENDPOINT_MODE`, `STEAM_CURSOR_LOOP_LIMIT`, `STEAM_DATA_DIR`, `STEAM_SAMPLE_SIZE`, `STEAM_MIN_RECOMMENDATIONS`, `STEAM_MAX_PAGES`, `STEAM_MAX_APPS`, `STEAM_MAX_GAMES`, `STEAM_GAP_DELAY`.
- Endpoint mode values:
  `proxy` for `gpaul.cc` routing, `direct` for the original Steam hosts. `run_notebook.py` and `python -m steam_crawler.pipeline` also accept `--endpoint-mode {proxy,direct}`, and the CLI flag takes priority when both are present.
- Stage 1 CSV schema:
  `appid`, `name`, `last_modified`, `price_change_number`, `raw_json`.
- Stage 2 CSV schema:
  `appid`, `success`, `type`, `category_ids`, `category_descriptions`, `recommendations_total`, `raw_json`.
- Stage 3 CSV schema:
  merged Stage 1 + Stage 2 columns, plus `eligible_for_sampling`.
- Stage 4 CSV schema:
  Stage 3 columns plus `sample_rank`, `random_seed`, `sampled_at`.
- Stage 5 CSV schema:
  `appid`, `recommendationid`, `author_steamid`, `timestamp_created`, `review_text`, `source_stream`, `raw_json`.
- Stage 4a analysis schema:
  `id`, `num_reviews`, `%positive_reviews`, `price`, `app_category`.
- Stage 5a analysis schema:
  `timestamp`, `user_id`, `app_id`, `review_id`, `review_score`, `review_rating`.
- Logs:
  `logs/run.log` for normal progress and `logs/errors.csv` for failures with `stage`, `appid`, `url`, `params_json`, `attempt`, `status_code`, `response_headers_json`, `response_body`, `exception_type`, `exception_message`, `retry_after_seconds`, `logged_at`.
- EDA notebook env expectations:
  `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` must be present for notebook-side data tasks; the notebook validates those credentials for `kaggle` and `kagglehub` use and maps `KAGGLE_API_TOKEN` to the Kaggle client's expected key env var internally.

## Implementation
- HTTP client:
  use one shared session with explicit `User-Agent`, sane timeouts, and per-endpoint throttling. Retry on network errors and `429/500/502/503/504`; non-retryable HTTP errors should fail immediately after one logged attempt. On `429`, first honor `Retry-After` if present as seconds or HTTP date; then check similar headers if any appear; otherwise apply a fixed `rate_limit_gap_delay_sec` cooling-off gap before the deterministic exponential backoff component (`2^0 = 1` second doubling per retry, capped by `max_backoff_sec`). Every failed response must be logged with full headers and body to both the run logger and `logs/errors.csv`.
- Stage 1:
  call the configured app-list endpoint with `max_results=5000`, follow `have_more_results` and `last_appid`, and write one row per app. In `proxy` mode this is `https://gpaul.cc/steamapi/IStoreService/GetAppList/v1/`; in `direct` mode this is `https://api.steampowered.com/IStoreService/GetAppList/v1/`. Resume by skipping this stage if the CSV already exists, unless the user forces a refresh.
- Stage 2:
  read Stage 1 and fetch appdetails one app at a time from the configured store appdetails endpoint using `cc=us` and `l=english` for stable category text. In `proxy` mode this is `https://gpaul.cc/steamstore/api/appdetails`; in `direct` mode this is `https://store.steampowered.com/api/appdetails`. Persist after every successful app so reruns can skip completed `appid`s. Store minified raw JSON in the CSV; flatten categories as pipe-separated IDs and descriptions.
- Stage 3:
  merge Stage 1 and Stage 2 into one metadata CSV, set `eligible_for_sampling = success && type == "game" && recommendations_total > 5000`, and keep both raw columns from the prior stages so this file is the canonical input for sampling.
- Stage 4:
  load Stage 3, filter eligible rows, sample without replacement using the fixed seed, and select `min(10000, eligible_count)` rows. Output a stable `sample_rank` so later reruns preserve order.
- Stage 5:
  for each sampled game, first page `filter=recent` until the recent quota is reached or the stream exhausts; then page `filter=all&day_range=365` until the helpful quota is reached or the stream exhausts; if the total is still below 1,000 unique rows, continue paging the recent stream as backfill. Use `language=all`, `review_type=all`, `purchase_type=all`, `num_per_page=100`, and keep the default off-topic filtering against the configured reviews endpoint. In `proxy` mode this is `https://gpaul.cc/steamstore/appreviews/{appid}`; in `direct` mode this is `https://store.steampowered.com/appreviews/{appid}`. Track per-app progress in `stage_05_progress.csv` so interruption does not require rereading the full review dataset. Unexpected exceptions should still persist a terminal `failed` progress row before being re-raised. Stop a review stream after the configured repeated-cursor limit when those cursors yield no new unique review IDs so one app cannot spin forever on a cursor cycle.
- Stage 4a:
  load `stage_04_selected_games.csv`, preserve Stage 4's sampled `appid`, `recommendations_total`, and `category_ids`, then fetch `appdetails` one app at a time using `filters=basic,price_overview` and fetch one reviews page per app with `num_per_page=1`. Convert `price_overview.final` from cents to currency units, fall back to `price_overview.initial` if `final` is absent, set free games to `0.0`, compute `%positive_reviews = total_positive / total_reviews * 100` from `query_summary`, and append rows incrementally to `stage_04a_selected_games.csv`. Missing values should be serialized as empty CSV cells rather than the literal string `<NA>`. Write the parquet copy only after the CSV patch step has finished.
- Stage 5a:
  load `stage_05_reviews_dataset.csv.gz` from the configured data directory and derive a narrow review dataset with one output row per Stage 5 review. Keep `timestamp` from `timestamp_created`, `user_id` from the nested review author steam id, `app_id` from the Stage 5 app id, `review_id` from `recommendationid`, `review_score` as `1` for `voted_up=true` and `0` for `voted_up=false`, and `review_rating` from `votes_up`. Write `stage_05a_reviews_dataset.csv.gz` first, then materialize `raw_reviews_dataset.parquet` from that gzipped CSV in a later step.

## Notebook Behavior
- `steam_crawler.ipynb` should have one runnable section per crawler stage plus one “run all missing stages” section. `run_notebook.py` should expose the same stage selection and smoke/full profile behavior for terminal execution.
- `eda_etl.ipynb` should mirror the project env-loading pattern, honor `STEAM_DATA_DIR`, ensure notebook dependencies are installed in the active kernel, validate the Kaggle credentials used by `kagglehub`, and provide analysis / ETL cells that operate on the staged outputs without changing the main pipeline.
- Every long loop must show a `tqdm.notebook` progress bar. Stage 2 uses total apps from Stage 1. Stage 5 uses selected-game count as the outer total and a per-game review counter as the inner total.
- After each stage, print a compact summary in the notebook: rows written, elapsed time, retry count, error count, and output path.
- If a stage input CSV already exists, the notebook should announce that it is reusing the cached output and skip recomputation unless explicitly overridden.
- The Stage 4a cells in `eda_etl.ipynb` should run as two explicit steps: first the resumable CSV patch, then a later parquet materialization cell that reads the completed CSV and writes `raw_selected_games.parquet`.
- The Stage 5a cells in `eda_etl.ipynb` should run after the Stage 4a section and follow the same two-step pattern: first build the gzipped CSV, then materialize `raw_reviews_dataset.parquet` from the completed Stage 5a CSV.
- `eda_etl.ipynb` should then derive `steamrec_app_metadata.{parquet,csv}` from the Stage 4a output by scaling `%positive_reviews` into `0-1`, filling missing `price` values with `0.0`, and keeping `app_category` as arrays of the original integer category ids.
- `eda_etl.ipynb` should derive `steamrec_app_category_mapping.{parquet,csv}` by sorting the original category ids in increasing order, assigning dense `app_category_id` values from `1..n`, and writing columns `app_category_id`, `app_category`, and `count`.
- `eda_etl.ipynb` should also derive `steamrec_item_mapping.{parquet,csv}` by sorting the original app ids in increasing order, assigning dense `item_id` values from `1..n`, and writing columns `app_id` and `item_id`.
- `eda_etl.ipynb` should also expose a Stage 5a user-diagnostics section that counts reviews per `user_id`, supports a review-count threshold knob, plots the review-count histogram on logarithmic axes with `0-1`, `1-2`, `2-3`, `3-4`, and `4-5` buckets before widening bucket sizes after `5`, and assigns chronological `position` values within each user history before any later ETL is chosen.
- In that position output, `app_id` should be remapped to the dense SteamRec `item_id` while keeping the column name `app_id`, and `app_category` should carry the mapped SteamRec category-id array for the corresponding app.
- `eda_etl.ipynb` should then derive `steamrec_interactions.{parquet,csv}` from `user_review_positions_df` by renaming `app_id` to `item_id`, renaming `review_rating` to `review_upvotes`, and preserving the mapped `app_category` arrays on each interaction row.
- If the local parquet stack cannot decode `raw_selected_games.parquet` or `raw_reviews_dataset.parquet`, the downstream ETL and diagnostics cells in `eda_etl.ipynb` should fall back to `stage_04a_selected_games.csv` or `stage_05a_reviews_dataset.csv.gz` instead of blocking the notebook.
- `eda_etl.ipynb` should end with one shared Kaggle upload cell and a Kaggle sanity-check cell that work against whichever raw and ETL parquet resources are available locally, including `steamrec_interactions.parquet`.

## Current Follow-up Gap
- Stage 2 still logs and skips `appdetails` requests that exhaust retries instead of emitting an explicit failed metadata row or failing the stage. The remaining fix is to choose one of those behaviors so Stage 3 cannot silently treat fetch exhaustion as absent metadata.

## Test Plan
- Unit-test retry delay parsing for numeric `Retry-After`, HTTP-date `Retry-After`, missing header fallback, and capped exponential backoff.
- Unit-test flatteners for categories, raw JSON serialization, and review row extraction.
- Integration smoke test on a tiny run:
  Stage 1 full paging mocked or capped, Stage 2 on a handful of real app IDs plus one invalid app ID, Stage 4 sample size reduced, Stage 5 review count reduced to 20 per game.
- Resume test:
  interrupt Stage 2 and Stage 5 midway, rerun, and verify no duplicate `appid` rows in metadata and no duplicate `recommendationid` rows per `appid` in reviews.
- Determinism test:
  rerun Stage 4 with the same seed and verify the same selected game order.

## Assumptions And References
- Per your choices, the `>5000 reviews` threshold will actually use `appdetails.recommendations.total`, not `appreviews.query_summary.total_reviews`.
- Review text output will be multilingual because the final review dataset will use `language=all`.
- If fewer than 10,000 games satisfy the threshold, Stage 4 will emit all eligible games.
- `appdetails` is treated as an undocumented store endpoint; do not assume multi-app batching because live validation with `appids=10,20` returned `null`.
- Source references used for the plan:
  [Steamworks IStoreService GetAppList docs](https://partner.steamgames.com/doc/webapi/IStoreService),
  [Steamworks User Reviews docs](https://partner.steamgames.com/doc/store/getreviews?l=english&language=english),
  [proxy appdetails shape check](https://gpaul.cc/steamstore/api/appdetails?appids=10&filters=categories,recommendations,type),
  [direct appdetails shape check](https://store.steampowered.com/api/appdetails?appids=10&filters=categories,recommendations,type).
