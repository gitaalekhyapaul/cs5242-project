# Steam Dataset Crawler Plan

## Summary
- Bootstrap a new Python crawler workspace inside `steam-crawler` because the folder currently only contains the Postman collection: [SteamAPI.postman_collection.json](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/SteamAPI.postman_collection.json).
- Build the crawler as reusable Python modules under `steam-crawler/src/steam_crawler` with a thin orchestration notebook at `steam-crawler/notebooks/steam_crawler.ipynb`. The notebook is the operator surface; the modules own HTTP, retry, flattening, resume, and CSV writing.
- Use 5 cacheable stages, each producing a CSV-format artifact and reusing the previous stage if present:
  1. `stage_01_apps_catalog.csv`
  2. `stage_02_app_details.csv.gz`
  3. `stage_03_apps_with_metadata.csv.gz`
  4. `stage_04_selected_games.csv`
  5. `stage_05_reviews_dataset.csv.gz`
- Stage 4 will sample up to 10,000 games where `recommendations_total > 5000`, using a fixed seed for reproducibility.
- Stage 5 will pull 1,000 reviews per selected game as `500 recent + 500 helpful`, deduplicated by `recommendationid` with backfill until 1,000 unique reviews or exhaustion.

## Interfaces
- Config comes from one notebook cell plus env vars:
  `STEAM_API_KEY`, `sample_size=10000`, `min_recommendations=5000`, `reviews_per_game=1000`, `recent_quota=500`, `helpful_quota=500`, `random_seed=5242`, `request_timeout_sec`, `max_retries`, `base_backoff_sec`, `max_backoff_sec`.
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
- Logs:
  `logs/run.log` for normal progress and `logs/errors.csv` for failures with `stage`, `appid`, `url`, `params_json`, `attempt`, `status_code`, `response_headers_json`, `response_body`, `exception_type`, `exception_message`, `retry_after_seconds`, `logged_at`.

## Implementation
- HTTP client:
  use one shared session with explicit `User-Agent`, sane timeouts, and per-host throttling. Retry on network errors and `429/500/502/503/504`. On `429`, first honor `Retry-After` if present as seconds or HTTP date; then check similar headers if any appear; otherwise fall back to exponential backoff with full jitter. Every failed response must be logged with full headers and body to both notebook output and `logs/errors.csv`.
- Stage 1:
  call `IStoreService/GetAppList/v1` with `max_results=50000`, follow `have_more_results` and `last_appid`, and write one row per app. Resume by skipping this stage if the CSV already exists, unless the user forces a refresh.
- Stage 2:
  read Stage 1 and fetch appdetails one app at a time from `store.steampowered.com/api/appdetails` using `cc=us` and `l=english` for stable category text. Persist after every successful app so reruns can skip completed `appid`s. Store minified raw JSON in the CSV; flatten categories as pipe-separated IDs and descriptions.
- Stage 3:
  merge Stage 1 and Stage 2 into one metadata CSV, set `eligible_for_sampling = success && type == "game" && recommendations_total > 5000`, and keep both raw columns from the prior stages so this file is the canonical input for sampling.
- Stage 4:
  load Stage 3, filter eligible rows, sample without replacement using the fixed seed, and select `min(10000, eligible_count)` rows. Output a stable `sample_rank` so later reruns preserve order.
- Stage 5:
  for each sampled game, first page `filter=recent` until 500 unique review IDs or exhaustion; then page `filter=all&day_range=365` for helpful reviews until 500 additional unique reviews or exhaustion; if overlap leaves the total below 1,000, keep paging the helpful stream first, then recent, until 1,000 unique rows or both streams stop yielding new reviews. Use `language=all`, `review_type=all`, `purchase_type=all`, `num_per_page=100`, and keep the default off-topic filtering. Track per-app progress in `stage_05_progress.csv` so interruption does not require rereading the full review dataset.

## Notebook Behavior
- The notebook should have one runnable section per stage plus one “run all missing stages” section.
- Every long loop must show a `tqdm.notebook` progress bar. Stage 2 uses total apps from Stage 1. Stage 5 uses selected-game count as the outer total and a per-game review counter as the inner total.
- After each stage, print a compact summary in the notebook: rows written, elapsed time, retry count, error count, and output path.
- If a stage input CSV already exists, the notebook should announce that it is reusing the cached output and skip recomputation unless explicitly overridden.

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
  [live appdetails shape check](https://store.steampowered.com/api/appdetails?appids=10&filters=categories,recommendations,type).
