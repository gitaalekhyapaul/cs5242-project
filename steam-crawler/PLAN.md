# Steam Dataset Crawler Plan

## Summary
- Bootstrap a new Python crawler workspace inside `steam-crawler` because the folder currently only contains the Postman collection: [SteamAPI.postman_collection.json](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler/SteamAPI.postman_collection.json).
- Build the crawler as reusable Python modules under `steam-crawler/src/steam_crawler` with two thin operator surfaces:
  `steam-crawler/notebooks/steam_crawler.ipynb` and `steam-crawler/run_notebook.py`. The modules own HTTP, retry, flattening, resume, and CSV writing.
- Use 5 cacheable stages, each producing a CSV-format artifact and reusing the previous stage if present:
  1. `stage_01_apps_catalog.csv`
  2. `stage_02_app_details.csv.gz`
  3. `stage_03_apps_with_metadata.csv.gz`
  4. `stage_04_selected_games.csv`
  5. `stage_05_reviews_dataset.csv.gz`
- Stage 4 will sample up to 10,000 games where `recommendations_total > 5000`, using a fixed seed for reproducibility.
- Stage 5 will pull 1,000 reviews per selected game as `500 recent + 500 helpful`, deduplicated by `recommendationid`, with recent backfill if the unique target is still not reached.
- The production runtime defaults to the proxy bases `https://gpaul.cc/steamapi` and `https://gpaul.cc/steamstore`, but can switch back to the original Steam hosts when `STEAM_ENDPOINT_MODE=direct` is set. If both env and CLI endpoint mode are set, the CLI flag wins.
- Stage 5's repeated-cursor stop-gap defaults to `STEAM_CURSOR_LOOP_LIMIT=10`; `--loop-limit` is available on both terminal runners, and the CLI flag wins if both are set.
- Stage outputs default to `<root>/data`, but `STEAM_DATA_DIR` or `--data-dir` can relocate the staged CSV / gzip outputs. If both are set, the CLI flag wins.
- Stage limits can also be driven by `STEAM_MAX_PAGES`, `STEAM_MAX_APPS`, `STEAM_SAMPLE_SIZE`, `STEAM_MAX_GAMES`, and `STEAM_GAP_DELAY`; the corresponding CLI flags win if both are set.

## Interfaces
- Config can come from notebook cells, `run_notebook.py`, or env vars:
  `STEAM_API_KEY`, `STEAM_RUN_MODE`, `STEAM_ENDPOINT_MODE`, `STEAM_CURSOR_LOOP_LIMIT`, `STEAM_DATA_DIR`, `STEAM_SAMPLE_SIZE`, `STEAM_MAX_PAGES`, `STEAM_MAX_APPS`, `STEAM_MAX_GAMES`, `STEAM_GAP_DELAY`.
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
- Logs:
  `logs/run.log` for normal progress and `logs/errors.csv` for failures with `stage`, `appid`, `url`, `params_json`, `attempt`, `status_code`, `response_headers_json`, `response_body`, `exception_type`, `exception_message`, `retry_after_seconds`, `logged_at`.

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

## Notebook Behavior
- The notebook should have one runnable section per stage plus one “run all missing stages” section. `run_notebook.py` should expose the same stage selection and smoke/full profile behavior for terminal execution.
- Every long loop must show a `tqdm.notebook` progress bar. Stage 2 uses total apps from Stage 1. Stage 5 uses selected-game count as the outer total and a per-game review counter as the inner total.
- After each stage, print a compact summary in the notebook: rows written, elapsed time, retry count, error count, and output path.
- If a stage input CSV already exists, the notebook should announce that it is reusing the cached output and skip recomputation unless explicitly overridden.

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
