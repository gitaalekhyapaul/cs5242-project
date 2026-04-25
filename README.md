# CS5242 MobileRec Project

This repository is now organized around one concrete goal: preparing the `recmeapp/mobilerec` dataset and training a SASRec baseline for next-item recommendation.

The immediate scope is:
- data preparation for sequential recommendation
- a reproducible SASRec baseline
- leave-one-out evaluation with `HR@10` and `NDCG@10`

## Project Focus

We use the MobileRec interactions dataset to model app recommendation as a sequential next-item prediction task. Each user's review history is converted into a chronological app sequence. For the baseline:
- each reviewed app is treated as a positive interaction
- users are split with chronological leave-one-out
- the second-last interaction is used for validation
- the last interaction is used for test

This is the standard setup used by the original SASRec paper and many later reproductions.

Useful references:
- SASRec paper/code: https://github.com/kang205/SASRec
- PyTorch SASRec reproduction: https://github.com/pmixer/SASRec.pytorch
- MobileRec dataset: https://huggingface.co/datasets/recmeapp/mobilerec
- Processed MobileRec dataset: [gitaalekhyapaul/cs5242-mobilerec-dataset](https://www.kaggle.com/datasets/gitaalekhyapaul/cs5242-mobilerec-dataset)
- Processed Steam dataset: [gitaalekhyapaul/steam-cs5242-dataset](https://www.kaggle.com/datasets/gitaalekhyapaul/steam-cs5242-dataset)

## Published Datasets

This repo currently exposes two processed datasets that are ready to consume from Kaggle:

- MobileRec sequential recommendation artifacts: [gitaalekhyapaul/cs5242-mobilerec-dataset](https://www.kaggle.com/datasets/gitaalekhyapaul/cs5242-mobilerec-dataset)
- Steam processed stage artifacts: [gitaalekhyapaul/steam-cs5242-dataset](https://www.kaggle.com/datasets/gitaalekhyapaul/steam-cs5242-dataset)

The MobileRec dataset is the main input for the SASRec baseline in this root project.
The Steam dataset is produced by the stage-based crawler under [`steam-crawler/`](./steam-crawler/PLAN.md) and is meant for the Steam-specific data collection and downstream analysis workflow.

## Repository Layout

```text
.
|-- .env.example
|-- pyproject.toml
|-- README.md
|-- models/
|   |-- prepare_mobilerec.py
|   |-- sasrec.py
|   |-- train_sasrec.py
|   `-- upload_processed_to_kaggle.py
|-- main/
|   |-- train_model.py
|   |-- finetune_tisasrec_m_transfer.py
|   `-- models/
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- configs/
|   `-- outputs/
|-- notebooks/
|   `-- CS5242_Project.ipynb
`-- steam-crawler/
```

Notes:
- `notebooks/CS5242_Project.ipynb` is kept as existing exploratory work.
- `steam-crawler/` is the stage-based Steam data pipeline used to build the published Steam processed dataset. The full operator runbook lives in [steam-crawler/README.md](./steam-crawler/README.md).

## Steam Crawler Reference

The root project is centered on MobileRec preparation and SASRec training, but the repo also contains a separate Steam ingestion workflow in [`steam-crawler/`](/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project/steam-crawler).

That crawler covers:

- staged Steam API collection with resumable CSV checkpoints
- optional Stage 4a and Stage 5a transforms for downstream analytics
- notebook and terminal operator surfaces
- publication of the processed Steam parquet snapshots to Kaggle

For the full crawler contract, environment setup, stage layout, and notebook/CLI usage, read [steam-crawler/README.md](./steam-crawler/README.md).

## Environment

This project is configured for:
- Python `3.10.12`
- `torch==2.9.1+cu129` on Linux
- `torch==2.9.1` on macOS arm64
- CUDA runtime `12.9`
- GPU validation target: `Tesla T4`

The root `pyproject.toml` is set up for `uv` with platform-specific PyTorch resolution:

- Linux resolves the CUDA 12.9 wheel from the official PyTorch index for cluster training.
- macOS arm64 resolves the standard PyPI wheel so local development tools like `uv sync` work on Apple Silicon.

## Setup With uv

Create or refresh the in-project environment:

```bash
cd /home/e0492463/cs5242-project-main
uv venv --python /usr/bin/python3 .venv
source .venv/bin/activate
uv sync
```

If you only want the already-created environment:

```bash
cd /home/e0492463/cs5242-project-main
source .venv/bin/activate
```

## Step 1: Prepare MobileRec

First download the raw files with the Hugging Face CLI, not from Python:

```bash
source .venv/bin/activate
hf download recmeapp/mobilerec interactions/mobilerec_final.csv app_meta/app_meta.csv \
  --repo-type dataset \
  --local-dir data/raw/mobilerec
```

Then build processed sequence artifacts:

```bash
source .venv/bin/activate
prepare-mobilerec
```

The default raw paths expected by `prepare-mobilerec` are:
- `data/raw/mobilerec/interactions/mobilerec_final.csv`
- `data/raw/mobilerec/app_meta/app_meta.csv`

The preprocessing step is designed to stay safer on this VM:
- raw download is done by the Hugging Face CLI, not by Python dataset loaders
- preprocessing uses disk-backed DuckDB instead of reading the whole CSV into pandas at once
- the default DuckDB memory cap is `2GB`

### What The Data Preparation Actually Does

`models/prepare_mobilerec.py` converts the raw MobileRec review logs into SASRec-ready sequential recommendation data with the following steps:

1. Load `mobilerec_final.csv` with DuckDB and keep only the fields needed for recommendation: `uid`, `app_package`, `app_category`, `rating`, and a usable interaction timestamp.
2. Clean invalid rows by dropping records with missing `uid`, missing `app_package`, or missing time information. The script first tries `unix_timestamp`; if that is unavailable, it falls back to parsing `formated_date`.
3. Filter low-activity users and keep only users with at least `5` interactions by default (`--min-user-interactions`). This ensures every remaining user has enough history for train, validation, and test splits.
4. Optionally subsample users deterministically with `--sample-users`, ordered by `uid`, for faster smoke tests.
5. Build integer id mappings:
   - `uid -> user_id`
   - `app_package -> item_id`
6. Create chronological user histories by sorting each user's interactions by `(timestamp, app_package)` and assigning a `position` index.
7. Export the cleaned interaction table to `interactions.parquet`, then aggregate each user history into sequence features for leave-one-out evaluation:
   - `train_sequence`: all but the last two items
   - `validation_sequence`: all but the last item
   - `test_sequence`: the full sequence
   - `validation_target`: the second-last item
   - `test_target`: the last item
8. Load `app_meta.csv`, remove duplicate `app_package` rows, and keep only metadata for items that still exist after filtering.
9. Write a compact `summary.json` with row counts, number of users/items, and validation/test coverage.

In short, the raw review table is turned into one ordered interaction log plus one per-user sequence table, which is the exact format used by the SASRec training script.

This writes:
- `data/processed/mobilerec/interactions.parquet`
- `data/processed/mobilerec/sequences.parquet`
- `data/processed/mobilerec/user_mapping.parquet`
- `data/processed/mobilerec/item_mapping.parquet`
- `data/processed/mobilerec/app_metadata.parquet`
- `data/processed/mobilerec/summary.json`

## Optional: Upload Processed Data To Kaggle

The root project now includes a terminal uploader at `models/upload_processed_to_kaggle.py`.
It follows the same Kaggle credential contract used in `steam-crawler/notebooks/eda.ipynb`:

- `KAGGLE_USERNAME`
- `KAGGLE_API_TOKEN`

Create the root env file first:

```bash
cp .env.example .env
```

Then upload any processed directory with a Kaggle dataset handle:

```bash
upload-processed-to-kaggle \
  --input-dir data/processed/mobilerec \
  --dataset-handle <your-kaggle-username>/<dataset-slug>
```

The script:

- loads credentials from the root `.env` by default
- lets the CLI override `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN`
- stages every non-hidden file under the input directory into a temporary upload bundle
- preserves the relative file layout inside the Kaggle dataset version
- maps `KAGGLE_API_TOKEN` to `KAGGLE_KEY` internally for the Kaggle client

The Kaggle upload cell in `eda_etl.ipynb` uses the same uploader helpers and also stages compatibility aliases for downstream Kaggle consumers:
- `final_app_category.parquet` from `app_category_mapping.parquet`
- `final_sequences.parquet` from `enriched_mobilerec_sequences.parquet`
- `final_item_mapping.parquet` from `item_mapping.parquet`

A separate Kaggle sanity-check cell downloads the Kaggle snapshot with retries and previews each expected parquet, so upload and verification can be rerun independently.

Useful flags:

- `--env-file /path/to/.env` to use a different env file
- `--kaggle-username <name>` to override `KAGGLE_USERNAME`
- `--kaggle-api-token <token>` to override `KAGGLE_API_TOKEN`
- `--version-notes "..."` to control the Kaggle version message

For a faster smoke run:

```bash
prepare-mobilerec --processed-dir data/processed/mobilerec-sample --sample-users 5000
```

If the VM is under pressure, you can make preprocessing even more conservative:

```bash
prepare-mobilerec --memory-limit 1GB --threads 2
```

## Step 2: Train SASRec Baseline

Run training on the prepared dataset:

```bash
source .venv/bin/activate
train-sasrec \
  --data-dir data/processed/mobilerec \
  --output-dir data/outputs/sasrec-baseline-full \
  --epochs 5 \
  --batch-size 256 \
  --max-len 50 \
  --eval-negative-samples 100
```

Outputs:
- `data/outputs/sasrec-baseline-full/best_model.pt`
- `data/outputs/sasrec-baseline-full/metrics.json`

For a smaller first run, prepare a sample dataset first and then train on it:

```bash
prepare-mobilerec --processed-dir data/processed/mobilerec-sample --sample-users 5000
train-sasrec --data-dir data/processed/mobilerec-sample --output-dir data/outputs/sasrec-sample --epochs 3
```

## Transfer Fine-Tuning: MobileRec To SteamRec

Use `main/finetune_tisasrec_m_transfer.py` to fine-tune the saved MobileRec
`tisasrec_m` checkpoint on SteamRec. The script fixes these choices by design:

- source model: `tisasrec_m`
- source mode: `penalize-negative`
- target dataset: `steamrec`
- trainable weights: `item_emb.weight` and `metadata_cat_emb.weight`
- frozen weights: the attention stack, feed-forward stack, time embeddings, position embeddings, numeric metadata projection, and fusion layer

The script loads all compatible same-shape MobileRec weights. It resets the
SteamRec item and genre/category embedding tables, then trains only those two
tables.

```bash
source .venv/bin/activate
python main/finetune_tisasrec_m_transfer.py \
  --source-checkpoint main/data/outputs/tisasrec_m/penalize-negative/best_model.pt \
  --epochs 5 \
  --batch-size 128 \
  --report-full-eval
```

Default outputs are written under:

```text
main/data/outputs/steamrec_transfer/tisasrec_m/penalize-negative/embedding_finetune/
```

The run directory keeps the same training artifacts as the normal trainer:

- `history.csv`
- `current_model.pt`
- `best_model.pt`
- `metrics.json`
- `transfer_load_report.json`

## TiSASRec Time Handling

For `tisasrec` and `tisasrec_m`, the trainer converts each user's raw
timestamps into personalized time ids before it builds relation matrices. For
each user, it finds the shortest nonzero adjacent timestamp gap. It then maps
each timestamp to:

```text
round((timestamp - user_min_timestamp) / user_min_nonzero_gap) + 1
```

If every timestamp in a user sequence is equal, the scale is `1`.

The relation matrix cache uses the personalized mode in its filename:

```text
relation_matrix_<dataset>_<max_len>_<time_span>_personalized.pickle
```

Older raw timestamp caches without the `_personalized` suffix are ignored by
the trainer.

## Baseline Assumptions

The current baseline intentionally keeps the setup simple:
- pure sequential recommendation using app ids only
- no metadata features yet
- all interactions treated as implicit positives
- one negative per position during training
- sampled ranking evaluation with 100 negatives per user during validation/test for the current formal run

This is good enough for:
- getting the pipeline running end to end
- verifying GPU training works
- establishing a baseline before adding time-aware or metadata-aware variants

## Current Baseline Result

The repository has already been validated end to end on the full processed MobileRec dataset with the following command:




source /home/e0492463/cs5242-project-main/.venv/bin/activate && train-sasrec --data-dir data/processed/mobilerec --output-dir data/outputs/sasrec-baseline-formal --epochs 5 --batch-size 256 --max-len 50 --eval-negative-samples 100 --report-full-eval --seed 42

{                                                                               35 [01:23<00:02, 31.61it/s, loss=1.0099]
  "config": {
    "data_dir": "data/processed/mobilerec",
    "output_dir": "data/outputs/sasrec-baseline-formal",
    "batch_size": 256,
    "epochs": 5,
    "max_len": 50,
    "hidden_size": 128,
    "num_blocks": 2,
    "num_heads": 2,
    "dropout": 0.2,
    "lr": 0.001,
    "weight_decay": 1e-05,
    "eval_negative_samples": 100,
    "report_full_eval": true,
    "seed": 42
  },
  "evaluation_protocol": {
    "seed": 42,
    "validation": {
      "mode": "sampled",
      "sequence": "train_sequence",
      "target": "validation_target",
      "negative_samples": 100,
      "negative_pool_excludes": "items already present in the input sequence"
    },
    "test": {
      "mode": "sampled",
      "sequence": "validation_sequence",
      "target": "test_target",
      "negative_samples": 100,
      "negative_pool_excludes": "items already present in the input sequence"
    },
    "full_test": {
      "enabled": true,
      "mode": "full_ranking",
      "sequence": "validation_sequence",
      "target": "test_target",
      "ranking_pool": "all catalog items excluding padding and items already present in the input sequence"
    }
  },
  "best_val_hr_at_10": 0.23830078373286523,
  "test_hr_at_10": 0.29528603321473307,
  "test_ndcg_at_10": 0.16047780840613754,
  "device": "cuda",
  "history": [
    {
      "epoch": 1,
      "train_loss": 1.044216931737973,
      "val_hr_at_10": 0.2009524204019077,
      "val_ndcg_at_10": 0.10573001672577338
    },
    {
      "epoch": 2,
      "train_loss": 1.0169477131292632,
      "val_hr_at_10": 0.17653486375731847,
      "val_ndcg_at_10": 0.09501029382183758
    },
    {
      "epoch": 3,
      "train_loss": 1.0111083860807053,
      "val_hr_at_10": 0.19240520431760105,
      "val_ndcg_at_10": 0.10329748695242258
    },
    {
      "epoch": 4,
      "train_loss": 1.007399996586648,
      "val_hr_at_10": 0.2119235378389998,
      "val_ndcg_at_10": 0.11320676166117256
    },
    {
      "epoch": 5,
      "train_loss": 1.0048197463182034,
      "val_hr_at_10": 0.23830078373286523,
      "val_ndcg_at_10": 0.1289146752740715
    }
  ],
  "full_test_hr_at_10": 0.008450088628803146,
  "full_test_ndcg_at_10": 0.00384851951195927
}

Artifacts:
- `data/outputs/sasrec-baseline-full/best_model.pt`
- `data/outputs/sasrec-baseline-full/metrics.json`

Observed result on this VM:
- device: `cuda`
best val HR@10 = 0.2383
sampled test HR@10 = 0.2953
sampled test NDCG@10 = 0.1605
full test HR@10 = 0.00845
full test NDCG@10 = 0.00385


Operational note:
- this full run completed safely on a `n1-standard-4` VM with a `Tesla T4`
- memory stayed within a comfortable range during training, so this machine size is sufficient for the current baseline workflow

## Next Steps

After the baseline is stable, the most natural improvements are:
- run TiSASRec experiments with personalized time intervals
- compare all-review interactions vs filtered positive interactions such as `rating >= 4`
- add app metadata embeddings from category or text fields
- standardize experiment configs and logging
