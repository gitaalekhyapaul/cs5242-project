# CS5242 Sequential Recommendation Project

This repo studies next-item recommendation from timestamped review histories.
It uses MobileRec for app recommendation and SteamRec for game recommendation.
The goal is to compare order-only self-attention, time-aware self-attention,
metadata-aware self-attention, and MobileRec-to-SteamRec transfer.

## Project Motivation

App and game catalogs are large. A user often has a short review history.
Can chronological review sequences predict the next app or game? This project
tests that question with two related domains and one shared training contract.

The root project owns model training, MobileRec preparation, transfer
fine-tuning, recommendation demos, and experiment reporting. The Steam data
crawler has its own runbook in
[`steam-crawler/README.md`](./steam-crawler/README.md).

## Published Datasets

- MobileRec processed dataset:
  [gitaalekhyapaul/cs5242-mobilerec-dataset](https://www.kaggle.com/datasets/gitaalekhyapaul/cs5242-mobilerec-dataset)
- SteamRec processed dataset:
  [gitaalekhyapaul/steam-cs5242-dataset](https://www.kaggle.com/datasets/gitaalekhyapaul/steam-cs5242-dataset)

## Repository Layout

```text
.
|-- .env.example
|-- pyproject.toml
|-- raw/                         # local MobileRec raw CSV staging
|-- processed/                   # local MobileRec processed staging
|-- main/
|   |-- data/
|   |   |-- mobilerec/           # trainer-ready MobileRec final_* files
|   |   |-- steamrec/            # trainer-ready SteamRec final_* files
|   |   `-- outputs/             # model checkpoints and metrics
|   |-- models/
|   |   |-- prepare_mobilerec.py
|   |   |-- upload_processed_to_kaggle.py
|   |   |-- sasrec.py
|   |   `-- tisasrec.py
|   |-- train_model.py
|   |-- finetune_tisasrec_m_transfer.py
|   |-- recommend.py
|   |-- mobilerec_eda_etl.ipynb
|   |-- eda.ipynb
|   `-- experiments/
|       |-- report.ipynb
|       |-- experiment_summary.csv
|       `-- experiment_summary.md
|-- report.ipynb
`-- steam-crawler/
```

## Methodology

The project treats app and game reviews as implicit interaction signals. Each
user history is sorted by time, then converted into a sequence of item ids.
The split is chronological:

- `train_sequence`: all items except the last two
- `validation_sequence`: all items except the last item
- `test_sequence`: the full sequence
- `validation_target`: the second-last item
- `test_target`: the last item

Training uses pointwise binary classification. The true next item is the
positive target. A sampled unseen item is the negative target. The trainer
supports three negative-label modes:

- `treat-as-positive`
- `filter-negative`
- `penalize-negative`

Evaluation reports `HR@10` and `NDCG@10`. Validation and sampled test ranking
use `100` sampled negatives by default. `--report-full-eval` adds full-catalog
test ranking against every item except padding and seen input items.

The model set is:

- `sasrec`: order-only self-attention with item and position embeddings.
- `tisasrec`: time-aware self-attention with personalized time-interval
  matrices.
- `tisasrec_m`: TiSASRec with numeric metadata and category or genre bags.

TiSASRec time ids are personalized per user. For each sequence, the trainer
finds the smallest nonzero adjacent timestamp gap, then maps each timestamp to:

```text
round((timestamp - user_min_timestamp) / user_min_nonzero_gap) + 1
```

The relation matrix cache uses:

```text
relation_matrix_<dataset>_<max_len>_<time_span>_personalized.pickle
```

## Data Contract

`main/train_model.py` reads one dataset folder at a time from
`main/data/<dataset>/`. The active datasets are `mobilerec` and `steamrec`.
Each dataset folder must contain:

- `final_sequences.parquet`
- `final_item_mapping.parquet`
- `final_app_category.parquet`

`final_sequences.parquet` uses this shared schema:

- `user_id`
- `sequence_length`
- `train_sequence`
- `validation_sequence`
- `test_sequence`
- `validation_target`
- `test_target`
- `timestamps`
- `ratings`
- `review_upvotes`
- `app_category`
- `app_num_reviews`
- `app_avg_rating`
- `app_price`

`main/data/mobilerec/final_sequences.parquet` currently has `700111` rows.
`main/data/steamrec/final_sequences.parquet` currently has `118048` rows.

The MobileRec final aliases come from `main/mobilerec_eda_etl.ipynb`.
The SteamRec final aliases come from
`steam-crawler/notebooks/eda_etl.ipynb`. The trainer-ready copies live under
`main/data/mobilerec/` and `main/data/steamrec/`.

## Setup

Use Python `3.10`. The root environment uses `uv` and the dependencies in
`pyproject.toml`.

```bash
cd "/Users/gitaalekhyapaul/Documents/[Local] CS5242/cs5242-project"
uv venv --python 3.10 .venv
source .venv/bin/activate
UV_CACHE_DIR=.uv-cache uv sync
```

Copy the root env template before Kaggle publication:

```bash
cp .env.example .env
```

The root `.env` supports:

- `KAGGLE_USERNAME`
- `KAGGLE_API_TOKEN`

## Prepare MobileRec

Download raw MobileRec CSV files with the Hugging Face CLI:

```bash
hf download recmeapp/mobilerec interactions/mobilerec_final.csv app_meta/app_meta.csv \
  --repo-type dataset \
  --local-dir raw/mobilerec
```

Build the base parquet artifacts:

```bash
python main/models/prepare_mobilerec.py \
  --raw-dir raw/mobilerec \
  --processed-dir processed \
  --memory-limit 2GB \
  --threads 4
```

The script writes:

- `processed/interactions.parquet`
- `processed/sequences.parquet`
- `processed/user_mapping.parquet`
- `processed/item_mapping.parquet`
- `processed/app_metadata.parquet`
- `processed/summary.json`

Use `main/mobilerec_eda_etl.ipynb` to attach metadata arrays, build enriched
MobileRec artifacts, and publish the Kaggle-facing final aliases.

For a smaller local preparation run:

```bash
python main/models/prepare_mobilerec.py \
  --raw-dir raw/mobilerec \
  --processed-dir processed-sample \
  --sample-users 5000
```

## Publish Processed MobileRec

The terminal uploader stages every non-hidden file in a directory and pushes a
Kaggle dataset version:

```bash
python main/models/upload_processed_to_kaggle.py \
  --input-dir processed \
  --dataset-handle <kaggle-username>/cs5242-mobilerec-dataset
```

Useful flags:

- `--env-file /path/to/.env`
- `--kaggle-username <name>`
- `--kaggle-api-token <token>`
- `--version-notes "..."`

The notebook publish cell is the preferred path for the public MobileRec
snapshot, since it stages the enriched files and `final_*` aliases together.

## Train Models

Run commands from the repo root with the root `.venv` active.

SASRec on MobileRec:

```bash
python main/train_model.py \
  --dataset mobilerec \
  --data-dir main/data \
  --model sasrec \
  --negative-items-handling treat-as-positive \
  --output-dir main/data/outputs \
  --epochs 100 \
  --batch-size 128 \
  --eval-negative-samples 100 \
  --report-full-eval
```

TiSASRec-M on MobileRec:

```bash
python main/train_model.py \
  --dataset mobilerec \
  --data-dir main/data \
  --model tisasrec_m \
  --negative-items-handling penalize-negative \
  --output-dir main/data/outputs \
  --epochs 100 \
  --batch-size 128 \
  --eval-negative-samples 100 \
  --report-full-eval
```

TiSASRec-M on SteamRec:

```bash
python main/train_model.py \
  --dataset steamrec \
  --data-dir main/data \
  --model tisasrec_m \
  --negative-items-handling penalize-negative \
  --output-dir main/data/outputs \
  --epochs 100 \
  --batch-size 128 \
  --eval-negative-samples 100 \
  --report-full-eval
```

`main/train_model.py` writes artifacts under:

```text
main/data/outputs/<model>/<negative-items-handling>/
```

The artifact names include the dataset:

- `history_<dataset>.csv`
- `current_model_<dataset>.pt`
- `best_model_<dataset>.pt`
- `metrics_<dataset>.json`

## Transfer Fine-Tuning

`main/finetune_tisasrec_m_transfer.py` fine-tunes a MobileRec TiSASRec-M
checkpoint on SteamRec. It loads compatible same-shape weights, resets the
SteamRec item and genre embedding tables, and trains only:

- `item_emb.weight`
- `metadata_cat_emb.weight`

Run:

```bash
python main/finetune_tisasrec_m_transfer.py \
  --source-checkpoint main/data/outputs/tisasrec_m/penalize-negative/best_model_mobilerec.pt \
  --epochs 60 \
  --batch-size 128 \
  --report-full-eval
```

The default output folder is:

```text
main/data/outputs/steamrec_transfer/tisasrec_m/penalize-negative/embedding_finetune/
```

Transfer artifacts:

- `history.csv`
- `current_model.pt`
- `best_model.pt`
- `metrics.json`
- `transfer_load_report.json`

## Recommendation Demo

The recommendation demo reads packaged metadata and model checkpoints from
`main/metadata/`.

```bash
cd main
python recommend.py
```

Use the fine-tuned SteamRec checkpoint:

```bash
cd main
python recommend.py --use-finetuned-model
```

## Experiment Reporting

`main/experiments/report.ipynb` scans experiment folders that contain
`history.csv` and `metrics.json`. It writes:

- `main/experiments/experiment_summary.csv`
- `main/experiments/experiment_summary.md`
- one training-curve PNG per experiment

Current summary highlights:

| experiment | dataset | model | negative mode | best HR@10 | best NDCG@10 | full HR@10 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `tisasrec_m_pn_mobilerec_100` | mobilerec | tisasrec_m | penalize-negative | 0.21818 | 0.10385 | 0.00400 |
| `tisasrec_m_ft_steamrec_60` | steamrec | tisasrec_m | penalize-negative | 0.14265 | 0.06615 | 0.00091 |
| `sasrec_tap_mobilerec_100` | mobilerec | sasrec | treat-as-positive | 0.14146 | 0.08500 | 0.00663 |
| `tisasrec_m_pn_steamrec_100` | steamrec | tisasrec_m | penalize-negative | 0.11581 | 0.05496 | 0.00202 |

The full table lives in
[`main/experiments/experiment_summary.md`](./main/experiments/experiment_summary.md).

## SteamRec Crawler

SteamRec data collection and Steam-specific ETL live under
[`steam-crawler/`](./steam-crawler/). That README covers:

- Steam API setup
- smoke and full crawler runs
- Stage 4a genre and price patching
- Stage 5a review transforms
- SteamRec `final_*` aliases
- SLURM execution

Keep this root README and `steam-crawler/README.md` in the same change set as
changes to data schemas, CLI flags, model entrypoints, or output artifacts.
