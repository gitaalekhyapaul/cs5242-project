from __future__ import annotations

import argparse
import json
import math
import random
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle

from models.sasrec import SASRec
from models.tisasrec import TiSASRec, TiSASRecWithoutMetadata

supported_datasets = ['mobilerec', 'steamrec']
supported_models = ['sasrec', 'tisasrec', 'tisasrec_m']
negative_items_handling_modes = ['treat-as-positive', 'filter-negative', 'penalize-negative']
NUM_METADATA = 5
TIME_NORMALIZATION = "personalized"

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SASRec baseline on prepared MobileRec sequences.")
    parser.add_argument("--dataset", type=str, default="mobilerec")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))

    parser.add_argument("--model", type=str, default="sasrec")
    parser.add_argument("--negative-items-handling", type=str, default="treat-as-positive")
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs"))

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument('--time-span', default=256, type=int)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steplr_gamma", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--eval-negative-samples", type=int, default=100)
    parser.add_argument(
        "--report-full-eval",
        action="store_true",
        help="Also report full-ranking test metrics against the entire item set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    # usage: --inference-only true
    parser.add_argument('--inference-only', default=False, type=str2bool)
    # usage: --training-only true
    parser.add_argument('--training-only', default=False, type=str2bool)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pad_feature_sequence(
    sequence: list[object],
    max_len: int,
    feature_dim: int = 1,
) -> np.ndarray:
    padded = np.zeros((max_len, feature_dim), dtype=np.int64)
    trimmed = list(sequence)[-max_len:]
    start = max_len - len(trimmed)
    for offset, value in enumerate(trimmed):
        values = np.asarray(value, dtype=np.int64).reshape(-1)
        values = values[(values > 0) & (values <= feature_dim)]
        width = min(values.size, feature_dim)
        if width:
            padded[start + offset, :width] = values[:width]
    return padded


def pad_numeric_sequence(sequence: list[object], max_len: int) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float32)
    trimmed = values[-max_len:]
    padded = np.zeros(max_len, dtype=np.float32)
    if trimmed.size:
        padded[-len(trimmed) :] = trimmed
    return padded


def pad_sequence(sequence: list[int], max_len: int) -> np.ndarray:
    trimmed = sequence[-max_len:]
    padded = np.zeros(max_len, dtype=np.int64)
    if trimmed:
        padded[-len(trimmed) :] = trimmed
    return padded


def personalize_time_sequence(timestamps: list[object]) -> list[int]:
    values = [int(timestamp) for timestamp in timestamps]
    if not values:
        return []

    time_diffs: list[int] = []
    for current_time, next_time in zip(values, values[1:]):
        diff = next_time - current_time
        if diff < 0:
            raise ValueError("Timestamps must be sorted by user before time normalization.")
        if diff > 0:
            time_diffs.append(diff)

    time_scale = min(time_diffs) if time_diffs else 1
    time_min = min(values)
    return [
        int(round((timestamp - time_min) / time_scale) + 1)
        for timestamp in values
    ]


def generate_time_matrix(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            time_matrix[i][j] = min(abs(time_seq[i]-time_seq[j]), time_span)
    return time_matrix


def generate_time_matrix_batch(time_seq: torch.Tensor, time_span: int, device: torch.device) -> torch.Tensor:
    time_seq = time_seq.to(device=device, dtype=torch.long)
    return (time_seq.unsqueeze(2) - time_seq.unsqueeze(1)).abs().clamp(max=time_span)


def relation_matrix_cache_path(data_dir: Path, dataset: str, max_len: int, time_span: int) -> Path:
    return data_dir / f"relation_matrix_{dataset}_{max_len}_{time_span}_{TIME_NORMALIZATION}.pickle"


def generate_relation_matrix(seqs, max_len, time_span):
    relation_matrix = dict()
    for row in seqs.itertuples():
        print(f'{row.Index} / {len(seqs)}')
        timestamps_sequence = personalize_time_sequence(list(row.timestamps))
        time_seq = np.zeros([max_len], dtype=np.int32)
        idx = max_len - 1

        for ele in reversed(timestamps_sequence[:-1]):
            time_seq[idx] = ele
            idx -= 1
            if idx == -1: break
        relation_matrix[row.Index] = generate_time_matrix(time_seq, time_span)
    return relation_matrix


def generate_combined_metadata_seq(metadata_seqs):
    # filter empty metadata seqs
    filtered_metadata_seqs = [ele for ele in metadata_seqs if len(ele) > 0]
    return np.stack(filtered_metadata_seqs, axis=1)


class TrainDataset(Dataset):
    def __init__(
        self,
        *,
        sequences: pd.DataFrame,
        max_len: int,
        num_items: int,
        num_categories: int,
        seed: int,
        negative_items_handling: str,
        relation_matrix: np.ndarray,
    ) -> None:
        self.rows: list[dict[str, object]] = []
        self.max_len = max_len
        self.num_items = num_items
        self.num_categories = num_categories
        self.rng = random.Random(seed)
        self.relation_matrix = relation_matrix

        for row in sequences.itertuples(index=False):
            train_sequence = list(row.train_sequence)
            timestamps_sequence = personalize_time_sequence(list(row.timestamps))

            # numerical metadata sequences
            ratings_sequence = list(row.ratings)
            ratings_seq_padded = pad_numeric_sequence(ratings_sequence[:-1], self.max_len)
            review_upvotes_sequence = list(row.review_upvotes)
            review_upvotes_seq_padded = pad_numeric_sequence(review_upvotes_sequence[:-1], self.max_len)
            app_num_reviews_sequence = list(row.app_num_reviews)
            app_num_reviews_seq_padded = pad_numeric_sequence(app_num_reviews_sequence[:-1], self.max_len)
            app_avg_rating_sequence = list(row.app_avg_rating)
            app_avg_rating_seq_padded = pad_numeric_sequence(app_avg_rating_sequence[:-1], self.max_len)
            app_price_sequence = list(row.app_price)
            app_price_seq_padded = pad_numeric_sequence(app_price_sequence[:-1], self.max_len)

            # category metadata sequence
            app_category_sequence = list(row.app_category)
            category_seq = app_category_sequence[:-1]

            if negative_items_handling == 'filter-negative':
                train_sequence = [
                    t for t, r in zip(train_sequence, ratings_sequence) if r == 1
                ]

            if len(train_sequence) < 2:
                continue

            history = train_sequence[:-1]
            time_seq = timestamps_sequence[:-1]
            metadata_seq_padded = generate_combined_metadata_seq([
                ratings_seq_padded,
                review_upvotes_seq_padded,
                app_num_reviews_seq_padded,
                app_avg_rating_seq_padded,
                app_price_seq_padded,
            ])

            targets = train_sequence[1:]
            target_ratings = ratings_sequence[1:]

            positive_items = train_sequence

            if negative_items_handling == 'treat-as-positive':
                pass
            elif negative_items_handling == 'penalize-negative':
                targets = [t * r for t, r in zip(targets, target_ratings)]
                positive_items = [
                    t for t, r in zip(train_sequence, ratings_sequence) if r == 1
                ]

            self.rows.append(
                {
                    "history": history,
                    "targets": targets,
                    "seen": set(positive_items),
                    "time_seq": time_seq,
                    "metadata_seq_padded": metadata_seq_padded,
                    "category_seq": category_seq,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _sample_negative(self, seen: set[int]) -> int:
        negative = self.rng.randint(1, self.num_items)
        while negative in seen:
            negative = self.rng.randint(1, self.num_items)
        return negative

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]

        if self.relation_matrix:
            time_matrix = self.relation_matrix[index]
        else:
            time_matrix = np.array([])

        time_seq = pad_sequence(list(row["time_seq"]), self.max_len)
        metadata_seq = row["metadata_seq_padded"]
        category_seq = pad_feature_sequence(list(row["category_seq"]), self.max_len, self.num_categories)

        input_seq = pad_sequence(list(row["history"]), self.max_len)
        pos_seq = pad_sequence(list(row["targets"]), self.max_len)
        neg_trimmed = [self._sample_negative(row["seen"]) for _ in row["targets"]]
        neg_seq = pad_sequence(neg_trimmed, self.max_len)

        return {
            "input_ids": torch.from_numpy(input_seq),
            "pos_ids": torch.from_numpy(pos_seq),
            "neg_ids": torch.from_numpy(neg_seq),
            "time_matrix": torch.from_numpy(time_matrix),
            "time_seq": torch.from_numpy(time_seq),
            "metadata_seq": torch.from_numpy(metadata_seq),
            "category_seq": torch.from_numpy(category_seq),
        }


class EvalDataset(Dataset):
    def __init__(
        self,
        *,
        sequences: pd.DataFrame,
        sequence_column: str,
        target_column: str,
        num_items: int,
        num_categories: int,
        negative_samples: int,
        max_len: int,
        seed: int,
    ) -> None:
        self.rows: list[dict[str, object]] = []
        self.num_categories = num_categories
        self.max_len = max_len

        rng = random.Random(seed)

        for row in sequences.itertuples(index=False):
            sequence = list(getattr(row, sequence_column))
            timestamps_sequence = personalize_time_sequence(list(row.timestamps))

            # numerical metadata sequences
            ratings_sequence = list(row.ratings)
            ratings_seq_padded = pad_numeric_sequence(ratings_sequence[:-1], self.max_len)
            review_upvotes_sequence = list(row.review_upvotes)
            review_upvotes_seq_padded = pad_numeric_sequence(review_upvotes_sequence[:-1], self.max_len)
            app_num_reviews_sequence = list(row.app_num_reviews)
            app_num_reviews_seq_padded = pad_numeric_sequence(app_num_reviews_sequence[:-1], self.max_len)
            app_avg_rating_sequence = list(row.app_avg_rating)
            app_avg_rating_seq_padded = pad_numeric_sequence(app_avg_rating_sequence[:-1], self.max_len)
            app_price_sequence = list(row.app_price)
            app_price_seq_padded = pad_numeric_sequence(app_price_sequence[:-1], self.max_len)

            # category metadata sequence
            app_category_sequence = list(row.app_category)
            category_seq = app_category_sequence[:-1]

            time_seq = timestamps_sequence[:-1]
            metadata_seq_padded = generate_combined_metadata_seq([
                ratings_seq_padded,
                review_upvotes_seq_padded,
                app_num_reviews_seq_padded,
                app_avg_rating_seq_padded,
                app_price_seq_padded,
            ])

            target = int(getattr(row, target_column))

            seen = set(sequence)

            # negatives: list[int] = []
            negatives: set[int] = set()

            while len(negatives) < negative_samples:
                sampled = rng.randint(1, num_items)
                if sampled not in seen and sampled not in negatives:
                    negatives.add(sampled)
            candidates = [target] + list(negatives)
            self.rows.append(
                {
                    "input_ids": pad_sequence(sequence, max_len),
                    "candidate_ids": np.asarray(candidates, dtype=np.int64),
                    "target": 0,
                    "time_seq": time_seq,
                    "metadata_seq_padded": metadata_seq_padded,
                    "category_seq": category_seq,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]

        time_seq = pad_sequence(list(row["time_seq"]), self.max_len)
        metadata_seq = row["metadata_seq_padded"]
        category_seq = pad_feature_sequence(list(row["category_seq"]), self.max_len, self.num_categories)

        return {
            "input_ids": torch.from_numpy(row["input_ids"]),
            "candidate_ids": torch.from_numpy(row["candidate_ids"]),
            "target": torch.tensor(row["target"], dtype=torch.long),
            "time_seq": torch.from_numpy(time_seq),
            "metadata_seq": torch.from_numpy(metadata_seq),
            "category_seq": torch.from_numpy(category_seq),
        }


class FullEvalDataset(Dataset):
    def __init__(
        self,
        *,
        sequences: pd.DataFrame,
        sequence_column: str,
        target_column: str,
        max_len: int,
        num_categories: int,
    ) -> None:
        self.rows: list[dict[str, object]] = []
        self.max_len = max_len
        self.num_categories = num_categories

        for row in sequences.itertuples(index=False):
            sequence = list(getattr(row, sequence_column))
            timestamps_sequence = personalize_time_sequence(list(row.timestamps))

            # numerical metadata sequences
            ratings_sequence = list(row.ratings)
            ratings_seq_padded = pad_numeric_sequence(ratings_sequence[:-1], self.max_len)
            review_upvotes_sequence = list(row.review_upvotes)
            review_upvotes_seq_padded = pad_numeric_sequence(review_upvotes_sequence[:-1], self.max_len)
            app_num_reviews_sequence = list(row.app_num_reviews)
            app_num_reviews_seq_padded = pad_numeric_sequence(app_num_reviews_sequence[:-1], self.max_len)
            app_avg_rating_sequence = list(row.app_avg_rating)
            app_avg_rating_seq_padded = pad_numeric_sequence(app_avg_rating_sequence[:-1], self.max_len)
            app_price_sequence = list(row.app_price)
            app_price_seq_padded = pad_numeric_sequence(app_price_sequence[:-1], self.max_len)

            # category metadata sequence
            app_category_sequence = list(row.app_category)
            category_seq = app_category_sequence[:-1]

            time_seq = timestamps_sequence[:-1]
            metadata_seq_padded = generate_combined_metadata_seq([
                ratings_seq_padded,
                review_upvotes_seq_padded,
                app_num_reviews_seq_padded,
                app_avg_rating_seq_padded,
                app_price_seq_padded,
            ])

            target = int(getattr(row, target_column))
            self.rows.append(
                {
                    "input_ids": pad_sequence(sequence, max_len),
                    "seen_ids": np.asarray(sorted(set(sequence)), dtype=np.int64),
                    "target": target,
                    "time_seq": time_seq,
                    "metadata_seq_padded": metadata_seq_padded,
                    "category_seq": category_seq,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]

        time_seq = pad_sequence(list(row["time_seq"]), self.max_len)
        metadata_seq = row["metadata_seq_padded"]
        category_seq = pad_feature_sequence(list(row["category_seq"]), self.max_len, self.num_categories)

        return {
            "input_ids": torch.from_numpy(row["input_ids"]),
            "seen_ids": torch.from_numpy(row["seen_ids"]),
            "target": torch.tensor(row["target"], dtype=torch.long),
            "time_seq": torch.from_numpy(time_seq),
            "metadata_seq": torch.from_numpy(metadata_seq),
            "category_seq": torch.from_numpy(category_seq),
        }


def collate_full_eval_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, object]:
    return {
        "input_ids": torch.stack([row["input_ids"] for row in batch], dim=0),
        "seen_ids": [row["seen_ids"] for row in batch],
        "target": torch.stack([row["target"] for row in batch], dim=0),
        "time_seq": torch.stack([row["time_seq"] for row in batch], dim=0),
        "metadata_seq": torch.stack([row["metadata_seq"] for row in batch], dim=0),
        "category_seq": torch.stack([row["category_seq"] for row in batch], dim=0),
    }


@dataclass
class Metrics:
    hr_at_10: float
    ndcg_at_10: float


def evaluate(
    model: SASRec | TiSASRec | TiSASRecWithoutMetadata,
    data_loader: DataLoader,
    device: torch.device,
    split_name: str,
    time_span: int,
) -> Metrics:

    model.eval()
    hit_scores: list[float] = []
    ndcg_scores: list[float] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"eval:{split_name}", leave=False):
            input_ids = batch["input_ids"].to(device)
            candidate_ids = batch["candidate_ids"].to(device)
            time_matrix = generate_time_matrix_batch(batch["time_seq"], time_span, device)
            metadata_seq = batch["metadata_seq"].to(device=device, dtype=torch.float32)
            category_seq = batch["category_seq"].to(device=device, dtype=torch.long)
            scores = model.score_candidates(
                input_ids=input_ids,
                candidate_ids=candidate_ids,
                time_matrix=time_matrix,
                metadata_seq=metadata_seq,
                category_seq=category_seq,
            )

            rankings = torch.argsort(scores, dim=1, descending=True)
            top_k = rankings[:, :10]
            hits = top_k.eq(0)
            hit_scores.extend(hits.any(dim=1).float().cpu().tolist())
            for row_hits in hits.cpu().tolist():
                if True in row_hits:
                    rank = row_hits.index(True) + 1
                    ndcg_scores.append(1.0 / math.log2(rank + 1))
                else:
                    ndcg_scores.append(0.0)

    return Metrics(
        hr_at_10=float(np.mean(hit_scores)),
        ndcg_at_10=float(np.mean(ndcg_scores)),
    )


def evaluate_full_ranking(
    model: SASRec | TiSASRec | TiSASRecWithoutMetadata,
    data_loader: DataLoader,
    device: torch.device,
    split_name: str,
    time_span: int,
) -> Metrics:
    model.eval()
    hit_scores: list[float] = []
    ndcg_scores: list[float] = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"eval:{split_name}:full", leave=False):
            input_ids = batch["input_ids"].to(device)
            seen_ids = batch["seen_ids"]
            targets = batch["target"].to(device)

            time_matrix = generate_time_matrix_batch(batch["time_seq"], time_span, device)
            metadata_seq = batch["metadata_seq"].to(device=device, dtype=torch.float32)
            category_seq = batch["category_seq"].to(device=device, dtype=torch.long)
            scores = model.score_all_items(
                input_ids=input_ids,
                metadata_seq=metadata_seq,
                category_seq=category_seq,
                time_matrix=time_matrix,
            )
            scores[:, 0] = float("-inf")
            for row_idx in range(scores.size(0)):
                row_seen = seen_ids[row_idx]
                original_target_score = scores[row_idx, targets[row_idx]].clone()
                if row_seen.numel() > 0:
                    scores[row_idx, row_seen.to(device)] = float("-inf")
                scores[row_idx, targets[row_idx]] = original_target_score

            target_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
            higher_scores = scores.gt(target_scores.unsqueeze(1)).sum(dim=1)
            ranks = higher_scores + 1

            hit_scores.extend(ranks.le(10).float().cpu().tolist())
            for rank in ranks.cpu().tolist():
                if rank <= 10:
                    ndcg_scores.append(1.0 / math.log2(rank + 1))
                else:
                    ndcg_scores.append(0.0)

    return Metrics(
        hr_at_10=float(np.mean(hit_scores)),
        ndcg_at_10=float(np.mean(ndcg_scores)),
    )


def main() -> None:
    args = parse_args()

    if args.dataset not in supported_datasets:
        raise ValueError(
            f"Invalid dataset argument: '{args.dataset}'. "
            f"Supported datasets are: {supported_datasets}."
        )

    if args.negative_items_handling not in negative_items_handling_modes:
        raise ValueError(
            f"Invalid negative item handling mode: '{args.negative_items_handling}'. "
            f"Supported modes are: {negative_items_handling_modes}."
        )

    set_seed(args.seed)
    (args.output_dir / args.model / args.negative_items_handling).mkdir(parents=True, exist_ok=True)

    sequences = pd.read_parquet(args.data_dir / args.dataset / "final_sequences.parquet")
    item_mapping = pd.read_parquet(args.data_dir / args.dataset / "final_item_mapping.parquet")
    category_mapping = pd.read_parquet(args.data_dir / args.dataset / "final_app_category.parquet")

    num_items = int(item_mapping["item_id"].max())
    num_categories = int(category_mapping["app_category_id"].max())
    # num_users = len(sequences)

    relation_matrix = None

    if args.model in ['tisasrec_m', 'tisasrec']:
        relation_matrix_path = relation_matrix_cache_path(
            args.data_dir,
            args.dataset,
            args.max_len,
            args.time_span,
        )
        try:
            with relation_matrix_path.open("rb") as fh:
                relation_matrix = pickle.load(fh)
        except:
            relation_matrix = generate_relation_matrix(sequences, args.max_len, args.time_span)
            relation_matrix_path.parent.mkdir(parents=True, exist_ok=True)
            with relation_matrix_path.open("wb") as fh:
                pickle.dump(relation_matrix, fh)

    print(f'relation matrix enabled: {bool(relation_matrix)}')

    print('Preparing train dataset')
    train_dataset = TrainDataset(
        sequences=sequences,
        max_len=args.max_len,
        num_items=num_items,
        num_categories=num_categories,
        seed=args.seed,
        negative_items_handling=args.negative_items_handling,
        relation_matrix=relation_matrix,
    )
    print('Prepared train dataset')

    print('Preparing val dataset')
    val_dataset = EvalDataset(
        sequences=sequences,
        sequence_column="train_sequence",
        target_column="validation_target",
        num_items=num_items,
        num_categories=num_categories,
        negative_samples=args.eval_negative_samples,
        max_len=args.max_len,
        seed=args.seed + 1,
    )
    print('Prepared val dataset')

    if not args.training_only:
        print('Preparing test dataset')
        test_dataset = EvalDataset(
            sequences=sequences,
            sequence_column="validation_sequence",
            target_column="test_target",
            num_items=num_items,
            num_categories=num_categories,
            negative_samples=args.eval_negative_samples,
            max_len=args.max_len,
            seed=args.seed + 2,
        )
        print('Prepared test dataset')

        print('Preparing full test dataset')
        full_test_dataset = FullEvalDataset(
            sequences=sequences,
            sequence_column="validation_sequence",
            target_column="test_target",
            max_len=args.max_len,
            num_categories=num_categories
        )
        print('Prepared full test dataset')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if not args.training_only:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        full_test_loader = DataLoader(
            full_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_full_eval_batch,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = None
    if args.model == 'sasrec':
        model = SASRec(
            num_items=num_items,
            max_len=args.max_len,
            hidden_size=args.hidden_size,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout=args.dropout,
        ).to(device)
    elif args.model == 'tisasrec_m':
        model = TiSASRec(
            num_items=num_items,
            num_categories=num_categories,
            num_metadata=NUM_METADATA,
            max_len=args.max_len,
            time_span=args.time_span,
            hidden_size=args.hidden_size,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device=device,
        ).to(device)
    elif args.model == 'tisasrec':
        model = TiSASRecWithoutMetadata(
            num_items=num_items,
            num_categories=num_categories,
            num_metadata=NUM_METADATA,
            max_len=args.max_len,
            time_span=args.time_span,
            hidden_size=args.hidden_size,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout=args.dropout,
            device=device,
        ).to(device)

    if not model:
        raise ValueError(
            f"Invalid model argument: '{args.model}'. "
            f"Supported models are: {supported_models}."
        )

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass # just ignore those failed init layers

    history_filepath = args.output_dir / args.model / args.negative_items_handling / "history.csv"
    history_headers = ['training_loss', 'hr@10', 'ndcg@10', 'lr', 'best_val_hr']

    # Create file and write header if it doesn't exist
    if not history_filepath.exists():
        with open(history_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(history_headers)

    best_val_hr = -1.0
    optimizer_lr = args.lr

    if history_filepath.exists():
        df = pd.read_csv(history_filepath)
        if not df.empty:
            # Get the 'best_val_hr' and 'lr' values from the very last row
            best_val_hr = df['best_val_hr'].iloc[-1]
            optimizer_lr = df['lr'].iloc[-1]
            print(f"Resuming from checkpoint: lr: {optimizer_lr}, best_val_hr: {best_val_hr}")

    best_checkpoint = args.output_dir / args.model / args.negative_items_handling / "best_model.pt"
    current_checkpoint = args.output_dir / args.model / args.negative_items_handling / "current_model.pt"

    if current_checkpoint.exists():
        try:
            model.load_state_dict(torch.load(current_checkpoint, map_location=device))
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(current_checkpoint)
    elif best_checkpoint.exists():
        try:
            model.load_state_dict(torch.load(best_checkpoint, map_location=device))
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(best_checkpoint)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_lr,
        betas=(0.9, 0.98), #* forget the past more quickly than default (0.9, 0.999)
        weight_decay=args.weight_decay,
        # eps=1e-9, #uncomment if exploding loss/early plateau
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.steplr_gamma)

    bce = nn.BCEWithLogitsLoss(reduction="none")

    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        if args.inference_only: break # skip training if in inference mode

        model.train()
        epoch_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"train:{epoch}", leave=False)
        for batch in progress:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            neg_ids = batch["neg_ids"].to(device)
            time_matrix = batch["time_matrix"].to(device=device, dtype=torch.long)
            if time_matrix.numel() == 0:
                time_matrix = generate_time_matrix_batch(batch["time_seq"], args.time_span, device)
            pos_logits, neg_logits = model.training_logits(
                input_ids=input_ids,
                pos_ids=torch.abs(pos_ids),
                neg_ids=neg_ids,
                time_matrix=time_matrix,
                metadata_seq=batch["metadata_seq"].to(device=device, dtype=torch.float32),
                category_seq=batch["category_seq"].to(device=device, dtype=torch.long),
            )

            negative_item_mask = pos_ids.lt(0)
            mask = pos_ids.ne(0)
            valid_positive_mask = mask & (~negative_item_mask)

            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)

            #* loss computation
            pos_loss = bce(pos_logits, pos_labels)
            neg_loss = bce(neg_logits, neg_labels)
            explicit_neg_loss = bce(pos_logits, neg_labels)

            loss = (
                (pos_loss * valid_positive_mask) +
                (neg_loss * mask) +
                # for negative marked items, apply explicit neg_loss
                (explicit_neg_loss * (mask & negative_item_mask))
            ).sum() / mask.sum().clamp(min=1)

            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            epoch_losses.append(loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}")

        scheduler.step()

        #* evaluate every epoch
        val_metrics = evaluate(model, val_loader, device, "val", time_span=args.time_span)

        train_loss = float(np.mean(epoch_losses))
        val_hr = val_metrics.hr_at_10
        val_ndcg = val_metrics.ndcg_at_10
        current_lr = scheduler.get_last_lr()[0]

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_hr_at_10": val_hr,
            "val_ndcg_at_10": val_ndcg,
        }

        with open(history_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                train_loss,
                val_hr,
                val_ndcg,
                current_lr,
                best_val_hr,
            ])

        history.append(epoch_record)
        if val_metrics.hr_at_10 > best_val_hr:
            best_val_hr = val_metrics.hr_at_10
            torch.save(model.state_dict(), best_checkpoint)

        torch.save(model.state_dict(), current_checkpoint)

    if args.training_only:
        return

    print('running test evaluation on best model')
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))

    test_metrics = evaluate(model, test_loader, device, "test", time_span=args.time_span)
    full_test_metrics = None
    if args.report_full_eval:
        full_test_metrics = evaluate_full_ranking(model, full_test_loader, device, "test", time_span=args.time_span)

    metrics_filepath = args.output_dir / args.model / args.negative_items_handling / "metrics.json"
    config = vars(args).copy()
    config["time_normalization"] = TIME_NORMALIZATION

    metrics = {
        "config": config,
        "evaluation_protocol": {
            "seed": args.seed,
            "validation": {
                "mode": "sampled",
                "sequence": "train_sequence",
                "target": "validation_target",
                "negative_samples": args.eval_negative_samples,
                "negative_pool_excludes": "items already present in the input sequence",
            },
            "test": {
                "mode": "sampled",
                "sequence": "validation_sequence",
                "target": "test_target",
                "negative_samples": args.eval_negative_samples,
                "negative_pool_excludes": "items already present in the input sequence",
            },
            "full_test": {
                "enabled": args.report_full_eval,
                "mode": "full_ranking",
                "sequence": "validation_sequence",
                "target": "test_target",
                "ranking_pool": "all catalog items excluding padding and items already present in the input sequence",
            },
        },
        "best_val_hr_at_10": best_val_hr,
        "test_hr_at_10": test_metrics.hr_at_10,
        "test_ndcg_at_10": test_metrics.ndcg_at_10,
        "device": str(device),
        "history": history,
    }
    if full_test_metrics is not None:
        metrics["full_test_hr_at_10"] = full_test_metrics.hr_at_10
        metrics["full_test_ndcg_at_10"] = full_test_metrics.ndcg_at_10
    metrics_filepath.write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
