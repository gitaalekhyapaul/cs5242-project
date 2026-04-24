from __future__ import annotations

import argparse
import json
import math
import random
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
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--eval-negative-samples", type=int, default=100)
    parser.add_argument(
        "--report-full-eval",
        action="store_true",
        help="Also report full-ranking test metrics against the entire item set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--inference-only', default=False, type=str2bool)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_sequence(sequence: list[int], max_len: int) -> np.ndarray:
    trimmed = sequence[-max_len:]
    padded = np.zeros(max_len, dtype=np.int64)
    if trimmed:
        padded[-len(trimmed) :] = trimmed
    return padded


def pad_feature_sequence(
    sequence: list[object],
    max_len: int,
    feature_dim: int = 1,
) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.int64)
    if values.size == 0:
        return np.zeros((max_len, feature_dim), dtype=np.int64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)

    trimmed = values[-max_len:, :feature_dim]
    padded = np.zeros((max_len, feature_dim), dtype=np.int64)
    if len(trimmed):
        padded[-len(trimmed) :, : trimmed.shape[1]] = trimmed
    return padded


def generate_time_matrix(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            time_matrix[i][j] = min(abs(time_seq[i]-time_seq[j]), time_span)
    return time_matrix


def generate_relation_matrix(seqs, max_len, time_span):
    relation_matrix = dict()
    for row in seqs.itertuples():
        print(f'{row.Index} / {len(seqs)}')
        timestamps_sequence = list(row.timestamps)
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
        sequences: pd.DataFrame,
        max_len: int,
        num_items: int,
        seed: int,
        negative_items_handling: str,
        relation_matrix: np.ndarray,
    ) -> None:
        self.rows: list[dict[str, object]] = []
        self.max_len = max_len
        self.num_items = num_items
        self.rng = random.Random(seed)
        self.relation_matrix = relation_matrix

        for row in sequences.itertuples(index=False):
            train_sequence = list(row.train_sequence)
            timestamps_sequence = list(row.timestamps)

            # numerical metadata sequences
            ratings_sequence = list(row.ratings)
            ratings_seq_padded = pad_sequence(ratings_sequence[:-1], self.max_len)
            review_upvotes_sequence = list(row.review_upvotes)
            review_upvotes_seq_padded = pad_sequence(review_upvotes_sequence[:-1], self.max_len)
            app_num_reviews_sequence = list(row.app_num_reviews)
            app_num_reviews_seq_padded = pad_sequence(app_num_reviews_sequence[:-1], self.max_len)
            app_avg_rating_sequence = list(row.app_avg_rating)
            app_avg_rating_seq_padded = pad_sequence(app_avg_rating_sequence[:-1], self.max_len)
            app_price_sequence = list(row.app_price)
            app_price_seq_padded = pad_sequence(app_price_sequence[:-1], self.max_len)

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
        category_seq = pad_feature_sequence(list(row["category_seq"]), self.max_len)

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
        sequences: pd.DataFrame,
        sequence_column: str,
        target_column: str,
        num_items: int,
        negative_samples: int,
        max_len: int,
        seed: int,
    ) -> None:
        self.rows: list[dict[str, object]] = []
        rng = random.Random(seed)
        for row in sequences.itertuples(index=False):
            sequence = list(getattr(row, sequence_column))
            target = int(getattr(row, target_column))
            targets = sequence[1:]

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
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "input_ids": torch.from_numpy(row["input_ids"]),
            "candidate_ids": torch.from_numpy(row["candidate_ids"]),
            "target": torch.tensor(row["target"], dtype=torch.long),
        }


class FullEvalDataset(Dataset):
    def __init__(
        self,
        sequences: pd.DataFrame,
        sequence_column: str,
        target_column: str,
        max_len: int,
    ) -> None:
        self.rows: list[dict[str, object]] = []
        for row in sequences.itertuples(index=False):
            sequence = list(getattr(row, sequence_column))
            target = int(getattr(row, target_column))
            self.rows.append(
                {
                    "input_ids": pad_sequence(sequence, max_len),
                    "seen_ids": np.asarray(sorted(set(sequence)), dtype=np.int64),
                    "target": target,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "input_ids": torch.from_numpy(row["input_ids"]),
            "seen_ids": torch.from_numpy(row["seen_ids"]),
            "target": torch.tensor(row["target"], dtype=torch.long),
        }


def collate_full_eval_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, object]:
    return {
        "input_ids": torch.stack([row["input_ids"] for row in batch], dim=0),
        "seen_ids": [row["seen_ids"] for row in batch],
        "target": torch.stack([row["target"] for row in batch], dim=0),
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

            # todo: add time_seq, metadata_seq, category_seq to eval and full eval datasets
            time_seq = batch["time_seq"].to(device)
            metadata_seq = batch["metadata_seq"].to(device)
            category_seq = batch["category_seq"].to(device)

            time_matrix = generate_time_matrix(time_seq, time_span)
            candidate_ids = batch["candidate_ids"].to(device)
            scores = model.score_candidates(input_ids=input_ids, candidate_ids=candidate_ids)

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

            # todo: add time_seq, metadata_seq, category_seq to eval and full eval datasets
            time_seq = batch["time_seq"].to(device)
            metadata_seq = batch["metadata_seq"].to(device)
            category_seq = batch["category_seq"].to(device)

            time_matrix = generate_time_matrix(time_seq, time_span)

            scores = model.score_all_items(input_ids=input_ids)
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
        try:
            relation_matrix = pickle.load(open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.max_len, args.time_span),'rb'))
        except:
            relation_matrix = generate_relation_matrix(sequences, args.max_len, args.time_span)
            pickle.dump(relation_matrix, open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.max_len, args.time_span),'wb'))

    print(f'relation matrix enabled: {bool(relation_matrix)}')

    print('Preparing train dataset')
    train_dataset = TrainDataset(
        sequences=sequences,
        max_len=args.max_len,
        num_items=num_items,
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
        negative_samples=args.eval_negative_samples,
        max_len=args.max_len,
        seed=args.seed + 1,
    )
    print('Prepared val dataset')

    print('Preparing test dataset')
    test_dataset = EvalDataset(
        sequences=sequences,
        sequence_column="validation_sequence",
        target_column="test_target",
        num_items=num_items,
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
    )
    print('Prepared full test dataset')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    full_test_loader = DataLoader(
        full_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_full_eval_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            num_metadata=args.num_metadata,
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

    best_val_hr = -1.0
    best_checkpoint = args.output_dir / args.model / args.negative_items_handling / "best_model.pt"

    if best_checkpoint.exists():
        try:
            model.load_state_dict(torch.load(best_checkpoint, map_location=device))
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(best_checkpoint)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98), #* forget the past more quickly than default (0.9, 0.999)
        weight_decay=args.weight_decay,
        # eps=1e-9, #uncomment if exploding loss/early plateau
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    #* if using standard loss computation
    bce = nn.BCEWithLogitsLoss(reduction="none")

    #* if using alternative loss computation
    # bce = nn.BCEWithLogitsLoss(reduction="mean")

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
            pos_logits, neg_logits = model.training_logits(
                input_ids=input_ids,
                pos_ids=torch.abs(pos_ids),
                neg_ids=neg_ids,
            )

            negative_item_mask = pos_ids < 0
            pos_labels_pos = torch.ones_like(pos_logits[~negative_item_mask])
            pos_labels_neg = torch.zeros_like(pos_logits[negative_item_mask])
            neg_labels = torch.zeros_like(neg_logits)

            #* loss computation
            mask = pos_ids.ne(0)
            pos_loss = bce(pos_logits[~negative_item_mask], pos_labels_pos)
            neg_loss = bce(neg_logits, neg_labels)
            neg_loss += bce(neg_logits[negative_item_mask], pos_labels_neg)
            loss = ((pos_loss + neg_loss) * mask).sum() / mask.sum().clamp(min=1)

            #* alternative loss computation
            # mask = pos_ids != 0 # mask padding items
            # loss = bce(pos_logits[mask & ~negative_item_mask], pos_labels_pos[mask].float())
            # loss += bce(neg_logits[mask], neg_labels[mask].float())
            # loss += bce(neg_logits[mask & negative_item_mask], pos_labels_neg[mask].float())

            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            epoch_losses.append(loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}")

        scheduler.step()

        #* evaluate every epoch
        val_metrics = evaluate(model, val_loader, device, "val", time_span=args.time_span)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_hr_at_10": val_metrics.hr_at_10,
            "val_ndcg_at_10": val_metrics.ndcg_at_10,
        }

        history.append(epoch_record)
        if val_metrics.hr_at_10 > best_val_hr:
            best_val_hr = val_metrics.hr_at_10
            torch.save(model.state_dict(), best_checkpoint)

    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    test_metrics = evaluate(model, test_loader, device, "test", time_span=args.time_span)
    full_test_metrics = None
    if args.report_full_eval:
        full_test_metrics = evaluate_full_ranking(model, full_test_loader, device, "test", time_span=args.time_span)

    metrics = {
        "config": vars(args),
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
    (args.output_dir / args.model / args.negative_items_handling / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
