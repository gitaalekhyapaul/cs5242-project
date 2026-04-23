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

from models.sasrec import SASRec
from models.tisasrec import TiSASRec, TiSASRecWithoutMetadata

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SASRec baseline on prepared MobileRec sequences.")
    parser.add_argument("--dataset", type=str, default="mobilerec")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))

    parser.add_argument("--model", type=str, default="sasrec")
    parser.add_argument("--output-dir", type=Path, default=Path("data/outputs"))

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=50)
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


class TrainDataset(Dataset):
    def __init__(self, sequences: pd.DataFrame, max_len: int, num_items: int, seed: int) -> None:
        self.rows: list[dict[str, object]] = []
        self.max_len = max_len
        self.num_items = num_items
        self.rng = random.Random(seed)
        for row in sequences.itertuples(index=False):
            train_sequence = list(row.train_sequence)
            if len(train_sequence) < 2:
                continue
            history = train_sequence[:-1]
            targets = train_sequence[1:]
            self.rows.append(
                {
                    "history": history,
                    "targets": targets,
                    "seen": set(train_sequence),
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
        input_seq = pad_sequence(list(row["history"]), self.max_len)
        pos_seq = pad_sequence(list(row["targets"]), self.max_len)
        neg_trimmed = [self._sample_negative(row["seen"]) for _ in row["targets"]]
        neg_seq = pad_sequence(neg_trimmed, self.max_len)
        return {
            "input_ids": torch.from_numpy(input_seq),
            "pos_ids": torch.from_numpy(pos_seq),
            "neg_ids": torch.from_numpy(neg_seq),
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
                    negatives.append(sampled)
            candidates = [target] + list(negatives)
            self.rows.append(
                {
                    #* for validation loss computation
                    "targets": targets,
                    "seen": seen,

                    "input_ids": pad_sequence(sequence, max_len),
                    "candidate_ids": np.asarray(candidates, dtype=np.int64),
                    "target": 0,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        pos_seq = pad_sequence(list(row["targets"]), self.max_len)
        neg_trimmed = [self._sample_negative(row["seen"]) for _ in row["targets"]]
        neg_seq = pad_sequence(neg_trimmed, self.max_len)

        return {
            "input_ids": torch.from_numpy(row["input_ids"]),
            "candidate_ids": torch.from_numpy(row["candidate_ids"]),
            "target": torch.tensor(row["target"], dtype=torch.long),

            #* for validation loss computation
            "pos_ids": torch.from_numpy(pos_seq),
            "neg_ids": torch.from_numpy(neg_seq),
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
    loss: float
    hr_at_10: float
    ndcg_at_10: float


def evaluate(
    model: SASRec | TiSASRec | TiSASRecWithoutMetadata,
    data_loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> Metrics:

    model.eval()
    hit_scores: list[float] = []
    ndcg_scores: list[float] = []
    losses: list[float] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"eval:{split_name}", leave=False):
            input_ids = batch["input_ids"].to(device)
            candidate_ids = batch["candidate_ids"].to(device)
            scores = model.score_candidates(input_ids=input_ids, candidate_ids=candidate_ids)

            pos_ids = batch["pos_ids"].to(device)
            neg_ids = batch["neg_ids"].to(device)
            pos_logits, neg_logits = model.training_logits(
                input_ids=input_ids,
                pos_ids=pos_ids,
                neg_ids=neg_ids,
            )

            bce = nn.BCEWithLogitsLoss(reduction="none")

            mask = pos_ids.ne(0)
            pos_loss = bce(pos_logits, torch.ones_like(pos_logits))
            neg_loss = bce(neg_logits, torch.zeros_like(neg_logits))
            loss = ((pos_loss + neg_loss) * mask).sum() / mask.sum().clamp(min=1)
            losses.append(loss)

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
        loss=float(np.mean(losses)),
        hr_at_10=float(np.mean(hit_scores)),
        ndcg_at_10=float(np.mean(ndcg_scores)),
    )


def evaluate_full_ranking(
    model: SASRec | TiSASRec | TiSASRecWithoutMetadata,
    data_loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> Metrics:
    model.eval()
    hit_scores: list[float] = []
    ndcg_scores: list[float] = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"eval:{split_name}:full", leave=False):
            input_ids = batch["input_ids"].to(device)
            seen_ids = batch["seen_ids"]
            targets = batch["target"].to(device)

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
    set_seed(args.seed)
    (args.output_dir / args.model).mkdir(parents=True, exist_ok=True)

    sequences = pd.read_parquet(args.data_dir / args.dataset / "final_sequences.parquet")
    item_mapping = pd.read_parquet(args.data_dir / args.dataset / "final_item_mapping.parquet")
    category_mapping = pd.read_parquet(args.data_dir / args.dataset / "final_app_category_mapping.parquet")

    num_items = int(item_mapping["item_id"].max())
    num_categories = int(category_mapping["app_category_id"].max())

    train_dataset = TrainDataset(
        sequences=sequences,
        max_len=args.max_len,
        num_items=num_items,
        seed=args.seed,
    )
    val_dataset = EvalDataset(
        sequences=sequences,
        sequence_column="train_sequence",
        target_column="validation_target",
        num_items=num_items,
        negative_samples=args.eval_negative_samples,
        max_len=args.max_len,
        seed=args.seed + 1,
    )
    test_dataset = EvalDataset(
        sequences=sequences,
        sequence_column="validation_sequence",
        target_column="test_target",
        num_items=num_items,
        negative_samples=args.eval_negative_samples,
        max_len=args.max_len,
        seed=args.seed + 2,
    )
    full_test_dataset = FullEvalDataset(
        sequences=sequences,
        sequence_column="validation_sequence",
        target_column="test_target",
        max_len=args.max_len,
    )

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

    model = SASRec(
        num_items=num_items,
        max_len=args.max_len,
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98), #* forget the past more quickly than default (0.9, 0.999)
        weight_decay=args.weight_decay,
        # eps=1e-9, #uncomment if exploding loss/early plateau
    )

    bce = nn.BCEWithLogitsLoss(reduction="none")

    #* if using alternative loss computation
    # bce = nn.BCEWithLogitsLoss(reduction="mean")

    best_val_hr = -1.0
    best_checkpoint = args.output_dir / args.model / "best_model.pt"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        if args.inference_only: break # skip training if in inference mode

        model.train()
        epoch_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"train:{epoch}", leave=False)
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            neg_ids = batch["neg_ids"].to(device)
            pos_logits, neg_logits = model.training_logits(
                input_ids=input_ids,
                pos_ids=pos_ids,
                neg_ids=neg_ids,
            )

            optimizer.zero_grad()

            mask = pos_ids.ne(0)
            pos_loss = bce(pos_logits, torch.ones_like(pos_logits))
            neg_loss = bce(neg_logits, torch.zeros_like(neg_logits))
            loss = ((pos_loss + neg_loss) * mask).sum() / mask.sum().clamp(min=1)

            #* alternative loss computation
            # mask = pos_ids != 0 # mask padding items
            # loss = bce(pos_logits[mask], pos_ids[mask].float())
            # loss += bce(neg_logits[mask], neg_ids[mask].float())

            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            epoch_losses.append(loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}")

        #* evaluate every epoch
        #todo: compute validation loss and append to epoch_record
        val_metrics = evaluate(model, val_loader, device, "val")
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_loss": val_metrics.loss,
            "val_hr_at_10": val_metrics.hr_at_10,
            "val_ndcg_at_10": val_metrics.ndcg_at_10,
        }

        history.append(epoch_record)
        if val_metrics.hr_at_10 > best_val_hr:
            best_val_hr = val_metrics.hr_at_10
            torch.save(model.state_dict(), best_checkpoint)

    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    test_metrics = evaluate(model, test_loader, device, "test")
    full_test_metrics = None
    if args.report_full_eval:
        full_test_metrics = evaluate_full_ranking(model, full_test_loader, device, "test")

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
    (args.output_dir / args.model / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
