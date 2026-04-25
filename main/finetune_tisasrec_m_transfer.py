from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.tisasrec import TiSASRec
from train_model import (
    EvalDataset,
    FullEvalDataset,
    NUM_METADATA,
    TIME_NORMALIZATION,
    TrainDataset,
    collate_full_eval_batch,
    evaluate,
    evaluate_full_ranking,
    generate_relation_matrix,
    generate_time_matrix_batch,
    relation_matrix_cache_path,
    set_seed,
    str2bool,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "tisasrec_m"
SOURCE_DATASET = "mobilerec"
TARGET_DATASET = "steamrec"
NEGATIVE_ITEMS_HANDLING = "penalize-negative"
RESET_TRANSFER_KEYS = {"item_emb.weight", "metadata_cat_emb.weight"}
TRAINABLE_TRANSFER_PREFIXES = ("item_emb.", "metadata_cat_emb.")
HISTORY_HEADERS = ["training_loss", "hr@10", "ndcg@10", "lr", "best_val_hr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune MobileRec TiSASRec-M weights on SteamRec by training only "
            "the target item and category embedding tables."
        )
    )
    parser.add_argument("--target-dataset", type=str, default=TARGET_DATASET)
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=SCRIPT_DIR
        / "data"
        / "outputs"
        / MODEL_NAME
        / NEGATIVE_ITEMS_HANDLING
        / "best_model.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "outputs" / "steamrec_transfer",
    )
    parser.add_argument("--experiment-name", type=str, default="embedding_finetune")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--time-span", type=int, default=256)
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
    parser.add_argument("--training-only", default=False, type=str2bool)
    parser.add_argument("--resume", default=True, type=str2bool)

    return parser.parse_args()


def json_default(value: Any) -> str:
    return str(value)


def load_torch_object(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a state dict or checkpoint dictionary.")

    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return normalize_state_dict_keys(candidate)

    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return normalize_state_dict_keys(checkpoint)

    raise ValueError("Checkpoint dictionary does not contain model weights.")


def normalize_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    return state_dict


def initialize_model_parameters(model: TiSASRec) -> None:
    for _, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except ValueError:
            pass
    model.clear_padding_item_embedding()


def transfer_compatible_weights(
    *,
    model: TiSASRec,
    source_state: dict[str, torch.Tensor],
) -> dict[str, object]:
    target_state = model.state_dict()
    loaded_keys: list[dict[str, object]] = []
    skipped_keys: list[dict[str, object]] = []

    for source_key, source_value in source_state.items():
        if source_key in RESET_TRANSFER_KEYS:
            skipped_keys.append(
                {
                    "key": source_key,
                    "reason": "target_vocab_reset",
                    "source_shape": list(source_value.shape),
                    "target_shape": list(target_state[source_key].shape)
                    if source_key in target_state
                    else None,
                }
            )
            continue

        if source_key not in target_state:
            skipped_keys.append(
                {
                    "key": source_key,
                    "reason": "missing_in_target_model",
                    "source_shape": list(source_value.shape),
                    "target_shape": None,
                }
            )
            continue

        target_value = target_state[source_key]
        if tuple(source_value.shape) != tuple(target_value.shape):
            skipped_keys.append(
                {
                    "key": source_key,
                    "reason": "shape_mismatch",
                    "source_shape": list(source_value.shape),
                    "target_shape": list(target_value.shape),
                }
            )
            continue

        target_state[source_key] = source_value.detach().clone()
        loaded_keys.append(
            {
                "key": source_key,
                "shape": list(source_value.shape),
            }
        )

    model.load_state_dict(target_state)
    model.clear_padding_item_embedding()

    if not loaded_keys:
        raise ValueError(
            "No compatible source checkpoint weights were loaded. "
            "Check that the source checkpoint and target model hyperparameters match."
        )

    return {
        "loaded_keys": loaded_keys,
        "skipped_keys": skipped_keys,
    }


def freeze_for_embedding_finetune(model: TiSASRec) -> list[str]:
    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_names: list[str] = []
    for name, param in model.named_parameters():
        if name.startswith(TRAINABLE_TRANSFER_PREFIXES):
            param.requires_grad = True
            trainable_names.append(name)

    if not trainable_names:
        raise ValueError("No embedding parameters were left trainable.")

    return trainable_names


def read_resume_state(history_path: Path, default_lr: float) -> tuple[float, float]:
    if not history_path.exists():
        return -1.0, default_lr

    history = pd.read_csv(history_path)
    if history.empty:
        return -1.0, default_lr

    return float(history["best_val_hr"].iloc[-1]), float(history["lr"].iloc[-1])


def prepare_history_file(history_path: Path, *, resume: bool) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and history_path.exists():
        return

    with history_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HISTORY_HEADERS)


def load_relation_matrix(
    *,
    data_dir: Path,
    target_dataset: str,
    sequences: pd.DataFrame,
    max_len: int,
    time_span: int,
) -> dict[int, np.ndarray]:
    relation_matrix_path = relation_matrix_cache_path(
        data_dir,
        target_dataset,
        max_len,
        time_span,
    )
    try:
        with relation_matrix_path.open("rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        relation_matrix = generate_relation_matrix(sequences, max_len, time_span)
        relation_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        with relation_matrix_path.open("wb") as fh:
            pickle.dump(relation_matrix, fh)
        return relation_matrix


def build_run_dir(args: argparse.Namespace) -> Path:
    return (
        args.output_dir
        / MODEL_NAME
        / NEGATIVE_ITEMS_HANDLING
        / args.experiment_name
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.target_dataset != TARGET_DATASET:
        raise ValueError(f"This transfer script currently targets only {TARGET_DATASET}.")

    if not args.source_checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {args.source_checkpoint}")

    run_dir = build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    sequences = pd.read_parquet(args.data_dir / args.target_dataset / "final_sequences.parquet")
    sequences = sequences.reset_index(drop=True)
    item_mapping = pd.read_parquet(args.data_dir / args.target_dataset / "final_item_mapping.parquet")
    category_mapping = pd.read_parquet(args.data_dir / args.target_dataset / "final_app_category.parquet")

    num_items = int(item_mapping["item_id"].max())
    num_categories = int(category_mapping["app_category_id"].max())

    relation_matrix = load_relation_matrix(
        data_dir=args.data_dir,
        target_dataset=args.target_dataset,
        sequences=sequences,
        max_len=args.max_len,
        time_span=args.time_span,
    )
    print(f"Relation matrix enabled: {bool(relation_matrix)}")

    print("Preparing train dataset")
    train_dataset = TrainDataset(
        sequences=sequences,
        max_len=args.max_len,
        num_items=num_items,
        num_categories=num_categories,
        seed=args.seed,
        negative_items_handling=NEGATIVE_ITEMS_HANDLING,
        relation_matrix=relation_matrix,
    )
    print("Prepared train dataset")

    if len(train_dataset) == 0:
        raise ValueError("SteamRec train dataset is empty after filtering.")

    print("Preparing val dataset")
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
    print("Prepared val dataset")

    if not args.training_only:
        print("Preparing test dataset")
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
        print("Prepared test dataset")

        print("Preparing full test dataset")
        full_test_dataset = FullEvalDataset(
            sequences=sequences,
            sequence_column="validation_sequence",
            target_column="test_target",
            max_len=args.max_len,
            num_categories=num_categories,
        )
        print("Prepared full test dataset")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    if not args.training_only:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        full_test_loader = DataLoader(
            full_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_full_eval_batch,
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

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

    initialize_model_parameters(model)

    source_checkpoint = load_torch_object(args.source_checkpoint, device)
    source_state = extract_state_dict(source_checkpoint)
    transfer_report = transfer_compatible_weights(
        model=model,
        source_state=source_state,
    )
    trainable_parameters = freeze_for_embedding_finetune(model)

    best_checkpoint = run_dir / "best_model.pt"
    current_checkpoint = run_dir / "current_model.pt"
    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "metrics.json"
    transfer_report_path = run_dir / "transfer_load_report.json"

    transfer_metadata = {
        "source_dataset": SOURCE_DATASET,
        "target_dataset": args.target_dataset,
        "model": MODEL_NAME,
        "negative_items_handling": NEGATIVE_ITEMS_HANDLING,
        "source_checkpoint": args.source_checkpoint,
        "reset_keys": sorted(RESET_TRANSFER_KEYS),
        "trainable_parameters": trainable_parameters,
        "frozen_parameter_count": sum(
            param.numel() for param in model.parameters() if not param.requires_grad
        ),
        "trainable_parameter_count": sum(
            param.numel() for param in model.parameters() if param.requires_grad
        ),
        "num_items": num_items,
        "num_categories": num_categories,
        **transfer_report,
    }
    transfer_report_path.write_text(
        json.dumps(transfer_metadata, indent=2, default=json_default)
    )

    prepare_history_file(history_path, resume=args.resume)
    best_val_hr = -1.0
    optimizer_lr = args.lr
    if args.resume:
        best_val_hr, optimizer_lr = read_resume_state(history_path, args.lr)

        if current_checkpoint.exists():
            model.load_state_dict(load_torch_object(current_checkpoint, device))
            print(f"Resumed from current checkpoint: {current_checkpoint}")
        elif best_checkpoint.exists():
            model.load_state_dict(load_torch_object(best_checkpoint, device))
            print(f"Resumed from best checkpoint: {best_checkpoint}")

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=optimizer_lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=args.steplr_gamma,
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")

    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
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
                time_matrix = generate_time_matrix_batch(
                    batch["time_seq"],
                    args.time_span,
                    device,
                )

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

            pos_loss = bce(pos_logits, pos_labels)
            neg_loss = bce(neg_logits, neg_labels)
            explicit_neg_loss = bce(pos_logits, neg_labels)

            loss = (
                (pos_loss * valid_positive_mask)
                + (neg_loss * mask)
                + (explicit_neg_loss * (mask & negative_item_mask))
            ).sum() / mask.sum().clamp(min=1)

            loss.backward()
            optimizer.step()
            model.clear_padding_item_embedding()

            loss_value = float(loss.detach().cpu().item())
            epoch_losses.append(loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}")

        scheduler.step()

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            "val",
            time_span=args.time_span,
        )

        train_loss = float(np.mean(epoch_losses))
        val_hr = val_metrics.hr_at_10
        val_ndcg = val_metrics.ndcg_at_10
        current_lr = scheduler.get_last_lr()[0]

        if val_hr > best_val_hr:
            best_val_hr = val_hr
            torch.save(model.state_dict(), best_checkpoint)

        torch.save(model.state_dict(), current_checkpoint)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_hr_at_10": val_hr,
            "val_ndcg_at_10": val_ndcg,
            "lr": current_lr,
            "best_val_hr_at_10": best_val_hr,
        }
        history.append(epoch_record)

        with history_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    train_loss,
                    val_hr,
                    val_ndcg,
                    current_lr,
                    best_val_hr,
                ]
            )

    if args.training_only:
        print(f"Training-only run complete. Artifacts written to: {run_dir}")
        return

    if not best_checkpoint.exists():
        if current_checkpoint.exists():
            best_checkpoint = current_checkpoint
        else:
            raise FileNotFoundError("No checkpoint is available for test evaluation.")

    print("Running test evaluation on best model")
    model.load_state_dict(load_torch_object(best_checkpoint, device))

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        "test",
        time_span=args.time_span,
    )
    full_test_metrics = None
    if args.report_full_eval:
        full_test_metrics = evaluate_full_ranking(
            model,
            full_test_loader,
            device,
            "test",
            time_span=args.time_span,
        )

    config = vars(args).copy()
    config["time_normalization"] = TIME_NORMALIZATION

    metrics = {
        "config": config,
        "transfer": transfer_metadata,
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
                "ranking_pool": (
                    "all catalog items excluding padding and items already "
                    "present in the input sequence"
                ),
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

    metrics_path.write_text(json.dumps(metrics, indent=2, default=json_default))
    print(json.dumps(metrics, indent=2, default=json_default))


if __name__ == "__main__":
    main()
