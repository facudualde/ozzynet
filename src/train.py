"""Training script for ConvNet on mel-spectrograms (Gtzan or Custom)."""

import argparse
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn import ConvNet
from confusion_matrix import (
    compute_confusion_matrix_songs,
    plot_confusion_matrix,
    print_confusion_matrix_report,
)
from dataset import Custom, Gtzan
from plot_utils import plot_history, plot_loss_history

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    # Deterministic seeds for torch + numpy + cuda so re-runs are reproducible.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    # One training pass; returns (avg_loss, accuracy_pct).
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y, _ in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = X.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    # Soft-vote per song; returns (voting_acc_pct, chunk_acc_pct, avg_chunk_loss).
    model.eval()
    song_probs: dict[str, list[torch.Tensor]] = defaultdict(list)
    song_labels: dict[str, int] = {}
    chunk_loss, chunk_correct, chunk_total = 0.0, 0, 0
    for X, y, sids in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        chunk_loss += criterion(logits, y).item() * X.size(0)
        chunk_correct += (logits.argmax(1) == y).sum().item()
        chunk_total += X.size(0)
        probs = torch.softmax(logits, dim=1).cpu()
        for i, sid in enumerate(sids):
            song_probs[sid].append(probs[i])
            song_labels[sid] = y[i].item()

    correct_songs = sum(
        (torch.stack(p).mean(0).argmax().item() == song_labels[sid])
        for sid, p in song_probs.items()
    )
    voting_acc = 100.0 * correct_songs / max(len(song_probs), 1)
    chunk_acc = 100.0 * chunk_correct / max(chunk_total, 1)
    return voting_acc, chunk_acc, chunk_loss / max(chunk_total, 1)


def final_evaluation(
    model: nn.Module,
    val_dataset: torch.utils.data.Dataset,
    batch_size: int,
    save_dir: str,
) -> None:
    # Song-level confusion matrix + textual report at the end of training.
    cm = compute_confusion_matrix_songs(model, val_dataset, batch_size, DEVICE)
    plot_confusion_matrix(
        cm, val_dataset.GENRES,
        save_path=f"{save_dir}/confusion_matrix.png",
        title="Song-level (soft voting)",
    )
    print_confusion_matrix_report(cm, val_dataset.GENRES, title="Song-level (soft voting)")


def loop(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    save_dir: str,
    dataset_name: str,
) -> tuple[str, str, dict]:
    # Train for `epochs`; track and save both the latest and best-by-voting-acc checkpoints.
    print("=" * 60)
    print("  Training ConvNet with soft-vote validation")
    print(f"  Device:  {DEVICE}")
    print(f"  Epochs:  {epochs}")
    print(f"  Batch:   {train_loader.batch_size}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Train:   {len(train_loader.dataset)} chunks / {len(train_loader)} batches")
    print(f"  Val:     {len(val_loader.dataset)} chunks / {len(val_loader)} batches")
    model.parameter_summary()
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = f"{save_dir}/{timestamp}_latest.pth"
    best_path = f"{save_dir}/{timestamp}_best.pth"
    best_acc = -1.0
    history = {
        "train_acc": [],
        "val_chunk_acc": [],
        "val_song_acc": [],
        "train_loss": [],
        "val_loss": [],
    }

    total_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 60)

        train_loader.dataset.reset_epoch_samples()

        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer)
        voting_acc, chunk_acc, val_loss = validate(val_loader, model, criterion)

        history["train_acc"].append(train_acc)
        history["val_chunk_acc"].append(chunk_acc)
        history["val_song_acc"].append(voting_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"  Train    avg_loss={train_loss:.4f}  acc={train_acc:.1f}%\n"
            f"  Val      voting_acc={voting_acc:.1f}%  chunk_acc={chunk_acc:.1f}%  avg_loss={val_loss:.4f}\n"
            f"  epoch {epoch} time: {format_duration(time.perf_counter() - epoch_start)}"
        )

        torch.save(model.state_dict(), latest_path)
        if voting_acc > best_acc:
            best_acc = voting_acc
            torch.save(model.state_dict(), best_path)
            print(f"  >> new best voting_acc={best_acc:.1f}% -> {best_path}")

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 60)
    print(f"  Done!  Total time: {format_duration(total_time)}")
    print(f"  Best voting acc: {best_acc:.1f}%  -> {best_path}")
    print(f"  Latest:          {latest_path}")
    print("=" * 60)
    return latest_path, best_path, history


def parse_args() -> argparse.Namespace:
    # CLI: positional batch_size/epochs (matches make train), optional tuning knobs.
    parser = argparse.ArgumentParser(description="Train ConvNet on mel-spectrograms.")
    parser.add_argument("batch_size", type=int)
    parser.add_argument("epochs", type=int)
    parser.add_argument("--dataset", choices=["gtzan", "custom"], default="gtzan",
                        help="Which dataset class to use.")
    parser.add_argument("--data_augmentation", action="store_true", help="Enable image-level augmentation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--random_chunks_number", type=int, default=10,
                        help="Random chunks sampled per song at the start of each epoch (>= 1).")
    args = parser.parse_args()
    if args.random_chunks_number < 1:
        parser.error("--random_chunks_number must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_cls = Gtzan if args.dataset == "gtzan" else Custom
    train_ds = dataset_cls(
        split="train", seed=args.seed, model="cnn",
        random_chunks_number=args.random_chunks_number,
        data_augmentation=args.data_augmentation,
    )
    val_ds = dataset_cls(
        split="val", seed=args.seed, model="cnn",
        # Validation uses the full chunk set per song for stable, comparable metrics.
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = ConvNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"checkpoints/from_scratch/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    _, _, history = loop(train_loader, val_loader, model, criterion, optimizer, args.epochs, save_dir, args.dataset)
    final_evaluation(model, val_ds, args.batch_size, save_dir)
    plot_history(history, save_dir, "ConvNet")
    plot_loss_history(history, save_dir, "ConvNet")
    print(f"Model + report saved under: {save_dir}")


if __name__ == "__main__":
    main()
