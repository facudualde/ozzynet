"""Training script for ConvNet on GTZAN mel-spectrograms."""

import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn import ConvNet
from dataset import GTZANDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def format_duration(seconds: float) -> str:
  return str(timedelta(seconds=int(seconds)))


def set_seed(seed: int) -> None:
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
  optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
  model.train()
  total_loss, total_correct, total = 0.0, 0, 0
  for X, y in loader:
    X, y = X.to(DEVICE), y.to(DEVICE)
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    bs = X.size(0)
    total_loss += loss.item() * bs
    total_correct += (logits.argmax(1) == y).sum().item()
    total += bs
  return total_loss / total, total_correct / total


@torch.no_grad()
def validate_chunks(
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
) -> tuple[float, float]:
  model.eval()
  total_loss, total_correct, total = 0.0, 0, 0
  for X, y in loader:
    X, y = X.to(DEVICE), y.to(DEVICE)
    logits = model(X)
    loss = criterion(logits, y)
    bs = X.size(0)
    total_loss += loss.item() * bs
    total_correct += (logits.argmax(1) == y).sum().item()
    total += bs
  return total_loss / total, total_correct / total


@torch.no_grad()
def validate_songs(
  model: nn.Module,
  batch_size: int,
) -> float:
  """Song-level accuracy via soft voting (mean of softmax probabilities)."""
  song_ds = GTZANDataset(
    split="val",
    seed=SEED,
    return_song_id=True,
  )

  def collate(batch):
    imgs, labels, sids = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels), list(sids)

  loader = DataLoader(
    song_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate,
  )

  model.eval()
  song_probs: dict[str, list[torch.Tensor]] = defaultdict(list)
  song_labels: dict[str, int] = {}

  for X, y, sids in loader:
    X = X.to(DEVICE)
    probs = torch.softmax(model(X), dim=1).cpu()
    for i, sid in enumerate(sids):
      song_probs[sid].append(probs[i])
      song_labels[sid] = y[i].item()

  correct = 0
  for sid, probs_list in song_probs.items():
    avg = torch.stack(probs_list).mean(0)
    if avg.argmax().item() == song_labels[sid]:
      correct += 1
  return correct / len(song_probs)

def loop(
  train_loader: DataLoader,
  val_loader: DataLoader,
  model: nn.Module,
  criterion: nn.Module,
  optimizer: torch.optim.Optimizer,
  scheduler,
  epochs: int,
  batch_size: int,
) -> None:
  print("=" * 60)
  print("  Training ConvNet on GTZAN (3-sec chunks)")
  print(f"  Device:  {DEVICE}")
  print(f"  Epochs:  {epochs}")
  print(f"  Batch:   {batch_size}")
  print(
    f"  Train:   {len(train_loader.dataset)} chunks "
    f"/ {len(train_loader)} batches"
  )
  print(
    f"  Val:     {len(val_loader.dataset)} chunks "
    f"/ {len(val_loader)} batches"
  )
  model.parameter_summary()
  print("=" * 60)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  save_dir = Path("checkpoints") / timestamp
  save_dir.mkdir(parents=True, exist_ok=True)
  best_song_acc = 0.0
  best_path: Path | None = None

  total_start = time.perf_counter()
  for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")
    print("-" * 60)
    epoch_start = time.perf_counter()

    train_loss, train_acc = train_one_epoch(
      model, train_loader, criterion, optimizer,
    )
    val_loss, val_chunk_acc = validate_chunks(
      model, val_loader, criterion,
    )
    song_acc = validate_songs(model, batch_size)

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    epoch_time = time.perf_counter() - epoch_start

    print(
      f"  train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f}\n"
      f"  val_loss:   {val_loss:.4f}  "
      f"val_acc (chunk): {val_chunk_acc:.4f}\n"
      f"  val_acc (song/soft-vote): {song_acc:.4f}\n"
      f"  lr: {current_lr:.6f}  time: {format_duration(epoch_time)}"
    )

    if song_acc > best_song_acc:
      best_song_acc = song_acc
      best_path = save_dir / f"best_song_acc_{song_acc:.4f}.pth"
      torch.save(model.state_dict(), best_path)
      print(
        f"  >> new best song_acc: {best_song_acc:.4f} "
        f"-> {best_path.name}"
      )

  last_path = save_dir / "last.pth"
  torch.save(model.state_dict(), last_path)
  total_time = time.perf_counter() - total_start

  print("\n" + "=" * 60)
  print(f"  Done!  Total time: {format_duration(total_time)}")
  print(f"  Best song_acc: {best_song_acc:.4f}  ({best_path})")
  print(f"  Last checkpoint: {last_path}")
  print("=" * 60)


def main() -> None:
  if len(sys.argv) < 3:
    print("Usage: python train.py [batch_size] [epochs]")
    sys.exit(1)
  batch_size = int(sys.argv[1])
  epochs = int(sys.argv[2])

  set_seed(SEED)

  train_ds = GTZANDataset(split="train", seed=SEED)
  val_ds = GTZANDataset(split="val", seed=SEED)

  train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
  )
  val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
  )

  model = ConvNet().to(DEVICE)
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-4,
  )
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs,
  )

  loop(
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    epochs,
    batch_size,
  )


if __name__ == "__main__":
  main()
