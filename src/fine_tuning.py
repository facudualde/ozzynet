import os
import sys
import time
from datetime import datetime, timedelta
import torch
from torch import nn
from torch.utils.data import DataLoader
from cnn import InceptionV3
from dataset import GTZANDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_duration(seconds: float) -> str:
  return str(timedelta(seconds=int(seconds)))

def train(train_dataloader, model, loss_fn, optimizer):
  model.train()
  size = len(train_dataloader.dataset)
  total_batches = len(train_dataloader)

  for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)

    pred = model(X)
    loss = loss_fn(pred, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 10 == 0:
      current = (batch + 1) * len(X)
      pct = 100.0 * (batch + 1) / total_batches
      bar_width = 20
      filled = int(bar_width * (batch + 1) / total_batches)
      bar = f"[{'#' * filled}{'-' * (bar_width - filled)}]"
      print(
        f"\r  {bar} {pct:5.1f}% "
        f"loss={loss.item():.4f} [{current:>5d}/{size:>5d}]",
        end="", flush=True,
      )

  print()

def validate(val_dataloader, model, loss_fn):
  size = len(val_dataloader.dataset)
  num_batches = len(val_dataloader)
  model.eval()
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, y in val_dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(
    f"Validation Error: \n" 
    f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
  )

def loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs):
  print("=" * 60)
  print("  Training InceptionV3 on GTZAN")
  print(f"  Device:  {device}")
  print(f"  Epochs:  {epochs}")
  print(f"  Batch:   {train_dataloader.batch_size}")
  print(f"  Train:   {len(train_dataloader.dataset)} samples / {len(train_dataloader)} batches")
  print(f"  Val:     {len(val_dataloader.dataset)} samples / {len(val_dataloader)} batches")
  model.parameter_summary()
  print("=" * 60)

  total_start = time.perf_counter()
  for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")
    print("-" * 60)
    epoch_start = time.perf_counter()
    train(train_dataloader, model, loss_fn, optimizer)
    validate(val_dataloader, model, loss_fn)
    epoch_time = time.perf_counter() - epoch_start
    print(f"  epoch {epoch} time: {format_duration(epoch_time)}")

  total_time = time.perf_counter() - total_start
  print("\n" + "=" * 60)
  print(f"  Done!  Total time: {format_duration(total_time)}")
  print("=" * 60)

  print()
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  save_dir = os.path.join("checkpoints", timestamp)
  os.makedirs(save_dir, exist_ok=True)
  save_path = os.path.join(save_dir, f"{timestamp}.pth")
  torch.save(model.state_dict(), save_path)
  print(f"Model saved to: {save_path}")

def setup():
  batch_size = int(sys.argv[1])
  epochs = int(sys.argv[2])

  train_ds = GTZANDataset(split="train", seed=42)
  val_ds   = GTZANDataset(split="val", seed=42)

  train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  val_dataloader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

  model = InceptionV3().to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-4)

  loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs)

if __name__ == "__main__":
  setup()
