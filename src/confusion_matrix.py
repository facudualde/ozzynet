from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix as _sklearn_cm
from torch.utils.data import DataLoader

def compute_confusion_matrix(model, loader, device, num_classes=10):
  model.eval()
  y_true, y_pred = [], []
  with torch.no_grad():
    for batch in loader:
      # Loader may yield (X, y) or (X, y, song_id) — be tolerant.
      X, y = batch[0], batch[1]
      X = X.to(device)
      logits = model(X)
      preds = logits.argmax(1).cpu().tolist()
      y_pred.extend(preds)
      y_true.extend(y.tolist())
  return _sklearn_cm(y_true, y_pred, labels=list(range(num_classes)))

def compute_confusion_matrix_songs(
  model, val_dataset, batch_size, device, num_classes=10,
):
  # Same song-level collate as train.py:validate_songs.
  def collate(batch):
    imgs, labels, sids = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels), list(sids)

  loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate,
  )

  model.eval()
  song_probs = defaultdict(list)
  song_labels = {}
  with torch.no_grad():
    for X, y, sids in loader:
      X = X.to(device)
      probs = F.softmax(model(X), dim=1).cpu()
      for i, sid in enumerate(sids):
        song_probs[sid].append(probs[i])
        song_labels[sid] = y[i].item()

  y_true, y_pred = [], []
  for sid, probs_list in song_probs.items():
    avg = torch.stack(probs_list).mean(0)
    y_pred.append(avg.argmax().item())
    y_true.append(song_labels[sid])
  return _sklearn_cm(y_true, y_pred, labels=list(range(num_classes)))


def print_confusion_matrix_report(cm, class_names, title=""):
  """Pretty-print a confusion matrix report.

  Parameters
  ----------
  cm : array-like of shape (num_classes, num_classes)
      Output of `compute_confusion_matrix` or
      `compute_confusion_matrix_songs`.
  class_names : list[str]
      Names for the rows and columns (e.g. DatasetFT.GENRES).
  title : str, optional
      Header shown at the top of the report.

  Output
  ------
  Prints to stdout:
    - Overall accuracy + total sample count.
    - Per-class recall with a 20-char bar chart.
    - Raw confusion matrix with row and column labels.
    - Top 5 most confused pairs (true → predicted).
  """
  cm = np.asarray(cm)
  n = len(class_names)
  total = int(cm.sum())
  correct = int(np.diag(cm).sum())
  overall_acc = correct / total if total else 0.0
  name_w = max(len(name) for name in class_names)
  bar_w = 20
  cell_w = 3

  bar = "=" * 60
  print(bar)
  print(
    f"  Confusion Matrix Report — {title}"
    if title else "  Confusion Matrix Report"
  )
  print(bar)

  print()
  print(f"  Total samples : {total}")
  print(f"  Correct       : {correct}/{total}")
  print(f"  Overall acc   : {overall_acc:.2%}")

  print()
  print("  Per-class accuracy (recall):")
  for i, name in enumerate(class_names):
    row_total = int(cm[i].sum())
    row_correct = int(cm[i, i])
    rec = row_correct / row_total if row_total else 0.0
    filled = int(round(rec * bar_w))
    glyph = "#" * filled + " " * (bar_w - filled)
    print(
      f"    {name:<{name_w}} "
      f"[{glyph}] {row_correct:>3d}/{row_total:<3d} = {rec:6.1%}"
    )

  abbrev = [nm[:3] for nm in class_names]
  print()
  print("  Confusion matrix (rows = true, cols = predicted):")
  header_cells = "".join(f"{a:>{cell_w + 1}}" for a in abbrev)
  print(f"    {' ' * (name_w + 1)}{header_cells}")
  for i, name in enumerate(class_names):
    cells = "".join(f"{int(cm[i, j]):>{cell_w + 1}}" for j in range(n))
    print(f"    {name:<{name_w}} {cells}")

  print()
  print("  Top 5 most confused pairs (true -> predicted):")
  pairs = []
  for i in range(n):
    row_total = max(int(cm[i].sum()), 1)
    for j in range(n):
      if i == j:
        continue
      v = int(cm[i, j])
      if v > 0:
        pairs.append((v / row_total, v, i, j))
  pairs.sort(reverse=True)
  if not pairs:
    print("    (no misclassifications)")
  else:
    for pct, v, i, j in pairs[:5]:
      print(
        f"    {class_names[i]:<{name_w}} -> {class_names[j]:<{name_w}}"
        f" :  {v}  ({pct:.1%} of {class_names[i]})"
      )

  print(bar)


def plot_confusion_matrix(cm, class_names, save_path, title=""):
  cm_arr = np.asarray(cm)
  cm_norm = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]
  fig, ax = plt.subplots(figsize=(8, 7))
  im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
  ax.figure.colorbar(im, ax=ax)
  ax.set(
    xticks=np.arange(len(class_names)),
    yticks=np.arange(len(class_names)),
    xticklabels=class_names,
    yticklabels=class_names,
    ylabel="True label",
    xlabel="Predicted label",
    title=title or "Confusion matrix",
  )
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")
  thresh = cm_norm.max() / 2.0
  for i in range(cm_arr.shape[0]):
    for j in range(cm_arr.shape[1]):
      ax.text(
        j, i, f"{int(cm_arr[i, j])}",
        ha="center", va="center",
        color="white" if cm_norm[i, j] > thresh else "black",
      )
  fig.tight_layout()
  plt.savefig(save_path, dpi=120)
  plt.close(fig)
