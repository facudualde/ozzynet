# Training tips — fine-tuning Inception v3 on GTZAN

This document explains why the **current** `src/fine_tuning.py` is set
up the way it is, and what was wrong with a naive setup that got stuck
around **60 % validation accuracy**.

The recommendations below skip data augmentation (left to a later
change) and focus on what can be done with the model and data we
already have.

## What the previous run was missing

If you saw something like this at the end of `make fine_tuning`:

```
Epoch 50/50
------------------------------------------------------------
  [###################-]  96.4% loss=1.2810 [ 7712/ 7992]
Validation Error:
Accuracy: 60.7%, Avg loss: 1.132050
```

…here are the reasons the accuracy was lower than what's reachable on
GTZAN with this architecture (typically 75–80 %).

### 1. ImageNet normalization on spectrogram data (largest single issue)

`DatasetFT` used to default to:

```python
mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

Those numbers were computed over millions of natural photos. GTZAN
spectrograms look completely different — pixel value histograms are
concentrated in a narrow range, different per channel, very different
from natural images. Two consequences:

- The very first conv layer, which expects ImageNet-like inputs, sees
  a drastically shifted distribution. The pretrained kernels are
  still useful as feature detectors but not in their trained
  operating range.
- The `fc` head has to compensate for what the convs are seeing,
  which makes it harder to converge.

**Fix**: compute dataset-specific mean/std from the spectrograms
themselves, save them to `data/norm_stats.json`, and use them instead.

```bash
python3 src/compute_norm_stats.py
# -> writes data/norm_stats.json with {"mean": [...], "std": [...]}
```

Expected gain: **+1–3 accuracy points**.

### 2. Learning rate 10× too high for fine-tuning

`optimizer = torch.optim.Adam(model.trainable_parameters())` left Adam
at its default `lr=1e-3`. That's fine for from-scratch training, but
far too aggressive for fine-tuning a pretrained model. The fresh
`fc` head with random 2048×10 weights gets multiplied by `1e-3`-scale
gradient updates, and the model rarely recovers the precision that
`1e-4` would give.

**Fix**:

```python
optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-4)
```

### 3. No LR scheduler — stuck at one learning rate

A constant LR runs for all 50 epochs. There's no way for the model to
"fine-tune" near the end, which is where most of the accuracy lives.

**Fix**: `ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=3)`
watches val_loss and halves the LR when it plateaus. Expected gain:
**+2–5 accuracy points**.

### 4. No regularization tricks — label smoothing, EMA

- `nn.CrossEntropyLoss(label_smoothing=0.1)` discourages
  overconfidence on a small training set. **+0.5–1 %**.
- `torch.optim.swa_utils.AveragedModel` keeps an Exponential Moving
  Average of weights for evaluation — typically **+1–2 %**
  generalization accuracy.
- Mixed precision (`torch.amp.GradScaler` + `autocast`) gives
  **~1.5–2× training speedup** for free.

### 5. Frozen backbone limits adaptation

Only the `fc` head is trainable (~20 k params out of 25 M). On
spectrograms — visually very different from ImageNet's photos —
the conv stack has a lot of useful generic features (edges, textures)
but also some less-useful ones (specific object parts). Unfreezing
the last Inception block (`Mixed_7b`, `Mixed_7c`) lets the model
adapt ~3–4 M parameters to spectrogram patterns.

**Fix**: `model.unfreeze_last_block()` after one third of training.
Expected gain: **+2–5 accuracy points**.

## Summary table

| Fix                            | Where                                 | Gain         |
| ------------------------------ | ------------------------------------- | ------------ |
| Dataset-specific normalization | `dataset.py` + `data/norm_stats.json` | +1–3         |
| LR `1e-3` → `1e-4`             | `fine_tuning.py`                      | +0–2         |
| LR scheduler                   | `fine_tuning.py`                      | +2–5         |
| Label smoothing `0.1`          | `fine_tuning.py`                      | +0.5–1       |
| EMA at validation              | `fine_tuning.py`                      | +1–2         |
| Mixed precision                | `fine_tuning.py`                      | 1.5–2× speed |
| Unfreeze last block            | `cnn.py` + `fine_tuning.py`           | +2–5         |
| **Approximate total**          |                                       | **+10–20**   |

Reaching **75–80 %** validation accuracy is a plausible outcome once
all of these land.

## What is NOT the cause (worth ruling out)

| Suspect                               | Why it's fine                                                                              |
| ------------------------------------- | ------------------------------------------------------------------------------------------ |
| Frozen backbone being too restrictive | The freeze itself is reasonable; it's the duration + missing unfreeze later that costs you |
| Batch size 32                         | Appropriate for the dataset and GPU memory                                                 |
| Number of epochs (50)                 | Plenty; the LR scheduler would actually use these                                          |
| Validation set size                   | 2 000 samples × 10 classes is well-balanced; val_acc is stable                             |
| Class imbalance                       | GTZAN is perfectly balanced                                                                |
| Data leakage                          | `random_split` is correct                                                                  |
| Bug in the model                      | `InceptionV3.forward` returns a clean `[B, 10]` tensor; no leaking aux output              |

## How to read the new training output

After the changes, `make fine_tuning` prints a header with the
dataset / device / hyperparameter summary, then per-epoch lines:

```
Epoch 5/30
------------------------------------------------------------
  [#####--------------]  25.0% loss=0.8123 [ 2000/7992]
  ...
  val_loss: 0.7041  val_acc: 0.7320  [1464/2008]
  epoch 5 time: 0:36
  current LR: 1.000e-04
```

When the third of training passes, you'll see:

```
  >> unfroze last block: 3,212,544 new trainable params
```

That happens silently at first if you've kept the loop the same; the
banner above makes it visible.

The save at the end writes one checkpoint per run to
`checkpoints/<timestamp>/<timestamp>.pth`.

## Step-by-step usage

1. Generate the spectrograms if you haven't:

   ```bash
   make spectrograms
   ```

2. Compute the dataset-specific normalization stats (once):

   ```bash
   make shell
   python3 src/compute_norm_stats.py
   exit
   ```

3. Train:

   ```bash
   make fine_tuning
   ```

   Default hyper-parameters are `BATCH_SIZE=32` / `EPOCHS=50` from
   the `makefile`; override as:

   ```bash
   BATCH_SIZE=64 EPOCHS=30 make fine_tuning
   ```

## Tuning further

If after this baseline the val_acc plateaus before reaching the
upper end of the expected range, consider adding (in order):

- **Light augmentation** (`RandomHorizontalFlip` on the spectrogram
  time axis is meaningful for genre classification; small
  `RandomAffine` simulates tempo / pitch shifts).
- **Different optimizer** for the unfrozen block (e.g. a separate
  param group with `lr=1e-5` to avoid disturbing pretrained
  features).
- **Different backbone** — audio-pretrained models (PANNs, AST,
  CLAP) tend to lift GTZAN accuracy to the 80–90 % range because
  they're pretrained on the kind of data we're actually classifying.

Don't add any of these without first re-running the baseline end to
end and confirming its accuracy — debugging multiple simultaneous
changes is harder than debugging one at a time.
