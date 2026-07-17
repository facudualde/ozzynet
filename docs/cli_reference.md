# CLI reference

How to invoke the scripts under `src/` from the makefile. Every target forwards extra flags via `FLAGS="..."`. Positional args like `BATCH_SIZE`, `EPOCHS`, and `CHECKPOINT` use make variables because they don't fit the `--flag` style.

For high-level workflow (setup → spectrograms → train → eval) see [`README.md`](../README.md).

## `make spectrograms` → `src/spectrograms.py`

Generates mel-spectrogram PNGs from `.wav`/`.mp3` files using librosa + Pillow. Pickled-state workers via `ProcessPoolExecutor`.

Force `sr=22050` so train and test chunks share dimensions regardless of input sample rate.

| Flag | Default | Description |
| --- | --- | --- |
| `--gtzan` | off | Read from `gtzan/(test/)songs` and write to `gtzan/(test/)spectrograms` (default: `dataset/`). |
| `--rgb` | off | Render RGB images (InceptionV3 pipeline). |
| `--ft` | off | Resize output to `299x299`. |
| `--pitch` | off | Pitch augmentation (+/-1 semitone). Does NOT affect hop. |
| `--hop <float>` | `3.0` | Stride in seconds. Range `(0, 3.0]`. Independent from `--pitch`. |
| `--type {train, test}` | `train` | `train` reads `<root>/songs`, `test` reads `<root>/test/songs`. `--pitch` is auto-disabled under `--type test` (with a warning). |
| `--max-workers <int>` | `4` | Parallel worker processes. |

Examples:

```
make spectrograms FLAGS="--gtzan --rgb --ft --pitch --hop 1.0"
make spectrograms FLAGS="--type test --gtzan --rgb --ft"
```

## `make train` → `src/train.py`

Trains a ConvNet from scratch on the spectrogram dataset. Saves `{ts}_latest.pth` and `{ts}_best.pth` to `checkpoints/from_scratch/<ts>/`.

Required make variables:

- `BATCH_SIZE=<int>`
- `EPOCHS=<int>`

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset {gtzan, custom}` | `gtzan` | Which dataset class to read from. |
| `--data_augmentation` | off | Enable image-level SpecAugment during training. |
| `--seed <int>` | `42` | Deterministic seed for split + weight init. |
| `--lr <float>` | `1e-5` | Optimizer learning rate. |
| `--weight_decay <float>` | `0.01` | AdamW weight decay. |
| `--num_workers <int>` | `2` | DataLoader workers. |
| `--random_chunks_number <int>` | `10` | Random chunks sampled per song at the start of each epoch (**train only**; validation always uses the full set for stable, comparable metrics). No-op for songs with ≤ this many chunks. Use smaller values (e.g. `5`) for stronger regularization on long Custom songs. Must be `>= 1`. |

Example:

```
make train BATCH_SIZE=64 EPOCHS=50 FLAGS="--data_augmentation"
```

## `make fine_tuning` → `src/fine_tuning.py`

Fine-tunes InceptionV3 on the spectrogram dataset. Saves checkpoints to `checkpoints/fine_tuning/<ts>/`.

Required make variables:

- `BATCH_SIZE=<int>`
- `EPOCHS=<int>`

Flags: same as `train` (see above).

Example:

```
make fine_tuning BATCH_SIZE=32 EPOCHS=10 FLAGS="--lr 1e-4"
```

## `make eval` → `src/eval.py`

Evaluates a `.pth` checkpoint on the held-out test set (`<root>/test/spectrograms/`). Prints a banner with checkpoint + dataset + model + sample/genre counts, chunk-level accuracy, song-level soft-vote accuracy, and a deduplicated list of misclassified songs grouped by `(true → pred)`.

The script reads `num_classes` from the state_dict to instantiate the right architecture, so it never has to guess the model from the path.

Validates at startup that the checkpoint's classifier output size matches the dataset's number of genres; a mismatch exits with `[ERR]` and a clear message.

Required make variable:

- `CHECKPOINT=<path-to-checkpoint>`

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset {gtzan, custom}` | `gtzan` | Which test root to read from. |
| `--model {cnn, inception}` | **required** | Architecture the checkpoint belongs to. |
| `--mode {song, chunk, both}` | `both` | `song` = soft-vote per song, `chunk` = per spectrogram, `both` = both. |
| `--batch-size <int>` | `32` | Inference batch size. |
| `--num-workers <int>` | `0` | DataLoader workers (CPU work). |

Examples:

```
make eval CHECKPOINT=checkpoints/fine_tuning/<ts>/<ts>_best.pth FLAGS="--model inception"
make eval CHECKPOINT=checkpoints/from_scratch/<ts>/<ts>_best.pth FLAGS="--model cnn --dataset custom"
```

## `make health` → `src/health.py`

GPU detect + CPU/GPU matmul benchmark. No flags, no env vars.

## Shared throughlines

- `FLAGS="..."` is forwarded verbatim by the makefile.
- Positional args (`BATCH_SIZE`, `EPOCHS`, `CHECKPOINT`) use make env vars because they don't fit the `--flag` style.
- All scripts accept `--num-workers` (except `health` and `eval`, which use `--max-workers` and `--num-workers` respectively). The spectrogram generator uses `--max-workers` for clarity.
- Audio is always loaded at `sr=22050` in `src/spectrograms.py` so train and test chunks share dimensions; downstream models expect `130x128` (ConvNet) or `299x299` (InceptionV3 with `--rgb --ft`).
