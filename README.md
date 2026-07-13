# ozzynet

AMD GPU (ROCm) PyTorch sandbox. Ships a Docker image with PyTorch + ROCm
pre-installed, plus a diagnostic script that benchmarks CPU vs. GPU and
verifies that PyTorch can see your AMD card.

## Prerequisites

- Linux host with an AMD GPU supported by ROCm.
- `amdgpu` driver loaded and `/dev/kfd` + `/dev/dri/renderD*` present (`ls /dev/kfd /dev/dri/`).
- Docker with Compose v2.
- Your user in the `video` and `render` groups: `sudo usermod -aG video,render $USER`, then log out/in.

## Quick start

```
1) git clone <repo-url> ozzynet && cd ozzynet
2) make setup        # writes .env from host uid/gid/groups
3) # edit .env and set HSA_OVERRIDE_GFX_VERSION for your GPU (table below)
4) make sync         # creates .venv/ with CPU torch + non-torch deps (editor only)
5) make up           # builds and starts the ROCm container
6) make health       # runs src/health.py: GPU detect + CPU/GPU matmul benchmark
```

To generate the spectrogram dataset once you're ready to train:

```
make spectrograms                                              # default: grayscale PNGs from dataset/songs into dataset/spectrograms
make spectrograms FLAGS="--gtzan"                              # GTZAN layout (gtzan/songs -> gtzan/spectrograms)
make spectrograms FLAGS="--rgb --ft --da"                      # InceptionV3 fine-tuning dataset from dataset/
make spectrograms FLAGS="--gtzan --rgb --ft --da"              # InceptionV3 fine-tuning dataset from gtzan/
```

The generator in `src/spectrograms.py` accepts these flags (combinable):

- `--gtzan`: swap input/output to `gtzan/songs` and `gtzan/spectrograms` (default: `dataset/`).
- `--rgb`: render RGB images via matplotlib (InceptionV3 pipeline).
- `--ft`: resize the output to 299x299.
- `--da`: pitch-shift (+/-1 semitone) and 1-second hop (overlap), otherwise 3-second stride.

To fine-tune Inception v3 on the generated spectrograms:

```
make fine_tuning     # runs src/fine_tuning.py; saves a timestamped checkpoint
BATCH_SIZE=64 EPOCHS=20 make fine_tuning   # override defaults
```

## Configuration: `HSA_OVERRIDE_GFX_VERSION`

ROCm 6.0+ dropped several consumer RDNA cards. Set this in `.env` to the
matching value:

| GPU family          | Models                                | GFX version |
| ------------------- | ------------------------------------- | ----------- |
| RDNA 1              | RX 5700 / 5700 XT                     | `10.1.0`    |
| RDNA 2 (Navi 21-24) | RX 6600 / 6700 XT / 6800 / 6900 XT    | `10.3.0`    |
| RDNA 2 (Navi 31-33) | RX 7900 XT / 7900 XTX                 | `11.0.0`    |
| RDNA 3              | RX 7600 / 7700 XT / 7800 XT / 7900 XT | `11.0.0`    |
| Vega                | RX Vega 56 / 64, Vega VII             | `9.0.0`     |

## Makefile targets

| Target              | What it does                                                                                                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `make setup`        | Writes `.env` from host uid/gid/groups; chmods entrypoint.                                                                                                                                                         |
| `make sync`         | Creates `.venv/`, installs `requirements.txt` deps, adds a CPU-only `torch` wheel for the editor. **Editor only — never run the project from the venv.**                                                           |
| `make up`           | Builds and starts the ROCm container.                                                                                                                                                                              |
| `make down`         | Stops and removes the container.                                                                                                                                                                                   |
| `make shell`        | Opens a bash session inside the container.                                                                                                                                                                         |
| `make health`       | Runs `src/health.py` (GPU detect + CPU/GPU matmul benchmark).                                                                                                                                                      |
| `make spectrograms` | Runs `src/spectrograms.py` to generate the spectrogram dataset. Default reads from `dataset/songs` and writes to `dataset/spectrograms`. Pass flags via `FLAGS="--gtzan"` (use `gtzan/` instead), `FLAGS="--rgb"`, `FLAGS="--ft"`, `FLAGS="--da"` — all combinable. See `docs/dataset.md` for how it works. |
| `make fine_tuning`  | Runs `src/fine_tuning.py` to fine-tune Inception v3 on the spectrogram dataset. Saves a timestamped checkpoint to `checkpoints/YYYYMMDD_HHMMSS/`. Override defaults with `BATCH_SIZE=N EPOCHS=N make fine_tuning`. |
| `make clean`        | Stops the container; removes `.env`, `.venv/`, and Python bytecode caches.                                                                                                                                         |

## Project structure

```
.
├── docker-compose.yml    # rocm/pytorch service with /dev/kfd + /dev/dri passthrough
├── Dockerfile            # extends base image with ffmpeg + pip install requirements.txt
├── entrypoint.sh         # creates matching groups, chroots to host user
├── makefile
├── requirements.txt      # pydub, librosa, matplotlib, numpy (torch from base image)
├── src/
│   ├── health.py         # GPU detect + CPU/GPU matmul benchmark
│   ├── spectrograms.py   # generates spectrograms/ from data/genres_original/
│   ├── dataset.py        # DatasetFT class with train/val splits (used by fine_tuning.py)
│   ├── cnn.py            # InceptionV3 wrapper for fine-tuning (frozen backbone, 10-class head)
│   └── fine_tuning.py    # fine-tuning loop (train, validate, save checkpoint)
├── docs/                 # architectural notes (dataset.md, fine_tuning.md)
├── checkpoints/          # generated by `make fine_tuning` (timestamped subdirs)
├── .env                  # generated by `make setup`, gitignored
├── .venv/                # generated by `make sync`, gitignored (editor only)
└── pyrightconfig.json    # optional; points Pyright at the editor venv
```

## IDE / editor setup

Code runs inside the container; your editor runs on the host. To silence
"Import 'torch' could not be resolved" diagnostics, run `make sync`. It
creates a `.venv/` with a CPU-only `torch` wheel matching the API surface
of the ROCm build, so autocomplete, hover, and go-to-definition all work
once your editor is pointed at `.venv/bin/python`.

**Don't run the project from the venv.**

## Troubleshooting

### No GPU detected

`torch.cuda.is_available()` returned False.

1. `ls /dev/kfd /dev/dri/` on the host (must list `kfd` and `renderD*`).
2. `rocm-smi` on the host must show your card.
3. `make shell`, then `ls /dev/kfd /dev/dri/` inside the container.
4. Check `HSA_OVERRIDE_GFX_VERSION` in `.env` matches your GPU.

### Segfault during GPU benchmark

GPU detected but the ROCm userspace isn't compiled for its ISA. Either
set `HSA_OVERRIDE_GFX_VERSION` to the matching value, or pin an older
`DOCKER_IMAGE` (e.g. `rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1`).

### `docker compose` fails with "permission denied"

Add yourself to the `docker` group (`sudo usermod -aG docker $USER`,
then log out/in) or prefix commands with `sudo`.
