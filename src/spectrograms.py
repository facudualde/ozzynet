import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import librosa
import matplotlib

matplotlib.use("Agg")  # no display; safe inside containers
import matplotlib.cm as cm
import numpy as np
from PIL import Image

# Two presets: default reads from dataset/songs -> dataset/spectrograms;
# --gtzan swaps to the canonical gtzan/songs -> gtzan/spectrograms layout.
# --type test swaps <root>/songs -> <root>/test/songs for held-out eval songs.
DATASET_INPUT_DIR = "dataset/songs"
DATASET_OUTPUT_DIR = "dataset/spectrograms"
GTZAN_INPUT_DIR = "gtzan/songs"
GTZAN_OUTPUT_DIR = "gtzan/spectrograms"
DATASET_TEST_INPUT_DIR = "dataset/test/songs"
DATASET_TEST_OUTPUT_DIR = "dataset/test/spectrograms"
GTZAN_TEST_INPUT_DIR = "gtzan/test/songs"
GTZAN_TEST_OUTPUT_DIR = "gtzan/test/spectrograms"

# Canonical GTZAN chunking: 3 seconds per chunk.
WINDOW_SECONDS = 3.0
DEFAULT_HOP_SECONDS = 3.0
# Force sample rate so train and test chunks share dimensions regardless of input.
SR = 22050

# Mel parameters: see docs in the original one-channel script.
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 8000
TOP_DB = 80

# Fine-tuning (InceptionV3) image size.
FT_IMG_SIZE = 299

# Matplotlib output size for the RGB pipeline (~299x299 at dpi=100).
IMG_SIZE_INCHES = (2.99, 2.99)
DPI = 100

MAX_WORKERS = 4


@dataclass(frozen=True)
class JobConfig:
    # Immutable bundle passed to each worker; avoids wide positional tuples.
    hop_seconds: float
    rgb: bool
    ft: bool
    pitch: bool


def mel_db(y: np.ndarray, sr: int) -> np.ndarray:
    # Single place where the mel spectrogram is computed.
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)


def render_grayscale(mel_db_arr: np.ndarray, target_size: tuple[int, int] | None) -> np.ndarray:
    # Map dB ([-TOP_DB, 0]) to uint8 ([0, 255]) and invert so energy = bright.
    img = ((mel_db_arr + TOP_DB) / TOP_DB * 255).clip(0, 255).astype(np.uint8)
    img = 255 - img
    if target_size is not None:
        img = np.array(Image.fromarray(img, mode="L").resize(target_size, Image.BILINEAR))
    return img


def render_rgb(mel_db_arr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    # Map dB to [0, 1] and apply viridis directly — avoids the matplotlib
    # round-trip to a PNG file, which broke inside ProcessPoolExecutor workers.
    normalized = ((mel_db_arr + TOP_DB) / TOP_DB).clip(0, 1)
    rgba = cm.viridis(normalized)  # (H, W, 4) float in [0, 1]
    rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
    if target_size is not None:
        rgb = np.array(Image.fromarray(rgb, mode="RGB").resize(target_size, Image.BILINEAR))
    return rgb


def save_spectrogram(
    y: np.ndarray,
    sr: int,
    output_path: str,
    cfg: JobConfig,
) -> None:
    # Compute mel once; pick the renderer + size based on flags.
    mel_db_arr = mel_db(y, sr)
    target_size = (FT_IMG_SIZE, FT_IMG_SIZE) if cfg.ft else None
    if cfg.rgb:
        rgb_img = render_rgb(mel_db_arr, target_size or (mel_db_arr.shape[1], mel_db_arr.shape[0]))
        Image.fromarray(rgb_img, mode="RGB").save(output_path)
    else:
        gray = render_grayscale(mel_db_arr, target_size)
        Image.fromarray(gray, mode="L").save(output_path)


def pitch_variants(y_full: np.ndarray, sr: int, cfg: JobConfig) -> dict[str, np.ndarray]:
    # Only --pitch produces the extra pitch-shifted copies.
    if not cfg.pitch:
        return {"original": y_full}
    return {
        "original": y_full,
        "pitch_up": librosa.effects.pitch_shift(y_full, sr=sr, n_steps=1.0),
        "pitch_down": librosa.effects.pitch_shift(y_full, sr=sr, n_steps=-1.0),
    }


def clean_song_dir(song_output_dir: str) -> None:
    # Wipe any leftover PNGs so re-runs don't mix old + new naming schemes.
    if not os.path.isdir(song_output_dir):
        return
    for entry in os.listdir(song_output_dir):
        path = os.path.join(song_output_dir, entry)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)


def process_song(args: tuple[str, str, JobConfig]) -> tuple[str, str | None]:
    # Unpack worker args; JobConfig carries the per-run flags.
    song_path, song_output_dir, cfg = args

    try:
        y_full, sr = librosa.load(song_path, sr=SR, mono=True)
    except Exception as exc:
        print(f"[WARN] failed to load {song_path}: {exc}")
        return song_path, None

    sr = int(sr)
    samples_per_window = int(WINDOW_SECONDS * sr)
    samples_per_hop = int(cfg.hop_seconds * sr)

    clean_song_dir(song_output_dir)
    os.makedirs(song_output_dir, exist_ok=True)

    # Walk the song until a full 3-second window no longer fits.
    segment_idx = 0
    start = 0
    while start + samples_per_window <= len(y_full):
        end = start + samples_per_window
        for suffix, y_audio in pitch_variants(y_full[start:end], sr, cfg).items():
            output_path = os.path.join(
                song_output_dir,
                f"{segment_idx}_{suffix}.png" if cfg.pitch else f"{segment_idx}.png",
            )
            save_spectrogram(y_audio, sr, output_path, cfg)
        start += samples_per_hop
        segment_idx += 1

    return song_path, song_output_dir


def _paths(type_: str, gtzan: bool) -> tuple[str, str]:
    # Resolve (input_dir, output_dir) from the --type/--gtzan combination.
    if type_ == "test":
        return (GTZAN_TEST_INPUT_DIR, GTZAN_TEST_OUTPUT_DIR) if gtzan else (DATASET_TEST_INPUT_DIR, DATASET_TEST_OUTPUT_DIR)
    return (GTZAN_INPUT_DIR, GTZAN_OUTPUT_DIR) if gtzan else (DATASET_INPUT_DIR, DATASET_OUTPUT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mel-spectrogram PNGs from GTZAN.")
    parser.add_argument("--rgb", action="store_true", help="Render RGB images (InceptionV3 style).")
    parser.add_argument("--ft", action="store_true", help="Resize output to 299x299 for fine-tuning.")
    parser.add_argument("--pitch", action="store_true", help="Pitch augmentation (+/-1 semitone). Does not affect hop.")
    parser.add_argument(
        "--hop",
        type=float,
        default=DEFAULT_HOP_SECONDS,
        help="Stride seconds (default 3.0). Range (0, 3.0].",
    )
    parser.add_argument(
        "--type",
        choices=["train", "test"],
        default="train",
        help="train reads <root>/songs, test reads <root>/test/songs (held-out eval).",
    )
    parser.add_argument(
        "--gtzan",
        action="store_true",
        help="Read from gtzan/(test/)songs and write to gtzan/(test/)spectrograms (default: dataset/...).",
    )
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="Parallel worker count.")
    args = parser.parse_args()
    if not 0 < args.hop <= WINDOW_SECONDS:
        parser.error(f"--hop must be in (0, {WINDOW_SECONDS}]")
    return args


def main() -> None:
    args = parse_args()

    # --pitch is meaningless on held-out test songs; warn and force-disable instead of erroring out.
    pitch_effective = args.pitch and args.type == "train"
    if args.pitch and args.type == "test":
        print("[WARN] --pitch is ignored with --type test (no augmentation on test songs)")

    input_dir, output_dir = _paths(args.type, args.gtzan)
    cfg = JobConfig(
        hop_seconds=args.hop,
        rgb=args.rgb,
        ft=args.ft,
        pitch=pitch_effective,
    )

    try:  # mirror the old script's "best-effort chown" for Docker volumes
        if os.path.exists(output_dir):
            os.system(f"chown -R $(id -u):$(id -g) {output_dir} 2>/dev/null")
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)

    work_items: list[tuple[str, str, JobConfig]] = []
    for genre in sorted(os.listdir(input_dir)):
        genre_input_dir = os.path.join(input_dir, genre)
        if not os.path.isdir(genre_input_dir):
            continue

        genre_output_dir = os.path.join(output_dir, genre)
        os.makedirs(genre_output_dir, exist_ok=True)

        for song_file in sorted(os.listdir(genre_input_dir)):
            if not (song_file.endswith(".wav") or song_file.endswith(".mp3")):
                continue

            song_path = os.path.join(genre_input_dir, song_file)
            song_name = os.path.splitext(song_file)[0]
            song_output_dir = os.path.join(genre_output_dir, song_name)
            os.makedirs(song_output_dir, exist_ok=True)

            work_items.append((song_path, song_output_dir, cfg))

    mode = "RGB" if cfg.rgb else "L"
    size = f"{FT_IMG_SIZE}x{FT_IMG_SIZE}" if cfg.ft else "native"
    preset = "gtzan" if args.gtzan else "dataset"
    print(
        f"Processing {len(work_items)} songs ({preset}/{args.type}) with {args.max_workers} workers "
        f"({mode} {size}, pitch={'on' if cfg.pitch else 'off'}, hop={cfg.hop_seconds}s)..."
    )

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for _, song_out in executor.map(process_song, work_items):
            if song_out is None:
                continue
            genre = os.path.basename(os.path.dirname(song_out))
            song_name = os.path.basename(song_out)
            print(f"Processed: {genre}/{song_name}")


if __name__ == "__main__":
    main()
