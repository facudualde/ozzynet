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

# Presets de directorios
DATASET_INPUT_DIR = "dataset/songs"
DATASET_OUTPUT_DIR = "dataset/spectrograms"
GTZAN_INPUT_DIR = "gtzan/songs"
GTZAN_OUTPUT_DIR = "gtzan/spectrograms"
DATASET_TEST_INPUT_DIR = "dataset/test/songs"
DATASET_TEST_OUTPUT_DIR = "dataset/test/spectrograms"
GTZAN_TEST_INPUT_DIR = "gtzan/test/songs"
GTZAN_TEST_OUTPUT_DIR = "gtzan/test/spectrograms"

WINDOW_SECONDS = 3.0
DEFAULT_HOP_SECONDS = 3.0
SR = 22050

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 8000
TOP_DB = 80

FT_IMG_SIZE = 299
IMG_SIZE_INCHES = (2.99, 2.99)
DPI = 100
MAX_WORKERS = 4


@dataclass(frozen=True)
class JobConfig:
    hop_seconds: float
    rgb: bool
    ft: bool
    pitch: bool
    # === NUEVOS PARÁMETROS ===
    target_samples: int | None  # Cuántos fragmentos fijos queremos
    is_gtzan: bool              # Saber si estamos usando GTZAN


def mel_db(y: np.ndarray, sr: int) -> np.ndarray:
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
    img = ((mel_db_arr + TOP_DB) / TOP_DB * 255).clip(0, 255).astype(np.uint8)
    img = 255 - img
    if target_size is not None:
        img = np.array(Image.fromarray(img, mode="L").resize(target_size, Image.BILINEAR))
    return img


def render_rgb(mel_db_arr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    normalized = ((mel_db_arr + TOP_DB) / TOP_DB).clip(0, 1)
    rgba = cm.viridis(normalized)
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
    mel_db_arr = mel_db(y, sr)
    target_size = (FT_IMG_SIZE, FT_IMG_SIZE) if cfg.ft else None
    if cfg.rgb:
        rgb_img = render_rgb(mel_db_arr, target_size or (mel_db_arr.shape[1], mel_db_arr.shape[0]))
        Image.fromarray(rgb_img, mode="RGB").save(output_path)
    else:
        gray = render_grayscale(mel_db_arr, target_size)
        Image.fromarray(gray, mode="L").save(output_path)


def pitch_variants(y_full: np.ndarray, sr: int, cfg: JobConfig) -> dict[str, np.ndarray]:
    if not cfg.pitch:
        return {"original": y_full}
    return {
        "original": y_full,
        "pitch_up": librosa.effects.pitch_shift(y_full, sr=sr, n_steps=1.0),
        "pitch_down": librosa.effects.pitch_shift(y_full, sr=sr, n_steps=-1.0),
    }


def clean_song_dir(song_output_dir: str) -> None:
    if not os.path.isdir(song_output_dir):
        return
    for entry in os.listdir(song_output_dir):
        path = os.path.join(song_output_dir, entry)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)


def process_song(args: tuple[str, str, JobConfig]) -> tuple[str, str | None]:
    song_path, song_output_dir, cfg = args

    try:
        y_full, sr = librosa.load(song_path, sr=SR, mono=True)
    except Exception as exc:
        print(f"[WARN] failed to load {song_path}: {exc}")
        return song_path, None

    sr = int(sr)
    samples_per_window = int(WINDOW_SECONDS * sr)
    total_samples = len(y_full)

    if total_samples < samples_per_window:
        print(f"[WARN] song too short {song_path}")
        return song_path, None

    clean_song_dir(song_output_dir)
    os.makedirs(song_output_dir, exist_ok=True)

    # === LÓGICA DE PARTICIÓN ADAPTATIVA VS TRADICIONAL ===
    if not cfg.is_gtzan and cfg.target_samples is not None:
        # Modo adaptativo puro para tu nuevo dataset personalizado
        available_space = total_samples - samples_per_window
        samples_per_hop = available_space / (cfg.target_samples - 1) if available_space > 0 else 0

        for step in range(cfg.target_samples):
            start_sample = int(round(step * samples_per_hop))
            end_sample = start_sample + samples_per_window

            if end_sample > total_samples:
                end_sample = total_samples
                start_sample = end_sample - samples_per_window

            y_segment = y_full[start_sample:end_sample]
            for suffix, y_audio in pitch_variants(y_segment, sr, cfg).items():
                output_path = os.path.join(
                    song_output_dir,
                    f"{step}_{suffix}.png" if cfg.pitch else f"{step}.png",
                )
                save_spectrogram(y_audio, sr, output_path, cfg)
    else:
        # Modo tradicional secuencial (Siempre usado en GTZAN o si omitís --samples)
        samples_per_hop = int(cfg.hop_seconds * sr)
        segment_idx = 0
        start = 0
        while start + samples_per_window <= total_samples:
            end = start + samples_per_window
            y_segment = y_full[start:end]
            for suffix, y_audio in pitch_variants(y_segment, sr, cfg).items():
                output_path = os.path.join(
                    song_output_dir,
                    f"{segment_idx}_{suffix}.png" if cfg.pitch else f"{segment_idx}.png",
                )
                save_spectrogram(y_audio, sr, output_path, cfg)
            start += samples_per_hop
            segment_idx += 1

    return song_path, song_output_dir


def _paths(type_: str, gtzan: bool) -> tuple[str, str]:
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
        help="Stride seconds (default 3.0). Range (0, 3.0]. Ignored if not using --gtzan and --samples is specified.",
    )
    # === ARGUMENTO NUEVO ===
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Target number of pure samples per song using adaptive hop (Ignored if --gtzan is active).",
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
    if args.samples <= 0:
        parser.error("--samples must be a positive integer greater than 0")
        
    return args


def main() -> None:
    args = parse_args()

    pitch_effective = args.pitch and args.type == "train"
    if args.pitch and args.type == "test":
        print("[WARN] --pitch is ignored with --type test (no augmentation on test songs)")

    # Imprimir advertencia si el usuario intenta combinar --hop y --samples de forma inválida
    if not args.gtzan and args.hop != DEFAULT_HOP_SECONDS:
        print(f"[WARN] --hop={args.hop}s is ignored because custom dataset mode is active. "
              f"Using adaptive calculation to get exactly {args.samples} samples.")

    input_dir, output_dir = _paths(args.type, args.gtzan)
    
    cfg = JobConfig(
        hop_seconds=args.hop,
        rgb=args.rgb,
        ft=args.ft,
        pitch=pitch_effective,
        target_samples=args.samples,
        is_gtzan=args.gtzan,
    )

    try:  
        if os.path.exists(output_dir):
            os.system(f"chown -R $(id -u):$(id -g) {output_dir} 2>/dev/null")
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)

    work_items: list[tuple[str, str, JobConfig]] = []
    
    if not os.path.exists(input_dir):
        print(f"[Error] Source directory '{input_dir}' does not exist.")
        return

    for genre in sorted(os.listdir(input_dir)):
        genre_input_dir = os.path.join(input_dir, genre)
        if not os.path.isdir(genre_input_dir):
            continue

        genre_output_dir = os.path.join(output_dir, genre)
        os.makedirs(genre_output_dir, exist_ok=True)

        for song_file in sorted(os.listdir(genre_input_dir)):
            if not (song_file.lower().endswith(".wav") or song_file.lower().endswith(".mp3")):
                continue

            song_path = os.path.join(genre_input_dir, song_file)
            song_name = os.path.splitext(song_file)[0]
            song_output_dir = os.path.join(genre_output_dir, song_name)
            os.makedirs(song_output_dir, exist_ok=True)

            work_items.append((song_path, song_output_dir, cfg))

    mode = "RGB" if cfg.rgb else "L"
    size = f"{FT_IMG_SIZE}x{FT_IMG_SIZE}" if cfg.ft else "native"
    preset = "gtzan" if args.gtzan else "dataset"
    
    # Ajustar el mensaje de inicio en la terminal según el método elegido
    if args.gtzan:
        strategy_str = f"hop={cfg.hop_seconds}s"
    else:
        strategy_str = f"adaptive target={cfg.target_samples} segments"

    print(
        f"Processing {len(work_items)} songs ({preset}/{args.type}) with {args.max_workers} workers "
        f"({mode} {size}, pitch={'on' if cfg.pitch else 'off'}, {strategy_str})..."
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