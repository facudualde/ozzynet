import os
from concurrent.futures import ProcessPoolExecutor
import librosa
import numpy as np
from PIL import Image

# Source: 1000 .wav files laid out as data/genres_original/<genre>/<song>.wav
INPUT_DIR = "data/genres_original"
# Sink:   spectrograms/<genre>/<song>/<chunk>.png (one PNG per 3-second chunk)
OUTPUT_DIR = "spectrograms"
WINDOW_LENGTH_MS = 3000
# === NUEVA CONFIGURACIÓN DE OVERLAP ===
HOP_LENGTH_MS = 1000  # Nos desplazamos 1 segundo en cada paso (Overlap de 2 segundos)
# ======================================
IMG_SIZE_INCHES = (2.99, 2.99)
DPI = 100
MAX_WORKERS = 4


def generate_spectrogram(y: np.ndarray, sr: int, output_path: str) -> None:
  # Usar escala Mel con 128 bancos de filtros
  mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
  mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
  fig, ax = plt.subplots(figsize=IMG_SIZE_INCHES, dpi=DPI)
  librosa.display.specshow(mel_spec_db, sr=sr, x_axis=None, y_axis=None, ax=ax, fmax=sr/2)
  ax.set_axis_off()
  ax.set_aspect("auto")
  fig.savefig(output_path, dpi=DPI, pad_inches=0)
  plt.close(fig)

  # Rescale from dB ([-TOP_DB, 0]) to uint8 ([0, 255]) so PIL can save it.
  # The math: shift by +TOP_DB so values are in [0, TOP_DB], divide by
  # TOP_DB to land in [0, 1], multiply by 255, clip and cast.
  mel_img = ((mel_db + TOP_DB) / TOP_DB * 255).clip(0, 255).astype(np.uint8)

  # Invert so high-energy regions become bright (255) and silence becomes
  # dark (0). This matches the convention ConvNet was trained against
  # — inverting once now vs. inverting every time at training.
  mel_img = 255 - mel_img

  # "L" mode = single-channel grayscale PNG. Small files, fast disk I/O.
  Image.fromarray(mel_img, mode="L").save(output_path)


def process_song(wav_path: str, song_output_dir: str) -> tuple[str, str | None]:
  # Load the whole song into memory at once (max ~30 s ≈ 660 k samples).
  # sr=None preserves the file's native sample rate (22050 for GTZAN).
  # mono=True mixes down to one channel — mono spectrograms are sufficient
  # for genre classification and keep the network input 1-channel.
  try:
    y_full, sr = librosa.load(wav_path, sr=None, mono=True)
  except Exception as exc:
    # GTZAN ships a few known-corrupt files (e.g. jazz.00054.wav) that fail
    # to decode. Return (wav_path, None) so main() can skip the success log;
    # the dataset class already tolerates missing songs.
    print(f"[WARN] failed to load {wav_path}: {exc}")
    return wav_path, None

def process_song(wav_path: str, song_output_dir: str) -> tuple[str, str]:
    # 1. Cargar el audio original completo
    y_full, sr = librosa.load(wav_path, sr=None)
    sr = int(sr)
    
    # 2. Calcular tamaños en número de muestras (samples)
    samples_per_window = int(WINDOW_LENGTH_MS / 1000 * sr)
    samples_per_hop = int(HOP_LENGTH_MS / 1000 * sr)

    # 3. Generar las variaciones de tono completas
    y_pitch_up = librosa.effects.pitch_shift(y_full, sr=sr, n_steps=1.0)
    y_pitch_down = librosa.effects.pitch_shift(y_full, sr=sr, n_steps=-1.0)

    variants = {
        "original": y_full,
        "pitch_up": y_pitch_up,
        "pitch_down": y_pitch_down
    }

    # 4. Procesar los segmentos usando la ventana móvil con Overlap
    start_sample = 0
    segment_count = 0
    
    # Determinamos la longitud total (todas las variantes miden lo mismo)
    total_samples = len(y_full)

    # El bucle avanza mientras podamos extraer una ventana completa de 3 segundos
    while start_sample + samples_per_window <= total_samples:
        end_sample = start_sample + samples_per_window

        for suffix, y_audio in variants.items():
            y_segment = y_audio[start_sample:end_sample]

            # El nombre guarda el índice del segmento correlativo y su variante
            output_path = os.path.join(song_output_dir, f"{segment_count}_{suffix}.png")
            generate_spectrogram(y_segment, sr, output_path)

        # DESPLAZAMIENTO: Avanzamos solo el tamaño del HOP (1 segundo) en vez de la ventana completa
        start_sample += samples_per_hop
        segment_count += 1

    return wav_path, song_output_dir


def main() -> None:
    try:
        if os.path.exists(OUTPUT_DIR):
            os.system(f"chown -R $(id -u):$(id -g) {OUTPUT_DIR} 2>/dev/null")
    except Exception:
        pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    work_items: list[tuple[str, str]] = []
    for genre in sorted(os.listdir(INPUT_DIR)):
        genre_input_dir = os.path.join(INPUT_DIR, genre)
        if not os.path.isdir(genre_input_dir):
            continue

        genre_output_dir = os.path.join(OUTPUT_DIR, genre)
        os.makedirs(genre_output_dir, exist_ok=True)

        for wav_file in sorted(os.listdir(genre_input_dir)):
            if not wav_file.endswith(".wav"):
                continue

            wav_path = os.path.join(genre_input_dir, wav_file)
            song_name = os.path.splitext(wav_file)[0]
            song_output_dir = os.path.join(genre_output_dir, song_name)
            os.makedirs(song_output_dir, exist_ok=True)

            work_items.append((wav_path, song_output_dir))

    print(f"Iniciando procesamiento de {len(work_items)} canciones con Overlap y Pitch Augmentation...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for wav_path, song_out in executor.map(process_song, *zip(*work_items)):
            genre = os.path.basename(os.path.dirname(song_out))
            song_name = os.path.basename(song_out)
            print(f"Processed (Overlap x3): {genre}/{song_name}")


if __name__ == "__main__":
    main()