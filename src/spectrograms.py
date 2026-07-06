import os
from concurrent.futures import ProcessPoolExecutor

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = "data/genres_original"
OUTPUT_DIR = "spectrograms"
WINDOW_LENGTH_MS = 3000
SEGMENTS_PER_SONG = 10
IMG_SIZE_INCHES = (2.99, 2.99)
DPI = 100
MAX_WORKERS = 4


def generate_spectrogram(y: np.ndarray, sr: int, output_path: str) -> None:
    # Usar escala Mel con 128 bancos de filtros (es el estándar de la industria)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=IMG_SIZE_INCHES, dpi=DPI)
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis=None, y_axis=None, ax=ax, fmax=sr/2)
    ax.set_axis_off()
    ax.set_aspect("auto")
    fig.savefig(output_path, dpi=DPI, pad_inches=0)
    plt.close(fig)


def process_song(wav_path: str, song_output_dir: str) -> tuple[str, str]:
    # 1. Cargar el audio original completo
    y_full, sr = librosa.load(wav_path, sr=None)
    sr = int(sr)
    samples_per_window = int(WINDOW_LENGTH_MS / 1000 * sr)

    # 2. Generar las variaciones de tono completas
    # n_steps=1 significa un semitono arriba, n_steps=-1 un semitono abajo
    y_pitch_up = librosa.effects.pitch_shift(y_full, sr=sr, n_steps=1.0)
    y_pitch_down = librosa.effects.pitch_shift(y_full, sr=sr, n_steps=-1.0)

    # Diccionario para iterar fácilmente sobre las 3 variantes
    variants = {
        "original": y_full,
        "pitch_up": y_pitch_up,
        "pitch_down": y_pitch_down
    }

    # 3. Procesar los segmentos para cada variante de la canción
    for count in range(0, SEGMENTS_PER_SONG):
        start_sample = count * samples_per_window
        end_sample = start_sample + samples_per_window

        for suffix, y_audio in variants.items():
            if end_sample > len(y_audio):
                y_segment = np.pad(y_audio[start_sample:], (0, end_sample - len(y_audio)))
            else:
                y_segment = y_audio[start_sample:end_sample]

            # El nombre del archivo ahora incluirá si es original, up o down (ej: 0_original.png, 0_pitch_up.png)
            output_path = os.path.join(song_output_dir, f"{count}_{suffix}.png")
            generate_spectrogram(y_segment, sr, output_path)

    return wav_path, song_output_dir


def main() -> None:
    # Intentar solucionar automáticamente problemas de permisos de root si el script corre en Docker
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

    print(f"Iniciando procesamiento de {len(work_items)} canciones con Data Augmentation...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for wav_path, song_out in executor.map(process_song, *zip(*work_items)):
            genre = os.path.basename(os.path.dirname(song_out))
            song_name = os.path.basename(song_out)
            print(f"Processed (x3 variants): {genre}/{song_name}")


if __name__ == "__main__":
    main()