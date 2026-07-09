import os
from concurrent.futures import ProcessPoolExecutor
import librosa
import numpy as np
from PIL import Image

# Source: 1000 .wav files laid out as data/genres_original/<genre>/<song>.wav
INPUT_DIR = "data/genres_original"
# Sink:   spectrograms/<genre>/<song>/<chunk>.png (one PNG per 3-second chunk)
OUTPUT_DIR = "spectrograms"

# Canonical GTZAN chunking: 30 seconds / 10 chunks = 3 seconds per chunk.
# Each chunk becomes one training sample; the per-chunk voting happens later
# in train.py to recover a song-level decision.
WINDOW_SECONDS    = 3.0
STRIDE_SECONDS = 3.0  # Step = 3
SONG_DURATION  = 30.0

SEGMENTS_PER_SONG = int((SONG_DURATION - WINDOW_SECONDS) / STRIDE_SECONDS) + 1

# Mel-spectrogram parameters tuned to:
#   - produce a ~128x129 image per chunk (matches ConvNet's resize target),
#   - keep the genre-discriminative band (vocals, drums, guitars live < 8 kHz),
#   - keep file size reasonable on disk (uint8 grayscale PNGs).
#
# n_fft=2048      → window length. At sr=22050, that's ~93 ms — long enough
#                   to resolve pitch in vocals/bass but short enough to track
#                   percussive onsets.
# hop_length=512  → stride. 75% overlap between windows, which yields ~129
#                   time frames per 3-second chunk.
# n_mels=128      → number of frequency bins. Matches the height we resize to
#                   in dataset.py.
# fmin=20, fmax=8 → keep the band where music energy actually lives; below
#                   20 Hz is rumble, above 8 kHz is mostly cymbals/air.
# top_db=80       → dynamic range cap. Anything 80 dB below the peak is
#                   clipped to silence before saving, since storing those
#                   near-zero floats would burn bytes for no ML signal.
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 128
FMIN       = 20
FMAX       = 8000
TOP_DB     = 80

# CPU-bound workload (librosa + PIL); 4 workers is a reasonable default
# for a 4-core machine. Tune via env var if running on bigger boxes.
MAX_WORKERS = 4

def generate_spectrogram(y: np.ndarray, sr: int, output_path: str) -> None:
  # Compute the mel-scaled power spectrogram.
  # power=2.0 means amplitude squared (energy), which is what power_to_db
  # expects to convert into decibels below.
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

  # Convert to decibels (perceived loudness scale), clipping anything
  # more than TOP_DB below the peak so the PNG won't waste bits on
  # values that are essentially silent.
  mel_db = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)

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

  # librosa returns numpy scalars (np.int64); downstream ints want Python int.
  sr = int(sr)

  # Number of audio samples that fit in one 3-second chunk. This is the
  # "slice size" used below to index into y_full.
  samples_per_window = int(WINDOW_SECONDS * sr)
  samples_per_stride = int(STRIDE_SECONDS * sr)

  # Walk the song in 3-second steps and emit a PNG per chunk.
  for i in range(SEGMENTS_PER_SONG):
    start = i * samples_per_stride
    end   = start + samples_per_window

    if end > len(y_full):
      # Last chunk would overrun the song (file slightly shorter than
      # 30 s on disk). Pad with zeros so the chunk has the expected length.
      # The padding shows up as a dark vertical band at the end of the
      # spectrogram; the network learns to ignore it.
      y_segment = np.pad(y_full[start:], (0, end - len(y_full)))
    else:
      y_segment = y_full[start:end]

    # Filename convention the dataset class relies on: 0..9.png per song.
    output_path = os.path.join(song_output_dir, f"{i}.png")
    generate_spectrogram(y_segment, sr, output_path)

  return wav_path, song_output_dir


def main() -> None:
  # Top-level output dir; idempotent so we don't crash on re-runs.
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  # Flat work list of (wav_path, song_output_dir) tuples dispatched to workers.
  work_items: list[tuple[str, str]] = []

  # Walk genres alphabetically, then songs alphabetically inside each genre.
  # Ordering doesn't matter for correctness — workers just pull from a queue.
  for genre in sorted(os.listdir(INPUT_DIR)):
    genre_input_dir = os.path.join(INPUT_DIR, genre)
    if not os.path.isdir(genre_input_dir):
      continue

    # Pre-create the genre output directory so workers don't fight for
    # mkdir and we fail fast if the disk is full / permissions are wrong.
    genre_output_dir = os.path.join(OUTPUT_DIR, genre)
    os.makedirs(genre_output_dir, exist_ok=True)

    for wav_file in sorted(os.listdir(genre_input_dir)):
      if not wav_file.endswith(".wav"):
        continue

      wav_path        = os.path.join(genre_input_dir, wav_file)
      song_name       = os.path.splitext(wav_file)[0]
      song_output_dir = os.path.join(genre_output_dir, song_name)

      # Pre-create the per-song directory for the same reason as above.
      os.makedirs(song_output_dir, exist_ok=True)

      work_items.append((wav_path, song_output_dir))

  # Friendly banner so the user sees the run kicked off.
  print(
    f"Processing {len(work_items)} songs with {MAX_WORKERS} workers "
    f"({SEGMENTS_PER_SONG} chunks of {WINDOW_SECONDS}s per song)..."
  )

  # ProcessPoolExecutor forks worker processes (avoids the GIL).
  # executor.map returns results in submission order, so output stays
  # alphabetical regardless of which worker actually ran each song.
  with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for _, song_out in executor.map(process_song, *zip(*work_items)):
      # process_song returns None for corrupt files; skip those in the log.
      if song_out is None:
        continue
      genre    = os.path.basename(os.path.dirname(song_out))
      song_name = os.path.basename(song_out)
      print(f"Processed: {genre}/{song_name}")


if __name__ == "__main__":
  main()
