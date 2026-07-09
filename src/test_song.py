import os
import sys
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from torchvision.transforms import v2 as transforms
from torchvision.models import Inception_V3_Weights
from cnn import InceptionV3

# Configuración idéntica al entrenamiento
WINDOW_LENGTH_MS = 3000
IMG_SIZE_INCHES = (2.99, 2.99)
DPI = 100
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def audio_to_spectrogram_tensor(y_segment: np.ndarray, sr: int, transform: transforms.Transform) -> torch.Tensor:
    """Convierte un segmento de audio en el tensor que espera InceptionV3 sin guardar en disco."""
    stft = librosa.stft(y_segment)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # Renderizar temporalmente en memoria usando matplotlib
    fig, ax = plt.subplots(figsize=IMG_SIZE_INCHES, dpi=DPI)
    librosa.display.specshow(stft_db, sr=sr, x_axis=None, y_axis=None, ax=ax)
    ax.set_axis_off()
    ax.set_aspect("auto")
    
    fig.canvas.draw()
    # Convertir el render de matplotlib a una imagen PIL RGB
    rgba_buffer = fig.canvas.buffer_rgba()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), rgba_buffer, "raw", "RGBA", 0, 1).convert("RGB")
    plt.close(fig)

    # Aplicar las transformaciones de InceptionV3
    return transform(img)

def main():
    if len(sys.argv) < 3:
        print("Uso: docker compose exec pytorch python3 src/test_song.py <ruta_audio> <ruta_checkpoint.pth>")
        sys.exit(1)

    audio_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    # 1. Configurar transformaciones de InceptionV3
    inceptionV3_w_t = Inception_V3_Weights.DEFAULT.transforms()
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(inceptionV3_w_t.mean, inceptionV3_w_t.std),
    ])

    # 2. Cargar el modelo entrenado
    print(f"Cargando modelo desde {checkpoint_path}...")
    model = InceptionV3().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 3. Cargar y segmentar el audio
    print(f"Cargando audio: {audio_path}...")
    y_full, sr = librosa.load(audio_path, sr=None)
    sr = int(sr)
    
    samples_per_window = int(WINDOW_LENGTH_MS / 1000 * sr)
    total_samples = len(y_full)
    
    # Calcular cuántos fragmentos de 3 segundos salen (sin solapamiento)
    num_segments = total_samples // samples_per_window
    if num_segments == 0:
        print("Error: El audio es demasiado corto (debe durar al menos 3 segundos).")
        sys.exit(1)

    print(f"Procesando canción en {num_segments} fragmentos de 3 segundos...")
    
    segment_predictions = []
    
    # 4. Pasar cada segmento por la red
    with torch.no_grad():
        for count in range(num_segments):
            start_sample = count * samples_per_window
            end_sample = start_sample + samples_per_window
            y_segment = y_full[start_sample:end_sample]

            # Generar el tensor directamente en memoria (sin escribir PNGs lentos en disco)
            x_tensor = audio_to_spectrogram_tensor(y_segment, sr, transform)
            x_tensor = x_tensor.unsqueeze(0).to(device) # Añadir dimensión de batch [1, 3, 299, 299]

            # Predicción
            output = model(x_tensor)
            pred_idx = output.argmax(1).item()
            segment_predictions.append(pred_idx)
            
            print(f"  -> Fragmento {count+1}/{num_segments}: Clasificado como '{GENRES[pred_idx]}'")

    # 5. Resolver por votación (Moda)
    votes = Counter(segment_predictions)
    winner_idx, winner_count = votes.most_common(1)[0]
    confidence = (winner_count / num_segments) * 100

    print("\n" + "="*40)
    print(" RESULTADO DE LA VOTACIÓN")
    print("="*40)
    for idx, count in votes.items():
        print(f" - {GENRES[idx]}: {count} votos ({(count/num_segments)*100:.1f}%)")
    print("-" * 40)
    print(f" GÉNERO PREDICHO FINAL: {GENRES[winner_idx].upper()} ({confidence:.1f}% de consenso)")
    print("="*40)

if __name__ == "__main__":
    main()