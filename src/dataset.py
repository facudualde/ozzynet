import random
from pathlib import Path
import torch
from torchvision.transforms import v2 as transforms
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision.models import Inception_V3_Weights

class GTZANDataset(TorchDataset):
    """Dataset GTZAN estructurado por canciones para evitar Data Leakage."""

    GENRES = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock",
    ]
    GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRES)}

    def __init__(
        self,
        root: str = "spectrograms",
        split: str = "train",  # Puede ser "train" o "val"
        val_ratio: float = 0.2,
        seed: int = 42,
        t: transforms.Transform | None = None,
    ) -> None:
        assert split in ["train", "val"], "El parámetro 'split' debe ser 'train' o 'val'"
        
        if t is None:
            inceptionV3_w_t = Inception_V3_Weights.DEFAULT.transforms()
            t = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(inceptionV3_w_t.mean, inceptionV3_w_t.std),
            ])
            
        self.root = Path(root)
        self.t: transforms.Transform = t
        self.split = split

        # Construir y filtrar las muestras según el split de forma atómica
        self.samples = self._build_split_samples(val_ratio, seed)

        if not self.samples:
            raise FileNotFoundError(f"No se encontraron espectrogramas para el split '{split}' en {self.root}")

    def _build_split_samples(self, val_ratio: float, seed: int) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        rng = random.Random(seed)

        for genre in self.GENRES:
            genre_dir = self.root / genre
            if not genre_dir.is_dir():
                continue

            # Obtener nombres de carpetas de canciones (ej: "blues.00000")
            song_names = sorted([d.name for d in genre_dir.iterdir() if d.is_dir()])
            rng.shuffle(song_names)

            # Separar estrictamente los nombres de las canciones
            val_len = int(len(song_names) * val_ratio)
            val_song_names = set(song_names[:val_len])

            for song_name in song_names:
                # Filtrar a nivel de canción según el split solicitado en el constructor
                if self.split == "val" and song_name not in val_song_names:
                    continue
                if self.split == "train" and song_name in val_song_names:
                    continue

                song_dir = genre_dir / song_name
                images = sorted(song_dir.glob("*.png"))
                for img_path in images:
                    samples.append((str(img_path), self.GENRE_TO_IDX[genre]))

        print(f"Instanciado split '{self.split}': {len(samples)} imágenes cargadas.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.t(image)
        return image, label