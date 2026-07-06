import random
from collections import defaultdict
from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision.models import Inception_V3_Weights


class GTZANDataset(TorchDataset):
    """Dataset GTZAN estructurado por canciones.

    Evita el Data Leakage utilizando espectrogramas de Mel estándar y soporta
    submuestreo dinámico de 10 fragmentos por canción por época.
    """

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
            # Transformaciones requeridas por InceptionV3 (ImageNet stats)
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

        # 1. Construir la lista maestra con todas las muestras asignadas a este split
        self.all_samples = self._build_split_samples(val_ratio, seed)

        if not self.all_samples:
            raise FileNotFoundError(f"No se encontraron espectrogramas para el split '{split}' en {self.root}")

        # 2. Inicializar la lista activa que usará __getitem__
        self.samples = list(self.all_samples)
        
        # 3. Si es el conjunto de entrenamiento, ejecutamos el primer filtro aleatorio de 10 fragmentos
        if self.split == "train":
            self.reset_epoch_samples()

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

            # Separar estrictamente las canciones
            val_len = int(len(song_names) * val_ratio)
            val_song_names = set(song_names[:val_len])

            for song_name in song_names:
                # Filtrar a nivel de canción según el split solicitado
                if self.split == "val" and song_name not in val_song_names:
                    continue
                if self.split == "train" and song_name in val_song_names:
                    continue

                song_dir = genre_dir / song_name
                images = sorted(song_dir.glob("*.png"))
                for img_path in images:
                    samples.append((str(img_path), self.GENRE_TO_IDX[genre]))

        print(f"Instanciado split '{self.split}': {len(samples)} imágenes totales en disco.")
        return samples

    def reset_epoch_samples(self) -> None:
        """Selecciona aleatoriamente 10 fragmentos de cada canción para la época actual."""
        if self.split == "val":
            # En validación no queremos aleatoriedad, evaluamos siempre sobre el conjunto completo
            self.samples = list(self.all_samples)
            return 

        sampled_list = []
        songs_dict = defaultdict(list)
        
        # Agrupamos las imágenes disponibles por el nombre de su canción
        for sample in self.all_samples: 
            song_name = Path(sample[0]).parent.name
            songs_dict[song_name].append(sample)
            
        # De cada canción individual, tomamos un subconjunto de 10 fragmentos al azar
        for song_name, samples_list in songs_dict.items():
            k = min(10, len(samples_list))
            sampled_list.extend(random.sample(samples_list, k))
            
        # Actualizamos la lista activa para esta iteración
        self.samples = sampled_list
        print(f"[Dataset] Muestras activas para esta época de entrenamiento: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        # Al usar .convert("RGB"), cargamos el espectrograma de Mel monocromático
        # duplicando su información en los tres canales, manteniendo feliz a InceptionV3.
        image = Image.open(img_path).convert("RGB")
        image = self.t(image)
        return image, label