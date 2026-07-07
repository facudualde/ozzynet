import random
from collections import defaultdict
from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms
import torchaudio
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision.models import Inception_V3_Weights


class DatasetFT(TorchDataset):
    """Dataset GTZAN estructurado por canciones.

    Evita el Data Leakage utilizando espectrogramas de Mel estándar y soporta
    submuestreo dinámico de 10 fragmentos por canción por época, tanto para
    entrenamiento como de manera estática para validación.
    """

    GENRES = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock",
    ]
    GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRES)}
    EXPECTED_CHUNKS = 10

    def __init__(
        self,
        fine_tuning: bool = False,
        root: str = "spectrograms",
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        t: transforms.Transform | None = None,
        return_song_id: bool = False,
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
        self.seed = seed  # Guardamos la semilla

        # 1. Construir la lista maestra con todas las muestras asignadas a este split
        self.all_samples = self._build_split_samples(val_ratio, seed)

        if not self.all_samples:
            raise FileNotFoundError(f"No se encontraron espectrogramas para el split '{split}' en {self.root}")

        # 2. Inicializar la lista activa que usará __getitem__
        self.samples = list(self.all_samples)
        
        # 3. Filtrar a 10 fragmentos por canción dependiendo del split
        if self.split == "train":
            self.reset_epoch_samples()
        elif self.split == "val":
            # Para validación fijamos los 10 fragmentos usando una función interna
            self._set_static_validation_samples()

        rng = random.Random(seed)
        samples: list[tuple[str, int]] = []

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
                if self.split == "val" and song_name not in val_song_names:
                    continue

                song_dir = genre_dir / song_name
                chunk_paths = sorted(song_dir.glob("*.png"))

        print(f"Instanciado split '{self.split}': {len(samples)} imágenes totales en disco.")
        return samples

    def _set_static_validation_samples(self) -> None:
        """Selecciona 10 fragmentos fijos por canción para el split de validación."""
        sampled_list = []
        songs_dict = defaultdict(list)
        
        # Usamos una semilla fija local para asegurar reproducibilidad absoluta en validación
        rng = random.Random(self.seed)
        
        for sample in self.all_samples: 
            song_name = Path(sample[0]).parent.name
            songs_dict[song_name].append(sample)
            
        for song_name, samples_list in songs_dict.items():
            k = min(10, len(samples_list))
            # Usamos el objeto rng local para el sampling determinista
            sampled_list.extend(rng.sample(samples_list, k))
            
        self.samples = sampled_list
        print(f"[Dataset] Muestras fijadas de forma estática para Validación: {len(self.samples)}")

    def reset_epoch_samples(self) -> None:
        """Selecciona aleatoriamente 10 fragmentos de cada canción para la época actual (solo train)."""
        if self.split == "val":
            # Protegemos el split de validación para que no cambie dinámicamente si es llamado por error
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

    def __getitem__(
      self,
      idx: int
    ) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        if self.fine_tuning:
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.open(img_path).convert("L")
        image = self.t(image)
        return image, label
