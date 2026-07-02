from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms
from PIL import Image
from torch.utils.data import Dataset as TorchDataset, random_split
from torchvision.models import Inception_V3_Weights

class GTZANDataset(TorchDataset):
    """GTZAN music genre classification dataset.
    Each sample is a 299x299 spectrogram labeled by genre. On
    construction, walks ``spectrograms/``, applies the default
    transform, and exposes ``self.train`` / ``self.val`` via
    ``random_split``.
    """

    GENRES = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock",
    ]
    GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRES)}

    def __init__(
        self,
        root: str = "spectrograms",
        val_ratio: float = 0.2,
        seed: int = 42,
        t: transforms.Transform | None =None,
    ) -> None:
        if t is None:
            inceptionV3_w_t = Inception_V3_Weights.DEFAULT.transforms()
            t = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                # Inception v3 expects these (its ImageNet pretraining stats).
                transforms.Normalize(
                    inceptionV3_w_t.mean,
                    inceptionV3_w_t.std,
                ),
            ])
        self.root = Path(root)
        self.t: transforms.Transform = t
        self.samples = self._build_samples()

        if not self.samples:
            raise FileNotFoundError(f"No spectrograms found under {self.root}")

        total = len(self.samples)
        val_len = int(total * val_ratio)
        train_len = total - val_len

        generator = torch.Generator().manual_seed(seed)
        self.train, self.val = random_split(
            self, [train_len, val_len], generator=generator
        )

    def __iter__(self):
        yield self.train
        yield self.val

    def _build_samples(self) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        for genre in self.GENRES:
            genre_dir = self.root / genre
            if not genre_dir.is_dir():
                continue
            for song_dir in sorted(genre_dir.iterdir()):
                if not song_dir.is_dir():
                    continue
                for img_path in sorted(song_dir.glob("*.png")):
                    samples.append((str(img_path), self.GENRE_TO_IDX[genre]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.t(image)
        return image, label
