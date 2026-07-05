import random
from pathlib import Path
import torch
from torchvision.transforms import v2 as transforms
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision.models import Inception_V3_Weights

class GTZANDataset(TorchDataset):
    """GTZAN dataset structured by song to avoid data leakage."""

    GENRES = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock",
    ]
    GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRES)}

    def __init__(
        self,
        root: str = "spectrograms",
        split: str = "train",  # Must be "train" or "val"
        val_ratio: float = 0.2,
        seed: int = 42,
        t: transforms.Transform | None = None,
    ) -> None:
        assert split in ["train", "val"], "the 'split' parameter must be 'train' or 'val'"
        
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

        # Build and filter samples by split, atomically per song.
        self.samples = self._build_split_samples(val_ratio, seed)

        if not self.samples:
            raise FileNotFoundError(f"No spectrograms found for split '{split}' under {self.root}")

    def _build_split_samples(self, val_ratio: float, seed: int) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        rng = random.Random(seed)

        for genre in self.GENRES:
            genre_dir = self.root / genre
            if not genre_dir.is_dir():
                continue

            # Get song folder names (e.g., "blues.00000").
            song_names = sorted([d.name for d in genre_dir.iterdir() if d.is_dir()])
            rng.shuffle(song_names)

            # Strictly split the song names into train vs. val.
            val_len = round(len(song_names) * val_ratio)
            val_song_names = set(song_names[:val_len])

            for song_name in song_names:
                # Filter at the song level according to the constructor's split argument.
                if self.split == "val" and song_name not in val_song_names:
                    continue
                if self.split == "train" and song_name in val_song_names:
                    continue

                song_dir = genre_dir / song_name
                images = sorted(song_dir.glob("*.png"))
                for img_path in images:
                    samples.append((str(img_path), self.GENRE_TO_IDX[genre]))

        print(f"Instantiated split '{self.split}': {len(samples)} images loaded.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.t(image)
        return image, label
