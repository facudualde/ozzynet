import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import v2 as transforms
from torchvision.models import Inception_V3_Weights

@dataclass(frozen=True)
class _Pipeline:
    # Bundle of knobs that travel together through __getitem__.
    image_mode: Literal["L", "RGB"]
    image_size: int | tuple[int, int]
    normalize_mean: list[float]
    normalize_std: list[float]
    augmentation_train: list


def _build_pipeline(model: str, data_augmentation: bool) -> _Pipeline:
    # Default transforms depend on which backbone is being trained.
    if model == "inception":
        w = Inception_V3_Weights.DEFAULT.transforms()
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=(-5, 5), translate=(0.05, 0.05))]
        return _Pipeline(image_mode="RGB", image_size=(299, 299), normalize_mean=list(w.mean), normalize_std=list(w.std), augmentation_train=aug if data_augmentation else [])
    if model == "cnn":
        aug = [torchaudio.transforms.FrequencyMasking(freq_mask_param=12), torchaudio.transforms.TimeMasking(time_mask_param=20)]
        return _Pipeline(image_mode="L", image_size=(130, 128), normalize_mean=[0.5], normalize_std=[0.5], augmentation_train=aug if data_augmentation else [])
    raise ValueError(f"model must be 'inception' or 'cnn', got {model!r}")


def _build_transforms(pipeline: _Pipeline, split: str, data_augmentation: bool) -> transforms.Compose:
    # Resize first so SpecAugment/affine operate on the target shape.
    ops: list = [transforms.Resize(pipeline.image_size), transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    if split == "train" and data_augmentation:
        ops.extend(pipeline.augmentation_train)
    ops.append(transforms.Normalize(mean=pipeline.normalize_mean, std=pipeline.normalize_std))
    return transforms.Compose(ops)


class _BaseSpectrogramDataset(TorchDataset):
    # Shared logic for Gtzan and Custom; subclasses only set DEFAULT_ROOT.
    GENRES: list[str] = []
    DEFAULT_ROOT: str = ""

    def __init__(
        self,
        root: str | None = None,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        model: Literal["inception", "cnn"] = "cnn",
        data_augmentation: bool = False,
        random_chunks_number: int | None = None,
        t: transforms.Transform | None = None,
    ) -> None:
        assert split in ("train", "val", "test")
        assert 0.0 < val_ratio < 1.0

        self.root = Path(root) if root else Path(self.DEFAULT_ROOT)
        self.split = split
        self.seed = seed
        self.model = model

        self.GENRES = self._discover_genres()
        self.GENRE_TO_IDX = {g: i for i, g in enumerate(self.GENRES)}

        pipeline = _build_pipeline(model, data_augmentation)
        self.image_mode = pipeline.image_mode
        self.t = t if t is not None else _build_transforms(pipeline, split, data_augmentation)

        # Build split first at song level; chunk filtering happens after.
        self.all_samples = self._build_split_samples(val_ratio, seed)
        self.samples = list(self.all_samples)

        # Stash for reset_epoch_samples; None disables per-epoch re-sampling.
        self._random_chunks_number = random_chunks_number

        # Apply per-song random chunk selection now (train only).
        if random_chunks_number is not None:
            self._apply_random_chunk_selection(random_chunks_number)

        if not self.samples:
            raise FileNotFoundError(f"No spectrograms found under {self.root}")

    def _discover_genres(self) -> list[str]:
        # Subfolders of the root are the classes; keeps Custom agnostic of contents.
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def _build_split_samples(self, val_ratio: float, seed: int) -> list[tuple[str, int]]:
        # One entry per chunk; songs are never split between train/val.
        samples: list[tuple[str, int]] = []
        rng = random.Random(seed)

        for genre in self.GENRES:
            genre_dir = self.root / genre
            if not genre_dir.is_dir():
                continue
            song_names = sorted(d.name for d in genre_dir.iterdir() if d.is_dir())
            rng.shuffle(song_names)
            n_val = int(len(song_names) * val_ratio)
            val_set = set(song_names[:n_val])

            for song_name in song_names:
                in_val = song_name in val_set
                if self.split == "test":
                    pass  # include all songs; val_ratio is ignored for test.
                elif self.split == "val" and not in_val:
                    continue
                elif self.split == "train" and in_val:
                    continue

                chunk_paths = sorted((self.root / genre / song_name).glob("*.png"))
                for chunk_path in chunk_paths:
                    samples.append((str(chunk_path), self.GENRE_TO_IDX[genre]))

        print(f"Instantiated split '{self.split}': {len(samples)} chunks loaded.")
        return samples

    def _apply_random_chunk_selection(self, n: int) -> None:
        # Sample up to n chunks per song_id without replacement.
        by_song: dict[str, list[tuple[str, int]]] = {}
        for path, label in self.all_samples:
            sid = Path(path).parent.name
            by_song.setdefault(sid, []).append((path, label))

        picked: list[tuple[str, int]] = []
        for entries in by_song.values():
            k = min(n, len(entries))
            picked.extend(random.sample(entries, k))
        self.samples = picked
        print(f"Random chunk selection (k={n}) applied: {len(self.samples)} chunks across {len(by_song)} songs.")

    def reset_epoch_samples(self) -> None:
        # Public hook called once per epoch; re-randomizes per-song chunks when configured.
        # No-op unless the dataset was constructed with random_chunks_number.
        if self._random_chunks_number is None:
            return
        self._apply_random_chunk_selection(self._random_chunks_number)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        image = self.t(Image.open(img_path).convert(self.image_mode))
        song_id = Path(img_path).parent.name
        return image, label, song_id


class Gtzan(_BaseSpectrogramDataset):
    # Reads mel-spectrograms under gtzan/spectrograms/.
    DEFAULT_ROOT = "gtzan/spectrograms"


class Custom(_BaseSpectrogramDataset):
    # Reads mel-spectrograms under dataset/spectrograms/; genres auto-discovered.
    DEFAULT_ROOT = "dataset/spectrograms"
