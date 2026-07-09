import random
from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms
import torchaudio
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

class DatasetCNN(TorchDataset):
    GENRES = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock",
    ]
    GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GENRES)}
    EXPECTED_CHUNKS = 10

    def __init__(
        self,
        root: str = "spectrograms",
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        t: transforms.Transform | None = None,
        return_song_id: bool = False,
    ) -> None:
        assert split in ("train", "val")

        # Store configuration as instance attributes so __getitem__
        # and _build_split_samples can reach them without re-passing.
        self.root = Path(root)
        self.split = split
        self.classes = self.GENRES
        self.return_song_id = return_song_id

        # Build the default transform pipeline unless the caller
        # supplied one explicitly (handy for tests / ablation).
        if t is None:
            # ConvNet branch: grayscale, 130x128, normalize to [-1, 1].
            transforms_list = [
                # Match ConvNet's expected input. Height 130 ~ mel bins,
                # width 128 ~ 3 sec of audio at hop_length=512, sr=22050.
                transforms.Resize((130, 128)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]

            # SpecAugment (frequency + time masking) only for train.
            # On val we want clean spectrograms so accuracy reflects
            # the real distribution, not the augmented one.
            #   freq_mask_param=12 → up to 12 contiguous mel bins zeroed
            #   time_mask_param=20 → up to 20 contiguous time frames zeroed
            if split == "train":
                transforms_list.extend([
                    torchaudio.transforms.FrequencyMasking(freq_mask_param=12),
                    torchaudio.transforms.TimeMasking(time_mask_param=20),
                ])

            transforms_list.append(
                # mean=0.5, std=0.5 maps [0, 1] → [-1, 1]. Chosen because
                # spectrograms inverted in spectrograms.py have most
                # energy away from 0; centering around 0 keeps ReLU
                # activations in their well-behaved region.
                transforms.Normalize(mean=[0.5], std=[0.5]),
            )

            # Compose into a single callable pipeline.
            t = transforms.Compose(transforms_list)

        self.t = t

        # Walk the filesystem to collect (path, label) tuples for this split.
        self.samples = self._build_split_samples(
            val_ratio=val_ratio,
            seed=seed,
        )

        # If nothing was found, fail loudly rather than silently producing
        # an empty dataset (which downstream would crash on len(ds)).
        if not self.samples:
            raise FileNotFoundError(
                f"No spectrograms found under {self.root}"
            )

    def _build_split_samples(
        self,
        val_ratio: float,
        seed: int,
    ) -> list[tuple[str, int]]:

        # Fresh RNG instance: ensures every call with the same `seed`
        # starts from the same internal state, regardless of what
        # happened elsewhere in the program.
        rng = random.Random(seed)
        samples: list[tuple[str, int]] = []

        # Split per genre rather than globally: with alphabetized
        # filenames, a global shuffle would put the first ~200 songs
        # almost entirely in `blues`, leaving other genres
        # over-represented in train and missing in val.
        for genre in self.GENRES:
            genre_dir = self.root / genre
            if not genre_dir.is_dir():
                continue

            # sorted() is deterministic given an unchanged filesystem,
            # so two calls to this function see the same initial ordering.
            song_names = sorted(
                d.name for d in genre_dir.iterdir() if d.is_dir()
            )

            # Same seed + same song_names → identical permutation.
            # This is what guarantees train and val are complementary.
            rng.shuffle(song_names)

            # round() avoids half-songs; the split is cheap enough that
            # being off-by-one doesn't matter.
            n_val = round(len(song_names) * val_ratio)

            # First n_val after the shuffle → val. The remainder → train.
            # Stored as a set for O(1) membership checks below.
            val_song_names = set(song_names[:n_val])

            for song_name in song_names:
                # Two filters that cover the population exactly once:
                #   train skips songs in val_song_names,
                #   val   skips songs NOT in val_song_names.
                # So no song lands in both splits, and no song is dropped.
                if self.split == "train" and song_name in val_song_names:
                    continue
                if self.split == "val" and song_name not in val_song_names:
                    continue

                song_dir = genre_dir / song_name
                chunk_paths = sorted(song_dir.glob("*.png"))

                # GTZAN ships a few known-corrupt files (e.g.
                # jazz.00054.wav fails to decode in librosa). When the
                # spectrogram generator hits one, it skips the whole
                # song, leaving a directory with 0 chunks. Warn but
                # tolerate so partial regenerations don't break loading.
                if len(chunk_paths) < self.EXPECTED_CHUNKS:
                    print(
                        f"[WARN] {genre}/{song_name}: "
                        f"expected {self.EXPECTED_CHUNKS} chunks, "
                        f"found {len(chunk_paths)} — using what's there"
                    )

                # One (path, label) entry per chunk. Song-level voting
                # is done later in train.py via `validate_songs`.
                for chunk_path in chunk_paths:
                    samples.append(
                        (str(chunk_path), self.GENRE_TO_IDX[genre])
                    )

        # Friendly log so we can verify counts from the shell.
        print(
            f"Instantiated split '{self.split}': "
            f"{len(samples)} chunks loaded."
        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        # Look up this sample's chunk path and label from the precomputed list.
        img_path, label = self.samples[idx]

        # Open the PNG. Use 3 channels (RGB) for InceptionV3, 1 channel
        # (grayscale) for ConvNet. `convert("RGB")` on a grayscale image
        # replicates the single channel across R, G, and B.
        image = Image.open(img_path).convert("L")

        # Apply the transform pipeline built in __init__:
        # resize + normalize (+ augment if train).
        image = self.t(image)

        # The song folder name (e.g. "rock.00037") is what train.py
        # uses to group the 10 chunks of a song for soft-vote validation.
        # Returned only when explicitly requested.
        if self.return_song_id:
            return image, label, Path(img_path).parent.name
        return image, label
