"""Evaluate a trained checkpoint on the GTZAN val split.

Usage:
    python src/eval.py <checkpoint>
    python src/eval.py <checkpoint> --mode song
    python src/eval.py <checkpoint> --mode chunk
    python src/eval.py <checkpoint> --mode both
    python src/eval.py <checkpoint> --batch-size 64
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cnn import ConvNet
from confusion_matrix import (
    compute_confusion_matrix,
    compute_confusion_matrix_songs,
    print_confusion_matrix_report,
)
from dataset import Gtzan


def load_model(checkpoint_path: str, device) -> ConvNet:
    """Construct ConvNet and load its weights from `checkpoint_path`."""
    model = ConvNet()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a GTZAN checkpoint and print a confusion matrix report.",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the .pth file (e.g. checkpoints/<run>/best_song_acc_*.pth).",
    )
    parser.add_argument(
        "--mode",
        choices=["song", "chunk", "both"],
        default="song",
        help="Validation granularity. Default: song-level with soft voting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size. Default: 32.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.checkpoint).exists():
        print(f"[ERR] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    genres = Gtzan.GENRES

    if args.mode in ("song", "both"):
        val_ds = Gtzan(split="val", model="cnn")
        cm = compute_confusion_matrix_songs(model, val_ds, args.batch_size, device)
        print_confusion_matrix_report(
            cm, genres, title="Song-level (soft voting)",
        )

    if args.mode in ("chunk", "both"):
        val_ds = Gtzan(split="val", model="cnn")
        loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        cm = compute_confusion_matrix(model, loader, device)
        print_confusion_matrix_report(cm, genres, title="Chunk-level")


if __name__ == "__main__":
    main()
