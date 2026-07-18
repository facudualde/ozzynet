"""Evaluate a trained checkpoint on the held-out test spectrograms.

Examples
--------
    make eval CHECKPOINT=checkpoints/from_scratch/<ts>/<ts>_best.pth FLAGS="--model cnn"
    make eval CHECKPOINT=checkpoints/fine_tuning/<ts>/<ts>_best.pth FLAGS="--model inception"
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn import ConvNet, InceptionV3
from confusion_matrix import plot_confusion_matrix, print_confusion_matrix_report
from dataset import Custom, Gtzan


def load_model(checkpoint_path: str, device: torch.device, model_kind: str) -> torch.nn.Module:
    # Peek at the state_dict to learn the classifier head size, then build the architecture.
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if model_kind == "cnn":
        num_classes = state_dict["fc2.weight"].shape[0]
        model = ConvNet(num_classes=num_classes)
    else:
        num_classes = state_dict["backbone.fc.3.weight"].shape[0]
        model = InceptionV3(num_classes=num_classes)
    model.load_state_dict(state_dict)
    return model.to(device)


def _accuracy_from_cm(cm) -> tuple[int, int, float]:
    # Pull overall accuracy out of a sklearn-style confusion matrix.
    total = int(cm.sum())
    correct = int(np.diag(cm).sum())
    pct = correct / total if total else 0.0
    return correct, total, pct


def _classifier_out_features(model: torch.nn.Module) -> int | None:
    # Reach into the classifier head to read its final-layer width.
    if hasattr(model, "fc2"):  # ConvNet
        return model.fc2.out_features
    if hasattr(model, "backbone") and hasattr(model.backbone, "fc"):  # InceptionV3
        last = model.backbone.fc[-1]
        if hasattr(last, "out_features"):
            return last.out_features
    return None


def _validate_class_count(model: torch.nn.Module, dataset) -> None:
    # Fail loud if the checkpoint's classifier head size doesn't match the dataset.
    out_features = _classifier_out_features(model)
    if out_features is None:
        return
    if out_features != len(dataset.GENRES):
        print(
            f"[ERR] checkpoint expects {out_features} classes but dataset has "
            f"{len(dataset.GENRES)} genres ({dataset.GENRES}). "
            f"Re-train the model on this dataset first.",
            file=sys.stderr,
        )
        sys.exit(1)


def _resolve_save_dir(checkpoint_path: str) -> str | None:
    # Place the confusion matrix next to the checkpoint if it lives under
    # the canonical training layout (checkpoints/from_scratch/ or
    # checkpoints/fine_tuning/). Returns None for arbitrary paths.
    ckpt_dir = Path(checkpoint_path).parent
    if ckpt_dir.parent.name in ("from_scratch", "fine_tuning"):
        return str(ckpt_dir)
    return None


def _is_test_dir_non_empty(test_root: str) -> bool:
    # True when the test spectrograms directory has at least one .png anywhere underneath.
    if not os.path.isdir(test_root):
        return False
    for genre in os.listdir(test_root):
        genre_dir = os.path.join(test_root, genre)
        if not os.path.isdir(genre_dir):
            continue
        for song in os.listdir(genre_dir):
            song_dir = os.path.join(genre_dir, song)
            if not os.path.isdir(song_dir):
                continue
            if any(f.lower().endswith(".png") for f in os.listdir(song_dir)):
                return True
    return False


def _infer(
    model: torch.nn.Module, dataset, batch_size: int, device: torch.device,
) -> tuple[list[tuple[str, int, int]], dict[str, tuple[int, int]]]:
    # Single inference pass yields both chunk-level records and song-level soft-vote predictions.
    records: list[tuple[str, int, int]] = []
    song_probs: dict[str, torch.Tensor] = {}
    song_true: dict[str, int] = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    with torch.no_grad():
        for X, y, sids in loader:
            X = X.to(device)
            probs = torch.softmax(model(X), dim=1).cpu()
            preds = probs.argmax(1).tolist()
            for i, sid in enumerate(sids):
                true = y[i].item()
                records.append((sid, true, preds[i]))
                song_true.setdefault(sid, true)
                song_probs[sid] = probs[i] if sid not in song_probs else song_probs[sid] + probs[i]
    song_preds = {sid: (true, song_probs[sid].argmax().item()) for sid, true in song_true.items()}
    return records, song_preds


def _print_misclassified(song_preds: dict[str, tuple[int, int]], genres: list[str]) -> None:
    # One row per misclassified song (deduplicated by song_id), grouped by (true -> pred).
    label_to_genre = {i: g for i, g in enumerate(genres)}
    wrong = [(sid, t, p) for sid, (t, p) in song_preds.items() if t != p]
    if not wrong:
        print("\n  No misclassified songs.")
        return
    print(f"\n  Misclassified songs ({len(wrong)} total, true class -> predicted class):")
    groups: dict[tuple[int, int], list[str]] = {}
    for sid, t, p in wrong:
        groups.setdefault((t, p), []).append(sid)
    for (t, p), songs in sorted(groups.items()):
        print(f"    {label_to_genre[t]} -> {label_to_genre[p]}  ({len(songs)} songs):")
        for sid in sorted(songs):
            print(f"      - {sid}")


def _chunk_cm_from_records(records: list[tuple[str, int, int]], num_classes: int) -> np.ndarray:
    # Build a chunk-level confusion matrix directly from records (sklearn-compatible).
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for _, true, pred in records:
        cm[true, pred] += 1
    return cm


def _song_cm_from_song_preds(
    song_preds: dict[str, tuple[int, int]], num_classes: int,
) -> np.ndarray:
    # Build a song-level confusion matrix from {sid: (true, pred)}.
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in song_preds.values():
        cm[true, pred] += 1
    return cm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on test spectrograms.")
    parser.add_argument("checkpoint", type=str, help="Path to the .pth file.")
    parser.add_argument(
        "--dataset", choices=["gtzan", "custom"], default="gtzan",
        help="Which test root to read from: gtzan (default) or custom.",
    )
    parser.add_argument(
        "--model", choices=["cnn", "inception"], required=True,
        help="Architecture of the checkpoint (cnn = ConvNet, inception = InceptionV3).",
    )
    parser.add_argument(
        "--mode", choices=["song", "chunk", "both"], default="both",
        help="song = soft-vote per song, chunk = per spectrogram, both = default.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.checkpoint).exists():
        print(f"[ERR] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, args.model)

    test_root = "gtzan/test/spectrograms" if args.dataset == "gtzan" else "dataset/test/spectrograms"
    cls = Gtzan if args.dataset == "gtzan" else Custom
    test_ds = cls(root=test_root, split="test", model=args.model)

    _validate_class_count(model, test_ds)

    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Dataset    : {args.dataset} (test)")
    print(f"  Model      : {args.model}")
    print(f"  Samples    : {len(test_ds)} chunks")
    print(f"  Genres     : {test_ds.GENRES}")
    print("=" * 60)

    # One inference pass feeds chunk cm, song cm, and the misclassified listing.
    records, song_preds = _infer(model, test_ds, args.batch_size, device)

    if args.mode in ("chunk", "both"):
        cm = _chunk_cm_from_records(records, num_classes=len(test_ds.GENRES))
        correct, total, pct = _accuracy_from_cm(cm)
        print(f"\n  Chunk-level accuracy: {correct}/{total} = {pct:.2%}")
        print_confusion_matrix_report(cm, test_ds.GENRES, title="Chunk-level (per spectrogram)")

    if args.mode in ("song", "both"):
        cm = _song_cm_from_song_preds(song_preds, num_classes=len(test_ds.GENRES))
        correct, total, pct = _accuracy_from_cm(cm)
        print(f"\n  Song-level accuracy (soft voting): {correct}/{total} = {pct:.2%}")
        print_confusion_matrix_report(cm, test_ds.GENRES, title="Song-level (soft voting)")
        # Persist song-level confusion matrix next to the checkpoint when both conditions hold.
        save_dir = _resolve_save_dir(args.checkpoint)
        if save_dir is not None and _is_test_dir_non_empty(test_root):
            cm_path = f"{save_dir}/confusion_matrix.png"
            plot_confusion_matrix(
                cm, test_ds.GENRES, save_path=cm_path,
                title=f"Song-level (soft voting) — {args.dataset} test (model={args.model})",
            )
            print(f"  >> Plot saved: {cm_path}")

    # Always at song level (one row per misclassified song, deduplicated by song_id).
    _print_misclassified(song_preds, test_ds.GENRES)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
