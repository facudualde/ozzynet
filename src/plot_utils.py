"""Shared plotting helpers for training/eval scripts."""

import matplotlib.pyplot as plt


def plot_history(history: dict, save_dir: str, model_name: str) -> None:
    # Three curves: train acc + val chunk acc + val song acc.
    epochs = range(1, len(history["train_acc"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train acc", marker="o")
    plt.plot(epochs, history["val_chunk_acc"], label="Val acc (chunk)", marker="o")
    plt.plot(epochs, history["val_song_acc"], label="Val acc (song)", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy curves - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = f"{save_dir}/training_curves.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  >> Plot saved: {out}")


def plot_loss_history(history: dict, save_dir: str, model_name: str) -> None:
    # Two curves: train loss + val chunk loss.
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss curves - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = f"{save_dir}/training_curves_loss.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  >> Plot saved: {out}")