"""
Plot training loss curves from the JSON log produced by train.py.

Usage:
    python plot_loss.py
    python plot_loss.py --log_file ./output/training_logs.json --out ./output/loss_curve.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_logs(log_file: str):
    with open(log_file) as f:
        records = json.load(f)

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for r in records:
        if "loss" in r:
            train_steps.append(r["step"])
            train_losses.append(r["loss"])
        if "eval_loss" in r:
            eval_steps.append(r["step"])
            eval_losses.append(r["eval_loss"])

    return train_steps, train_losses, eval_steps, eval_losses


def smooth(values, weight: float = 0.8):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


def plot(log_file: str, out_path: str):
    train_steps, train_losses, eval_steps, eval_losses = load_logs(log_file)

    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig, axes = plt.subplots(1, 1, figsize=(6, 4))

    # --- Left: Training loss ---
    ax = axes if isinstance(axes, plt.Axes) else axes[0]
    # ax.plot(train_steps, train_losses, color="steelblue", alpha=0.35, linewidth=1, label="raw")
    ax.plot(train_steps, train_losses, color="steelblue", linewidth=2)
    # if len(train_losses) > 5:
    #     ax.plot(train_steps, smooth(train_losses, 0.85),
    #             color="steelblue", linewidth=2, label="smoothed (EMA)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss  (OFT · Qwen2.5-1.5B · GSM8K)")
    # ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

    # --- Right: Eval loss per epoch ---
    # ax = axes[1]
    # if eval_losses:
    #     epoch_labels = [f"Epoch {i+1}" for i in range(len(eval_losses))]
    #     ax.plot(range(1, len(eval_losses) + 1), eval_losses,
    #             marker="o", color="coral", linewidth=2, markersize=7, label="eval loss")
    #     ax.set_xticks(range(1, len(eval_losses) + 1))
    #     ax.set_xticklabels(epoch_labels)
    #     ax.set_ylabel("Loss")
    #     ax.set_title("Validation Loss per Epoch")
    #     ax.legend()
    #     ax.grid(True, linestyle="--", alpha=0.4)
    #     ax.set_ylim(bottom=0)

    #     # Annotate min eval loss
    #     min_idx = int(np.argmin(eval_losses))
    #     ax.annotate(
    #         f"best: {eval_losses[min_idx]:.4f}",
    #         xy=(min_idx + 1, eval_losses[min_idx]),
    #         xytext=(min_idx + 1 + 0.2, eval_losses[min_idx] + 0.02),
    #         fontsize=9,
    #         arrowprops=dict(arrowstyle="->", color="gray"),
    #     )
    # else:
    #     ax.text(0.5, 0.5, "No eval logs found", ha="center", va="center",
    #             transform=ax.transAxes, fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved loss curve to {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="./output/training_logs.json")
    parser.add_argument("--out", type=str, default="./output/loss_curve.png")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Log file not found: {args.log_file}")
        print("Run train.py first to generate logs.")
        return

    plot(args.log_file, args.out)

    # --- Print summary stats ---
    train_steps, train_losses, eval_steps, eval_losses = load_logs(args.log_file)
    print(f"\nTraining summary:")
    print(f"  Initial train loss : {train_losses[0]:.4f}")
    print(f"  Final train loss   : {train_losses[-1]:.4f}")
    print(f"  Reduction          : {train_losses[0] - train_losses[-1]:.4f} ({(1 - train_losses[-1]/train_losses[0])*100:.1f}%)")
    if eval_losses:
        print(f"  Best eval loss     : {min(eval_losses):.4f} (epoch {eval_losses.index(min(eval_losses))+1})")


if __name__ == "__main__":
    main()
