import os
import json
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    checkpoints_dir = Path("checkpoints")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    if not checkpoints_dir.exists() or not list(checkpoints_dir.iterdir()):
        print("[!] No checkpoints directory or experiments found. Train some models first!")
        return

    # Find all history.json files
    history_files = list(checkpoints_dir.glob("*/history.json"))
    if not history_files:
        print("[!] No training history.json logs found in checkpoints/ subdirectories.")
        return

    print(f"[+] Found {len(history_files)} experiment histories. Loading and plotting...")

    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    for hist_path in history_files:
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            exp_name = data.get("experiment", hist_path.parent.name)
            epochs_data = data.get("epochs", [])

            if not epochs_data:
                continue

            epochs = [e["epoch"] for e in epochs_data]
            train_losses = [e["train_loss"] for e in epochs_data]
            val_losses = [e["val_loss"] for e in epochs_data]
            val_ppls = [e["val_ppl"] for e in epochs_data]

            # Plot Validation Loss on ax1
            ax1.plot(epochs, val_losses, marker="o", label=f"{exp_name} (Val)")
            # Plot Training Loss as a dashed line of same color
            line = ax1.lines[-1]
            ax1.plot(epochs, train_losses, linestyle="--", color=line.get_color(), alpha=0.6, label=f"{exp_name} (Train)")

            # Plot Validation Perplexity on ax2
            ax2.plot(epochs, val_ppls, marker="o", label=exp_name)

            print(f"  - Loaded '{exp_name}': {len(epochs)} epochs, final val loss: {val_losses[-1]:.4f}")
        except Exception as e:
            print(f"  [!] Failed to load {hist_path}: {e}")

    # Style first plot (Loss)
    ax1.set_title("Training & Validation Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Style second plot (Perplexity)
    ax2.set_title("Validation Perplexity Curves")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    output_path = data_dir / "experiment_loss_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\n[OK] Comparison graph successfully saved to: {output_path}")


if __name__ == "__main__":
    main()
