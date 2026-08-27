"""
Multi-Experiment Loss and Perplexity Curve Visualizer for APL SLM.
Reads training history JSON logs and produces side-by-side comparative plots.
"""

import os
import json
import sys
import argparse
from pathlib import Path
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def smooth_curve(points: List[float], factor: float = 0.6) -> List[float]:
    """Applies exponential moving average smoothing."""
    smoothed: List[float] = []
    for point in points:
        if smoothed:
            previous = smoothed[-1]
            smoothed.append(previous * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed


def plot_experiments(
    checkpoints_dir: Path = Path("checkpoints"),
    output_path: Path = Path("data/experiment_loss_comparison.png"),
    smooth: bool = False,
    selected_experiments: Optional[List[str]] = None,
):
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoints_dir.exists():
        print(f"[!] Checkpoints directory '{checkpoints_dir}' not found.")
        return

    history_files = list(checkpoints_dir.glob("*/history.json"))
    if not history_files:
        print("[!] No training history.json logs found in checkpoints/ subdirectories.")
        return

    print(f"[+] Found {len(history_files)} experiment histories. Loading and plotting...")

    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    valid_plots = 0

    for hist_path in sorted(history_files):
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            exp_name = data.get("experiment", hist_path.parent.name) if isinstance(data, dict) else hist_path.parent.name
            if selected_experiments and exp_name not in selected_experiments and hist_path.parent.name not in selected_experiments:
                continue

            epochs_data = data.get("epochs", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            if not epochs_data:
                continue

            epochs = [e["epoch"] for e in epochs_data if "epoch" in e]
            train_losses = [e["train_loss"] for e in epochs_data if "train_loss" in e]
            val_losses = [e["val_loss"] for e in epochs_data if "val_loss" in e]
            val_ppls = [e.get("val_ppl", 0.0) for e in epochs_data if "val_loss" in e]

            if not epochs or not val_losses:
                continue

            plot_val_losses = smooth_curve(val_losses) if smooth else val_losses
            plot_train_losses = smooth_curve(train_losses) if (smooth and train_losses) else train_losses

            # Plot Validation Loss on ax1
            ax1.plot(epochs, plot_val_losses, marker="o" if not smooth else None, label=f"{exp_name} (Val)")
            line = ax1.lines[-1]
            if len(plot_train_losses) == len(epochs):
                ax1.plot(epochs, plot_train_losses, linestyle="--", color=line.get_color(), alpha=0.6, label=f"{exp_name} (Train)")

            # Plot Validation Perplexity on ax2
            plot_val_ppls = smooth_curve(val_ppls) if smooth else val_ppls
            if len(plot_val_ppls) == len(epochs):
                ax2.plot(epochs, plot_val_ppls, marker="o" if not smooth else None, label=exp_name)

            valid_plots += 1
            print(f"  - Loaded '{exp_name}': {len(epochs)} epochs, final val loss: {val_losses[-1]:.4f}")
        except Exception as e:
            print(f"  [!] Failed to load {hist_path}: {e}")

    if valid_plots == 0:
        plt.close()
        print("[!] No valid epoch history logs found to plot.")
        return

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
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\n[OK] Comparison graph successfully saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot APL SLM Training & Validation Curves")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Directory containing experiment checkpoints")
    parser.add_argument("--output", type=str, default="data/experiment_loss_comparison.png", help="Output image file path")
    parser.add_argument("--smooth", action="store_true", help="Apply exponential moving average smoothing")
    parser.add_argument("--experiments", nargs="*", default=None, help="List of specific experiment names to plot")
    args = parser.parse_args()

    plot_experiments(
        checkpoints_dir=Path(args.checkpoints_dir),
        output_path=Path(args.output),
        smooth=args.smooth,
        selected_experiments=args.experiments,
    )


if __name__ == "__main__":
    main()
