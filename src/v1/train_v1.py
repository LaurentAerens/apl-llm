"""
Architecture Version 1 Training Entrypoint & Wrapper.
Dispatches to the unified APLTrainer configured for v1 (Baseline Causal Transformer).
"""

import sys
import argparse
from pathlib import Path

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from trainer import APLTrainer
from dataset import APLDataset

# Backward compatibility alias
APLDataset_v1 = APLDataset


def train_v1(args):
    args.version = 1
    trainer = APLTrainer(args)
    return trainer.train()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from train import parse_args
    args = parse_args()
    args.version = 1
    train_v1(args)


if __name__ == "__main__":
    main()
