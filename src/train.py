"""
Universal Training Entrypoint & Version Dispatcher for APL SLM.
Dispatches to the unified APLTrainer for architecture versions (v1, v2, v3).
"""

import sys
import argparse
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src root is in sys.path
src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from trainer import APLTrainer
from config import MODEL_PRESETS


def determine_version(args) -> int:
    """Infers architecture version from flags, base checkpoint, or experiment name."""
    if args.version is not None:
        return args.version

    # If fine-tuning, inspect the checkpoint to determine architecture version
    if args.finetune_from:
        ckpt_path = Path(args.finetune_from)
        if ckpt_path.exists():
            try:
                from model import AutoModel
                _, _, saved_ver = AutoModel.from_checkpoint(ckpt_path, device=torch.device("cpu"))
                print(f"[+] Detected architecture version {saved_ver} from checkpoint: {ckpt_path}")
                return saved_ver
            except Exception:
                pass

    # Check experiment name hints
    if args.exp_name:
        exp_lower = args.exp_name.lower()
        if "-v1" in exp_lower or "v1" in exp_lower:
            return 1
        if "-v2" in exp_lower or "v2" in exp_lower:
            return 2
        if "-v3" in exp_lower or "v3" in exp_lower:
            return 3

    # Default to current flagship architecture (v3)
    return 3


def parse_args():
    parser = argparse.ArgumentParser(description="Train APL SLM (Multi-Version Dispatcher)")
    parser.add_argument("--model_version", "--version", "-v", dest="version", type=int, default=None, choices=[1, 2, 3], help="Architecture version (1=Baseline, 2=Depth Conditioned, 3=Modern RoPE+SwiGLU)")
    parser.add_argument("--data_file", "--dataset_path", dest="data_file", type=str, default="data/apl_corpus.txt", help="Path to corpus file")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment output name")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--depth_loss_weight", type=float, default=0.2, help="Weight for depth loss penalty (v2/v3)")
    parser.add_argument("--n_layer", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--device", type=str, default=None, help="Hardware device (cpu or cuda)")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Training precision")
    preset_choices = list(MODEL_PRESETS.keys()) + ["(none)", "none"]
    parser.add_argument("--model_preset", type=str, default="small", choices=preset_choices, help="Model preset")
    parser.add_argument("--early_stopping_patience", type=int, default=6, help="Early stopping patience in epochs")
    parser.add_argument("--finetune_from", type=str, default=None, help="Base checkpoint to fine-tune from")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with truncated corpus")
    return parser.parse_args()


def main():
    args = parse_args()
    args.version = determine_version(args)

    trainer = APLTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
