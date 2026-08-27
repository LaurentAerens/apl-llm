"""
Universal Training Entrypoint & Version Dispatcher for APL SLM.
Dispatches to the appropriate architecture-specific training pipeline:
- v1 (Pure Causal Baseline) -> src/v1/train_v1.py
- v2 (Depth Conditioned Transformer) -> src/v2/train_v2.py
- v3 (Modern RoPE + SwiGLU + QK-Norm) -> src/v3/train_v3.py
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

from v1.train_v1 import train_v1
from v2.train_v2 import train_v2
from v3.train_v3 import train_v3


def determine_version(args) -> int:
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


def main():
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
    parser.add_argument("--model_preset", type=str, default="small", help="Model preset (small, medium, large, deep, wide, xlarge, huge, giant, none)")
    parser.add_argument("--early_stopping_patience", type=int, default=6, help="Early stopping patience in epochs")
    parser.add_argument("--finetune_from", type=str, default=None, help="Base checkpoint to fine-tune from")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with truncated corpus")
    args = parser.parse_args()

    args.version = determine_version(args)

    # Preset hyperparameter configurations
    presets = {
        "small": {"n_layer": 4, "n_head": 4, "n_embd": 64},
        "medium": {"n_layer": 6, "n_head": 8, "n_embd": 256},
        "large": {"n_layer": 8, "n_head": 12, "n_embd": 384},
        "deep": {"n_layer": 12, "n_head": 8, "n_embd": 256},
        "wide": {"n_layer": 6, "n_head": 16, "n_embd": 512},
        "xlarge": {"n_layer": 12, "n_head": 16, "n_embd": 512},
        "huge": {"n_layer": 16, "n_head": 16, "n_embd": 768},
        "giant": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
        "(none)": {"n_layer": 4, "n_head": 4, "n_embd": 64},
        "none": {"n_layer": 4, "n_head": 4, "n_embd": 64},
    }
    preset_key = (args.model_preset or "small").lower()
    if preset_key in presets:
        p = presets[preset_key]
        if args.n_layer is None:
            args.n_layer = p["n_layer"]
        if args.n_head is None:
            args.n_head = p["n_head"]
        if args.n_embd is None:
            args.n_embd = p["n_embd"]
    else:
        if args.n_layer is None:
            args.n_layer = 4
        if args.n_head is None:
            args.n_head = 4
        if args.n_embd is None:
            args.n_embd = 64

    if args.exp_name is None:
        args.exp_name = f"{preset_key.capitalize()}-v{args.version}"

    print("=" * 65)
    print(f"🚀 Launching APL SLM Training — Version v{args.version} ({args.exp_name})")
    print(f"   Architecture: {args.n_layer}L / {args.n_head}H / {args.n_embd}D")
    print(f"   Hyperparameters: LR={args.lr}, Epochs={args.epochs}, Batch={args.batch_size}, GradAccum={args.grad_accum_steps}, Warmup={args.warmup_epochs}")
    print("=" * 65 + "\n")

    if args.version == 1:
        train_v1(args)
    elif args.version == 2:
        train_v2(args)
    else:
        train_v3(args)


if __name__ == "__main__":
    main()
