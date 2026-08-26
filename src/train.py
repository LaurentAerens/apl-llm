import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from v1.train_v1 import train_v1
from v2.train_v2 import train_v2
from v3.train_v3 import train_v3


def main():
    parser = argparse.ArgumentParser(description="Train APL SLM (Multi-Version)")
    parser.add_argument("--version", type=int, default=3, choices=[1, 2, 3], help="Architecture version (1=Baseline, 2=Depth Conditioned, 3=Modern RoPE+SwiGLU)")
    parser.add_argument("--data_file", type=str, default="data/apl_corpus.txt", help="Path to corpus file")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment output name")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--depth_loss_weight", type=float, default=0.2, help="Weight for depth loss penalty (v2/v3)")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--device", type=str, default=None, help="Hardware device (cpu or cuda)")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Training precision")
    parser.add_argument("--model_preset", type=str, default="small", help="Model preset (small, medium, large, deep, wide, none)")
    parser.add_argument("--early_stopping_patience", type=int, default=6, help="Early stopping patience in epochs")
    parser.add_argument("--finetune_from", type=str, default=None, help="Base checkpoint to fine-tune from")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = parser.parse_args()

    # Preset hyperparameter configurations
    presets = {
        "small": {"n_layer": 4, "n_head": 4, "n_embd": 64},
        "medium": {"n_layer": 6, "n_head": 8, "n_embd": 256},
        "large": {"n_layer": 8, "n_head": 12, "n_embd": 384},
        "deep": {"n_layer": 12, "n_head": 8, "n_embd": 256},
        "wide": {"n_layer": 6, "n_head": 16, "n_embd": 512},
    }
    if args.model_preset.lower() in presets:
        p = presets[args.model_preset.lower()]
        args.n_layer, args.n_head, args.n_embd = p["n_layer"], p["n_head"], p["n_embd"]

    if args.exp_name is None:
        args.exp_name = f"{args.model_preset.capitalize()}-v{args.version}"

    print("=" * 60)
    print(f"🚀 Launching APL SLM Training — Version v{args.version} ({args.exp_name})")
    print(f"   Architecture: {args.n_layer}L / {args.n_head}H / {args.n_embd}D")
    print(f"   Hyperparameters: LR={args.lr}, Epochs={args.epochs}, Batch={args.batch_size}, Warmup={args.warmup_epochs}")
    print("=" * 60)

    if args.version == 1:
        train_v1(args)
    elif args.version == 2:
        train_v2(args)
    else:
        train_v3(args)


if __name__ == "__main__":
    main()
