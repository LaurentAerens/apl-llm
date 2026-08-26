import os
import argparse
import time
import json
import math
import sys
from pathlib import Path
from typing import Union, Optional, Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tokenizer import APLTokenizer
from v1.model_v1 import APL_SLM_v1, APL_SLMConfig_v1
from synthetic_generator import APLSyntheticGenerator


class APLDataset_v1(Dataset):
    """Chunked sequence dataset for pure causal language modeling (v1)."""

    def __init__(
        self,
        text: str,
        tokenizer: APLTokenizer,
        seq_len: int = 512,
        tokens_path: Union[str, Path] = "data/apl_corpus_tokens.pt",
        force_recompute: bool = False,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        tokens_file = Path(tokens_path)

        loaded_from_cache = False
        if tokens_file.exists() and not force_recompute:
            try:
                self.tokens = torch.load(tokens_file, weights_only=False)
                loaded_from_cache = True
                print(f"[+] Loaded precomputed token file: {tokens_file} ({len(self.tokens):,} tokens)")
            except Exception as e:
                print(f"[!] Precomputed token load failed ({e}). Recomputing...")

        if not loaded_from_cache:
            print("[+] Generating token file from corpus...")
            raw_tokens = tokenizer.encode(text, add_special_tokens=False)
            self.tokens = torch.tensor(raw_tokens, dtype=torch.long)
            tokens_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.tokens, tokens_file)
            print(f"[OK] Precomputed token file saved to: {tokens_file} ({len(self.tokens):,} tokens)")

        num_samples = (len(self.tokens) - 1) // self.seq_len
        self.num_samples = max(1, num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len

        inputs = self.tokens[start_idx:end_idx]
        targets = self.tokens[start_idx + 1 : end_idx + 1]

        if len(inputs) < self.seq_len:
            pad_len = self.seq_len - len(inputs)
            inputs = torch.cat([inputs, torch.full((pad_len,), self.tokenizer.pad_id, dtype=torch.long)])
        if len(targets) < self.seq_len:
            pad_len = self.seq_len - len(targets)
            targets = torch.cat([targets, torch.full((pad_len,), self.tokenizer.pad_id, dtype=torch.long)])

        return inputs, targets


def train_v1(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[+] Initializing APL SLM v1 Training on device: {device}")

    tokenizer = APLTokenizer()
    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"[!] Data file {data_path} not found. Generating synthetic APL corpus...")
        corpus_text = APLSyntheticGenerator.generate_synthetic_corpus(count=10000)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(corpus_text)
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            corpus_text = f.read()

    full_dataset = APLDataset_v1(
        text=corpus_text,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        tokens_path="data/apl_corpus_tokens.pt",
    )

    split_idx = int(len(full_dataset) * 0.9)
    train_indices = list(range(0, max(1, split_idx)))
    val_indices = list(range(max(1, split_idx), len(full_dataset)))
    if not val_indices:
        val_indices = train_indices[: max(1, len(train_indices) // 5)]

    train_data = torch.utils.data.Subset(full_dataset, train_indices)
    val_data = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    config = APL_SLMConfig_v1(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        version=1,
    )

    model = APL_SLM_v1(config).to(device)
    print(f"[+] APL SLM v1 initialized with {model.count_parameters():,} trainable parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    checkpoint_dir = Path("checkpoints") / args.exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "apl_slm_best.pt"
    last_ckpt_path = checkpoint_dir / "apl_slm.pt"
    history_path = checkpoint_dir / "history.json"
    tokenizer.save(checkpoint_dir / "tokenizer.json")

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"experiment": args.exp_name, "version": 1, "epochs": []}

    if args.resume and best_ckpt_path.exists():
        print(f"[+] Resuming from existing checkpoint: {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        for x_tok, y_tok in train_loader:
            x_tok, y_tok = x_tok.to(device), y_tok.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(x_tok)
            loss = criterion(logits.view(-1, logits.size(-1)), y_tok.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_ppl = math.exp(min(avg_train_loss, 20.0))

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_tok, y_tok in val_loader:
                x_tok, y_tok = x_tok.to(device), y_tok.to(device)
                logits, _, _ = model(x_tok)
                loss = criterion(logits.view(-1, logits.size(-1)), y_tok.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_ppl = math.exp(min(avg_val_loss, 20.0))
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] Completed in {epoch_time:.2f}s | LR: {current_lr:.6f}\n"
            f"  - Train Loss: {avg_train_loss:.4f} | Train Perplexity: {train_ppl:.2f}\n"
            f"  - Val Loss:   {avg_val_loss:.4f} | Val Perplexity:   {val_ppl:.2f}"
        )

        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "train_ppl": round(train_ppl, 2),
                "val_loss": round(avg_val_loss, 4),
                "val_ppl": round(val_ppl, 2),
                "lr": current_lr,
                "duration": round(epoch_time, 2),
            }
        )
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "model_version": 1,
            "args": vars(args),
            "epoch": epoch,
            "val_loss": avg_val_loss,
            "val_ppl": val_ppl,
        }
        torch.save(checkpoint_dict, last_ckpt_path)

        if avg_val_loss < best_val_loss:
            old_best = best_val_loss
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(checkpoint_dict, best_ckpt_path)
            if old_best == float("inf"):
                print(f"  [OK] Initial best model saved to: {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")
            else:
                print(f"  [OK] Validation Loss improved from {old_best:.4f} to {best_val_loss:.4f}. Saved best checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"[!] Early stopping triggered after {epoch} epochs.")
                break

    print(f"\n[OK] Training completed. Best validation loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train APL SLM v1 (Baseline)")
    parser.add_argument("--data_file", type=str, default="data/apl_corpus.txt")
    parser.add_argument("--exp_name", type=str, default="Small-v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--precision", type=str, default="auto")
    parser.add_argument("--model_preset", type=str, default="small")
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Model presets
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

    train_v1(args)


if __name__ == "__main__":
    main()
