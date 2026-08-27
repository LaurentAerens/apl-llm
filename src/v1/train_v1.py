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
    # Presets mapping
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
    preset_key = getattr(args, "model_preset", "small") or "small"
    if preset_key.lower() in presets:
        p = presets[preset_key.lower()]
        n_layer = getattr(args, "n_layer", None) or p["n_layer"]
        n_head = getattr(args, "n_head", None) or p["n_head"]
        n_embd = getattr(args, "n_embd", None) or p["n_embd"]
    else:
        n_layer = getattr(args, "n_layer", None) or 4
        n_head = getattr(args, "n_head", None) or 4
        n_embd = getattr(args, "n_embd", None) or 64

    exp_name = getattr(args, "exp_name", None) or "Small-v1"
    grad_accum_steps = max(1, getattr(args, "grad_accum_steps", 1))
    early_stopping_patience = getattr(args, "early_stopping_patience", 6)
    precision = getattr(args, "precision", "auto")

    if getattr(args, "finetune_from", None) and args.lr == 5e-4:
        args.lr = 5e-5
        print(f"[+] Fine-tuning mode active: Automatically adjusted default learning rate to {args.lr}")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[+] Using Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"[+] GPU Detected: {gpu_name}")
    elif device.type == "cpu":
        import multiprocessing
        cores = multiprocessing.cpu_count()
        torch.set_num_threads(cores)
        print(f"[+] CPU training optimized: using {cores} threads")

    tokenizer = APLTokenizer()
    data_path = Path(getattr(args, "data_file", "data/apl_corpus.txt"))
    if not data_path.exists():
        print(f"[!] Data file {data_path} not found. Generating synthetic APL corpus...")
        corpus_text = APLSyntheticGenerator.generate_synthetic_corpus(count=10000)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(corpus_text)
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            corpus_text = f.read()

    if getattr(args, "demo", False):
        corpus_text = corpus_text[:100000]
        print(f"[!] Demo Mode Active: Truncating dataset to first {len(corpus_text):,} characters")

    tokens_path = "data/apl_corpus_tokens_demo.pt" if getattr(args, "demo", False) else "data/apl_corpus_tokens.pt"
    full_dataset = APLDataset_v1(
        text=corpus_text,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        tokens_path=tokens_path,
        force_recompute=getattr(args, "demo", False),
    )

    split_idx = int(len(full_dataset) * 0.9)
    train_indices = list(range(0, max(1, split_idx)))
    val_indices = list(range(max(1, split_idx), len(full_dataset)))
    if not val_indices:
        val_indices = train_indices[: max(1, len(train_indices) // 5)]

    train_data = torch.utils.data.Subset(full_dataset, train_indices)
    val_data = torch.utils.data.Subset(full_dataset, val_indices)

    use_pin = (device.type == "cuda")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=use_pin)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=use_pin)

    config = APL_SLMConfig_v1(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=getattr(args, "dropout", 0.0),
        version=1,
    )

    model = APL_SLM_v1(config).to(device)

    if getattr(args, "finetune_from", None) and Path(args.finetune_from).exists():
        print(f"[+] Loading weights from base checkpoint for fine-tuning: {args.finetune_from}")
        base_ckpt = torch.load(args.finetune_from, map_location=device, weights_only=False)
        model.load_state_dict(base_ckpt["model_state_dict"], strict=False)

    print(f"[+] Initialized APL_SLM v1 ({n_layer}L, {n_head}H, {n_embd}D). Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    warmup_steps = args.warmup_epochs * max(1, len(train_loader) // grad_accum_steps)
    total_steps = args.epochs * max(1, len(train_loader) // grad_accum_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Precision & Autocast setup
    use_autocast = False
    autocast_dtype = torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    if precision == "auto":
        if device.type == "cpu":
            use_autocast = True
            autocast_dtype = torch.bfloat16
        elif device.type == "cuda" and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(device)
            if major >= 8 and torch.cuda.is_bf16_supported():
                use_autocast = True
                autocast_dtype = torch.bfloat16
            elif major >= 7:
                use_autocast = True
                autocast_dtype = torch.float16
                scaler = torch.cuda.amp.GradScaler(enabled=True)
    elif precision == "bf16":
        use_autocast = True
        autocast_dtype = torch.bfloat16
    elif precision == "fp16":
        use_autocast = True
        autocast_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    checkpoint_dir = Path("checkpoints") / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "apl_slm_best.pt"
    last_ckpt_path = checkpoint_dir / "apl_slm.pt"
    history_path = checkpoint_dir / "history.json"
    tokenizer.save(checkpoint_dir / "tokenizer.json")

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    if getattr(args, "resume", False) and (best_ckpt_path.exists() or last_ckpt_path.exists()):
        resume_path = best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path
        print(f"[+] Resuming from existing checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    hist_data = json.load(f)
                    history = hist_data.get("epochs", []) if isinstance(hist_data, dict) else hist_data
            except Exception:
                pass

    print("\n" + "=" * 60)
    print(f"🚀 Starting APL SLM v1 Training Loop ({exp_name})")
    print("=" * 60)

    start_time = time.time()
    total_batches = len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        last_log_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x_tok, y_tok) in enumerate(train_loader):
            x_tok, y_tok = x_tok.to(device), y_tok.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                logits, _, _ = model(x_tok)
                raw_loss = criterion(logits.view(-1, logits.size(-1)), y_tok.view(-1))
                loss = raw_loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_accum_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == total_batches)
            if is_accum_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_train_loss += raw_loss.item()

            now = time.time()
            if (now - last_log_time >= 1.0) or (batch_idx + 1 == total_batches) or (batch_idx == 0):
                step = batch_idx + 1
                pct = (step / total_batches) * 100.0
                elapsed_ep = now - epoch_start
                rate = step / max(0.01, elapsed_ep)
                eta = (total_batches - step) / max(0.01, rate)
                curr_loss = raw_loss.item()
                curr_lr = scheduler.get_last_lr()[0]
                print(
                    f"  [Epoch {epoch}/{args.epochs} | Step {step:5d}/{total_batches} ({pct:5.1f}%)] "
                    f"Loss: {curr_loss:.4f} | LR: {curr_lr:.6f} | {rate:.1f} it/s | ETA: {eta:.1f}s",
                    flush=True,
                )
                last_log_time = now

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_ppl = math.exp(min(avg_train_loss, 20.0))

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for x_tok, y_tok in val_loader:
                x_tok, y_tok = x_tok.to(device), y_tok.to(device)
                with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                    logits, _, _ = model(x_tok)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_tok.view(-1))
                total_val_loss += loss.item()
                val_steps += 1

        avg_val_loss = total_val_loss / max(1, val_steps)
        val_ppl = math.exp(min(avg_val_loss, 20.0))
        epoch_duration = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        print(f"\n[Epoch {epoch}/{args.epochs}] Duration: {epoch_duration:.1f}s | LR: {current_lr:.6f}")
        print(f"  - Train Loss: {avg_train_loss:.4f} | Train Perplexity: {train_ppl:.2f}")
        print(f"  - Val Loss:   {avg_val_loss:.4f} | Val Perplexity:   {val_ppl:.2f}")

        history.append({
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "train_ppl": float(train_ppl),
            "val_loss": float(avg_val_loss),
            "val_ppl": float(val_ppl),
            "lr": float(current_lr),
            "duration": float(epoch_duration),
        })
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({"experiment": exp_name, "version": 1, "epochs": history}, f, indent=2)

        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "model_version": 1,
            "args": vars(args) if hasattr(args, "__dict__") else {},
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
                print(f"  [+] Validation Loss improved from {old_best:.4f} to {best_val_loss:.4f}! Saved best checkpoint.")
        else:
            patience_counter += 1
            print(f"  [-] Patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch}.")
                break

    elapsed = time.time() - start_time
    print(f"\n[OK] v1 Training Complete in {elapsed:.2f}s! Best Validation Loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train APL SLM v1 (Baseline)")
    parser.add_argument("--model_version", "--version", "-v", dest="version", type=int, default=1)
    parser.add_argument("--data_file", "--dataset_path", dest="data_file", type=str, default="data/apl_corpus.txt")
    parser.add_argument("--exp_name", type=str, default="Small-v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--model_preset", type=str, default="small")
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--finetune_from", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    train_v1(args)


if __name__ == "__main__":
    main()
