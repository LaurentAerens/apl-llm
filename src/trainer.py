"""
Unified Training Pipeline and Trainer Engine for APL Small Language Models.
Coordinates dataset loading, model initialization, mixed-precision training,
cosine learning rate scheduling, validation evaluation, early stopping, and checkpointing.
"""

import os
import sys
import time
import json
import math
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tokenizer import APLTokenizer
from config import (
    APLModelConfig,
    create_config_for_version,
    resolve_preset_dimensions,
    MODEL_PRESETS,
)
from dataset import APLDataset, create_dataloaders
from synthetic_generator import APLSyntheticGenerator
from v1.model_v1 import APL_SLM_v1
from v2.model_v2 import APL_SLM_v2
from v3.model_v3 import APL_SLM_v3


class APLTrainer:
    """
    Unified training orchestrator for all APL SLM architecture versions (v1, v2, v3).
    """

    def __init__(self, args: Any):
        self.args = args

        # Architecture and Presets
        self.version = getattr(args, "version", 3) or 3
        preset_key = getattr(args, "model_preset", "small") or "small"
        n_layer, n_head, n_embd = resolve_preset_dimensions(
            preset_key,
            n_layer=getattr(args, "n_layer", None),
            n_head=getattr(args, "n_head", None),
            n_embd=getattr(args, "n_embd", None),
        )
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.exp_name = getattr(args, "exp_name", None) or f"{preset_key.capitalize()}-v{self.version}"
        self.epochs = int(getattr(args, "epochs", 30))
        self.batch_size = int(getattr(args, "batch_size", 16))
        self.seq_len = int(getattr(args, "seq_len", 512))
        self.lr = float(getattr(args, "lr", 5e-4))
        self.warmup_epochs = int(getattr(args, "warmup_epochs", 2))
        self.grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
        self.early_stopping_patience = int(getattr(args, "early_stopping_patience", 6))
        self.precision = getattr(args, "precision", "auto")
        self.depth_loss_weight = float(getattr(args, "depth_loss_weight", 0.2))
        self.data_file = getattr(args, "data_file", "data/apl_corpus.txt")
        self.finetune_from = getattr(args, "finetune_from", None)
        self.resume = bool(getattr(args, "resume", False))
        self.demo = bool(getattr(args, "demo", False))

        # Adjust LR if fine-tuning
        if self.finetune_from and self.lr == 5e-4:
            self.lr = 5e-5

        # Device setup
        device_arg = getattr(args, "device", None)
        if device_arg:
            self.device = torch.device(device_arg)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cpu":
            import multiprocessing
            cores = multiprocessing.cpu_count()
            torch.set_num_threads(cores)

        self.tokenizer = APLTokenizer()

    def _setup_scaler(self, enabled: bool):
        """Initializes GradScaler with modern PyTorch syntax and backward-compatibility."""
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                return torch.amp.GradScaler("cuda", enabled=enabled)
            except Exception:
                pass
        return torch.cuda.amp.GradScaler(enabled=enabled)

    def _setup_precision(self) -> tuple[bool, torch.dtype, Any]:
        use_autocast = False
        autocast_dtype = torch.float32
        scaler = self._setup_scaler(enabled=False)

        if self.precision == "auto":
            if self.device.type == "cpu":
                use_autocast = True
                autocast_dtype = torch.bfloat16
            elif self.device.type == "cuda" and torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability(self.device)
                if major >= 8 and torch.cuda.is_bf16_supported():
                    use_autocast = True
                    autocast_dtype = torch.bfloat16
                elif major >= 7:
                    use_autocast = True
                    autocast_dtype = torch.float16
                    scaler = self._setup_scaler(enabled=True)
        elif self.precision == "bf16":
            use_autocast = True
            autocast_dtype = torch.bfloat16
        elif self.precision == "fp16":
            use_autocast = True
            autocast_dtype = torch.float16
            scaler = self._setup_scaler(enabled=(self.device.type == "cuda"))

        return use_autocast, autocast_dtype, scaler

    def _load_corpus_text(self) -> str:
        data_path = Path(self.data_file)
        if not data_path.exists():
            print(f"[!] Data file {data_path} not found. Generating synthetic APL corpus...")
            corpus_text = APLSyntheticGenerator.generate_synthetic_corpus(count=10000)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(corpus_text)
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                corpus_text = f.read()

        if self.demo:
            corpus_text = corpus_text[:100000]
            print(f"[!] Demo Mode Active: Truncated dataset to first {len(corpus_text):,} characters")

        return corpus_text

    def _instantiate_model(self) -> tuple[nn.Module, APLModelConfig]:
        config = create_config_for_version(
            version=self.version,
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=self.seq_len,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=getattr(self.args, "dropout", 0.0),
        )

        if self.version == 1:
            model = APL_SLM_v1(config).to(self.device)
        elif self.version == 2:
            model = APL_SLM_v2(config).to(self.device)
        else:
            model = APL_SLM_v3(config).to(self.device)

        if self.finetune_from and Path(self.finetune_from).exists():
            print(f"[+] Loading weights from base checkpoint for fine-tuning: {self.finetune_from}")
            base_ckpt = torch.load(self.finetune_from, map_location=self.device, weights_only=False)
            model.load_state_dict(base_ckpt["model_state_dict"], strict=False)

        return model, config

    def train(self):
        use_autocast, autocast_dtype, scaler = self._setup_precision()
        corpus_text = self._load_corpus_text()

        suffix = "_demo.pt" if self.demo else ".pt"
        corpus_stem = Path(self.data_file).stem
        tokens_path = Path("data") / f"{corpus_stem}_tokens{suffix}"
        depths_path = Path("data") / f"{corpus_stem}_depths{suffix}"

        full_dataset = APLDataset(
            text=corpus_text,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            tokens_path=tokens_path,
            depths_path=depths_path,
            force_recompute=self.demo,
        )

        train_loader, val_loader = create_dataloaders(
            full_dataset,
            batch_size=self.batch_size,
            val_split=0.1,
            pin_memory=(self.device.type == "cuda"),
        )

        model, config = self._instantiate_model()
        n_params = model.count_parameters() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters())

        print("=" * 65)
        print(f"[+] Initialized APL SLM v{self.version} ({self.exp_name})")
        print(f"   Architecture: {self.n_layer}L / {self.n_head}H / {self.n_embd}D ({n_params:,} parameters)")
        print(f"   Device: {self.device} | Precision: {self.precision}")
        print(f"   Dataset: {len(full_dataset):,} chunks ({len(train_loader)} train batches, {len(val_loader)} val batches)")
        print("=" * 65)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        tok_criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
        depth_criterion = nn.CrossEntropyLoss() if self.version in (2, 3) else None

        warmup_steps = self.warmup_epochs * max(1, len(train_loader) // self.grad_accum_steps)
        total_steps = self.epochs * max(1, len(train_loader) // self.grad_accum_steps)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        checkpoint_dir = Path("checkpoints") / self.exp_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = checkpoint_dir / "apl_slm_best.pt"
        last_ckpt_path = checkpoint_dir / "apl_slm.pt"
        history_path = checkpoint_dir / "history.json"
        self.tokenizer.save(checkpoint_dir / "tokenizer.json")

        start_epoch = 1
        best_val_loss = float("inf")
        patience_counter = 0
        history = []

        if self.resume and (best_ckpt_path.exists() or last_ckpt_path.exists()):
            resume_path = best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path
            print(f"[+] Resuming from existing checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
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

        start_time = time.time()
        total_batches = len(train_loader)

        for epoch in range(start_epoch, self.epochs + 1):
            epoch_start = time.time()
            model.train()
            total_train_loss = 0.0
            last_log_time = time.time()
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, (x_tok, y_tok, x_depth, y_depth) in enumerate(train_loader):
                x_tok, y_tok = x_tok.to(self.device), y_tok.to(self.device)
                x_depth, y_depth = x_depth.to(self.device), y_depth.to(self.device)

                with torch.amp.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=use_autocast):
                    if self.version in (2, 3):
                        logits, depth_logits, _ = model(x_tok, depth_ids=x_depth)
                        loss_tok = tok_criterion(logits.view(-1, logits.size(-1)), y_tok.view(-1))
                        loss_depth = depth_criterion(depth_logits.view(-1, depth_logits.size(-1)), y_depth.view(-1))
                        raw_loss = loss_tok + self.depth_loss_weight * loss_depth
                    else:
                        logits, _, _ = model(x_tok)
                        raw_loss = tok_criterion(logits.view(-1, logits.size(-1)), y_tok.view(-1))

                    loss = raw_loss / self.grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                is_accum_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or ((batch_idx + 1) == total_batches)
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
                        f"  [Epoch {epoch}/{self.epochs} | Step {step:5d}/{total_batches} ({pct:5.1f}%)] "
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
                for x_tok, y_tok, x_depth, y_depth in val_loader:
                    x_tok, y_tok = x_tok.to(self.device), y_tok.to(self.device)
                    x_depth, y_depth = x_depth.to(self.device), y_depth.to(self.device)

                    with torch.amp.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=use_autocast):
                        if self.version in (2, 3):
                            logits, depth_logits, _ = model(x_tok, depth_ids=x_depth)
                            l_tok = tok_criterion(logits.view(-1, logits.size(-1)), y_tok.view(-1))
                            l_depth = depth_criterion(depth_logits.view(-1, depth_logits.size(-1)), y_depth.view(-1))
                            loss = l_tok + self.depth_loss_weight * l_depth
                        else:
                            logits, _, _ = model(x_tok)
                            loss = tok_criterion(logits.view(-1, logits.size(-1)), y_tok.view(-1))

                    total_val_loss += loss.item()
                    val_steps += 1

            avg_val_loss = total_val_loss / max(1, val_steps)
            val_ppl = math.exp(min(avg_val_loss, 20.0))
            epoch_duration = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]

            print(f"\n[Epoch {epoch}/{self.epochs}] Duration: {epoch_duration:.1f}s | LR: {current_lr:.6f}")
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
                json.dump({"experiment": self.exp_name, "version": self.version, "epochs": history}, f, indent=2)

            checkpoint_dict = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "model_version": self.version,
                "args": vars(self.args) if hasattr(self.args, "__dict__") else {},
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
                print(f"  [-] Patience: {patience_counter}/{self.early_stopping_patience}")
                if patience_counter >= self.early_stopping_patience:
                    print(f"\n[!] Early stopping triggered at epoch {epoch}.")
                    break

        elapsed = time.time() - start_time
        print(f"\n[OK] v{self.version} Training Complete in {elapsed:.2f}s! Best Validation Loss: {best_val_loss:.4f}")
        return model, best_val_loss

