import os
import sys
import time
import json
import queue
import threading
import subprocess
import re
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk

# Ensure the 'src/' directory is in the import path
sys.path.append(str(Path(__file__).parent))

# Import LLM autocomplete functions directly
try:
    from v1.autocomplete_v1 import autocomplete_v1
    from v2.autocomplete_v2 import autocomplete_v2
    from v3.autocomplete_v3 import autocomplete_v3
    from autocomplete import load_model, autocomplete
except ImportError:
    autocomplete_v1, autocomplete_v2, autocomplete_v3 = None, None, None
    load_model, autocomplete = None, None

from queue_manager import TrainingQueueManager, TrainingJob


class APL_SLM_GUI:
    # Recommended hyperparameter presets tailored to each model architecture
    MODEL_PRESETS = {
        "small": {
            "base_name": "Small",
            "exp_name": "Small-v3",
            "epochs": "40",
            "batch_size": "16",
            "lr": "5e-4",
            "warmup_epochs": "2",
            "patience": "6",
            "info": "🍏 Small (4 Layers / 4 Heads / 64 Dim • ~380k params)\nRecommended: LR 5e-4, Warmup 2 epochs, Patience 6 — Lightweight & fast CPU baseline",
        },
        "medium": {
            "base_name": "Medium",
            "exp_name": "Medium-v3",
            "epochs": "30",
            "batch_size": "16",
            "lr": "3e-4",
            "warmup_epochs": "3",
            "patience": "8",
            "info": "🍏 Medium (6 Layers / 8 Heads / 256 Dim • ~2.4M params)\nRecommended: LR 3e-4, Warmup 3 epochs, Patience 8 — Balanced capacity & training speed",
        },
        "large": {
            "base_name": "Large",
            "exp_name": "Large-v3",
            "epochs": "25",
            "batch_size": "16",
            "lr": "1.5e-4",
            "warmup_epochs": "5",
            "patience": "10",
            "info": "🍏 Large (8 Layers / 12 Heads / 384 Dim • ~7.1M params)\nRecommended: LR 1.5e-4, Warmup 5 epochs, Patience 10 — High reasoning & syntax precision",
        },
        "deep": {
            "base_name": "Deep",
            "exp_name": "Deep-v3",
            "epochs": "30",
            "batch_size": "16",
            "lr": "2e-4",
            "warmup_epochs": "4",
            "patience": "10",
            "info": "🌊 Deep (12 Layers / 8 Heads / 256 Dim • ~4.7M params)\nRecommended: LR 2e-4, Warmup 4 epochs, Patience 10 — Deep hierarchical sequence modeling",
        },
        "wide": {
            "base_name": "Wide",
            "exp_name": "Wide-v3",
            "epochs": "25",
            "batch_size": "16",
            "lr": "1.5e-4",
            "warmup_epochs": "5",
            "patience": "10",
            "info": "📐 Wide (6 Layers / 16 Heads / 512 Dim • ~9.5M params)\nRecommended: LR 1.5e-4, Warmup 5 epochs, Patience 10 — Broad multi-head representation capacity",
        },
        "xlarge": {
            "base_name": "XLarge",
            "exp_name": "XLarge-v3",
            "epochs": "20",
            "batch_size": "16",
            "lr": "1.0e-4",
            "warmup_epochs": "4",
            "patience": "10",
            "info": "🌟 XLarge (12 Layers / 12 Heads / 512 Dim • ~85.5M params)\nRecommended: LR 1.0e-4, Warmup 4 epochs, Patience 10 — Large capacity for whole-file APL script synthesis",
        },
        "huge": {
            "base_name": "Huge",
            "exp_name": "Huge-v3",
            "epochs": "15",
            "batch_size": "8",
            "lr": "8e-5",
            "warmup_epochs": "5",
            "patience": "10",
            "info": "🏛️ Huge (16 Layers / 16 Heads / 768 Dim • ~135.5M params)\nRecommended: LR 8e-5, Warmup 5 epochs, Patience 10 — High-capacity transformer for complex array idioms",
        },
        "giant": {
            "base_name": "Giant",
            "exp_name": "Giant-v3",
            "epochs": "15",
            "batch_size": "8",
            "lr": "6e-5",
            "warmup_epochs": "5",
            "patience": "12",
            "info": "🌌 Giant (24 Layers / 16 Heads / 1024 Dim • ~202M params)\nRecommended: LR 6e-5, Warmup 5 epochs, Patience 12 — Maximum capacity architecture for enterprise APL codebases",
        },
        "(none)": {
            "base_name": "Custom",
            "exp_name": "custom-model",
            "epochs": "30",
            "batch_size": "16",
            "lr": "3e-4",
            "warmup_epochs": "3",
            "patience": "8",
            "info": "🔧 Custom / Manual: Enter custom model architecture and hyperparameter settings",
        },
    }

    @staticmethod
    def get_next_experiment_name(
        base_name: str,
        checkpoints_dir: Path = Path("checkpoints"),
        queued_jobs: list = None,
    ) -> str:
        """
        Calculates the next auto-incremented experiment name (e.g. Small-v3.0, Small-v3.1, Small-v3.2)
        by scanning existing directories and checkpoint files in checkpoints/ as well as any queued jobs.
        """
        clean_base = re.sub(r"\.\d+$", "", base_name.strip())
        pattern = re.compile(rf"^{re.escape(clean_base)}(\.(\d+))?$", re.IGNORECASE)
        found_indices = []
        if checkpoints_dir.exists():
            for item in checkpoints_dir.iterdir():
                name = item.stem if item.is_file() else item.name
                m = pattern.match(name)
                if m:
                    if m.group(2) is not None:
                        try:
                            found_indices.append(int(m.group(2)))
                        except ValueError:
                            pass
                    else:
                        found_indices.append(0)

        if queued_jobs:
            for job in queued_jobs:
                job_exp = getattr(job, "exp_name", "") if not isinstance(job, dict) else job.get("exp_name", "")
                m = pattern.match(job_exp.strip())
                if m:
                    if m.group(2) is not None:
                        try:
                            found_indices.append(int(m.group(2)))
                        except ValueError:
                            pass
                    else:
                        found_indices.append(0)

        if not found_indices:
            return f"{clean_base}.0"
        next_index = max(found_indices) + 1
        return f"{clean_base}.{next_index}"

    def __init__(self, root):
        self.root = root
        self.root.title("🍏 APL SLM Studio (Green Apple Edition)")
        self.root.geometry("1140x780")
        self.root.minsize(980, 680)

        # Threading and Subprocess state
        self.log_queue = queue.Queue()
        self.running_process = None
        self.running_thread = None
        self.process_active = False

        # Training Queue & Batch Scheduler
        self.queue_manager = TrainingQueueManager()
        self.batch_running = False
        self.batch_stop_requested = False

        # Cached autocomplete model
        self.cached_model = None
        self.cached_tokenizer = None
        self.cached_device = None
        self.cached_checkpoint_path = None

        # 🍏 Iconic APL Green Apple & Emerald Dark Theme Palette
        self.colors = {
            "bg_dark": "#07150e",        # Deep Forest Obsidian
            "bg_side": "#0e2419",        # Rich Pine Green Sidebar
            "bg_card": "#163526",        # Dark Emerald Card Surface
            "bg_card_light": "#1d4633",  # Elevated Emerald Card
            "accent": "#22c55e",         # Granny Smith Green Apple
            "accent_hover": "#4ade80",   # Bright Crisp Apple Green
            "accent_apple": "#10b981",   # Classic Emerald Jade
            "apple_leaf": "#84cc16",     # Lime Leaf Accent
            "apple_red": "#ef4444",      # Apple Red (Kill / Danger)
            "apple_gold": "#eab308",     # Golden Apple (Best Model Stars)
            "text": "#f0fdf4",           # Crisp Mint White
            "text_muted": "#86efac",     # Pale Mint Sage
            "terminal_bg": "#030a06",    # Deepest Moss Obsidian
            "terminal_fg": "#4ade80",    # Phosphor Green Apple Text
            "danger": "#ef4444",
            "warning": "#f59e0b",
        }

        # Apply root background color
        self.root.configure(bg=self.colors["bg_dark"])

        # Configure standard styling
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=self.colors["bg_card"], background=self.colors["bg_card"], foreground=self.colors["text"])
        style.configure("TCheckbutton", background=self.colors["bg_dark"], foreground=self.colors["text"])

        # Create Layout
        self.create_layout()

        # Load list of checkpoints
        self.refresh_checkpoints()

        # Start log queue polling
        self.poll_log_queue()

    def create_layout(self):
        # 1. Left Sidebar Navigation
        self.sidebar = tk.Frame(self.root, bg=self.colors["bg_side"], width=230)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        # 🍏 Logo Header
        logo_frame = tk.Frame(self.sidebar, bg=self.colors["bg_side"])
        logo_frame.pack(pady=(20, 15), padx=10, fill=tk.X)

        title_lbl = tk.Label(
            logo_frame,
            text="🍏 APL·SLM",
            font=("Segoe UI", 18, "bold"),
            bg=self.colors["bg_side"],
            fg=self.colors["accent_hover"],
        )
        title_lbl.pack(anchor="center")

        subtitle_lbl = tk.Label(
            logo_frame,
            text="GREEN APPLE STUDIO",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors["bg_side"],
            fg=self.colors["apple_leaf"],
        )
        subtitle_lbl.pack(anchor="center", pady=(2, 0))

        # Nav Buttons container
        self.nav_frame = tk.Frame(self.sidebar, bg=self.colors["bg_side"])
        self.nav_frame.pack(fill=tk.BOTH, expand=True, padx=5)

        self.tabs = {}
        self.active_tab = None

        # Build Sidebar Navigation Buttons with Apple & APL Icons
        nav_items = [
            ("Dataset", "🍏 Dataset", self.create_dataset_tab),
            ("Tokenizer", "🔤 Tokenizer", self.create_tokenizer_tab),
            ("Training", "🚀 Training", self.create_training_tab),
            ("Scheduler", "📋 Scheduler", self.create_scheduler_tab),
            ("Fine-tune", "🎯 Fine-Tune", self.create_finetune_tab),
            ("Plots", "📈 Metrics", self.create_plots_tab),
            ("Autocomplete", "⚡ Autocomplete", self.create_autocomplete_tab),
            ("Benchmarks", "🏆 Benchmarks", self.create_benchmarks_tab),
        ]

        self.nav_buttons = {}
        for tab_id, label, creator in nav_items:
            tab_frame = tk.Frame(self.root, bg=self.colors["bg_dark"])
            self.tabs[tab_id] = (tab_frame, creator)

            btn = tk.Button(
                self.nav_frame,
                text=label,
                font=("Segoe UI", 11, "bold"),
                bg=self.colors["bg_side"],
                fg=self.colors["text"],
                activebackground=self.colors["bg_card"],
                activeforeground=self.colors["accent_hover"],
                bd=0,
                height=2,
                anchor="w",
                padx=18,
                cursor="hand2",
                command=lambda tid=tab_id: self.switch_tab(tid),
            )
            btn.pack(fill=tk.X, pady=2)
            self.nav_buttons[tab_id] = btn

        # Footer Status Panel in Sidebar
        self.status_panel = tk.Frame(self.sidebar, bg=self.colors["bg_side"], bd=0)
        self.status_panel.pack(side=tk.BOTTOM, fill=tk.X, pady=15, padx=12)

        self.status_title = tk.Label(
            self.status_panel,
            text="ENGINE STATUS",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors["bg_side"],
            fg=self.colors["text_muted"],
        )
        self.status_title.pack(anchor="w")

        self.status_desc = tk.Label(
            self.status_panel,
            text="● Idle",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg_side"],
            fg=self.colors["accent"],
        )
        self.status_desc.pack(anchor="w")

        # 2. Main Content View Area (right side)
        self.content_area = tk.Frame(self.root, bg=self.colors["bg_dark"])
        self.content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Default to Dataset Tab
        self.switch_tab("Dataset")

    def switch_tab(self, tab_name):
        if self.active_tab == tab_name:
            return

        if self.active_tab:
            active_frame, _ = self.tabs[self.active_tab]
            active_frame.pack_forget()
            self.nav_buttons[self.active_tab].configure(bg=self.colors["bg_side"], fg=self.colors["text"])

        self.active_tab = tab_name
        target_frame, creator = self.tabs[tab_name]

        if not target_frame.winfo_children():
            creator(target_frame)

        target_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=20)
        self.nav_buttons[tab_name].configure(bg=self.colors["bg_card"], fg=self.colors["accent_hover"])

        self.refresh_checkpoints()
        if tab_name == "Plots":
            self.load_and_display_plot()

    # ----------------------------------------------------
    # TAB 0: Dataset Collection
    # ----------------------------------------------------
    def create_dataset_tab(self, frame):
        header = tk.Label(frame, text="🍏 Collect & Build APL Dataset", font=("Segoe UI", 18, "bold"), bg=self.colors["bg_dark"], fg=self.colors["accent_hover"])
        header.pack(anchor="w", pady=(0, 10))

        desc = tk.Label(
            frame,
            text="Collects open-source APL repositories from GitHub and GitLab under verified permissive licenses, generates ATTRIBUTION.md, and augments with synthetic algorithmic idioms and dfns.",
            font=("Segoe UI", 10),
            bg=self.colors["bg_dark"],
            fg=self.colors["text_muted"],
            wraplength=740,
            justify=tk.LEFT,
        )
        desc.pack(anchor="w", pady=(0, 15))

        card = tk.Frame(frame, bg=self.colors["bg_card"], bd=0, padx=16, pady=16)
        card.pack(fill=tk.X, pady=(0, 15))

        # Data Sources Checkboxes
        tk.Label(card, text="Data Sources:", font=("Segoe UI", 10, "bold"), bg=self.colors["bg_card"], fg=self.colors["text"]).grid(row=0, column=0, sticky="w", pady=6)

        sources_frame = tk.Frame(card, bg=self.colors["bg_card"])
        sources_frame.grid(row=0, column=1, sticky="w", padx=10, pady=6)

        self.src_github_var = tk.BooleanVar(value=True)
        self.src_gitlab_var = tk.BooleanVar(value=True)
        self.src_synthetic_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(sources_frame, text="GitHub", variable=self.src_github_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(sources_frame, text="GitLab", variable=self.src_gitlab_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(sources_frame, text="Synthetic Idioms", variable=self.src_synthetic_var).pack(side=tk.LEFT)

        # Mode Selector
        tk.Label(card, text="Collection Mode:", font=("Segoe UI", 10), bg=self.colors["bg_card"], fg=self.colors["text"]).grid(row=1, column=0, sticky="w", pady=6)
        self.ds_mode = ttk.Combobox(card, values=["curated", "search", "all"], width=15, state="readonly")
        self.ds_mode.current(0)
        self.ds_mode.grid(row=1, column=1, sticky="w", padx=10, pady=6)

        # Limit
        tk.Label(card, text="Max Repos to Scan (0 for All):", font=("Segoe UI", 10), bg=self.colors["bg_card"], fg=self.colors["text"]).grid(row=2, column=0, sticky="w", pady=6)
        self.ds_limit = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=10)
        self.ds_limit.insert(0, "50")
        self.ds_limit.grid(row=2, column=1, sticky="w", padx=10, pady=6)

        # Github Token (PAT)
        tk.Label(card, text="GitHub Personal Access Token (Optional):", font=("Segoe UI", 10), bg=self.colors["bg_card"], fg=self.colors["text"]).grid(row=3, column=0, sticky="w", pady=6)
        self.ds_token = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=35, show="*")
        self.ds_token.grid(row=3, column=1, sticky="w", padx=10, pady=6)

        # Run Controls
        btn_frame = tk.Frame(frame, bg=self.colors["bg_dark"])
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_run_ds = tk.Button(
            btn_frame,
            text="🍏 Build Dataset (GitHub + GitLab)",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["accent"],
            fg=self.colors["bg_dark"],
            activebackground=self.colors["accent_hover"],
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.run_dataset_collection,
        )
        self.btn_run_ds.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_gen_synth = tk.Button(
            btn_frame,
            text="✨ Generate Synthetic Idioms",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["apple_leaf"],
            fg=self.colors["bg_dark"],
            activebackground=self.colors["accent_hover"],
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.run_synthetic_dataset_generation,
        )
        self.btn_gen_synth.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_kill_ds = tk.Button(
            btn_frame,
            text="Kill Process",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["apple_red"],
            fg=self.colors["text"],
            activebackground="#f87171",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.kill_active_process,
        )
        self.btn_kill_ds.pack(side=tk.LEFT)

        self.create_terminal_log(frame)

    # ----------------------------------------------------
    # TAB 1: Tokenizer
    # ----------------------------------------------------
    def create_tokenizer_tab(self, frame):
        header = tk.Label(frame, text="🔤 APL Glyph Tokenizer Diagnostics", font=("Segoe UI", 18, "bold"), bg=self.colors["bg_dark"], fg=self.colors["accent_hover"])
        header.pack(anchor="w", pady=(0, 10))

        desc = tk.Label(
            frame,
            text="Inspects the full Unicode APL glyph vocabulary (~170 tokens) and structural depth trackers for parentheses (), brackets [], and dfns {}.",
            font=("Segoe UI", 10),
            bg=self.colors["bg_dark"],
            fg=self.colors["text_muted"],
            wraplength=740,
            justify=tk.LEFT,
        )
        desc.pack(anchor="w", pady=(0, 15))

        card = tk.Frame(frame, bg=self.colors["bg_card"], bd=0, padx=16, pady=16)
        card.pack(fill=tk.X, pady=(0, 15))

        tk.Label(card, text="Corpus Dataset File Path:", font=("Segoe UI", 10, "bold"), bg=self.colors["bg_card"], fg=self.colors["text"]).grid(row=0, column=0, sticky="w", pady=6)
        self.tok_corpus = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=50)
        self.tok_corpus.insert(0, "data/apl_corpus.txt")
        self.tok_corpus.grid(row=0, column=1, sticky="w", padx=10, pady=6)

        btn_frame = tk.Frame(frame, bg=self.colors["bg_dark"])
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_inspect_tok = tk.Button(
            btn_frame,
            text="🍏 Inspect Tokenizer & Vocab",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["accent"],
            fg=self.colors["bg_dark"],
            activebackground=self.colors["accent_hover"],
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.inspect_tokenizer,
        )
        self.btn_inspect_tok.pack(side=tk.LEFT)

        self.create_terminal_log(frame)

    # ----------------------------------------------------
    # TAB 2: Model Training
    # ----------------------------------------------------
    def create_training_tab(self, frame):
        header = tk.Label(frame, text="🚀 Train APL SLM Autocomplete", font=("Segoe UI", 18, "bold"), bg=self.colors["bg_dark"], fg=self.colors["accent_hover"])
        header.pack(anchor="w", pady=(0, 10))

        card = tk.Frame(frame, bg=self.colors["bg_card"], bd=0, padx=16, pady=16)
        card.pack(fill=tk.X, pady=(0, 15))

        # Row 0 & 1: Model Preset, Arch Version, Total Epochs, Batch Size
        tk.Label(card, text="Model Preset (Auto-config):", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_hover"]).grid(row=0, column=0, sticky="w", pady=3)
        self.trn_preset = ttk.Combobox(card, values=["small", "medium", "large", "deep", "wide", "xlarge", "huge", "giant", "(none)"], width=13, state="readonly")
        self.trn_preset.current(0)
        self.trn_preset.grid(row=1, column=0, sticky="w", padx=(0, 15), pady=(0, 10))
        self.trn_preset.bind("<<ComboboxSelected>>", self.on_model_preset_changed)

        tk.Label(card, text="Architecture Version:", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_hover"]).grid(row=0, column=1, sticky="w", pady=3)
        self.trn_version = ttk.Combobox(
            card,
            values=[
                "v3 (Modern RoPE + SwiGLU + QK-Norm)",
                "v2 (Structural Depth Conditioned)",
                "v1 (Classic Transformer Baseline)",
            ],
            width=38,
            state="readonly",
        )
        self.trn_version.current(0)
        self.trn_version.grid(row=1, column=1, sticky="w", padx=(0, 15), pady=(0, 10))
        self.trn_version.bind("<<ComboboxSelected>>", self.on_model_preset_changed)

        tk.Label(card, text="Total Epochs:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=0, column=2, sticky="w", pady=3)
        self.trn_epochs = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=10)
        self.trn_epochs.insert(0, "40")
        self.trn_epochs.grid(row=1, column=2, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Batch Size:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=0, column=3, sticky="w", pady=3)
        self.trn_batch = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=10)
        self.trn_batch.insert(0, "16")
        self.trn_batch.grid(row=1, column=3, sticky="w", padx=(0, 15), pady=(0, 10))

        # Row 2 & 3: Learning Rate, Warmup Epochs, Grad Accum Steps, Training Precision
        tk.Label(card, text="Learning Rate:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=0, sticky="w", pady=3)
        self.trn_lr = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=13)
        self.trn_lr.insert(0, "5e-4")
        self.trn_lr.grid(row=3, column=0, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Warmup Epochs:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=1, sticky="w", pady=3)
        self.trn_warmup = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=20)
        self.trn_warmup.insert(0, "2")
        self.trn_warmup.grid(row=3, column=1, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Grad Accum Steps:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=2, sticky="w", pady=3)
        self.trn_accum = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=10)
        self.trn_accum.insert(0, "1")
        self.trn_accum.grid(row=3, column=2, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Training Precision:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=3, sticky="w", pady=3)
        self.trn_prec = ttk.Combobox(card, values=["auto", "bf16", "fp16", "fp32"], width=10, state="readonly")
        self.trn_prec.current(0)
        self.trn_prec.grid(row=3, column=3, sticky="w", padx=(0, 15), pady=(0, 10))

        # Row 4 & 5: Experiment Name, Hardware Device, Early Stop Patience & Checkboxes
        tk.Label(card, text="Experiment Name:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=4, column=0, sticky="w", pady=3)
        self.trn_exp = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=13)
        self.trn_exp.insert(0, "Small-v3.0")
        self.trn_exp.grid(row=5, column=0, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Hardware Device:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=4, column=1, sticky="w", pady=3)
        self.trn_device = ttk.Combobox(card, values=["(auto)", "cpu", "cuda"], width=20, state="readonly")
        self.trn_device.current(0)
        self.trn_device.grid(row=5, column=1, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Early Stop Patience:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=4, column=2, sticky="w", pady=3)
        self.trn_patience = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=10)
        self.trn_patience.insert(0, "6")
        self.trn_patience.grid(row=5, column=2, sticky="w", padx=(0, 15), pady=(0, 10))

        chk_box = tk.Frame(card, bg=self.colors["bg_card"])
        chk_box.grid(row=5, column=3, sticky="w", pady=(0, 10))

        self.trn_resume_val = tk.BooleanVar(value=True)
        self.chk_resume = ttk.Checkbutton(chk_box, text="Resume", variable=self.trn_resume_val, style="TCheckbutton")
        self.chk_resume.pack(side=tk.LEFT, padx=(0, 8))

        self.trn_demo_val = tk.BooleanVar(value=False)
        self.chk_demo = ttk.Checkbutton(chk_box, text="Demo", variable=self.trn_demo_val, style="TCheckbutton")
        self.chk_demo.pack(side=tk.LEFT)

        # Row 6: Preset Architecture & Recommendation Info Banner
        self.lbl_preset_info = tk.Label(
            card,
            text=self.MODEL_PRESETS["small"]["info"],
            font=("Segoe UI", 9),
            bg=self.colors["bg_card_light"],
            fg=self.colors["accent_hover"],
            justify=tk.LEFT,
            padx=12,
            pady=8,
            anchor="w",
        )
        self.lbl_preset_info.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(5, 0))

        self.on_model_preset_changed()

        # Run Controls
        btn_frame = tk.Frame(frame, bg=self.colors["bg_dark"])
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_run_train = tk.Button(
            btn_frame,
            text="🍏 Start Model Training",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["accent"],
            fg=self.colors["bg_dark"],
            activebackground=self.colors["accent_hover"],
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.run_model_training,
        )
        self.btn_run_train.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_queue_train = tk.Button(
            btn_frame,
            text="➕ Add to Queue",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg_card_light"],
            fg=self.colors["text"],
            activebackground=self.colors["bg_card"],
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.add_training_to_queue,
        )
        self.btn_queue_train.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_kill_train = tk.Button(
            btn_frame,
            text="Kill Process",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["apple_red"],
            fg=self.colors["text"],
            activebackground="#f87171",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.kill_active_process,
        )
        self.btn_kill_train.pack(side=tk.LEFT)

        # Live Training Statistics Cards
        stats_container = tk.Frame(frame, bg=self.colors["bg_dark"])
        stats_container.pack(fill=tk.X, pady=(0, 10))

        # Left Card: Last Epoch Stats
        card_last = tk.Frame(stats_container, bg=self.colors["bg_card"], padx=12, pady=10)
        card_last.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        tk.Label(
            card_last,
            text="⏱️ LAST EPOCH STATS",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg_card"],
            fg=self.colors["accent_hover"],
        ).pack(anchor="w", pady=(0, 5))

        self.lbl_trn_last_epoch = tk.Label(
            card_last, text="Epoch: - / -", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["text"]
        )
        self.lbl_trn_last_epoch.pack(anchor="w")

        grid_last = tk.Frame(card_last, bg=self.colors["bg_card"])
        grid_last.pack(fill=tk.X, pady=(4, 0))

        self.lbl_trn_train_loss = tk.Label(grid_last, text="Train Loss: -", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_trn_train_loss.grid(row=0, column=0, sticky="w", padx=(0, 15))

        self.lbl_trn_train_ppl = tk.Label(grid_last, text="Train PPL: -", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_trn_train_ppl.grid(row=0, column=1, sticky="w")

        self.lbl_trn_val_loss = tk.Label(grid_last, text="Val Loss:   -", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_trn_val_loss.grid(row=1, column=0, sticky="w", padx=(0, 15))

        self.lbl_trn_val_ppl = tk.Label(grid_last, text="Val PPL:   -", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_trn_val_ppl.grid(row=1, column=1, sticky="w")

        self.lbl_trn_lr_time = tk.Label(card_last, text="LR: - | Duration: -", font=("Segoe UI", 8), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_trn_lr_time.pack(anchor="w", pady=(4, 0))

        # Right Card: Best Epoch Stats
        card_best = tk.Frame(stats_container, bg=self.colors["bg_card"], padx=12, pady=10)
        card_best.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            card_best,
            text="⭐ BEST MODEL STATS",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg_card"],
            fg=self.colors["apple_gold"],
        ).pack(anchor="w", pady=(0, 5))

        self.lbl_trn_best_epoch = tk.Label(
            card_best, text="Best Epoch: -", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["text"]
        )
        self.lbl_trn_best_epoch.pack(anchor="w")

        grid_best = tk.Frame(card_best, bg=self.colors["bg_card"])
        grid_best.pack(fill=tk.X, pady=(4, 0))

        self.lbl_trn_best_loss = tk.Label(grid_best, text="Best Val Loss: -", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_hover"])
        self.lbl_trn_best_loss.grid(row=0, column=0, sticky="w", padx=(0, 15))

        self.lbl_trn_best_ppl = tk.Label(grid_best, text="Best Val PPL: -", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_trn_best_ppl.grid(row=0, column=1, sticky="w")

        self.lbl_trn_best_status = tk.Label(card_best, text="Status: Waiting for training...", font=("Segoe UI", 8), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_trn_best_status.pack(anchor="w", pady=(4, 0))

        self.create_terminal_log(frame)

    # ----------------------------------------------------
    # TAB: Training Scheduler & Batch Queue Manager
    # ----------------------------------------------------
    def create_scheduler_tab(self, frame):
        header = tk.Label(frame, text="📋 Training Scheduler & Queue Manager", font=("Segoe UI", 18, "bold"), bg=self.colors["bg_dark"], fg=self.colors["accent_hover"])
        header.pack(anchor="w", pady=(0, 4))

        desc = tk.Label(
            frame,
            text="Queue multiple training and fine-tuning experiments across any architecture version (v1-v3). Run them sequentially overnight or in batches.",
            font=("Segoe UI", 10),
            bg=self.colors["bg_dark"],
            fg=self.colors["text_muted"],
            wraplength=750,
            justify=tk.LEFT,
        )
        desc.pack(anchor="w", pady=(0, 8))

        # Status Summary Header Cards
        summary_card = tk.Frame(frame, bg=self.colors["bg_card"], padx=12, pady=8)
        summary_card.pack(fill=tk.X, pady=(0, 8))

        self.lbl_queue_total = tk.Label(summary_card, text="Total: 0", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["text"])
        self.lbl_queue_total.pack(side=tk.LEFT, padx=(0, 18))

        self.lbl_queue_pending = tk.Label(summary_card, text="⏳ Pending: 0", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["warning"])
        self.lbl_queue_pending.pack(side=tk.LEFT, padx=(0, 18))

        self.lbl_queue_running = tk.Label(summary_card, text="🚀 Running: 0", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_hover"])
        self.lbl_queue_running.pack(side=tk.LEFT, padx=(0, 18))

        self.lbl_queue_completed = tk.Label(summary_card, text="✅ Completed: 0", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["terminal_fg"])
        self.lbl_queue_completed.pack(side=tk.LEFT, padx=(0, 18))

        self.lbl_queue_failed = tk.Label(summary_card, text="❌ Failed: 0", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["danger"])
        self.lbl_queue_failed.pack(side=tk.LEFT)

        # Control Action Buttons Bar
        action_bar = tk.Frame(frame, bg=self.colors["bg_dark"])
        action_bar.pack(fill=tk.X, pady=(0, 8))

        self.btn_start_batch = tk.Button(
            action_bar,
            text="▶ Start Batch Runner",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["accent"],
            fg=self.colors["bg_dark"],
            activebackground=self.colors["accent_hover"],
            bd=0,
            padx=14,
            pady=5,
            cursor="hand2",
            command=self.start_batch_runner,
        )
        self.btn_start_batch.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_stop_batch = tk.Button(
            action_bar,
            text="⏸ Pause / Stop Queue",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["danger"],
            fg=self.colors["text"],
            activebackground="#f87171",
            bd=0,
            padx=14,
            pady=5,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.stop_batch_runner,
        )
        self.btn_stop_batch.pack(side=tk.LEFT, padx=(0, 8))

        btn_add_custom = tk.Button(
            action_bar,
            text="➕ New Job",
            font=("Segoe UI", 9),
            bg=self.colors["bg_card"],
            fg=self.colors["text"],
            bd=0,
            padx=10,
            pady=5,
            cursor="hand2",
            command=self.open_add_job_dialog,
        )
        btn_add_custom.pack(side=tk.LEFT, padx=(0, 6))

        btn_up = tk.Button(
            action_bar,
            text="⬆ Up",
            font=("Segoe UI", 9),
            bg=self.colors["bg_card"],
            fg=self.colors["text"],
            bd=0,
            padx=8,
            pady=5,
            cursor="hand2",
            command=lambda: self.move_queue_job(-1),
        )
        btn_up.pack(side=tk.LEFT, padx=(0, 4))

        btn_down = tk.Button(
            action_bar,
            text="⬇ Down",
            font=("Segoe UI", 9),
            bg=self.colors["bg_card"],
            fg=self.colors["text"],
            bd=0,
            padx=8,
            pady=5,
            cursor="hand2",
            command=lambda: self.move_queue_job(1),
        )
        btn_down.pack(side=tk.LEFT, padx=(0, 6))

        btn_reset_failed = tk.Button(
            action_bar,
            text="🔄 Retry Failed",
            font=("Segoe UI", 9),
            bg=self.colors["bg_card"],
            fg=self.colors["text"],
            bd=0,
            padx=8,
            pady=5,
            cursor="hand2",
            command=self.reset_failed_jobs,
        )
        btn_reset_failed.pack(side=tk.LEFT, padx=(0, 6))

        btn_delete = tk.Button(
            action_bar,
            text="🗑 Delete",
            font=("Segoe UI", 9),
            bg=self.colors["bg_card"],
            fg=self.colors["danger"],
            bd=0,
            padx=8,
            pady=5,
            cursor="hand2",
            command=self.delete_queue_job,
        )
        btn_delete.pack(side=tk.LEFT, padx=(0, 6))

        btn_clear = tk.Button(
            action_bar,
            text="🧹 Clear Done",
            font=("Segoe UI", 9),
            bg=self.colors["bg_card"],
            fg=self.colors["text_muted"],
            bd=0,
            padx=8,
            pady=5,
            cursor="hand2",
            command=self.clear_completed_jobs,
        )
        btn_clear.pack(side=tk.LEFT, padx=(0, 6))

        btn_refresh = tk.Button(
            action_bar,
            text="🔄 Refresh",
            font=("Segoe UI", 9),
            bg=self.colors["bg_card"],
            fg=self.colors["text"],
            bd=0,
            padx=8,
            pady=5,
            cursor="hand2",
            command=self.refresh_queue_table,
        )
        btn_refresh.pack(side=tk.RIGHT)

        # Queue Table (Treeview)
        table_frame = tk.Frame(frame, bg=self.colors["bg_dark"])
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        cols = ("idx", "id", "status", "name", "ver", "preset", "epochs", "last_ep", "best_ep", "batch", "lr", "patience", "device", "dataset", "finetune")
        self.tree_queue = ttk.Treeview(table_frame, columns=cols, show="headings", height=7, selectmode="browse")

        self.tree_queue.heading("idx", text="#")
        self.tree_queue.heading("id", text="ID")
        self.tree_queue.heading("status", text="Status")
        self.tree_queue.heading("name", text="Experiment Name")
        self.tree_queue.heading("ver", text="Ver")
        self.tree_queue.heading("preset", text="Preset")
        self.tree_queue.heading("epochs", text="Epochs")
        self.tree_queue.heading("last_ep", text="Last Ep")
        self.tree_queue.heading("best_ep", text="Best Ep")
        self.tree_queue.heading("batch", text="Batch")
        self.tree_queue.heading("lr", text="LR")
        self.tree_queue.heading("patience", text="Patience")
        self.tree_queue.heading("device", text="Device")
        self.tree_queue.heading("dataset", text="Dataset")
        self.tree_queue.heading("finetune", text="Fine-Tune From")

        self.tree_queue.column("idx", width=28, anchor="center")
        self.tree_queue.column("id", width=55, anchor="center")
        self.tree_queue.column("status", width=80, anchor="center")
        self.tree_queue.column("name", width=125, anchor="w")
        self.tree_queue.column("ver", width=35, anchor="center")
        self.tree_queue.column("preset", width=60, anchor="center")
        self.tree_queue.column("epochs", width=48, anchor="center")
        self.tree_queue.column("last_ep", width=105, anchor="center")
        self.tree_queue.column("best_ep", width=105, anchor="center")
        self.tree_queue.column("batch", width=45, anchor="center")
        self.tree_queue.column("lr", width=55, anchor="center")
        self.tree_queue.column("patience", width=55, anchor="center")
        self.tree_queue.column("device", width=50, anchor="center")
        self.tree_queue.column("dataset", width=115, anchor="w")
        self.tree_queue.column("finetune", width=115, anchor="w")

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree_queue.yview)
        self.tree_queue.configure(yscroll=scrollbar.set)
        self.tree_queue.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_queue.tag_configure("pending", foreground=self.colors["warning"])
        self.tree_queue.tag_configure("running", foreground=self.colors["accent_hover"])
        self.tree_queue.tag_configure("completed", foreground=self.colors["terminal_fg"])
        self.tree_queue.tag_configure("failed", foreground=self.colors["danger"])

        self.refresh_queue_table()
        self.create_terminal_log(frame)

    # ----------------------------------------------------
    # TAB 3: Fine-Tuning
    # ----------------------------------------------------
    def create_finetune_tab(self, frame):
        header = tk.Label(frame, text="🎯 Fine-Tune APL SLM Checkpoints", font=("Segoe UI", 18, "bold"), bg=self.colors["bg_dark"], fg=self.colors["accent_hover"])
        header.pack(anchor="w", pady=(0, 10))

        desc = tk.Label(
            frame,
            text="Specialize a pre-trained model checkpoint on targeted APL algorithmic tasks, dfn contracts, or domain-specific corpora.",
            font=("Segoe UI", 10),
            bg=self.colors["bg_dark"],
            fg=self.colors["text_muted"],
            wraplength=740,
            justify=tk.LEFT,
        )
        desc.pack(anchor="w", pady=(0, 10))

        # Synthetic Generator Card
        synth_card = tk.Frame(frame, bg=self.colors["bg_card"], bd=0, padx=16, pady=14)
        synth_card.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            synth_card,
            text="🧬 Synthetic Fine-Tuning Dataset Generator",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg_card"],
            fg=self.colors["accent_hover"],
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        tk.Label(synth_card, text="Idiom Samples Count:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=1, column=0, sticky="w", pady=2)
        self.ft_synth_count = tk.Entry(synth_card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=15)
        self.ft_synth_count.insert(0, "50000")
        self.ft_synth_count.grid(row=2, column=0, sticky="w", padx=(0, 15), pady=(0, 4))

        tk.Label(synth_card, text="Output File Path:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=1, column=1, sticky="w", pady=2)
        self.ft_synth_output = tk.Entry(synth_card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=30)
        self.ft_synth_output.insert(0, "data/synthetic_idioms.txt")
        self.ft_synth_output.grid(row=2, column=1, sticky="w", padx=(0, 15), pady=(0, 4))

        self.btn_gen_synth_ft = tk.Button(
            synth_card,
            text="⚡ Generate Synthetic Dataset",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors["apple_leaf"],
            fg=self.colors["bg_dark"],
            activebackground=self.colors["accent_hover"],
            bd=0,
            padx=14,
            pady=4,
            cursor="hand2",
            command=self.run_synthetic_dataset_generation_ft,
        )
        self.btn_gen_synth_ft.grid(row=2, column=2, sticky="w", pady=(0, 4))

        # Fine-Tuning Execution Card
        card = tk.Frame(frame, bg=self.colors["bg_card"], bd=0, padx=16, pady=14)
        card.pack(fill=tk.X, pady=(0, 12))

        # Row 0 & 1: Base Checkpoint & Strategy
        tk.Label(card, text="Base Model Checkpoint to Fine-Tune:", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_hover"]).grid(row=0, column=0, columnspan=2, sticky="w", pady=3)
        self.combo_ft_checkpoint = ttk.Combobox(card, width=45, state="readonly")
        self.combo_ft_checkpoint.grid(row=1, column=0, columnspan=2, sticky="w", padx=(0, 15), pady=(0, 10))
        self.combo_ft_checkpoint.bind("<<ComboboxSelected>>", self.on_ft_checkpoint_selected)

        tk.Label(card, text="Fine-Tuning Strategy / Task:", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_hover"]).grid(row=0, column=2, columnspan=2, sticky="w", pady=3)
        self.combo_ft_strategy = ttk.Combobox(
            card,
            values=[
                "Structural Contracts & Idioms (Synthetic Idioms)",
                "General Domain Adaptation (Custom Corpus)",
            ],
            width=40,
            state="readonly",
        )
        self.combo_ft_strategy.current(0)
        self.combo_ft_strategy.grid(row=1, column=2, columnspan=2, sticky="w", padx=(0, 15), pady=(0, 10))
        self.combo_ft_strategy.bind("<<ComboboxSelected>>", self.on_ft_strategy_changed)

        # Row 2 & 3: Exp Name, Epochs, LR, Warmup
        tk.Label(card, text="Fine-Tuned Exp Name:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=0, sticky="w", pady=3)
        self.ft_exp = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=20)
        self.ft_exp.insert(0, "Small-v3-finetuned")
        self.ft_exp.grid(row=3, column=0, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Epochs:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=1, sticky="w", pady=3)
        self.ft_epochs = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=10)
        self.ft_epochs.insert(0, "5")
        self.ft_epochs.grid(row=3, column=1, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Learning Rate:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=2, sticky="w", pady=3)
        self.ft_lr = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=15)
        self.ft_lr.insert(0, "5e-5")
        self.ft_lr.grid(row=3, column=2, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Warmup Epochs:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=3, sticky="w", pady=3)
        self.ft_warmup = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=10)
        self.ft_warmup.insert(0, "1")
        self.ft_warmup.grid(row=3, column=3, sticky="w", padx=(0, 15), pady=(0, 10))

        # Row 4 & 5: Depth Penalty, Device, Precision, Dataset Path
        tk.Label(card, text="Depth Loss Penalty:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=4, column=0, sticky="w", pady=3)
        self.ft_depth_penalty = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=20)
        self.ft_depth_penalty.insert(0, "0.2")
        self.ft_depth_penalty.grid(row=5, column=0, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Hardware Device:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=4, column=1, sticky="w", pady=3)
        self.ft_device = ttk.Combobox(card, values=["(auto)", "cpu", "cuda"], width=12, state="readonly")
        self.ft_device.current(0)
        self.ft_device.grid(row=5, column=1, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Precision:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=4, column=2, sticky="w", pady=3)
        self.ft_prec = ttk.Combobox(card, values=["auto", "bf16", "fp16", "fp32"], width=12, state="readonly")
        self.ft_prec.current(0)
        self.ft_prec.grid(row=5, column=2, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Dataset Path:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=4, column=3, sticky="w", pady=3)
        self.ft_dataset_path = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=25)
        self.ft_dataset_path.insert(0, "data/synthetic_idioms.txt")
        self.ft_dataset_path.grid(row=5, column=3, sticky="w", padx=(0, 15), pady=(0, 10))

        # Row 6: Resume Checkpoint
        self.ft_resume_val = tk.BooleanVar(value=False)
        self.chk_ft_resume = ttk.Checkbutton(card, text="Resume Fine-Tuning Checkpoint (--resume)", variable=self.ft_resume_val, style="TCheckbutton")
        self.chk_ft_resume.grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # Run Controls
        btn_frame = tk.Frame(frame, bg=self.colors["bg_dark"])
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_run_ft = tk.Button(
            btn_frame,
            text="🍏 Start Fine-Tuning",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["accent"],
            fg=self.colors["bg_dark"],
            activebackground=self.colors["accent_hover"],
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.run_finetuning,
        )
        self.btn_run_ft.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_queue_ft = tk.Button(
            btn_frame,
            text="➕ Add to Queue",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg_card_light"],
            fg=self.colors["text"],
            activebackground=self.colors["bg_card"],
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.add_finetune_to_queue,
        )
        self.btn_queue_ft.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_kill_ft = tk.Button(
            btn_frame,
            text="Kill Process",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["apple_red"],
            fg=self.colors["text"],
            activebackground="#f87171",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.kill_active_process,
        )
        self.btn_kill_ft.pack(side=tk.LEFT)

        # Live Fine-Tuning Statistics Cards
        ft_stats_container = tk.Frame(frame, bg=self.colors["bg_dark"])
        ft_stats_container.pack(fill=tk.X, pady=(0, 10))

        # Left Card: Last Epoch Stats
        ft_card_last = tk.Frame(ft_stats_container, bg=self.colors["bg_card"], padx=12, pady=10)
        ft_card_last.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        tk.Label(
            ft_card_last,
            text="⏱️ LAST EPOCH STATS",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg_card"],
            fg=self.colors["accent_hover"],
        ).pack(anchor="w", pady=(0, 5))

        self.lbl_ft_last_epoch = tk.Label(
            ft_card_last, text="Epoch: - / -", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["text"]
        )
        self.lbl_ft_last_epoch.pack(anchor="w")

        grid_ft_last = tk.Frame(ft_card_last, bg=self.colors["bg_card"])
        grid_ft_last.pack(fill=tk.X, pady=(4, 0))

        self.lbl_ft_train_loss = tk.Label(grid_ft_last, text="Train Loss: -", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_ft_train_loss.grid(row=0, column=0, sticky="w", padx=(0, 15))

        self.lbl_ft_train_ppl = tk.Label(grid_ft_last, text="Train PPL: -", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_ft_train_ppl.grid(row=0, column=1, sticky="w")

        self.lbl_ft_val_loss = tk.Label(grid_ft_last, text="Val Loss:   -", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_ft_val_loss.grid(row=1, column=0, sticky="w", padx=(0, 15))

        self.lbl_ft_val_ppl = tk.Label(grid_ft_last, text="Val PPL:   -", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_ft_val_ppl.grid(row=1, column=1, sticky="w")

        self.lbl_ft_lr_time = tk.Label(ft_card_last, text="LR: - | Duration: -", font=("Segoe UI", 8), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_ft_lr_time.pack(anchor="w", pady=(4, 0))

        # Right Card: Best Epoch Stats
        ft_card_best = tk.Frame(ft_stats_container, bg=self.colors["bg_card"], padx=12, pady=10)
        ft_card_best.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            ft_card_best,
            text="⭐ BEST MODEL STATS",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg_card"],
            fg=self.colors["apple_gold"],
        ).pack(anchor="w", pady=(0, 5))

        self.lbl_ft_best_epoch = tk.Label(
            ft_card_best, text="Best Epoch: -", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["text"]
        )
        self.lbl_ft_best_epoch.pack(anchor="w")

        grid_ft_best = tk.Frame(ft_card_best, bg=self.colors["bg_card"])
        grid_ft_best.pack(fill=tk.X, pady=(4, 0))

        self.lbl_ft_best_loss = tk.Label(grid_ft_best, text="Best Val Loss: -", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_hover"])
        self.lbl_ft_best_loss.grid(row=0, column=0, sticky="w", padx=(0, 15))

        self.lbl_ft_best_ppl = tk.Label(grid_best, text="Best Val PPL: -", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_ft_best_ppl.grid(row=0, column=1, sticky="w")

        self.lbl_ft_best_status = tk.Label(ft_card_best, text="Status: Waiting for fine-tuning...", font=("Segoe UI", 8), bg=self.colors["bg_card"], fg=self.colors["text_muted"])
        self.lbl_ft_best_status.pack(anchor="w", pady=(4, 0))

        self.create_terminal_log(frame)

    # ----------------------------------------------------
    # TAB 4: Plots
    # ----------------------------------------------------
    def create_plots_tab(self, frame):
        header = tk.Label(frame, text="📈 Experiment Metrics & Curves", font=("Segoe UI", 18, "bold"), bg=self.colors["bg_dark"], fg=self.colors["accent_hover"])
        header.pack(anchor="w", pady=(0, 10))

        btn_frame = tk.Frame(frame, bg=self.colors["bg_dark"])
        btn_frame.pack(fill=tk.X, pady=(0, 15))

        self.btn_run_plot = tk.Button(
            btn_frame,
            text="🍏 Regenerate Comparison Plots",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["accent"],
            fg=self.colors["bg_dark"],
            activebackground=self.colors["accent_hover"],
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.run_plot_generation,
        )
        self.btn_run_plot.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_clear_plot = tk.Button(
            btn_frame,
            text="🧹 Clear / Reset Plot",
            font=("Segoe UI", 10),
            bg=self.colors["bg_card"],
            fg=self.colors["text_muted"],
            activebackground=self.colors["bg_side"],
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self.clear_plot,
        )
        self.btn_clear_plot.pack(side=tk.LEFT)

        self.img_frame = tk.Frame(frame, bg=self.colors["bg_card"], bd=0, padx=10, pady=10)
        self.img_frame.pack(fill=tk.BOTH, expand=True)

        self.plot_label = tk.Label(
            self.img_frame,
            text="Loading plots...",
            font=("Segoe UI", 11),
            bg=self.colors["bg_card"],
            fg=self.colors["text_muted"],
        )
        self.plot_label.pack(expand=True, fill=tk.BOTH)

        self.load_and_display_plot()

    def clear_plot(self):
        plot_path = Path("data/experiment_loss_comparison.png")
        if plot_path.exists():
            plot_path.unlink()
        self.load_and_display_plot()

    def load_and_display_plot(self):
        plot_path = Path("data/experiment_loss_comparison.png")
        if plot_path.exists():
            try:
                img = Image.open(plot_path)
                max_w, max_h = 780, 440
                img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                self.plot_photo = ImageTk.PhotoImage(img)
                self.plot_label.config(image=self.plot_photo, text="")
            except Exception as e:
                self.plot_label.config(image="", text=f"Failed to render plot: {e}")
        else:
            self.plot_label.config(
                image="",
                text="📊 No comparison plot found.\n\nWhen you train new experiments, their training & validation\nloss curves will be saved and plotted here.",
            )

    # ----------------------------------------------------
    # TAB 5: Autocomplete Playground
    # ----------------------------------------------------
    def create_autocomplete_tab(self, frame):
        header = tk.Label(frame, text="⚡ APL SLM Autocomplete Studio", font=("Segoe UI", 18, "bold"), bg=self.colors["bg_dark"], fg=self.colors["accent_hover"])
        header.pack(anchor="w", pady=(0, 10))

        model_panel = tk.Frame(frame, bg=self.colors["bg_card"], bd=0, padx=16, pady=12)
        model_panel.pack(fill=tk.X, pady=(0, 12))

        tk.Label(model_panel, text="Model Checkpoint:", font=("Segoe UI", 10, "bold"), bg=self.colors["bg_card"], fg=self.colors["text"]).pack(side=tk.LEFT, padx=(0, 10))

        self.combo_checkpoint = ttk.Combobox(model_panel, width=50, state="readonly")
        self.combo_checkpoint.pack(side=tk.LEFT, padx=(0, 15))
        self.combo_checkpoint.bind("<<ComboboxSelected>>", self.on_checkpoint_selected)

        self.btn_load_model = tk.Button(
            model_panel,
            text="🍏 Load Model",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors["accent"],
            fg=self.colors["bg_dark"],
            activebackground=self.colors["accent_hover"],
            bd=0,
            padx=14,
            pady=4,
            cursor="hand2",
            command=self.load_selected_model_in_background,
        )
        self.btn_load_model.pack(side=tk.LEFT)

        self.lbl_model_status = tk.Label(
            model_panel,
            text="● No model loaded",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors["bg_card"],
            fg=self.colors["warning"],
        )
        self.lbl_model_status.pack(side=tk.RIGHT, padx=10)

        ctrl_frame = tk.Frame(frame, bg=self.colors["bg_dark"])
        ctrl_frame.pack(fill=tk.X, pady=(0, 12))

        sliders_card = tk.Frame(ctrl_frame, bg=self.colors["bg_card"], bd=0, padx=16, pady=14, width=380)
        sliders_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        tk.Label(sliders_card, text="Max Tokens to Generate:", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).pack(anchor="w")
        self.sld_max_tokens = tk.Scale(sliders_card, from_=8, to=256, resolution=8, orient=tk.HORIZONTAL, bg=self.colors["bg_card"], fg=self.colors["text"], bd=0, highlightthickness=0)
        self.sld_max_tokens.set(64)
        self.sld_max_tokens.pack(fill=tk.X, pady=(0, 8))

        tk.Label(sliders_card, text="Sampling Temperature:", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).pack(anchor="w")
        self.sld_temp = tk.Scale(sliders_card, from_=0.0, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, bg=self.colors["bg_card"], fg=self.colors["text"], bd=0, highlightthickness=0)
        self.sld_temp.set(0.7)
        self.sld_temp.pack(fill=tk.X, pady=(0, 8))

        tk.Label(sliders_card, text="Top K Sampling Filter:", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).pack(anchor="w")
        self.sld_top_k = tk.Scale(sliders_card, from_=1, to=20, resolution=1, orient=tk.HORIZONTAL, bg=self.colors["bg_card"], fg=self.colors["text"], bd=0, highlightthickness=0)
        self.sld_top_k.set(5)
        self.sld_top_k.pack(fill=tk.X)

        prompt_card = tk.Frame(ctrl_frame, bg=self.colors["bg_card"], bd=0, padx=16, pady=14)
        prompt_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        prompt_hdr = tk.Frame(prompt_card, bg=self.colors["bg_card"])
        prompt_hdr.pack(fill=tk.X, pady=(0, 4))
        tk.Label(prompt_hdr, text="Input APL Code Prefix:", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_hover"]).pack(side=tk.LEFT)
        tk.Label(prompt_hdr, text="(Click glyphs below to insert)", font=("Segoe UI", 8), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).pack(side=tk.RIGHT)

        glyph_frame = tk.Frame(prompt_card, bg=self.colors["bg_card"])
        glyph_frame.pack(fill=tk.X, pady=(0, 6))

        quick_glyphs = [
            "⍳", "⍴", "⌽", "⊖", "⍉", "↑", "↓", "⊂", "⊃", "⊆", "⊇",
            "⍋", "⍒", "∊", "⍷", "⍸", "∪", "∩", "⌸", "⌹", "⊥", "⊤",
            "⍺", "⍵", "∇", "⋄", "←", "→", "¨", "⍨", "⍣", "⍤", "⍥",
            "○", "⌈", "⌊", "≢", "≡", "≠", "≤", "≥", "+", "×", "÷", "*", "|",
        ]

        row1_glyphs = quick_glyphs[:23]
        row2_glyphs = quick_glyphs[23:]

        row1_f = tk.Frame(glyph_frame, bg=self.colors["bg_card"])
        row1_f.pack(fill=tk.X, pady=1)
        for g in row1_glyphs:
            tk.Button(
                row1_f,
                text=g,
                font=("Consolas", 10, "bold"),
                bg=self.colors["bg_card_light"],
                fg=self.colors["accent_hover"],
                activebackground=self.colors["accent"],
                activeforeground=self.colors["bg_dark"],
                bd=0,
                width=2,
                cursor="hand2",
                command=lambda char=g: self.insert_glyph(char),
            ).pack(side=tk.LEFT, padx=1)

        row2_f = tk.Frame(glyph_frame, bg=self.colors["bg_card"])
        row2_f.pack(fill=tk.X, pady=1)
        for g in row2_glyphs:
            tk.Button(
                row2_f,
                text=g,
                font=("Consolas", 10, "bold"),
                bg=self.colors["bg_card_light"],
                fg=self.colors["accent_hover"],
                activebackground=self.colors["accent"],
                activeforeground=self.colors["bg_dark"],
                bd=0,
                width=2,
                cursor="hand2",
                command=lambda char=g: self.insert_glyph(char),
            ).pack(side=tk.LEFT, padx=1)

        self.txt_prompt = tk.Text(
            prompt_card,
            height=3,
            bg=self.colors["terminal_bg"],
            fg=self.colors["terminal_fg"],
            insertbackground=self.colors["text"],
            bd=0,
            font=("Consolas", 13),
        )
        self.txt_prompt.insert("1.0", "{+/⍵")
        self.txt_prompt.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        self.btn_autocomplete = tk.Button(
            prompt_card,
            text="🍏 Generate Autocomplete",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["accent"],
            fg=self.colors["bg_dark"],
            activebackground=self.colors["accent_hover"],
            bd=0,
            padx=16,
            pady=5,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.run_autocomplete_generation,
        )
        self.btn_autocomplete.pack(anchor="e")

        res_card = tk.Frame(frame, bg=self.colors["bg_card"], bd=0, padx=16, pady=14)
        res_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(res_card, text="Autocomplete Output Results:", font=("Segoe UI", 10, "bold"), bg=self.colors["bg_card"], fg=self.colors["text"]).pack(anchor="w", pady=(0, 5))

        self.txt_result = tk.Text(
            res_card,
            bg=self.colors["terminal_bg"],
            fg=self.colors["text"],
            bd=0,
            font=("Consolas", 13),
            state=tk.DISABLED,
        )
        self.txt_result.pack(fill=tk.BOTH, expand=True)

    def insert_glyph(self, char):
        if hasattr(self, "txt_prompt"):
            self.txt_prompt.insert(tk.INSERT, char)
            self.txt_prompt.focus_set()

    # ----------------------------------------------------
    # TAB 6: Benchmarks
    # ----------------------------------------------------
    def create_benchmarks_tab(self, frame):
        header = tk.Label(frame, text="🏆 APL SLM Benchmark Comparison Suite", font=("Segoe UI", 18, "bold"), bg=self.colors["bg_dark"], fg=self.colors["accent_hover"])
        header.pack(anchor="w", pady=(0, 10))

        btn_frame = tk.Frame(frame, bg=self.colors["bg_dark"])
        btn_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(btn_frame, text="Select Checkpoint:", font=("Segoe UI", 10, "bold"), bg=self.colors["bg_dark"], fg=self.colors["text"]).pack(side=tk.LEFT, padx=(0, 10))

        self.combo_bench_checkpoint = ttk.Combobox(btn_frame, width=40, state="readonly")
        self.combo_bench_checkpoint.pack(side=tk.LEFT, padx=(0, 15))

        self.btn_run_bench = tk.Button(
            btn_frame,
            text="🍏 Run Benchmarks",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["accent"],
            fg=self.colors["bg_dark"],
            activebackground=self.colors["accent_hover"],
            bd=0,
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.run_benchmarks,
        )
        self.btn_run_bench.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_kill_bench = tk.Button(
            btn_frame,
            text="Kill",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["apple_red"],
            fg=self.colors["text"],
            activebackground="#f87171",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.kill_active_process,
        )
        self.btn_kill_bench.pack(side=tk.LEFT)

        self.create_terminal_log(frame)

    # ----------------------------------------------------
    # Terminal Logger Console
    # ----------------------------------------------------
    def create_terminal_log(self, parent_frame):
        log_frame = tk.Frame(parent_frame, bg=self.colors["terminal_bg"], bd=0, padx=6, pady=6)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(15, 0))

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.txt_log = tk.Text(
            log_frame,
            bg=self.colors["terminal_bg"],
            fg=self.colors["terminal_fg"],
            insertbackground=self.colors["text"],
            bd=0,
            font=("Consolas", 10),
            yscrollcommand=scrollbar.set,
            wrap=tk.WORD,
        )
        self.txt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.txt_log.yview)

    # ----------------------------------------------------
    # Subprocess Executor Engine
    # ----------------------------------------------------
    def execute_command(self, cmd_args):
        if self.process_active:
            messagebox.showwarning("Busy", "A background task is already running. Please wait or stop it first.")
            return

        self.process_active = True
        self.status_desc.config(text="● Running...", fg=self.colors["warning"])

        self._active_exp_name = None
        if "--exp_name" in cmd_args:
            idx = cmd_args.index("--exp_name")
            if idx + 1 < len(cmd_args):
                self._active_exp_name = cmd_args[idx + 1]

        if hasattr(self, "txt_log") and self.txt_log.winfo_exists():
            self.txt_log.config(state=tk.NORMAL)
            self.txt_log.delete("1.0", tk.END)
            self.txt_log.config(state=tk.DISABLED)

        self.set_buttons_running_state(True)

        self.running_thread = threading.Thread(target=self.subprocess_worker_thread, args=(cmd_args,), daemon=True)
        self.running_thread.start()

    def subprocess_worker_thread(self, cmd_args):
        try:
            executable_args = [sys.executable, "-u"] + cmd_args
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            self.running_process = subprocess.Popen(
                executable_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                env=env,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            for line in self.running_process.stdout:
                self.log_queue.put(("log", line))

            self.running_process.wait()
            exit_code = self.running_process.returncode
            self.log_queue.put(("finished", exit_code))
        except Exception as e:
            self.log_queue.put(("error", str(e)))

    def poll_log_queue(self):
        try:
            while True:
                msg_type, val = self.log_queue.get_nowait()
                if msg_type == "log":
                    self.append_log(val)
                elif msg_type == "finished":
                    self.on_process_finished(val)
                elif msg_type == "error":
                    self.append_log(f"\n[FATAL ERROR] Failed to start subprocess: {val}\n")
                    self.on_process_finished(-1)
        except queue.Empty:
            pass

        self.root.after(100, self.poll_log_queue)

    def reset_training_stats_ui(self):
        self._trn_curr_ep = "-"
        self._trn_total_ep = "-"
        self._trn_dur = "-"
        self._trn_lr = "-"
        self._trn_train_loss = "-"
        self._trn_train_ppl = "-"
        self._trn_val_loss = "-"
        self._trn_val_ppl = "-"
        self._trn_best_val_loss = float("inf")
        self._trn_best_ep = "-"

        if hasattr(self, "lbl_trn_last_epoch"):
            self.lbl_trn_last_epoch.config(text="Epoch: - / -")
            self.lbl_trn_train_loss.config(text="Train Loss: -")
            self.lbl_trn_train_ppl.config(text="Train PPL: -")
            self.lbl_trn_val_loss.config(text="Val Loss:   -")
            self.lbl_trn_val_ppl.config(text="Val PPL:   -")
            self.lbl_trn_lr_time.config(text="LR: - | Duration: -")

            self.lbl_trn_best_epoch.config(text="Best Epoch: -")
            self.lbl_trn_best_loss.config(text="Best Val Loss: -", fg=self.colors["accent_hover"])
            self.lbl_trn_best_ppl.config(text="Best Val PPL: -")
            self.lbl_trn_best_status.config(text="Status: Training in progress...", fg=self.colors["text_muted"])

        if hasattr(self, "lbl_ft_last_epoch"):
            self.lbl_ft_last_epoch.config(text="Epoch: - / -")
            self.lbl_ft_train_loss.config(text="Train Loss: -")
            self.lbl_ft_train_ppl.config(text="Train PPL: -")
            self.lbl_ft_val_loss.config(text="Val Loss:   -")
            self.lbl_ft_val_ppl.config(text="Val PPL:   -")
            self.lbl_ft_lr_time.config(text="LR: - | Duration: -")

            self.lbl_ft_best_epoch.config(text="Best Epoch: -")
            self.lbl_ft_best_loss.config(text="Best Val Loss: -", fg=self.colors["accent_hover"])
            self.lbl_ft_best_ppl.config(text="Best Val PPL: -")
            self.lbl_ft_best_status.config(text="Status: Fine-tuning in progress...", fg=self.colors["text_muted"])

    def parse_training_log_line(self, line: str):
        updated = False

        # 1. Match: [Epoch 12/40] Completed in 4.2s | LR: 0.000350 or [Epoch 12/40 | Step 10/100 (10.0%)]
        m_epoch = re.search(r"\[Epoch\s+(\d+)/(\d+)\]", line)
        if m_epoch:
            ep, total_ep = m_epoch.groups()
            self._trn_curr_ep = ep
            self._trn_total_ep = total_ep
            updated = True
            if hasattr(self, "lbl_trn_last_epoch"):
                self.lbl_trn_last_epoch.config(text=f"Epoch: {ep} / {total_ep}")
            if hasattr(self, "lbl_ft_last_epoch"):
                self.lbl_ft_last_epoch.config(text=f"Epoch: {ep} / {total_ep}")

        m_epoch_detail = re.search(r"(?:Duration|Completed in)[:\s]+([\d\.]+)s\s*\|\s*LR:\s*([\d\.e\-]+)", line)
        if m_epoch_detail:
            dur, lr = m_epoch_detail.groups()
            self._trn_dur = dur
            self._trn_lr = lr

        # 2. Match: - Train Loss: 0.4512 | Train Perplexity: 1.57
        m_train = re.search(r"Train Loss:\s*([\d\.]+)(?:\s*\|\s*Train Perplexity:\s*([\d\.]+|inf))?", line)
        if m_train:
            t_loss = m_train.group(1)
            t_ppl = m_train.group(2) or "-"
            self._trn_train_loss = t_loss
            self._trn_train_ppl = t_ppl
            if hasattr(self, "lbl_trn_train_loss"):
                self.lbl_trn_train_loss.config(text=f"Train Loss: {t_loss}")
                self.lbl_trn_train_ppl.config(text=f"Train PPL: {t_ppl}")
            if hasattr(self, "lbl_ft_train_loss"):
                self.lbl_ft_train_loss.config(text=f"Train Loss: {t_loss}")
                self.lbl_ft_train_ppl.config(text=f"Train PPL: {t_ppl}")

        # 3. Match: - Val Loss:   0.4820 | Val Perplexity:   1.62
        m_val = re.search(r"Val Loss:\s*([\d\.]+)(?:\s*\|\s*Val Perplexity:\s*([\d\.]+|inf))?", line)
        if m_val:
            v_loss = m_val.group(1)
            v_ppl = m_val.group(2) or "-"
            self._trn_val_loss = v_loss
            self._trn_val_ppl = v_ppl
            updated = True
            lr_str = getattr(self, "_trn_lr", "-")
            dur_str = getattr(self, "_trn_dur", "-")
            if hasattr(self, "lbl_trn_val_loss"):
                self.lbl_trn_val_loss.config(text=f"Val Loss:   {v_loss}")
                self.lbl_trn_val_ppl.config(text=f"Val PPL:   {v_ppl}")
                self.lbl_trn_lr_time.config(text=f"LR: {lr_str} | Duration: {dur_str}s")
            if hasattr(self, "lbl_ft_val_loss"):
                self.lbl_ft_val_loss.config(text=f"Val Loss:   {v_loss}")
                self.lbl_ft_val_ppl.config(text=f"Val PPL:   {v_ppl}")
                self.lbl_ft_lr_time.config(text=f"LR: {lr_str} | Duration: {dur_str}s")

            try:
                flt_val = float(v_loss)
                if flt_val < getattr(self, "_trn_best_val_loss", float("inf")):
                    self._trn_best_val_loss = flt_val
                    self._trn_best_ep = getattr(self, "_trn_curr_ep", "?")
            except Exception:
                pass

        # 4. Match Best Val Loss improvements
        m_best_imp = re.search(r"Validation Loss improved (?:from [\d\.]+ )?to ([\d\.]+)", line)
        m_best_init = re.search(r"Initial best model saved.*Val Loss:\s*([\d\.]+)", line)
        best_loss_found = None
        if m_best_imp:
            best_loss_found = m_best_imp.group(1)
        elif m_best_init:
            best_loss_found = m_best_init.group(1)

        if best_loss_found:
            ep_str = getattr(self, "_trn_curr_ep", "?")
            v_ppl_str = getattr(self, "_trn_val_ppl", "-")
            updated = True
            try:
                self._trn_best_val_loss = float(best_loss_found)
                self._trn_best_ep = ep_str
            except Exception:
                pass

            if hasattr(self, "lbl_trn_best_epoch"):
                self.lbl_trn_best_epoch.config(text=f"Best Epoch: Epoch {ep_str}")
                self.lbl_trn_best_loss.config(text=f"Best Val Loss: {best_loss_found}", fg=self.colors["accent_hover"])
                self.lbl_trn_best_ppl.config(text=f"Best Val PPL: {v_ppl_str}")
                self.lbl_trn_best_status.config(text="Status: Saved best checkpoint ⭐", fg=self.colors["apple_gold"])
            if hasattr(self, "lbl_ft_best_epoch"):
                self.lbl_ft_best_epoch.config(text=f"Best Epoch: Epoch {ep_str}")
                self.lbl_ft_best_loss.config(text=f"Best Val Loss: {best_loss_found}", fg=self.colors["accent_hover"])
                self.lbl_ft_best_ppl.config(text=f"Best Val PPL: {v_ppl_str}")
                self.lbl_ft_best_status.config(text="Status: Saved best checkpoint ⭐", fg=self.colors["apple_gold"])

        if updated and hasattr(self, "tree_queue") and self.tree_queue.winfo_exists():
            self.refresh_queue_table()

    def append_log(self, text):
        if hasattr(self, "txt_log") and self.txt_log.winfo_exists():
            self.txt_log.config(state=tk.NORMAL)
            self.txt_log.insert(tk.END, text)
            self.txt_log.see(tk.END)
            self.txt_log.config(state=tk.DISABLED)
        self.parse_training_log_line(text)

    def kill_active_process(self):
        if self.running_process:
            self.append_log("\n[!] Killing active process...\n")
            self.running_process.terminate()
            try:
                self.running_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.running_process.kill()

    def on_process_finished(self, exit_code):
        self.last_exit_code = exit_code
        self.process_active = False
        self.running_process = None
        self.running_thread = None
        self.set_buttons_running_state(False)

        if exit_code == 0:
            self.status_desc.config(text="● Ready (Success)", fg=self.colors["accent_hover"])
            self.append_log("\n[OK] Task Completed Successfully.\n")
        else:
            self.status_desc.config(text="● Stopped/Failed", fg=self.colors["danger"])
            self.append_log(f"\n[FAIL] Process exited with code {exit_code}.\n")

        self.refresh_checkpoints()
        if self.active_tab == "Plots":
            self.load_and_display_plot()
        if hasattr(self, "tree_queue"):
            self.refresh_queue_table()

    def set_buttons_running_state(self, is_running):
        state = tk.DISABLED if is_running else tk.NORMAL
        kill_state = tk.NORMAL if is_running else tk.DISABLED

        for btn_name in ["btn_run_ds", "btn_gen_synth", "btn_gen_synth_ft", "btn_inspect_tok", "btn_run_train", "btn_queue_train", "btn_run_ft", "btn_queue_ft", "btn_run_plot", "btn_run_bench"]:
            if hasattr(self, btn_name):
                getattr(self, btn_name).config(state=state)

        for kill_btn in ["btn_kill_ds", "btn_kill_train", "btn_kill_ft", "btn_kill_bench"]:
            if hasattr(self, kill_btn):
                getattr(self, kill_btn).config(state=kill_state)

    def refresh_checkpoints(self):
        checkpoints = ["all"] if self.active_tab == "Benchmarks" else []
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            for f in checkpoints_dir.glob("*.pt"):
                checkpoints.append(str(f))
            for sub in sorted(checkpoints_dir.iterdir()):
                if sub.is_dir():
                    for f in sub.glob("*.pt"):
                        checkpoints.append(str(f))

        clean_checkpoints = [c for c in checkpoints if c != "all"]

        if hasattr(self, "combo_checkpoint") and self.combo_checkpoint.winfo_exists():
            self.combo_checkpoint.config(values=clean_checkpoints)
            if clean_checkpoints and not self.combo_checkpoint.get():
                best_cands = [c for c in clean_checkpoints if "best" in c]
                self.combo_checkpoint.set(best_cands[0] if best_cands else clean_checkpoints[0])

        if hasattr(self, "combo_ft_checkpoint") and self.combo_ft_checkpoint.winfo_exists():
            self.combo_ft_checkpoint.config(values=clean_checkpoints)
            if clean_checkpoints and not self.combo_ft_checkpoint.get():
                self.combo_ft_checkpoint.set(clean_checkpoints[0])
                self.on_ft_checkpoint_selected()

        if hasattr(self, "combo_bench_checkpoint") and self.combo_bench_checkpoint.winfo_exists():
            bench_opts = ["all"] + clean_checkpoints
            self.combo_bench_checkpoint.config(values=bench_opts)
            if not self.combo_bench_checkpoint.get():
                self.combo_bench_checkpoint.set("all")

    def on_model_preset_changed(self, event=None):
        preset = self.trn_preset.get().strip().lower()
        cfg = self.MODEL_PRESETS.get(preset) or self.MODEL_PRESETS.get("(none)")
        if not cfg:
            return

        v_str = self.trn_version.get() if hasattr(self, "trn_version") else "v3"
        ver_tag = "v3"
        if "v1" in v_str:
            ver_tag = "v1"
        elif "v2" in v_str:
            ver_tag = "v2"

        base_prefix = f"{cfg.get('base_name', preset.capitalize())}-{ver_tag}"
        queued = getattr(self.queue_manager, "jobs", None) if hasattr(self, "queue_manager") else None
        next_exp_name = self.get_next_experiment_name(base_prefix, queued_jobs=queued)

        def _set_val(widget_attr, val):
            if hasattr(self, widget_attr):
                w = getattr(self, widget_attr)
                w.delete(0, tk.END)
                w.insert(0, str(val))

        _set_val("trn_exp", next_exp_name)
        _set_val("trn_epochs", cfg["epochs"])
        _set_val("trn_batch", cfg["batch_size"])
        _set_val("trn_lr", cfg["lr"])
        _set_val("trn_warmup", cfg["warmup_epochs"])
        _set_val("trn_patience", cfg["patience"])

        if hasattr(self, "trn_accum"):
            accum_val = 2 if preset in ["large", "xlarge", "huge", "giant"] else 1
            self.trn_accum.delete(0, tk.END)
            self.trn_accum.insert(0, str(accum_val))

        if hasattr(self, "lbl_preset_info"):
            info_text = cfg["info"]
            if "v1" in v_str:
                info_text = f"⚡ Architecture v1 (Classic Baseline Transformer):\n{info_text.splitlines()[-1]} (Pure token causal LM without depth node)"
            elif "v2" in v_str:
                info_text = f"🔄 Architecture v2 (Structural Depth Conditioned):\n{info_text.splitlines()[-1]} (Includes parenthesis, bracket, and dfn depth embedding)"
            else:
                info_text = f"🍏 Architecture v3 (Modern RoPE + SwiGLU + QK-Norm):\n{info_text.splitlines()[-1]} (Pre-RMSNorm, SwiGLU, RoPE Rotary Position Embedding & single-token KV caching)"
            self.lbl_preset_info.config(text=info_text)

    def on_checkpoint_selected(self, event=None):
        ckpt_path = self.combo_checkpoint.get()
        if not ckpt_path:
            return
        name_lower = ckpt_path.lower()
        if any(k in name_lower for k in ["large", "wide", "deep"]):
            self.sld_temp.set(0.7)
            self.sld_top_k.set(5)
            self.sld_max_tokens.set(128)
        elif "medium" in name_lower:
            self.sld_temp.set(0.7)
            self.sld_top_k.set(4)
            self.sld_max_tokens.set(96)
        else:
            self.sld_temp.set(0.7)
            self.sld_top_k.set(5)
            self.sld_max_tokens.set(64)

    def load_selected_model_in_background(self):
        ckpt_path = self.combo_checkpoint.get()
        if not ckpt_path:
            messagebox.showwarning("Error", "Please select a model checkpoint first.")
            return

        if load_model is None:
            messagebox.showerror("Error", "Could not import load_model from autocomplete.py.")
            return

        self.btn_load_model.config(state=tk.DISABLED)
        self.lbl_model_status.config(text="● Loading model...", fg=self.colors["warning"])

        threading.Thread(target=self._load_model_worker, args=(ckpt_path,), daemon=True).start()

    def _load_model_worker(self, ckpt_path):
        try:
            model, tokenizer, device = load_model(ckpt_path)
            self.cached_model = model
            self.cached_tokenizer = tokenizer
            self.cached_device = device
            self.cached_checkpoint_path = ckpt_path

            ver = getattr(model, "version", 1)
            n_params = model.count_parameters() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters())
            parent_name = Path(ckpt_path).parent.name
            file_name = Path(ckpt_path).name
            status_text = f"🍏 Loaded: {parent_name}/{file_name} (v{ver} • {n_params:,} params)"
            self.root.after(0, lambda: self.lbl_model_status.config(text=status_text, fg=self.colors["accent_hover"]))
            self.root.after(0, lambda: self.btn_autocomplete.config(state=tk.NORMAL))
        except Exception as e:
            self.root.after(0, lambda: self.lbl_model_status.config(text="● Failed to load", fg=self.colors["danger"]))
            self.root.after(0, lambda: messagebox.showerror("Load Failed", f"Failed to load checkpoint:\n{e}"))
        finally:
            self.root.after(0, lambda: self.btn_load_model.config(state=tk.NORMAL))

    def run_dataset_collection(self):
        cmd = ["src/dataset_collector.py"]
        mode = self.ds_mode.get()
        if mode:
            cmd += ["--mode", mode]
        limit = self.ds_limit.get().strip()
        if limit:
            cmd += ["--limit", limit]
        token = self.ds_token.get().strip()
        if token:
            cmd += ["--token", token]

        selected_sources = []
        if hasattr(self, "src_github_var") and self.src_github_var.get():
            selected_sources.append("github")
        if hasattr(self, "src_gitlab_var") and self.src_gitlab_var.get():
            selected_sources.append("gitlab")
        if hasattr(self, "src_synthetic_var") and self.src_synthetic_var.get():
            selected_sources.append("synthetic")

        if selected_sources:
            cmd += ["--sources"] + selected_sources

        self.execute_command(cmd)

    def run_synthetic_dataset_generation(self):
        cmd = ["src/synthetic_generator.py", "--count", "50000", "--output", "data/synthetic_idioms.txt"]
        self.execute_command(cmd)

    def run_synthetic_dataset_generation_ft(self):
        count = self.ft_synth_count.get().strip() or "50000"
        output = self.ft_synth_output.get().strip() or "data/synthetic_idioms.txt"
        cmd = ["src/synthetic_generator.py", "--count", count, "--output", output]
        self.execute_command(cmd)

    def inspect_tokenizer(self):
        from tokenizer import APLTokenizer
        tok = APLTokenizer()
        self.append_log(f"\n[🍏 APL Glyph Tokenizer Diagnostics]\nTotal Vocabulary Size: {tok.vocab_size} tokens\n")
        self.append_log(f"Special Tokens: {tok.SPECIAL_TOKENS}\n")
        self.append_log(f"Unicode APL Glyphs ({len(tok.APL_GLYPHS)}): {' '.join(tok.APL_GLYPHS)}\n")
        self.append_log(f"Delimiters ({len(tok.DELIMITERS)}): {' '.join(tok.DELIMITERS)}\n\n")

    def run_model_training(self):
        self.reset_training_stats_ui()
        v_str = self.trn_version.get() if hasattr(self, "trn_version") else "v3"
        ver_num = 3
        if "v1" in v_str:
            ver_num = 1
            cmd = ["src/v1/train_v1.py"]
        elif "v2" in v_str:
            ver_num = 2
            cmd = ["src/v2/train_v2.py"]
        else:
            ver_num = 3
            cmd = ["src/v3/train_v3.py"]

        exp_name = self.trn_exp.get().strip()
        if exp_name:
            cmd += ["--exp_name", exp_name]
        epochs = self.trn_epochs.get().strip()
        if epochs:
            cmd += ["--epochs", epochs]
        batch_size = self.trn_batch.get().strip()
        if batch_size:
            cmd += ["--batch_size", batch_size]
        lr = self.trn_lr.get().strip()
        if lr:
            cmd += ["--lr", lr]
        if hasattr(self, "trn_warmup"):
            warmup = self.trn_warmup.get().strip()
            if warmup:
                cmd += ["--warmup_epochs", warmup]
        if hasattr(self, "trn_accum"):
            accum = self.trn_accum.get().strip()
            if accum:
                cmd += ["--grad_accum_steps", accum]
        patience = self.trn_patience.get().strip()
        if patience:
            cmd += ["--early_stopping_patience", patience]
        prec = self.trn_prec.get()
        if prec:
            cmd += ["--precision", prec]
        preset = self.trn_preset.get()
        if preset and preset != "(none)":
            cmd += ["--model_preset", preset]
        device = self.trn_device.get()
        if device and device != "(auto)":
            cmd += ["--device", device]
        if self.trn_resume_val.get():
            cmd.append("--resume")
        if self.trn_demo_val.get():
            cmd.append("--demo")

        self.execute_command(cmd)

    def on_ft_checkpoint_selected(self, event=None):
        if not hasattr(self, "combo_ft_checkpoint") or not self.combo_ft_checkpoint.winfo_exists():
            return
        ckpt = self.combo_ft_checkpoint.get().strip()
        if ckpt:
            p = Path(ckpt)
            model_name = p.parent.name if p.parent.name != "checkpoints" else p.stem
            if not model_name.endswith("-finetuned"):
                self.ft_exp.delete(0, tk.END)
                self.ft_exp.insert(0, f"{model_name}-finetuned")

    def on_ft_strategy_changed(self, event=None):
        if not hasattr(self, "combo_ft_strategy") or not self.combo_ft_strategy.winfo_exists():
            return
        strategy = self.combo_ft_strategy.get()
        if "Synthetic Idioms" in strategy:
            self.ft_dataset_path.delete(0, tk.END)
            self.ft_dataset_path.insert(0, "data/synthetic_idioms.txt")
        else:
            self.ft_dataset_path.delete(0, tk.END)
            self.ft_dataset_path.insert(0, "data/apl_corpus.txt")

    def run_finetuning(self):
        self.reset_training_stats_ui()
        base_ckpt = self.combo_ft_checkpoint.get().strip()
        if not base_ckpt:
            messagebox.showwarning("Error", "Please select a base model checkpoint to fine-tune from.")
            return

        exp_name = self.ft_exp.get().strip() or "model-finetuned"
        epochs = self.ft_epochs.get().strip() or "5"
        lr = self.ft_lr.get().strip() or "5e-5"
        warmup = self.ft_warmup.get().strip() or "1"
        depth_penalty = self.ft_depth_penalty.get().strip() or "0.2"
        dataset_path = self.ft_dataset_path.get().strip() or "data/synthetic_idioms.txt"

        if dataset_path == "data/synthetic_idioms.txt" and not Path(dataset_path).exists():
            from synthetic_generator import APLSyntheticGenerator
            text = APLSyntheticGenerator.generate_synthetic_corpus(count=50000)
            Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
            with open(dataset_path, "w", encoding="utf-8") as f:
                f.write(text)

        cmd = [
            "src/train.py",
            "--exp_name", exp_name,
            "--finetune_from", base_ckpt,
            "--data_file", dataset_path,
            "--epochs", epochs,
            "--warmup_epochs", warmup,
            "--lr", lr,
            "--depth_loss_weight", depth_penalty,
        ]

        device = self.ft_device.get()
        if device and device != "(auto)":
            cmd += ["--device", device]

        prec = self.ft_prec.get()
        if prec:
            cmd += ["--precision", prec]

        if hasattr(self, "ft_resume_val") and self.ft_resume_val.get():
            cmd.append("--resume")

        self.execute_command(cmd)

    def add_training_to_queue(self):
        v_str = self.trn_version.get() if hasattr(self, "trn_version") else "v3"
        ver = 3
        if "v1" in v_str:
            ver = 1
        elif "v2" in v_str:
            ver = 2
        elif "v3" in v_str:
            ver = 3

        preset = self.trn_preset.get().strip().lower() if hasattr(self, "trn_preset") else "small"
        exp_name = self.trn_exp.get().strip() if hasattr(self, "trn_exp") else ""
        if not exp_name:
            cfg = self.MODEL_PRESETS.get(preset) or self.MODEL_PRESETS.get("small")
            base_prefix = f"{cfg.get('base_name', preset.capitalize())}-v{ver}"
            exp_name = self.get_next_experiment_name(base_prefix, queued_jobs=getattr(self.queue_manager, "jobs", None))

        try:
            epochs = int(self.trn_epochs.get().strip()) if hasattr(self, "trn_epochs") and self.trn_epochs.get().strip() else 30
            batch_size = int(self.trn_batch.get().strip()) if hasattr(self, "trn_batch") and self.trn_batch.get().strip() else 16
            lr = float(self.trn_lr.get().strip()) if hasattr(self, "trn_lr") and self.trn_lr.get().strip() else 5e-4
            warmup = int(self.trn_warmup.get().strip()) if hasattr(self, "trn_warmup") and self.trn_warmup.get().strip() else 2
            accum = int(self.trn_accum.get().strip()) if hasattr(self, "trn_accum") and self.trn_accum.get().strip() else 1
            patience = int(self.trn_patience.get().strip()) if hasattr(self, "trn_patience") and self.trn_patience.get().strip() else 6
        except ValueError as e:
            messagebox.showerror("Invalid Parameter", f"Please check numerical input values:\n{e}")
            return

        precision = self.trn_prec.get().strip() if hasattr(self, "trn_prec") else "auto"
        device_val = self.trn_device.get().strip() if hasattr(self, "trn_device") else "(auto)"
        device = device_val if device_val in ["cpu", "cuda"] else None
        resume = bool(self.trn_resume_val.get()) if hasattr(self, "trn_resume_val") else False

        job = TrainingJob(
            exp_name=exp_name,
            version=ver,
            model_preset=preset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            warmup_epochs=warmup,
            grad_accum_steps=accum,
            early_stopping_patience=patience,
            precision=precision,
            dataset_path="data/apl_corpus.txt",
            resume=resume,
            device=device,
        )

        self.queue_manager.add_job(job)
        self.refresh_queue_table()
        self.append_log(f"[+] Enqueued Training Job '{exp_name}' (v{ver}, {preset}, {epochs} epochs, batch {batch_size}, lr {lr}).\n")

        self.on_model_preset_changed()
        messagebox.showinfo("Job Enqueued", f"Experiment '{exp_name}' added to Training Queue.")

    def add_finetune_to_queue(self):
        ckpt = self.combo_ft_checkpoint.get().strip() if hasattr(self, "combo_ft_checkpoint") else ""
        if not ckpt:
            messagebox.showwarning("Missing Checkpoint", "Please select a base checkpoint to fine-tune from.")
            return

        dataset_path = self.ft_dataset_path.get().strip() if hasattr(self, "ft_dataset_path") and self.ft_dataset_path.get().strip() else "data/synthetic_idioms.txt"

        ver = 3
        if "v1" in ckpt.lower():
            ver = 1
        elif "v2" in ckpt.lower():
            ver = 2
        elif "v3" in ckpt.lower():
            ver = 3

        exp_name = self.ft_exp.get().strip() if hasattr(self, "ft_exp") and self.ft_exp.get().strip() else ""
        if not exp_name or exp_name == "Small-v3-finetuned":
            ckpt_stem = Path(ckpt).parent.name if Path(ckpt).parent.name not in ["checkpoints", ""] else Path(ckpt).stem
            base_prefix = f"FineTune-{ckpt_stem}"
            exp_name = self.get_next_experiment_name(base_prefix, queued_jobs=getattr(self.queue_manager, "jobs", None))

        try:
            epochs = int(self.ft_epochs.get().strip()) if hasattr(self, "ft_epochs") and self.ft_epochs.get().strip() else 5
            lr = float(self.ft_lr.get().strip()) if hasattr(self, "ft_lr") and self.ft_lr.get().strip() else 5e-5
            warmup = int(self.ft_warmup.get().strip()) if hasattr(self, "ft_warmup") and self.ft_warmup.get().strip() else 1
            depth_penalty = float(self.ft_depth_penalty.get().strip()) if hasattr(self, "ft_depth_penalty") and self.ft_depth_penalty.get().strip() else 0.2
        except ValueError as e:
            messagebox.showerror("Invalid Parameter", f"Please check numerical input values:\n{e}")
            return

        device_val = self.ft_device.get().strip() if hasattr(self, "ft_device") else "(auto)"
        device = device_val if device_val in ["cpu", "cuda"] else None
        prec = self.ft_prec.get().strip() if hasattr(self, "ft_prec") else "auto"
        resume = bool(self.ft_resume_val.get()) if hasattr(self, "ft_resume_val") else False

        job = TrainingJob(
            exp_name=exp_name,
            version=ver,
            model_preset="small",
            epochs=epochs,
            batch_size=16,
            lr=lr,
            warmup_epochs=warmup,
            early_stopping_patience=6,
            precision=prec,
            dataset_path=dataset_path,
            finetune_from=ckpt,
            depth_loss_weight=depth_penalty,
            resume=resume,
            device=device,
        )

        self.queue_manager.add_job(job)
        self.refresh_queue_table()
        self.append_log(f"[+] Enqueued Fine-Tuning Job '{exp_name}' from '{ckpt}'.\n")
        messagebox.showinfo("Job Enqueued", f"Fine-Tuning job '{exp_name}' added to Training Queue.")

    def get_experiment_epoch_stats(self, exp_name: str) -> tuple[str, str]:
        if not exp_name:
            return "-", "-"

        last_ep_str = "-"
        best_ep_str = "-"

        active_exp = getattr(self, "_active_exp_name", None)
        if active_exp and (active_exp == exp_name or active_exp in exp_name):
            curr_ep = getattr(self, "_trn_curr_ep", None)
            curr_loss = getattr(self, "_trn_val_loss", None)
            if curr_ep and curr_ep != "-":
                if curr_loss and curr_loss != "-":
                    try:
                        last_ep_str = f"Ep {curr_ep} ({float(curr_loss):.3f})"
                    except Exception:
                        last_ep_str = f"Ep {curr_ep}"
                else:
                    last_ep_str = f"Ep {curr_ep}"
            best_ep = getattr(self, "_trn_best_ep", None)
            best_loss = getattr(self, "_trn_best_val_loss", None)
            if best_ep and best_ep != "-" and best_loss is not None and best_loss != float("inf"):
                best_ep_str = f"Ep {best_ep} ({best_loss:.3f})"
            elif best_ep and best_ep != "-":
                best_ep_str = f"Ep {best_ep}"

        ckpt_dir = Path("checkpoints") / exp_name
        history_file = ckpt_dir / "history.json"

        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries = []
                if isinstance(data, list):
                    entries = data
                elif isinstance(data, dict):
                    entries = data.get("epochs") or data.get("history") or []

                if entries and isinstance(entries, list):
                    last_entry = entries[-1]
                    last_ep = last_entry.get("epoch", len(entries))
                    last_v_loss = last_entry.get("val_loss")
                    if last_v_loss is not None and isinstance(last_v_loss, (int, float)):
                        last_ep_str = f"Ep {last_ep} ({last_v_loss:.3f})"
                    else:
                        last_ep_str = f"Ep {last_ep}"

                    valid_entries = [e for e in entries if isinstance(e, dict) and e.get("val_loss") is not None]
                    if valid_entries:
                        best_entry = min(valid_entries, key=lambda x: x.get("val_loss", float("inf")))
                        b_ep = best_entry.get("epoch", "-")
                        b_loss = best_entry.get("val_loss")
                        if b_loss is not None and b_loss != float("inf"):
                            best_ep_str = f"Ep {b_ep} ({b_loss:.3f})"
                        else:
                            best_ep_str = f"Ep {b_ep}"
                    return last_ep_str, best_ep_str
            except Exception:
                pass

        if last_ep_str != "-":
            return last_ep_str, best_ep_str

        candidates = [
            ckpt_dir / "apl_slm_best.pt",
            ckpt_dir / "apl_slm.pt",
            ckpt_dir / "best_model.pt",
            ckpt_dir / "last_model.pt",
        ]
        for pt in candidates:
            if pt.exists():
                try:
                    import torch
                    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
                    ep = ckpt.get("epoch")
                    loss = ckpt.get("val_loss") or ckpt.get("best_val_loss") or ckpt.get("loss")
                    if ep is not None:
                        if loss is not None and isinstance(loss, (int, float)):
                            last_ep_str = f"Ep {ep} ({loss:.3f})"
                            best_ep_str = f"Ep {ep} ({loss:.3f})"
                        else:
                            last_ep_str = f"Ep {ep}"
                            best_ep_str = f"Ep {ep}"
                        return last_ep_str, best_ep_str
                except Exception:
                    pass

        return last_ep_str, best_ep_str

    def refresh_queue_table(self):
        if not hasattr(self, "tree_queue") or not self.tree_queue.winfo_exists():
            return

        jobs = self.queue_manager.load()

        total = len(jobs)
        pending = sum(1 for j in jobs if j.status == "pending")
        running = sum(1 for j in jobs if j.status == "running")
        completed = sum(1 for j in jobs if j.status == "completed")
        failed = sum(1 for j in jobs if j.status in ("failed", "cancelled"))

        self.lbl_queue_total.config(text=f"Total: {total}")
        self.lbl_queue_pending.config(text=f"⏳ Pending: {pending}")
        self.lbl_queue_running.config(text=f"🚀 Running: {running}")
        self.lbl_queue_completed.config(text=f"✅ Completed: {completed}")
        self.lbl_queue_failed.config(text=f"❌ Failed: {failed}")

        for item in self.tree_queue.get_children():
            self.tree_queue.delete(item)

        status_icons = {
            "pending": "⏳ Pending",
            "running": "🚀 Running",
            "completed": "✅ Done",
            "failed": "❌ Failed",
            "cancelled": "⏹ Cancelled",
        }

        for idx, j in enumerate(jobs):
            status_text = status_icons.get(j.status, j.status)
            ft_text = Path(j.finetune_from).name if j.finetune_from else "-"
            ds_text = Path(j.dataset_path).name if j.dataset_path else "-"
            preset_text = j.model_preset if j.model_preset != "(none)" else "-"
            dev_text = j.device if j.device else "auto"
            pat_text = f"{j.early_stopping_patience} ep"

            last_ep_text, best_ep_text = self.get_experiment_epoch_stats(j.exp_name)

            self.tree_queue.insert(
                "",
                tk.END,
                values=(
                    idx + 1,
                    j.job_id,
                    status_text,
                    j.exp_name,
                    f"v{j.version}",
                    preset_text,
                    j.epochs,
                    last_ep_text,
                    best_ep_text,
                    j.batch_size,
                    f"{j.lr:.1e}",
                    pat_text,
                    dev_text,
                    ds_text,
                    ft_text,
                ),
                tags=(j.status,),
            )

    def move_queue_job(self, direction: int):
        if not hasattr(self, "tree_queue"):
            return
        selected = self.tree_queue.selection()
        if not selected:
            messagebox.showwarning("Select Job", "Please select a job from the queue to move.")
            return
        item_vals = self.tree_queue.item(selected[0], "values")
        job_id = item_vals[1]
        if self.queue_manager.move_job(job_id, direction):
            self.refresh_queue_table()
            for item in self.tree_queue.get_children():
                if self.tree_queue.item(item, "values")[1] == job_id:
                    self.tree_queue.selection_set(item)
                    break

    def delete_queue_job(self):
        if not hasattr(self, "tree_queue"):
            return
        selected = self.tree_queue.selection()
        if not selected:
            messagebox.showwarning("Select Job", "Please select a job from the queue to delete.")
            return
        item_vals = self.tree_queue.item(selected[0], "values")
        job_id = item_vals[1]
        exp_name = item_vals[3]
        if messagebox.askyesno("Confirm Delete", f"Remove job '{exp_name}' ({job_id}) from queue?"):
            self.queue_manager.remove_job(job_id)
            self.refresh_queue_table()

    def clear_completed_jobs(self):
        cleared = self.queue_manager.clear_completed()
        self.refresh_queue_table()
        messagebox.showinfo("Queue Cleared", f"Removed {cleared} completed/failed jobs from queue.")

    def reset_failed_jobs(self):
        count = self.queue_manager.reset_failed()
        self.refresh_queue_table()
        messagebox.showinfo("Jobs Reset", f"Reset {count} failed/cancelled jobs back to pending status.")

    def start_batch_runner(self):
        if self.batch_running or self.process_active:
            messagebox.showwarning("Runner Active", "A training process or batch runner is already active.")
            return

        pending = self.queue_manager.get_next_pending()
        if not pending:
            messagebox.showinfo("Queue Empty", "No pending jobs in queue to run. Add experiments to queue first.")
            return

        self.batch_running = True
        self.batch_stop_requested = False
        self.btn_start_batch.config(state=tk.DISABLED)
        self.btn_stop_batch.config(state=tk.NORMAL)

        threading.Thread(target=self._batch_runner_loop, daemon=True).start()

    def stop_batch_runner(self):
        self.batch_stop_requested = True
        if self.running_process and self.process_active:
            if messagebox.askyesno("Stop Running Job", "Batch runner stop requested. Do you want to terminate the currently running job immediately?"):
                self.kill_active_process()
            else:
                self.append_log("\n[!] Batch runner will pause after current training job finishes.\n")
        else:
            self.batch_running = False
            self.btn_start_batch.config(state=tk.NORMAL)
            self.btn_stop_batch.config(state=tk.DISABLED)
            self.append_log("\n[!] Batch runner paused.\n")

    def _batch_runner_loop(self):
        self.append_log("\n" + "=" * 65 + "\n")
        self.append_log("🚀 APL BATCH SCHEDULER STARTED\n")
        self.append_log("=" * 65 + "\n")

        while self.batch_running and not self.batch_stop_requested:
            self.queue_manager.load()
            job = self.queue_manager.get_next_pending()
            if not job:
                self.append_log("\n[+] All pending jobs in queue completed.\n")
                break

            self.queue_manager.update_job(job.job_id, status="running", started_at=time.time())
            self.root.after(0, self.refresh_queue_table)

            cmd_args = job.to_command_args()
            self.append_log(f"\n[+] Executing Queue Job [{job.job_id}]: {job.exp_name} (v{job.version})\n")

            self.root.after(0, self.reset_training_stats_ui)
            self.execute_command(cmd_args)

            while self.process_active:
                time.sleep(0.5)

            exit_code = getattr(self, "last_exit_code", 0)
            if exit_code == 0:
                self.queue_manager.update_job(job.job_id, status="completed", finished_at=time.time())
            else:
                self.queue_manager.update_job(job.job_id, status="failed", finished_at=time.time(), error_message=f"Exit code {exit_code}")

            self.root.after(0, self.refresh_queue_table)

        self.batch_running = False
        self.root.after(0, lambda: self.btn_start_batch.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.btn_stop_batch.config(state=tk.DISABLED))
        self.append_log("\n[+] Batch Runner finished.\n")

    def open_add_job_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("➕ Enqueue Custom APL Training Job")
        dlg.geometry("560x540")
        dlg.configure(bg=self.colors["bg_dark"])
        dlg.transient(self.root)
        dlg.grab_set()

        lbl_title = tk.Label(dlg, text="Configure Queued Experiment", font=("Segoe UI", 12, "bold"), bg=self.colors["bg_dark"], fg=self.colors["accent_hover"])
        lbl_title.pack(pady=(12, 8))

        form = tk.Frame(dlg, bg=self.colors["bg_card"], padx=15, pady=12)
        form.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 12))

        # Row 0: Preset & Version
        tk.Label(form, text="Preset (Size):", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_hover"]).grid(row=0, column=0, sticky="w", pady=4)
        combo_preset = ttk.Combobox(form, values=["small", "medium", "large", "deep", "wide", "xlarge", "huge", "giant", "(none)"], width=12, state="readonly")
        combo_preset.current(0)
        combo_preset.grid(row=0, column=1, sticky="w", padx=(0, 10), pady=4)

        tk.Label(form, text="Version:", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_hover"]).grid(row=0, column=2, sticky="w", pady=4)
        combo_ver = ttk.Combobox(form, values=["v3", "v2", "v1"], width=6, state="readonly")
        combo_ver.current(0)
        combo_ver.grid(row=0, column=3, sticky="w", pady=4)

        # Row 1: Exp Name & Epochs
        tk.Label(form, text="Exp Name:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=1, column=0, sticky="w", pady=4)
        ent_name = tk.Entry(form, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=20)
        ent_name.grid(row=1, column=1, sticky="w", padx=(0, 10), pady=4)

        tk.Label(form, text="Epochs:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=1, column=2, sticky="w", pady=4)
        ent_epochs = tk.Entry(form, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=8)
        ent_epochs.grid(row=1, column=3, sticky="w", pady=4)

        # Row 2: Batch & LR
        tk.Label(form, text="Batch Size:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=0, sticky="w", pady=4)
        ent_batch = tk.Entry(form, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=10)
        ent_batch.grid(row=2, column=1, sticky="w", padx=(0, 10), pady=4)

        tk.Label(form, text="Learning Rate:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=2, sticky="w", pady=4)
        ent_lr = tk.Entry(form, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=10)
        ent_lr.grid(row=2, column=3, sticky="w", pady=4)

        # Row 3: Grad Accum & Warmup
        tk.Label(form, text="Grad Accum:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=3, column=0, sticky="w", pady=4)
        ent_accum = tk.Entry(form, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=10)
        ent_accum.grid(row=3, column=1, sticky="w", padx=(0, 10), pady=4)

        tk.Label(form, text="Warmup Ep:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=3, column=2, sticky="w", pady=4)
        ent_warmup = tk.Entry(form, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=8)
        ent_warmup.grid(row=3, column=3, sticky="w", pady=4)

        # Row 4: Early Stopping Patience & Precision
        tk.Label(form, text="Early Stop (Patience):", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=4, column=0, sticky="w", pady=4)
        ent_patience = tk.Entry(form, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=10)
        ent_patience.grid(row=4, column=1, sticky="w", padx=(0, 10), pady=4)

        tk.Label(form, text="Precision:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=4, column=2, sticky="w", pady=4)
        combo_prec = ttk.Combobox(form, values=["auto", "fp16", "bf16", "fp32"], width=7, state="readonly")
        combo_prec.current(0)
        combo_prec.grid(row=4, column=3, sticky="w", pady=4)

        # Row 5: Device & Resume Checkpoint
        tk.Label(form, text="Device:", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_hover"]).grid(row=5, column=0, sticky="w", pady=4)
        combo_device = ttk.Combobox(form, values=["(auto)", "cuda", "cpu"], width=10, state="readonly")
        combo_device.current(0)
        combo_device.grid(row=5, column=1, sticky="w", padx=(0, 10), pady=4)

        chk_resume_var = tk.BooleanVar(value=True)
        chk_resume = ttk.Checkbutton(form, text="Resume Checkpoint", variable=chk_resume_var, style="TCheckbutton")
        chk_resume.grid(row=5, column=2, columnspan=2, sticky="w", pady=4)

        # Row 6: Dataset path
        tk.Label(form, text="Dataset Path:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=6, column=0, sticky="w", pady=4)
        ent_dataset = tk.Entry(form, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=32)
        ent_dataset.insert(0, "data/apl_corpus.txt")
        ent_dataset.grid(row=6, column=1, columnspan=3, sticky="w", pady=4)

        # Row 7: Fine-tune from
        tk.Label(form, text="Fine-Tune Path:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=7, column=0, sticky="w", pady=4)
        ent_ft = tk.Entry(form, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=32)
        ent_ft.grid(row=7, column=1, columnspan=3, sticky="w", pady=4)

        def _update_dialog_preset(event=None):
            preset_name = combo_preset.get().strip().lower()
            ver_name = combo_ver.get().strip().lower()
            cfg = self.MODEL_PRESETS.get(preset_name) or self.MODEL_PRESETS.get("small")

            base_prefix = f"{cfg.get('base_name', preset_name.capitalize())}-{ver_name}"
            next_name = self.get_next_experiment_name(base_prefix, queued_jobs=getattr(self.queue_manager, "jobs", None))

            ent_name.delete(0, tk.END)
            ent_name.insert(0, next_name)

            ent_epochs.delete(0, tk.END)
            ent_epochs.insert(0, cfg.get("epochs", "30"))

            ent_batch.delete(0, tk.END)
            ent_batch.insert(0, cfg.get("batch_size", "16"))

            ent_lr.delete(0, tk.END)
            ent_lr.insert(0, cfg.get("lr", "5e-4"))

            ent_warmup.delete(0, tk.END)
            ent_warmup.insert(0, cfg.get("warmup_epochs", "2"))

            ent_patience.delete(0, tk.END)
            ent_patience.insert(0, cfg.get("patience", "6"))

            accum_val = 2 if preset_name in ["large", "xlarge", "huge", "giant"] else 1
            ent_accum.delete(0, tk.END)
            ent_accum.insert(0, str(accum_val))

        combo_preset.bind("<<ComboboxSelected>>", _update_dialog_preset)
        combo_ver.bind("<<ComboboxSelected>>", _update_dialog_preset)
        _update_dialog_preset()

        def _on_submit():
            v_val = 3
            if "1" in combo_ver.get():
                v_val = 1
            elif "2" in combo_ver.get():
                v_val = 2
            elif "3" in combo_ver.get():
                v_val = 3

            ft_val = ent_ft.get().strip() or None
            dev_val = combo_device.get().strip()
            device_str = dev_val if dev_val in ["cuda", "cpu"] else None

            job = TrainingJob(
                exp_name=ent_name.get().strip() or "Custom-Exp",
                version=v_val,
                model_preset=combo_preset.get(),
                epochs=int(ent_epochs.get().strip() or "30"),
                batch_size=int(ent_batch.get().strip() or "16"),
                lr=float(ent_lr.get().strip() or "5e-4"),
                warmup_epochs=int(ent_warmup.get().strip() or "2"),
                grad_accum_steps=int(ent_accum.get().strip() or "1"),
                early_stopping_patience=int(ent_patience.get().strip() or "6"),
                precision=combo_prec.get().strip() or "auto",
                dataset_path=ent_dataset.get().strip() or "data/apl_corpus.txt",
                finetune_from=ft_val,
                resume=chk_resume_var.get(),
                device=device_str,
            )
            self.queue_manager.add_job(job)
            self.refresh_queue_table()
            self.append_log(f"[+] Enqueued Custom Job '{job.exp_name}' (v{job.version}, {job.model_preset}, patience {job.early_stopping_patience}, device={job.device or 'auto'}).\n")
            dlg.destroy()

        btn_box = tk.Frame(dlg, bg=self.colors["bg_dark"])
        btn_box.pack(fill=tk.X, padx=15, pady=(0, 12))

        btn_save = tk.Button(btn_box, text="➕ Enqueue Job", font=("Segoe UI", 10, "bold"), bg=self.colors["accent"], fg=self.colors["bg_dark"], bd=0, padx=14, pady=5, cursor="hand2", command=_on_submit)
        btn_save.pack(side=tk.LEFT, padx=(0, 8))

        btn_cancel = tk.Button(btn_box, text="Cancel", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text"], bd=0, padx=10, pady=5, cursor="hand2", command=dlg.destroy)
        btn_cancel.pack(side=tk.LEFT)

    def run_plot_generation(self):
        self.plot_label.config(image="", text="Regenerating graphs...")
        cmd = ["src/plot_experiments.py"]
        self.execute_command(cmd)

    def run_benchmarks(self):
        checkpoint = self.combo_bench_checkpoint.get() or "all"
        cmd = ["src/benchmark_autocomplete.py", "--checkpoint", checkpoint]
        self.execute_command(cmd)

    def run_autocomplete_generation(self):
        if not self.cached_model:
            messagebox.showwarning("Error", "Model is not loaded. Please select a checkpoint and click 'Load Model'.")
            return

        prompt = self.txt_prompt.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Error", "Please input an APL prompt.")
            return

        max_tokens = int(self.sld_max_tokens.get())
        temp = float(self.sld_temp.get())
        top_k = int(self.sld_top_k.get())

        self.btn_autocomplete.config(state=tk.DISABLED)
        self.txt_result.config(state=tk.NORMAL)
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, "🍏 Generating autocompletion sequence...")
        self.txt_result.config(state=tk.DISABLED)

        threading.Thread(
            target=self._autocomplete_worker,
            args=(prompt, max_tokens, temp, top_k),
            daemon=True,
        ).start()

    def _autocomplete_worker(self, prompt, max_tokens, temp, top_k):
        try:
            start_time = time.time()
            v = getattr(self.cached_model, "version", 3)
            if v == 3 and autocomplete_v3 is not None:
                completed_code = autocomplete_v3(
                    self.cached_model, self.cached_tokenizer, self.cached_device, prompt,
                    max_new_tokens=max_tokens, temperature=temp, top_k=top_k, verbose=False,
                )
            elif v == 2 and autocomplete_v2 is not None:
                completed_code = autocomplete_v2(
                    self.cached_model, self.cached_tokenizer, self.cached_device, prompt,
                    max_new_tokens=max_tokens, temperature=temp, top_k=top_k, verbose=False,
                )
            else:
                completed_code = autocomplete_v1(
                    self.cached_model, self.cached_tokenizer, self.cached_device, prompt,
                    max_new_tokens=max_tokens, temperature=temp, top_k=top_k, verbose=False,
                )
            elapsed = time.time() - start_time
            added_code = completed_code[len(prompt):] if completed_code.startswith(prompt) else completed_code

            self.root.after(0, lambda: self._show_autocomplete_results(prompt, added_code, elapsed))
        except Exception as e:
            self.root.after(0, lambda: self._show_autocomplete_error(str(e)))

    def _show_autocomplete_results(self, prompt, completed_suffix, duration):
        self.txt_result.config(state=tk.NORMAL)
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, f"⍝ Prompt:\n{prompt}\n\n")
        self.txt_result.insert(tk.END, f"⍝ 🍏 Completed in {duration:.2f}s:\n")
        self.txt_result.insert(tk.END, prompt)

        start_idx = self.txt_result.index(tk.INSERT)
        self.txt_result.insert(tk.END, completed_suffix)
        end_idx = self.txt_result.index(tk.INSERT)

        self.txt_result.tag_add("completion", start_idx, end_idx)
        self.txt_result.tag_config("completion", foreground=self.colors["accent_hover"])

        self.txt_result.config(state=tk.DISABLED)
        self.btn_autocomplete.config(state=tk.NORMAL)

    def _show_autocomplete_error(self, err_msg):
        self.txt_result.config(state=tk.NORMAL)
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, f"[ERROR] Failed to run autocompletion:\n{err_msg}")
        self.txt_result.config(state=tk.DISABLED)
        self.btn_autocomplete.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = APL_SLM_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
