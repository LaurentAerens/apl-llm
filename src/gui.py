import os
import sys
import time
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
    def get_next_experiment_name(base_name: str, checkpoints_dir: Path = Path("checkpoints")) -> str:
        """
        Calculates the next auto-incremented experiment name (e.g. Small-v3.0, Small-v3.1, Small-v3.2)
        by scanning existing directories and checkpoint files in checkpoints/.
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
            text="Collects open-source APL repositories from GitHub under verified permissive licenses, generates ATTRIBUTION.md, and augments with synthetic algorithmic idioms and dfns.",
            font=("Segoe UI", 10),
            bg=self.colors["bg_dark"],
            fg=self.colors["text_muted"],
            wraplength=740,
            justify=tk.LEFT,
        )
        desc.pack(anchor="w", pady=(0, 15))

        card = tk.Frame(frame, bg=self.colors["bg_card"], bd=0, padx=16, pady=16)
        card.pack(fill=tk.X, pady=(0, 15))

        # Mode Selector
        tk.Label(card, text="Collection Mode:", font=("Segoe UI", 10, "bold"), bg=self.colors["bg_card"], fg=self.colors["text"]).grid(row=0, column=0, sticky="w", pady=6)
        self.ds_mode = ttk.Combobox(card, values=["curated", "search", "all"], width=15, state="readonly")
        self.ds_mode.current(0)
        self.ds_mode.grid(row=0, column=1, sticky="w", padx=10, pady=6)

        # Limit
        tk.Label(card, text="Max Repos to Scan (0 for All):", font=("Segoe UI", 10), bg=self.colors["bg_card"], fg=self.colors["text"]).grid(row=1, column=0, sticky="w", pady=6)
        self.ds_limit = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=10)
        self.ds_limit.insert(0, "50")
        self.ds_limit.grid(row=1, column=1, sticky="w", padx=10, pady=6)

        # Github Token (PAT)
        tk.Label(card, text="GitHub Personal Access Token (Optional):", font=("Segoe UI", 10), bg=self.colors["bg_card"], fg=self.colors["text"]).grid(row=2, column=0, sticky="w", pady=6)
        self.ds_token = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=35, show="*")
        self.ds_token.grid(row=2, column=1, sticky="w", padx=10, pady=6)

        # Run Controls
        btn_frame = tk.Frame(frame, bg=self.colors["bg_dark"])
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_run_ds = tk.Button(
            btn_frame,
            text="🍏 Build Dataset (Scrape GitHub)",
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
        self.trn_preset = ttk.Combobox(card, values=["small", "medium", "large", "deep", "wide", "(none)"], width=13, state="readonly")
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

        # Row 2 & 3: Learning Rate, Warmup Epochs, Early Stop Patience, Training Precision
        tk.Label(card, text="Learning Rate:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=0, sticky="w", pady=3)
        self.trn_lr = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=13)
        self.trn_lr.insert(0, "5e-4")
        self.trn_lr.grid(row=3, column=0, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Warmup Epochs:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=1, sticky="w", pady=3)
        self.trn_warmup = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=20)
        self.trn_warmup.insert(0, "2")
        self.trn_warmup.grid(row=3, column=1, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Early Stop Patience:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=2, sticky="w", pady=3)
        self.trn_patience = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=10)
        self.trn_patience.insert(0, "6")
        self.trn_patience.grid(row=3, column=2, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Training Precision:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=2, column=3, sticky="w", pady=3)
        self.trn_prec = ttk.Combobox(card, values=["auto", "bf16", "fp16", "fp32"], width=10, state="readonly")
        self.trn_prec.current(0)
        self.trn_prec.grid(row=3, column=3, sticky="w", padx=(0, 15), pady=(0, 10))

        # Row 4 & 5: Experiment Name, Hardware Device & Checkboxes
        tk.Label(card, text="Experiment Name:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=4, column=0, sticky="w", pady=3)
        self.trn_exp = tk.Entry(card, bg=self.colors["bg_dark"], fg=self.colors["text"], insertbackground=self.colors["text"], bd=0, width=13)
        self.trn_exp.insert(0, "Small-v3.0")
        self.trn_exp.grid(row=5, column=0, sticky="w", padx=(0, 15), pady=(0, 10))

        tk.Label(card, text="Hardware Device:", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).grid(row=4, column=1, sticky="w", pady=3)
        self.trn_device = ttk.Combobox(card, values=["(auto)", "cpu", "cuda"], width=20, state="readonly")
        self.trn_device.current(0)
        self.trn_device.grid(row=5, column=1, sticky="w", padx=(0, 15), pady=(0, 10))

        self.trn_resume_val = tk.BooleanVar(value=True)
        self.chk_resume = ttk.Checkbutton(card, text="Resume Checkpoint", variable=self.trn_resume_val, style="TCheckbutton")
        self.chk_resume.grid(row=5, column=2, sticky="w", pady=(0, 10))

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
        self.btn_run_plot.pack(side=tk.LEFT)

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
                text="No comparison plot found.\n\nClick 'Regenerate Comparison Plots' to scan history logs and plot validation curves.",
            )

    # ----------------------------------------------------
    # TAB 5: Autocomplete Playground
    # ----------------------------------------------------
    def create_autocomplete_tab(self, frame):
        header = tk.Label(frame, text="⚡ APL SLM Autocomplete Studio", font=("Segoe UI", 18, "bold"), bg=self.colors["bg_dark"], fg=self.colors["accent_hover"])
        header.pack(anchor="w", pady=(0, 10))

        # Model Selector Panel
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

        # Controls & Prompt Card
        ctrl_frame = tk.Frame(frame, bg=self.colors["bg_dark"])
        ctrl_frame.pack(fill=tk.X, pady=(0, 12))

        # Sliders Card
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

        # Prompt input panel
        prompt_card = tk.Frame(ctrl_frame, bg=self.colors["bg_card"], bd=0, padx=16, pady=14)
        prompt_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        prompt_hdr = tk.Frame(prompt_card, bg=self.colors["bg_card"])
        prompt_hdr.pack(fill=tk.X, pady=(0, 4))
        tk.Label(prompt_hdr, text="Input APL Code Prefix:", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_hover"]).pack(side=tk.LEFT)
        tk.Label(prompt_hdr, text="(Click glyphs below to insert)", font=("Segoe UI", 8), bg=self.colors["bg_card"], fg=self.colors["text_muted"]).pack(side=tk.RIGHT)

        # 🍏 Rich APL Glyph Ribbon
        glyph_frame = tk.Frame(prompt_card, bg=self.colors["bg_card"])
        glyph_frame.pack(fill=tk.X, pady=(0, 6))

        # Categorized glyph palettes
        quick_glyphs = [
            "⍳", "⍴", "⌽", "⊖", "⍉", "↑", "↓", "⊂", "⊃", "⊆", "⊇",
            "⍋", "⍒", "∊", "⍷", "⍸", "∪", "∩", "⌸", "⌹", "⊥", "⊤",
            "⍺", "⍵", "∇", "⋄", "←", "→", "¨", "⍨", "⍣", "⍤", "⍥",
            "○", "⌈", "⌊", "≢", "≡", "≠", "≤", "≥", "+", "×", "÷", "*", "|",
        ]
        
        # Display in neat flow rows
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

        # Results area
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

    def parse_training_log_line(self, line: str):
        # 1. Match: [Epoch 12/40] Completed in 4.2s | LR: 0.000350
        m_epoch = re.search(r"\[Epoch\s+(\d+)/(\d+)\]\s+Completed in\s+([\d\.]+)s\s*\|\s*LR:\s*([\d\.e\-]+)", line)
        if m_epoch:
            ep, total_ep, dur, lr = m_epoch.groups()
            self._trn_curr_ep = ep
            self._trn_total_ep = total_ep
            self._trn_dur = dur
            self._trn_lr = lr
            if hasattr(self, "lbl_trn_last_epoch"):
                self.lbl_trn_last_epoch.config(text=f"Epoch: {ep} / {total_ep}")

        # 2. Match: - Train Loss: 0.4512 | Train Perplexity: 1.57
        m_train = re.search(r"Train Loss:\s*([\d\.]+)\s*\|\s*Train Perplexity:\s*([\d\.]+|inf)", line)
        if m_train:
            t_loss, t_ppl = m_train.groups()
            self._trn_train_loss = t_loss
            self._trn_train_ppl = t_ppl
            if hasattr(self, "lbl_trn_train_loss"):
                self.lbl_trn_train_loss.config(text=f"Train Loss: {t_loss}")
                self.lbl_trn_train_ppl.config(text=f"Train PPL: {t_ppl}")

        # 3. Match: - Val Loss:   0.4820 | Val Perplexity:   1.62
        m_val = re.search(r"Val Loss:\s*([\d\.]+)\s*\|\s*Val Perplexity:\s*([\d\.]+|inf)", line)
        if m_val:
            v_loss, v_ppl = m_val.groups()
            self._trn_val_loss = v_loss
            self._trn_val_ppl = v_ppl
            lr_str = getattr(self, "_trn_lr", "-")
            dur_str = getattr(self, "_trn_dur", "-")
            if hasattr(self, "lbl_trn_val_loss"):
                self.lbl_trn_val_loss.config(text=f"Val Loss:   {v_loss}")
                self.lbl_trn_val_ppl.config(text=f"Val PPL:   {v_ppl}")
                self.lbl_trn_lr_time.config(text=f"LR: {lr_str} | Duration: {dur_str}s")

        # 4. Match Best Val Loss improvements
        m_best_imp = re.search(r"Validation Loss improved from [\d\.]+ to ([\d\.]+)", line)
        m_best_init = re.search(r"Initial best model saved.*Val Loss:\s*([\d\.]+)", line)
        best_loss_found = None
        if m_best_imp:
            best_loss_found = m_best_imp.group(1)
        elif m_best_init:
            best_loss_found = m_best_init.group(1)

        if best_loss_found:
            ep_str = getattr(self, "_trn_curr_ep", "?")
            v_ppl_str = getattr(self, "_trn_val_ppl", "-")
            if hasattr(self, "lbl_trn_best_epoch"):
                self.lbl_trn_best_epoch.config(text=f"Best Epoch: Epoch {ep_str}")
                self.lbl_trn_best_loss.config(text=f"Best Val Loss: {best_loss_found}", fg=self.colors["accent_hover"])
                self.lbl_trn_best_ppl.config(text=f"Best Val PPL: {v_ppl_str}")
                self.lbl_trn_best_status.config(text="Status: Saved best checkpoint ⭐", fg=self.colors["apple_gold"])

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

    def set_buttons_running_state(self, is_running):
        state = tk.DISABLED if is_running else tk.NORMAL
        kill_state = tk.NORMAL if is_running else tk.DISABLED

        for btn_name in ["btn_run_ds", "btn_gen_synth", "btn_gen_synth_ft", "btn_inspect_tok", "btn_run_train", "btn_run_ft", "btn_run_plot", "btn_run_bench"]:
            if hasattr(self, btn_name):
                getattr(self, btn_name).config(state=state)

        for kill_btn in ["btn_kill_ds", "btn_kill_train", "btn_kill_ft", "btn_kill_bench"]:
            if hasattr(self, kill_btn):
                getattr(self, kill_btn).config(state=kill_state)

    def refresh_checkpoints(self):
        checkpoints = []
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            for f in checkpoints_dir.glob("*.pt"):
                checkpoints.append(str(f))
            for sub in sorted(checkpoints_dir.iterdir()):
                if sub.is_dir():
                    for f in sub.glob("*.pt"):
                        checkpoints.append(str(f))

        if hasattr(self, "combo_checkpoint") and self.combo_checkpoint.winfo_exists():
            self.combo_checkpoint.config(values=checkpoints)
            if checkpoints and not self.combo_checkpoint.get():
                best_cands = [c for c in checkpoints if "best" in c]
                self.combo_checkpoint.set(best_cands[0] if best_cands else checkpoints[0])

        if hasattr(self, "combo_ft_checkpoint") and self.combo_ft_checkpoint.winfo_exists():
            self.combo_ft_checkpoint.config(values=checkpoints)
            if checkpoints and not self.combo_ft_checkpoint.get():
                self.combo_ft_checkpoint.set(checkpoints[0])
                self.on_ft_checkpoint_selected()

        if hasattr(self, "combo_bench_checkpoint") and self.combo_bench_checkpoint.winfo_exists():
            bench_opts = ["all"] + checkpoints
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
        next_exp_name = self.get_next_experiment_name(base_prefix)

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

        self.execute_command(cmd)

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
