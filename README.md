# 📐 APL SLM: Hyper-Focused Small Language Model & IntelliSense

An ultra-lightweight, high-performance Causal Transformer Small Language Model (~380,000 parameters) specialized exclusively in **APL code autocompletion and IntelliSense**.

Built with single-token **KV Caching**, Pre-**RMSNorm**, **SwiGLU**, **Rotary Position Embeddings (RoPE)**, and a specialized Unicode glyph tokenizer with structural parenthesis, bracket, and dfn (`{}`) depth conditioning.

---

## 🏛️ Architecture Evolution & Versioning

For complete architectural details, formulas, and comparison matrices, see [ARCHITECTURE_VERSIONS.md](ARCHITECTURE_VERSIONS.md).

* **`v1` (Baseline)**: Standard autoregressive Causal Transformer (`LayerNorm` Post-LN, GELU, learned absolute positional embeddings `wpe`).
* **`v2` (Depth Conditioned)**: Conditioned on structural parenthesis `()`, bracket `[]`, and dfn `{}` nesting depths with an auxiliary depth prediction head.
* **`v3` (Modern RoPE + SwiGLU)**: Modern high-performance transformer with Pre-**RMSNorm**, **SwiGLU**, Rotary Position Embeddings (**RoPE**), **QK-Norm**, and single-token KV caching.

---

## 🚀 Key Features

* **Interactive GUI Control Center (`src/gui.py`):** Full-featured dark-theme desktop studio with dataset scraper, tokenizer diagnostics, training dashboard, fine-tuning, plot visualizer, autocomplete playground, and benchmarks.
* **Hyper-Focused Architecture:** Tailored exclusively for array programming without multi-gigabyte conversational overhead (~380k parameters).
* **Unicode APL Character Tokenizer:** Full support for standard ISO / Dyalog APL glyphs (⍳, ⍴, ⌽, ⍉, ⍋, ⍒, ⍺, ⍵, ∇, ⋄, ←, etc.) across ~170 tokens.
* **Structural Depth Conditioning:** Tracks nesting depth of parentheses `()`, brackets `[]`, and dfns `{}` for syntax-aware balanced code generation.
* **KV-Cached Generation:** Single-token autoregressive decoding ($\mathcal{O}(1)$ step projection time) running in sub-millisecond speeds on local CPU.
* **Open Source Dataset Scraper:** Scrapes GitHub and GitLab APL repositories under verified permissive open-source licenses and auto-generates `data/ATTRIBUTION.md`.
* **VS Code Integration:** Includes the companion `extension/` package (**apl-intellisense-slm**) providing inline ghost text completions.

---

## 🛠️ Installation

Requirements: Python 3.10+ and PyTorch.

```bash
cd c:\Users\laerens\source\repos\Personal\apl-llm
py -3 -m pip install -r requirements.txt
```

---

## 💻 Usage

### 0. Launch GUI Control Center
Run the desktop GUI Control Center for all dataset scraping, training, metric plots, and autocompletion:

```bash
py -3 src/gui.py
```

---

### 1. Collect Dataset & Generate License Attribution
Scrape open-source APL repositories from GitHub and GitLab and augment with synthetic algorithmic idioms and dfns:

```bash
# Scrape GitHub + GitLab and generate synthetic idioms
py -3 src/dataset_collector.py --mode search --limit 50 --sources github gitlab synthetic
```

*Place your GitHub token in `settings.local` or pass `--token <PAT>` to bypass rate limits.*

---

### 2. Train the Model
Train any architecture generation (`--version 1`, `2`, or `3`) using presets (`small`, `medium`, `large`, `deep`, `wide`):

```bash
# Train default modern v3 model with Small preset
py -3 src/train.py --version 3 --model_preset small --epochs 40 --exp_name Small-v3.0

# Train v2 depth conditioned model
py -3 src/train.py --version 2 --model_preset small --epochs 40 --exp_name Small-v2.0
```

Checkpoints will be saved automatically to `checkpoints/<exp_name>/apl_slm_best.pt` with `tokenizer.json` and `history.json`.

---

### 3. Run Autocomplete CLI
Run interactive inference or test prompts from the terminal (auto-detects checkpoint architecture version):

```bash
py -3 src/autocomplete.py --prompt "{+/⍵"
```

---

### 4. Run Benchmark Suite
Evaluate functional accuracy and syntax balancing across single checkpoints or all models:

```bash
# Benchmark specific checkpoint
py -3 src/benchmark_autocomplete.py --checkpoint checkpoints/Small-v3.0/apl_slm_best.pt

# Benchmark all checkpoints in checkpoints/
py -3 src/benchmark_autocomplete.py --checkpoint all
```

---

### 5. Plot Loss & Perplexity Curves
Generate side-by-side training/validation loss and perplexity curves from all experiment histories:

```bash
py -3 src/plot_experiments.py
```

Saves visualization to `data/experiment_loss_comparison.png`.

---

### 6. VS Code Extension Packaging (apl-intellisense-slm)
Package into a standalone `.vsix` installer for VS Code:

```bash
cd extension
vsce package
```

This generates `apl-intellisense-slm-0.1.0.vsix` ready to install via **Extensions > Install from VSIX...** in VS Code.

---

## 📁 Repository Structure

```text
apl-llm/
├── ARCHITECTURE_VERSIONS.md    # Multi-generation architecture guide & specs
├── LICENSE
├── README.md
├── requirements.txt
├── settings.local.example
├── data/
│   ├── apl_corpus.txt          # Training corpus
│   ├── experiment_loss_comparison.png # Comparison plots
│   └── ATTRIBUTION.md          # Open-source license attributions
├── checkpoints/                # Model checkpoints & histories
├── src/
│   ├── gui.py                  # Tkinter Desktop GUI Control Center
│   ├── model.py                # AutoModel factory & root model aliases
│   ├── tokenizer.py            # Unicode APL glyph tokenizer with depth tracking
│   ├── train.py                # Unified multi-version training CLI
│   ├── autocomplete.py         # Multi-version autocompletion engine
│   ├── benchmark_autocomplete.py # Benchmark evaluation suite
│   ├── plot_experiments.py     # Multi-experiment loss & perplexity plotter
│   ├── dataset_collector.py    # GitHub repo scraper & license validator
│   ├── synthetic_generator.py  # Synthetic APL idiom & dfn generator
│   ├── autocomplete_server.py  # Fast JSON stdio daemon for VS Code
│   ├── v1/                     # Version 1 (Baseline Causal Transformer)
│   │   ├── model_v1.py
│   │   ├── train_v1.py
│   │   └── autocomplete_v1.py
│   ├── v2/                     # Version 2 (Structural Depth Conditioned)
│   │   ├── model_v2.py
│   │   ├── train_v2.py
│   │   └── autocomplete_v2.py
│   └── v3/                     # Version 3 (Modern RoPE + SwiGLU + QK-Norm)
│       ├── model_v3.py
│       ├── train_v3.py
│       └── autocomplete_v3.py
└── extension/
    ├── package.json            # apl-intellisense-slm manifest
    ├── extension.js            # Inline completion provider
    ├── README.md
    └── media/
```

---

## 📜 License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
