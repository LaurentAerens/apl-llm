# 📐 APL IntelliSense SLM for VS Code

An ultra-lightweight, local Small Language Model (SLM, ~380k parameters) specialized exclusively in **APL code autocompletion and IntelliSense**, powered by PyTorch, RoPE, and single-token Key-Value (KV) caching.

---

## ✨ Features

- ⚡ **Microsecond-Latency Completions**: Powered by step-by-step KV-cached autoregressive decoding.
- 🎯 **Hyper-Focused Model**: ~380k parameter Transformer trained on APL syntax, dfns, and array idioms.
- 🔒 **100% Local & Private**: Runs completely on your local machine using PyTorch.
- 🔄 **Auto-Managed Background Daemon**: Starts transparently when editing .apl files.
- 🛠️ **Configurable Interpreter & Weights**: Point the extension to your custom trained .pt checkpoints.

---

## 🚀 Quick Start

1. Install the extension in VS Code.
2. Ensure Python 3.10+ and PyTorch are installed (pip install torch).
3. Open any .apl, .apln, .aplf, or .dyalog source file.
4. Start typing! Ghost completions will appear. Press Tab to accept suggestions.

---

## ⚙️ Extension Settings

| Setting | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| pl-slm.pythonPath | string | "python" | Path to Python executable with PyTorch installed. |
| pl-slm.modelPath | string | "checkpoints/apl_slm_best.pt" | Path to model weights .pt file. |
| pl-slm.maxTokens | integer | 128 | Max autocompleted tokens per completion. |

---

## 💬 Commands

- APL IntelliSense SLM: Restart Daemon - Restarts the local Python inference server.

---

## 📜 License

Licensed under the [GNU General Public License v3.0](LICENSE).
