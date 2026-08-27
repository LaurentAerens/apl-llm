import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from autocomplete import load_model, autocomplete
from tokenizer import APLTokenizer


BENCHMARK_PROMPTS = [
    # Category 1: Statistics & Aggregations
    {"name": "Mean Function", "category": "Statistics", "prompt": "{+/⍵", "expected_contain": ["≢", "}"]},
    {"name": "Explicit Mean", "category": "Statistics", "prompt": "{(+/⍵)÷", "expected_contain": ["≢", "⍵"]},
    {"name": "Variance / Sum of Squares", "category": "Statistics", "prompt": "{+/(⍵-", "expected_contain": ["2", "}"]},
    {"name": "Standard Deviation", "category": "Statistics", "prompt": "{(({+/⍵", "expected_contain": ["0.5", "}"]},

    # Category 2: Sorting & Indices
    {"name": "Ascending Grade Sort", "category": "Sorting", "prompt": "V[", "expected_contain": ["⍋", "]"]},
    {"name": "Descending Grade Sort", "category": "Sorting", "prompt": "X[", "expected_contain": ["⍒", "]"]},
    {"name": "Matrix Row Sort", "category": "Sorting", "prompt": "M[⍋M[;", "expected_contain": [";", "]"]},
    {"name": "Index Lookup", "category": "Sorting", "prompt": "keys⍳", "expected_contain": ["target"]},

    # Category 3: Array Algorithms & Predicates
    {"name": "Palindrome Predicate", "category": "Algorithms", "prompt": "{⍵≡", "expected_contain": ["⌽", "}"]},
    {"name": "Membership Test", "category": "Algorithms", "prompt": "{⍺∊", "expected_contain": ["⍵", "}"]},
    {"name": "Unique Elements", "category": "Algorithms", "prompt": "{∪", "expected_contain": ["⍵", "}"]},
    {"name": "Prime Number Sieve", "category": "Algorithms", "prompt": "{(~⍵∊", "expected_contain": ["⍳", "}"]},

    # Category 4: Arithmetic & Math Reductions
    {"name": "Factorial Reduction", "category": "Math", "prompt": "{×/1+⍳", "expected_contain": ["⍵", "}"]},
    {"name": "Maximum Scan/Reduction", "category": "Math", "prompt": "{⌈/", "expected_contain": ["⍵", "}"]},
    {"name": "Minimum Scan/Reduction", "category": "Math", "prompt": "{⌊/", "expected_contain": ["⍵", "}"]},
    {"name": "Cumulative Sum Scan", "category": "Math", "prompt": "{+\\", "expected_contain": ["⍵", "}"]},

    # Category 5: Tacit Programming & Trains
    {"name": "Tacit Mean (Fork)", "category": "Tacit", "prompt": "(+/ ÷", "expected_contain": ["≢", ")"]},
    {"name": "Tacit Range (Fork)", "category": "Tacit", "prompt": "(⌈/ -", "expected_contain": ["⌊/", ")"]},
    {"name": "Tacit Signum Fork", "category": "Tacit", "prompt": "(× ÷", "expected_contain": ["|", ")"]},
    {"name": "Tacit Composition", "category": "Tacit", "prompt": "(+ ∘", "expected_contain": ["×", ")"]},

    # Category 6: Matrix & Linear Algebra
    {"name": "Inner Product / Dot Product", "category": "LinearAlgebra", "prompt": "{+/⍺×", "expected_contain": ["⍵", "}"]},
    {"name": "Matrix Transpose", "category": "LinearAlgebra", "prompt": "{⍉", "expected_contain": ["⍵", "}"]},
    {"name": "Matrix Inverse", "category": "LinearAlgebra", "prompt": "{⌹", "expected_contain": ["⍵", "}"]},
    {"name": "Outer Product Table", "category": "LinearAlgebra", "prompt": "X∘.", "expected_contain": ["+", "Y"]},
]


def evaluate_balance(code: str, tokenizer: APLTokenizer) -> bool:
    tokens = tokenizer.encode(code)
    depths = tokenizer.compute_depth_sequences(tokens)
    final_depth = depths[-1] if depths else 0
    info = tokenizer.get_token_info(tokens[-1]) if tokens else None
    if info:
        final_depth = max(0, final_depth + info.paren_delta + info.bracket_delta + info.dfn_delta)
    return final_depth == 0


def get_checkpoint_val_loss(checkpoint_path: Path) -> float:
    """Reads validation loss from history.json or the checkpoint dictionary."""
    ckpt_dir = checkpoint_path.parent
    history_file = ckpt_dir / "history.json"
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("epochs", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            valid_losses = [e.get("val_loss") for e in entries if isinstance(e, dict) and e.get("val_loss") is not None]
            if valid_losses:
                return float(min(valid_losses))
        except Exception:
            pass

    # Fallback to checkpoint file
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        loss = ckpt.get("val_loss") or ckpt.get("best_val_loss") or ckpt.get("loss")
        if loss is not None and isinstance(loss, (int, float)):
            return float(loss)
    except Exception:
        pass

    return float("inf")


def benchmark_single_checkpoint(checkpoint_path: str, verbose: bool = True) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        print(f"[!] Checkpoint not found: {checkpoint_path}")
        return {
            "passed": 0,
            "balanced": 0,
            "total": len(BENCHMARK_PROMPTS),
            "acc_pct": 0.0,
            "bal_pct": 0.0,
            "checkpoint": str(checkpoint_path),
            "val_loss": float("inf"),
            "params": 0,
            "version": "?",
        }

    model, tokenizer, device = load_model(checkpoint_path)
    ver = getattr(model, "version", "?")
    n_params = model.count_parameters() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters())
    val_loss = get_checkpoint_val_loss(path)

    if verbose:
        print("=" * 70)
        print(f"📊 Benchmarking Checkpoint: {path.parent.name}/{path.name}")
        print(f"   Architecture: v{ver} | Parameters: {n_params:,} | Best Val Loss: {val_loss:.4f} | Device: {device}")
        print("=" * 70)

    passed = 0
    balanced_count = 0

    for idx, test in enumerate(BENCHMARK_PROMPTS, 1):
        prompt = test["prompt"]
        full_code = autocomplete(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=32,
            temperature=0.1,
            top_k=5,
            verbose=False,
        )
        completion = full_code[len(prompt):] if full_code.startswith(prompt) else full_code

        contains_expected = all(token in completion for token in test["expected_contain"])
        is_balanced = evaluate_balance(full_code, tokenizer)

        if contains_expected:
            passed += 1
        if is_balanced:
            balanced_count += 1

        if verbose:
            status = "✓ PASS" if contains_expected else "✗ FAIL"
            bal_str = "✓ Balanced" if is_balanced else "✗ Unbalanced"
            print(f"[{idx:02d}] {test['name']:<28} ({test['category']}) | {status:6} | {bal_str:12}")
            print(f"     Prompt:     {prompt}")
            print(f"     Completion: {completion.strip()}\n")

    total = len(BENCHMARK_PROMPTS)
    acc_pct = (passed / total) * 100
    bal_pct = (balanced_count / total) * 100

    if verbose:
        print("----------------------------------------------------------------------")
        print(f"  Accuracy:  {passed}/{total} ({acc_pct:.1f}%)")
        print(f"  Balance:   {balanced_count}/{total} ({bal_pct:.1f}%)")
        print("----------------------------------------------------------------------\n")

    return {
        "passed": passed,
        "balanced": balanced_count,
        "total": total,
        "acc_pct": acc_pct,
        "bal_pct": bal_pct,
        "checkpoint": f"{path.parent.name}/{path.name}",
        "exp_name": path.parent.name,
        "version": f"v{ver}",
        "params": n_params,
        "val_loss": val_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="APL Model Benchmark Suite")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="all",
        help="Path to model checkpoint, or 'all' to compare all experiments under checkpoints/ (default: all)",
    )
    args = parser.parse_args()

    if args.checkpoint == "all":
        checkpoints_dir = Path("checkpoints")
        if not checkpoints_dir.exists() or not list(checkpoints_dir.iterdir()):
            print("[!] No checkpoints found in checkpoints/.")
            return

        print("=" * 75)
        print(f"[+] RUNNING BATCH COMPARATIVE BENCHMARK FOR ALL EXPERIMENTS ({len(BENCHMARK_PROMPTS)} TESTS)")
        print("=" * 75)

        results = []
        for exp_dir in sorted(checkpoints_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            best_model = exp_dir / "apl_slm_best.pt"
            std_model = exp_dir / "apl_slm.pt"
            model_path = best_model if best_model.exists() else (std_model if std_model.exists() else None)

            if not model_path:
                continue

            print(f"[+] Evaluating experiment: {exp_dir.name}...")
            try:
                res = benchmark_single_checkpoint(str(model_path), verbose=False)
                results.append(res)
            except Exception as e:
                print(f"  [!] Failed to evaluate {exp_dir.name}: {e}")

        # Also check root checkpoints/ if any pt files exist directly
        for pt_file in sorted(checkpoints_dir.glob("*.pt")):
            try:
                res = benchmark_single_checkpoint(str(pt_file), verbose=False)
                res["exp_name"] = pt_file.stem
                results.append(res)
            except Exception as e:
                print(f"  [!] Failed to evaluate {pt_file.name}: {e}")

        if not results:
            print("[!] No valid checkpoints evaluated.")
            return

        # Sort by accuracy percentage descending, then balance descending
        results.sort(key=lambda r: (r["acc_pct"], r["bal_pct"]), reverse=True)

        print("\n" + "=" * 90)
        print("APL SLM COMPARATIVE BENCHMARK RESULTS")
        print("=" * 90)
        print(f"| {'Experiment Name':<24} | {'Ver':<5} | {'Params':<10} | {'Best Val Loss':<13} | {'Accuracy':<14} | {'Syntax Balance':<14} |")
        print(f"| {'-'*24} | {'-'*5} | {'-'*10} | {'-'*13} | {'-'*14} | {'-'*14} |")
        for res in results:
            val_loss_str = f"{res['val_loss']:.4f}" if res["val_loss"] != float("inf") else "N/A"
            acc_str = f"{res['acc_pct']:.1f}% ({res['passed']}/{res['total']})"
            bal_str = f"{res['bal_pct']:.1f}% ({res['balanced']}/{res['total']})"
            print(f"| {res['exp_name']:<24} | {res['version']:<5} | {res['params']:<10,} | {val_loss_str:<13} | {acc_str:<14} | {bal_str:<14} |")
        print("=" * 90)

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        bench_md = data_dir / "benchmark_results.md"
        with open(bench_md, "w", encoding="utf-8") as f:
            f.write(f"# APL SLM Model Benchmarking Comparison ({len(BENCHMARK_PROMPTS)} Tests)\n\n")
            f.write("Evaluation of APL syntax autocompletion, operator reasoning, bracket/parenthesis/dfn structural balance across trained model checkpoints.\n\n")
            f.write("| Experiment Name | Architecture | Parameters | Best Val Loss | Accuracy | Syntax Balance |\n")
            f.write("|---|---|---|---|---|---|\n")
            for res in results:
                val_loss_str = f"{res['val_loss']:.4f}" if res["val_loss"] != float("inf") else "N/A"
                acc_str = f"**{res['acc_pct']:.1f}%** ({res['passed']}/{res['total']})"
                bal_str = f"{res['bal_pct']:.1f}% ({res['balanced']}/{res['total']})"
                f.write(f"| `{res['exp_name']}` | {res['version']} | {res['params']:,} | {val_loss_str} | {acc_str} | {bal_str} |\n")

            f.write("\n## Experiment Performance Breakdown\n\n")
            for res in results:
                f.write(f"### `{res['exp_name']}` ({res['version']})\n")
                f.write(f"- **Parameters**: {res['params']:,}\n")
                f.write(f"- **Validation Loss**: {res['val_loss']:.4f}\n" if res["val_loss"] != float("inf") else "- **Validation Loss**: N/A\n")
                f.write(f"- **Completion Accuracy**: {res['acc_pct']:.1f}% ({res['passed']}/{res['total']})\n")
                f.write(f"- **Structural Syntax Balance**: {res['bal_pct']:.1f}% ({res['balanced']}/{res['total']})\n\n")

        print(f"\n[OK] Comparison results saved to: {bench_md}")
    else:
        benchmark_single_checkpoint(args.checkpoint, verbose=True)


if __name__ == "__main__":
    main()
