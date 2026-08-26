import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from autocomplete import load_model, autocomplete
from tokenizer import APLTokenizer

BENCHMARK_PROMPTS = [
    # Category: Statistics
    {"category": "Statistics", "prompt": "{+/⍵", "expected_contain": ["≢", "}"]},
    {"category": "Statistics", "prompt": "{(+/⍵)÷", "expected_contain": ["≢", "⍵"]},
    # Category: Sorting
    {"category": "Sorting", "prompt": "V[", "expected_contain": ["⍋", "]"]},
    {"category": "Sorting", "prompt": "X[", "expected_contain": ["⍒", "]"]},
    # Category: Palindrome / Algorithms
    {"category": "Algorithms", "prompt": "{⍵≡", "expected_contain": ["⌽", "}"]},
    {"category": "Algorithms", "prompt": "{⍺∊", "expected_contain": ["⍵", "}"]},
    # Category: Math
    {"category": "Math", "prompt": "{×/1+⍳", "expected_contain": ["⍵", "}"]},
    {"category": "Math", "prompt": "{⌈/", "expected_contain": ["⍵", "}"]},
    # Category: Tacit Fork
    {"category": "Tacit", "prompt": "(+/ ÷", "expected_contain": ["≢", ")"]},
    {"category": "Tacit", "prompt": "(⌈/ -", "expected_contain": ["⌊/", ")"]},
    # Category: Linear Algebra
    {"category": "LinearAlgebra", "prompt": "{+/⍺×", "expected_contain": ["⍵", "}"]},
    {"category": "LinearAlgebra", "prompt": "{⍉", "expected_contain": ["⍵", "}"]},
]


def evaluate_balance(code: str, tokenizer: APLTokenizer) -> bool:
    tokens = tokenizer.encode(code)
    depths = tokenizer.compute_depth_sequences(tokens)
    final_depth = depths[-1] if depths else 0
    info = tokenizer.get_token_info(tokens[-1]) if tokens else None
    if info:
        final_depth = max(0, final_depth + info.paren_delta + info.bracket_delta + info.dfn_delta)
    return final_depth == 0


def benchmark_single_checkpoint(checkpoint_path: str) -> Dict[str, float]:
    path = Path(checkpoint_path)
    if not path.is_file():
        print(f"[!] Checkpoint not found: {checkpoint_path}")
        return {"passed": 0, "balanced": 0, "total": len(BENCHMARK_PROMPTS)}

    model, tokenizer, device = load_model(checkpoint_path)
    ver = getattr(model, "version", "?")
    n_params = model.count_parameters() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters())

    print("=" * 70)
    print(f"📊 Benchmarking Checkpoint: {path.parent.name}/{path.name}")
    print(f"   Architecture: v{ver} | Parameters: {n_params:,} | Device: {device}")
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

        status = "✓ PASS" if contains_expected else "✗ FAIL"
        bal_str = "✓ Balanced" if is_balanced else "✗ Unbalanced"
        print(f"[{idx:02d}] {test['category']:15} | {status:6} | {bal_str:12}")
        print(f"     Prompt:     {prompt}")
        print(f"     Completion: {completion.strip()}\n")

    total = len(BENCHMARK_PROMPTS)
    acc_pct = (passed / total) * 100
    bal_pct = (balanced_count / total) * 100

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
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="APL Model Benchmark Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline/apl_slm_best.pt", help="Model path or 'all'")
    args = parser.parse_args()

    if args.checkpoint == "all":
        checkpoints_dir = Path("checkpoints")
        ckpt_files = list(checkpoints_dir.glob("**/*.pt"))
        if not ckpt_files:
            print("[!] No checkpoints found in checkpoints/.")
            return

        # Prioritize best checkpoints
        best_ckpts = [c for c in ckpt_files if "best" in c.name]
        targets = best_ckpts if best_ckpts else ckpt_files

        results = []
        for ckpt in targets:
            res = benchmark_single_checkpoint(str(ckpt))
            results.append(res)

        print("\n" + "=" * 70)
        print("🏆 ALL CHECKPOINTS BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Checkpoint':<35} | {'Accuracy':<15} | {'Syntax Balance':<15}")
        print("-" * 70)
        for r in results:
            acc_str = f"{r['passed']}/{r['total']} ({r['acc_pct']:.1f}%)"
            bal_str = f"{r['balanced']}/{r['total']} ({r['bal_pct']:.1f}%)"
            print(f"{r['checkpoint']:<35} | {acc_str:<15} | {bal_str:<15}")
        print("=" * 70)
    else:
        benchmark_single_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
