"""
Synthetic APL Code Generator.
Generates valid synthetic APL expressions, dfns, tacit trains, matrix transforms, and algorithms.
"""

import sys
import random
import argparse
from pathlib import Path
from typing import List, Optional


class APLSyntheticGenerator:
    """Generates valid synthetic APL expressions, dfns, tacit trains, and algorithms."""

    REDUCTIONS = [
        "+/", "×/", "⌈/", "⌊/", "∧/", "∨/", "≠/", "=/", "~/", "|/"
    ]

    SCANS = [
        "+\\", "×\\", "⌈\\", "⌊\\", "∧\\", "∨\\"
    ]

    DFN_PATTERNS = [
        # Math & Statistics
        "{+/⍵÷≢⍵}",                       # Mean
        "{(+/⍵)÷≢⍵}",                     # Mean with parens
        "{(+/⍵*2)÷≢⍵}",                   # Root mean square base
        "{(+/((⍵-((+/⍵)÷≢⍵))*2))÷≢⍵}",    # Variance
        "{×/1+⍳⍵}",                       # Factorial
        "{1≥⍵:1 ⋄ ⍵×∇ ⍵-1}",              # Recursive Factorial
        "{⍵≤1:⍵ ⋄ (∇ ⍵-1)+(∇ ⍵-2)}",      # Recursive Fibonacci
        "{⌈/⍵}",                          # Maximum
        "{⌊/⍵}",                          # Minimum
        "{(⌈/⍵)-(⌊/⍵)}",                  # Range (max - min)
        "{|⍵}",                           # Absolute value
        "{(⍵>0)-(⍵<0)}",                  # Signum
        "{0=2|⍵}",                        # Is even mask
        "{1=2|⍵}",                        # Is odd mask

        # Array & Manipulation
        "{∪⍵}",                           # Unique elements
        "{⍵≡⌽⍵}",                         # Is Palindrome
        "{⌽⍵}",                           # Reverse vector
        "{⊖⍵}",                           # Reverse along first axis
        "{⍉⍵}",                           # Matrix transpose
        "{⍺+.×⍵}",                        # Matrix multiplication
        "{+/⍺×⍵}",                        # Dot product
        "{⍵[⍋⍵]}",                        # Sort ascending
        "{⍵[⍒⍵]}",                        # Sort descending
        "{(⍵>{val})/⍵}",                  # Filter greater than
        "{(0={val}|⍵)/⍵}",                # Filter divisible by val
        "{⍺←0 ⋄ ⍺+⍵}",                    # Default argument
        "{⍺←1 ⋄ ⍺×⍵}",                    # Default scaling argument
        "{⍵=0:1 ⋄ 10×∇ ⍵-1}",             # Recursive power of 10
        "{(⍴⍵)⍴{val}}",                   # Reshape constant
        "{(⍳⍴⍵)⌽⍵}"                       # Progressive rotate
    ]

    TACIT_FORKS = [
        "(+/ ÷ ≢)",                         # Mean
        "(⌈/ - ⌊/)",                         # Range
        "(+/ × ≢)",
        "(⌽ ≡ ⊢)",                          # Palindrome check
        "(, ⍪ ⊢)",
        "(⍴ ⍴ ⍳)"
    ]

    @classmethod
    def generate_random_vector(cls, length: int = 5, max_val: int = 20, rng: Optional[random.Random] = None) -> str:
        r = rng or random
        nums = [str(r.randint(0, max_val)) for _ in range(length)]
        return " ".join(nums)

    @classmethod
    def generate_idiom(cls, rng: Optional[random.Random] = None) -> str:
        r = rng or random
        choice = r.randint(1, 6)

        if choice == 1:
            # Reduction on vector
            op = r.choice(cls.REDUCTIONS)
            vec = cls.generate_random_vector(r.randint(3, 8), rng=r)
            return f"{op} {vec}\n"

        elif choice == 2:
            # Scan on vector
            op = r.choice(cls.SCANS)
            vec = cls.generate_random_vector(r.randint(3, 8), rng=r)
            return f"{op} {vec}\n"

        elif choice == 3:
            # Dfn applied to vector or scalar
            pattern = r.choice(cls.DFN_PATTERNS)
            val = r.randint(2, 10)
            dfn = pattern.replace("{val}", str(val))
            arg = cls.generate_random_vector(r.randint(3, 6), rng=r) if ("≢" in dfn or "⌽" in dfn) else str(r.randint(1, 10))
            return f"{dfn} {arg}\n"

        elif choice == 4:
            # Tacit fork applied to vector
            fork = r.choice(cls.TACIT_FORKS)
            vec = cls.generate_random_vector(r.randint(3, 8), rng=r)
            return f"{fork} {vec}\n"

        elif choice == 5:
            # Matrix reshape & operations
            rows, cols = r.randint(2, 5), r.randint(2, 5)
            total = rows * cols
            return f"M ← {rows} {cols}⍴⍳{total}\n+⌿ M\n"

        else:
            # Variable assignment and calculation
            var = r.choice(["A", "B", "X", "Y", "V", "data", "nums"])
            vec = cls.generate_random_vector(r.randint(4, 10), rng=r)
            return f"{var} ← {vec}\nmean ← {{+/⍵÷≢⍵}} {var}\nsorted ← {var}[⍋{var}]\n"

    @classmethod
    def generate_synthetic_corpus(cls, count: int = 5000, seed: Optional[int] = None) -> str:
        rng = random.Random(seed) if seed is not None else random.Random()
        lines = [cls.generate_idiom(rng=rng) for _ in range(count)]
        return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Synthetic APL Corpus Generator")
    parser.add_argument("--count", type=int, default=1000, help="Number of synthetic expressions to generate")
    parser.add_argument("--output", type=str, default="data/apl_corpus.txt", help="Output corpus file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[+] Generating {args.count:,} synthetic APL idioms (seed={args.seed})...")
    corpus = APLSyntheticGenerator.generate_synthetic_corpus(count=args.count, seed=args.seed)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"[OK] Generated {len(corpus):,} characters in: {out_path}")


if __name__ == "__main__":
    main()
