import random
from typing import List

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
        "{{+/⍵÷≢⍵}}",                       # Mean
        "{{(+/⍵)÷≢⍵}}",                     # Mean with parens
        "{{(+/⍵*2)÷≢⍵}}",                   # Root mean square base
        "{{(+/((⍵-((+/⍵)÷≢⍵))*2))÷≢⍵}}",    # Variance
        "{{×/1+⍳⍵}}",                       # Factorial
        "{{1≥⍵:1 ⋄ ⍵×∇ ⍵-1}}",              # Recursive Factorial
        "{{⍵≤1:⍵ ⋄ (∇ ⍵-1)+(∇ ⍵-2)}}",      # Recursive Fibonacci
        "{{⌈/⍵}}",                          # Maximum
        "{{⌊/⍵}}",                          # Minimum
        "{{(⌈/⍵)-(⌊/⍵)}}",                  # Range (max - min)
        "{{|⍵}}",                           # Absolute value
        "{{(⍵>0)-(⍵<0)}}",                  # Signum
        "{{0=2|⍵}}",                        # Is even mask
        "{{1=2|⍵}}",                        # Is odd mask
        
        # Array & Manipulation
        "{{∪⍵}}",                           # Unique elements
        "{{⍵≡⌽⍵}}",                         # Is Palindrome
        "{{⌽⍵}}",                           # Reverse vector
        "{{⊖⍵}}",                           # Reverse along first axis
        "{{⍉⍵}}",                           # Matrix transpose
        "{{⍺+.×⍵}}",                        # Matrix multiplication
        "{{+/⍺×⍵}}",                        # Dot product
        "{{⍵[⍋⍵]}}",                        # Sort ascending
        "{{⍵[⍒⍵]}}",                        # Sort descending
        "{{(⍵>{val})/⍵}}",                  # Filter greater than
        "{{(0={val}|⍵)/⍵}}",                # Filter divisible by val
        "{{⍺←0 ⋄ ⍺+⍵}}",                    # Default argument
        "{{⍺←1 ⋄ ⍺×⍵}}",                    # Default scaling argument
        "{{⍵=0:1 ⋄ 10×∇ ⍵-1}}",             # Recursive power of 10
        "{{(⍴⍵)⍴{val}}}",                   # Reshape constant
        "{{(⍳⍴⍵)⌽⍵}}"                       # Progressive rotate
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
    def generate_random_vector(cls, length: int = 5, max_val: int = 20) -> str:
        nums = [str(random.randint(0, max_val)) for _ in range(length)]
        return " ".join(nums)

    @classmethod
    def generate_idiom(cls) -> str:
        choice = random.randint(1, 6)
        
        if choice == 1:
            # Reduction on vector
            op = random.choice(cls.REDUCTIONS)
            vec = cls.generate_random_vector(random.randint(3, 8))
            return f"{op} {vec}\n"
            
        elif choice == 2:
            # Scan on vector
            op = random.choice(cls.SCANS)
            vec = cls.generate_random_vector(random.randint(3, 8))
            return f"{op} {vec}\n"
            
        elif choice == 3:
            # Dfn applied to vector or scalar
            pattern = random.choice(cls.DFN_PATTERNS)
            val = random.randint(2, 10)
            dfn = pattern.format(val=val)
            arg = cls.generate_random_vector(random.randint(3, 6)) if "≢" in dfn or "⌽" in dfn else str(random.randint(1, 10))
            return f"{dfn} {arg}\n"
            
        elif choice == 4:
            # Tacit fork applied to vector
            fork = random.choice(cls.TACIT_FORKS)
            vec = cls.generate_random_vector(random.randint(3, 8))
            return f"{fork} {vec}\n"
            
        elif choice == 5:
            # Matrix reshape & operations
            r, c = random.randint(2, 5), random.randint(2, 5)
            total = r * c
            return f"M ← {r} {c}⍴⍳{total}\n+⌿ M\n"
            
        else:
            # Variable assignment and calculation
            var = random.choice(["A", "B", "X", "Y", "V", "data", "nums"])
            vec = cls.generate_random_vector(random.randint(4, 10))
            return f"{var} ← {vec}\nmean ← {{+/⍵÷≢⍵}} {var}\nsorted ← {var}[⍋{var}]\n"

    @classmethod
    def generate_synthetic_corpus(cls, count: int = 5000) -> str:
        lines = []
        for _ in range(count):
            lines.append(cls.generate_idiom())
        return "\n".join(lines)
