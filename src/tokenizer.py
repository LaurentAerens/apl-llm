"""
Specialized Unicode Glyph Tokenizer for APL (ISO / Dyalog) with Structural Depth Tracking.
Handles character-level glyph mapping, delimiter balancing, and running nested syntax depth.
"""

import json
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class TokenInfo:
    """Metadata describing a single token's structural delta for brackets, parens, and dfns."""
    id: int
    text: str
    paren_delta: int
    bracket_delta: int
    dfn_delta: int

    @property
    def composite_delta(self) -> int:
        return self.paren_delta + self.bracket_delta + self.dfn_delta


class APLTokenizer:
    """
    Specialized character/glyph tokenizer for APL (A Programming Language).
    Tracks running composite structural depth across parentheses `()`, brackets `[]`, and dfns `{}`.
    """

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # Complete Unicode APL glyph primitives (Dyalog / ISO standard)
    APL_GLYPHS = [
        # Arithmetic, Mathematical & Functions
        "+", "-", "×", "÷", "*", "⍟", "|", "⌈", "⌊", "○", "!", "?", "¯",
        # Logical & Comparison
        "=", "≠", "≤", "≥", "<", ">", "∧", "∨", "⍲", "⍱", "~", "≡", "≢",
        # Structural, Selection & Set operations
        "⍳", "⍴", ",", "⍪", "⌽", "⊖", "⍉", "↑", "↓", "⊂", "⊃", "⊆", "⊇",
        "⌷", "⍋", "⍒", "∊", "⍷", "⍸", "∪", "∩", "⌸", "⌹", "⊥", "⊤", "⍕", "⍎",
        # Operators, Adverbs & Modifiers
        "/", "\\", "⌿", "⍀", "¨", "⍨", "⍣", "⍤", "⍥", "⍠", "∘", ".", "@", "⌶", "⍶", "⍹",
        # Dfns, Variables & Control
        "⍺", "⍵", "∇", "⋄", "←", "→", "⍝", "⍫", "⍬", "⍭", "⍮", "⍯", "⍰", "⍞", "⍡", "⍢",
    ]

    DELIMITERS = ["(", ")", "[", "]", "{", "}", ":", ";"]

    ASCII_CHARS = list(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_'\":$#%&~@^!?-+=/\\*<>,.[]{}();"
    )

    WHITESPACE_TOKENS = ["\n", " ", "\t", "\r"]

    def __init__(self, vocab: Optional[List[str]] = None):
        if vocab is not None:
            self.vocab = list(vocab)
        else:
            seen = set()
            self.vocab = []
            for token in (self.SPECIAL_TOKENS + self.APL_GLYPHS + self.DELIMITERS + self.ASCII_CHARS + self.WHITESPACE_TOKENS):
                if token not in seen:
                    seen.add(token)
                    self.vocab.append(token)

        self.char_to_id: Dict[str, int] = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.id_to_char: Dict[int, str] = {idx: ch for idx, ch in enumerate(self.vocab)}

        self.pad_id = self.char_to_id.get("<pad>", 0)
        self.bos_id = self.char_to_id.get("<bos>", 1)
        self.eos_id = self.char_to_id.get("<eos>", 2)
        self.unk_id = self.char_to_id.get("<unk>", 3)

        self._build_token_deltas()

    def _build_token_deltas(self):
        special_set = set(self.SPECIAL_TOKENS)
        self.tokens_info: List[TokenInfo] = []
        self.paren_deltas: List[int] = []
        self.bracket_deltas: List[int] = []
        self.dfn_deltas: List[int] = []

        for idx, token_str in enumerate(self.vocab):
            if token_str in special_set:
                p_delta = b_delta = d_delta = 0
            else:
                p_delta = token_str.count("(") - token_str.count(")")
                b_delta = token_str.count("[") - token_str.count("]")
                d_delta = token_str.count("{") - token_str.count("}")

            info = TokenInfo(
                id=idx,
                text=token_str,
                paren_delta=p_delta,
                bracket_delta=b_delta,
                dfn_delta=d_delta,
            )
            self.tokens_info.append(info)
            self.paren_deltas.append(p_delta)
            self.bracket_deltas.append(b_delta)
            self.dfn_deltas.append(d_delta)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_token_info(self, token_id: int) -> TokenInfo:
        if 0 <= token_id < len(self.tokens_info):
            return self.tokens_info[token_id]
        return TokenInfo(id=token_id, text="", paren_delta=0, bracket_delta=0, dfn_delta=0)

    def get_token_delta(self, token_id: int) -> int:
        info = self.get_token_info(token_id)
        return info.composite_delta

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes string text into APL token IDs."""
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_id)

        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_id))

        if add_special_tokens:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs into string text."""
        chars = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id} if skip_special_tokens else set()

        for tid in token_ids:
            if tid in special_ids:
                continue
            if 0 <= tid < len(self.vocab):
                chars.append(self.id_to_char[tid])
        return "".join(chars)

    def compute_depth_sequences(self, token_ids: List[int], max_depth: int = 32) -> List[int]:
        """
        Calculates running composite structural depth (parens + brackets + dfns)
        at each token position.
        """
        depths = []
        current_depth = 0
        for tid in token_ids:
            clamped = min(max(0, current_depth), max_depth - 1)
            depths.append(clamped)
            info = self.get_token_info(tid)
            current_depth = max(0, current_depth + info.composite_delta)
        return depths

    def is_balanced(self, code: str) -> bool:
        """
        Validates whether parentheses `()`, brackets `[]`, and dfns `{}` in the code
        are fully balanced and properly nested without underflow.
        """
        p_depth = 0
        b_depth = 0
        d_depth = 0
        stack = []

        matching = {")": "(", "]": "[", "}": "{"}

        for ch in code:
            if ch in ("(", "[", "{"):
                stack.append(ch)
                if ch == "(":
                    p_depth += 1
                elif ch == "[":
                    b_depth += 1
                elif ch == "{":
                    d_depth += 1
            elif ch in matching:
                expected_open = matching[ch]
                if not stack or stack[-1] != expected_open:
                    return False
                stack.pop()
                if ch == ")":
                    p_depth -= 1
                elif ch == "]":
                    b_depth -= 1
                elif ch == "}":
                    d_depth -= 1

        return len(stack) == 0 and p_depth == 0 and b_depth == 0 and d_depth == 0

    def get_unclosed_delimiters(self, code: str) -> List[str]:
        """Returns list of unclosed open delimiters in FIFO order."""
        stack = []
        matching = {")": "(", "]": "[", "}": "{"}
        for ch in code:
            if ch in ("(", "[", "{"):
                stack.append(ch)
            elif ch in matching:
                if stack and stack[-1] == matching[ch]:
                    stack.pop()
        return stack

    def save(self, filepath: Union[str, Path]):
        """Persists vocabulary and delimiter delta tables to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.vocab,
            "paren_deltas": self.paren_deltas,
            "bracket_deltas": self.bracket_deltas,
            "dfn_deltas": self.dfn_deltas,
            "tokens_info": [asdict(info) for info in self.tokens_info],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "APLTokenizer":
        """Loads tokenizer vocabulary from JSON file."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = data.get("vocab", [])
        tokenizer = cls(vocab=vocab if vocab else None)
        return tokenizer


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tok = APLTokenizer()
    print(f"APLTokenizer initialized with {tok.vocab_size} tokens.")
    test_expr = "{+/⍵÷≢⍵} 1 2 3 4 5"
    encoded = tok.encode(test_expr)
    decoded = tok.decode(encoded)
    depths = tok.compute_depth_sequences(encoded)
    balanced = tok.is_balanced(test_expr)

    print(f"Original: {test_expr}")
    print(f"Encoded:  {encoded[:10]}... ({len(encoded)} tokens)")
    print(f"Decoded:  {decoded}")
    print(f"Depths:   {depths[:10]}...")
    print(f"Balanced: {balanced}")
    assert test_expr == decoded, "Encoding/decoding round-trip failed!"
    assert tok.is_balanced("{(+/⍵)÷≢⍵}") is True
    assert tok.is_balanced("{(+/⍵)÷≢⍵") is False
    assert tok.is_balanced(") (") is False
    print("All tokenizer sanity assertions passed!")
