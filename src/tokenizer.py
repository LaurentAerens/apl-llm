import json
from pathlib import Path
from typing import List, Union, Optional
from dataclasses import dataclass

@dataclass
class TokenInfo:
    id: int
    text: str
    paren_delta: int
    bracket_delta: int
    dfn_delta: int

class APLTokenizer:
    """Specialized character/glyph tokenizer for APL (A Programming Language) with structural depth tracking."""

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
        "⍺", "⍵", "∇", "⋄", "←", "→", "⍝", "⍫", "⍬", "⍭", "⍮", "⍯", "⍰", "⍞", "⍡", "⍢"
    ]
    
    DELIMITERS = ["(", ")", "[", "]", "{", "}", ":", ";"]
    
    ASCII_CHARS = list(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_'\":$#%&~@^!?-+=/\\*<>,.[]{}();"
    )
    
    WHITESPACE_TOKENS = ["\n", " ", "\t", "\r"]

    def __init__(self):
        # Build unique ordered vocabulary
        seen = set()
        self.vocab = []
        for token in (self.SPECIAL_TOKENS + self.APL_GLYPHS + self.DELIMITERS + self.ASCII_CHARS + self.WHITESPACE_TOKENS):
            if token not in seen:
                seen.add(token)
                self.vocab.append(token)

        self.char_to_id = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.id_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}

        self.pad_id = self.char_to_id["<pad>"]
        self.bos_id = self.char_to_id["<bos>"]
        self.eos_id = self.char_to_id["<eos>"]
        self.unk_id = self.char_to_id["<unk>"]
        
        self._build_token_deltas()

    def _build_token_deltas(self):
        special_set = set(self.SPECIAL_TOKENS)
        self.tokens_info = []
        self.paren_deltas = []
        self.bracket_deltas = []
        self.dfn_deltas = []
        
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
                dfn_delta=d_delta
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

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes string text into APL token IDs."""
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_id)

        for char in text:
            if char in self.char_to_id:
                tokens.append(self.char_to_id[char])
            else:
                tokens.append(self.unk_id)

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

    def get_token_delta(self, token_id: int) -> int:
        info = self.get_token_info(token_id)
        return info.paren_delta + info.bracket_delta + info.dfn_delta

    def compute_depth_sequences(self, token_ids: List[int]) -> List[int]:
        """Calculates running composite structural depth (parens + brackets + dfns)."""
        depths = []
        current_depth = 0
        for tid in token_ids:
            depths.append(current_depth)
            info = self.get_token_info(tid)
            current_depth = max(0, current_depth + info.paren_delta + info.bracket_delta + info.dfn_delta)
        return depths

    def save(self, filepath: Union[str, Path]):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        tokens_info_list = [
            {
                "id": info.id,
                "text": info.text,
                "paren_delta": info.paren_delta,
                "bracket_delta": info.bracket_delta,
                "dfn_delta": info.dfn_delta,
            }
            for info in self.tokens_info
        ]
        data = {
            "vocab": self.vocab,
            "paren_deltas": self.paren_deltas,
            "bracket_deltas": self.bracket_deltas,
            "dfn_deltas": self.dfn_deltas,
            "tokens_info": tokens_info_list,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]):
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.vocab = data.get("vocab", tokenizer.vocab)
        tokenizer.char_to_id = {ch: idx for idx, ch in enumerate(tokenizer.vocab)}
        tokenizer.id_to_char = {idx: ch for idx, ch in enumerate(tokenizer.vocab)}
        tokenizer._build_token_deltas()
        return tokenizer
