"""
Specialized Unicode Glyph Tokenizer for APL (Extension Distribution).
"""

from pathlib import Path
import sys

# Try importing from src if available in parent repo
parent_src = Path(__file__).resolve().parent.parent / "src"
if parent_src.exists() and str(parent_src) not in sys.path:
    sys.path.insert(0, str(parent_src))

from tokenizer import APLTokenizer, TokenInfo

__all__ = ["APLTokenizer", "TokenInfo"]
