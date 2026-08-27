"""
Autocomplete generation interface for APL SLM (Extension Distribution).
"""

from pathlib import Path
import sys

parent_src = Path(__file__).resolve().parent.parent / "src"
if parent_src.exists() and str(parent_src) not in sys.path:
    sys.path.insert(0, str(parent_src))

from autocomplete import APLCompleter, autocomplete, load_model

__all__ = ["APLCompleter", "autocomplete", "load_model"]
