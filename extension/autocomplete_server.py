"""
Fast JSON-RPC stdio daemon for VS Code extension (Extension Distribution).
"""

from pathlib import Path
import sys

parent_src = Path(__file__).resolve().parent.parent / "src"
if parent_src.exists() and str(parent_src) not in sys.path:
    sys.path.insert(0, str(parent_src))

from autocomplete_server import main

if __name__ == "__main__":
    main()
