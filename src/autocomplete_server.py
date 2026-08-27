"""
APL Autocomplete Daemon Server.
Fast JSON-RPC stdio daemon used by the VS Code companion extension for real-time ghost text IntelliSense.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Optional

# Ensure src directory is in sys.path
src_dir = os.path.dirname(__file__)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from autocomplete import APLCompleter


def resolve_checkpoint_path(checkpoint_arg: str) -> str:
    """Finds the best available model checkpoint path."""
    path = Path(checkpoint_arg)
    if path.is_file():
        return str(path)

    # Search candidates
    search_dirs = [
        Path(src_dir).parent / "checkpoints",
        Path.cwd() / "checkpoints",
        Path(src_dir) / "checkpoints",
        Path.cwd(),
    ]
    candidates = []
    for s_dir in search_dirs:
        if s_dir.exists():
            candidates.extend(list(s_dir.glob("**/*.pt")))

    if candidates:
        best_c = [c for c in candidates if "best" in c.name]
        return str(best_c[0] if best_c else candidates[0])

    return checkpoint_arg


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="APL Autocomplete Daemon Server")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/apl_slm_best.pt", help="Model checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Inference device (cpu or cuda)")
    args, _ = parser.parse_known_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    try:
        completer = APLCompleter(checkpoint_path, device=args.device)
        ver = getattr(completer.model, "version", "?")
        sys.stderr.write(f"[APL-SLM] Completer loaded successfully (v{ver}) from: {checkpoint_path}\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[APL-SLM] Failed to initialize model: {e}\n")
        sys.stderr.flush()
        sys.exit(1)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            req_id = req.get("id", req.get("req_id", 0))
            prompt = req.get("prompt", "")
            max_tokens = int(req.get("max_tokens", 64))
            temp = float(req.get("temp", req.get("temperature", 0.2)))
            top_k = int(req.get("top_k", 5))
            top_p = float(req.get("top_p", 0.9))

            completion = completer.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
            )

            res = {
                "id": req_id,
                "completion": completion,
            }
            sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"[APL-SLM] Error processing request: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
