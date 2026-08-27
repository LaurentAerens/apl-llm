import sys
import os
import json
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from autocomplete import APLCompleter


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="APL Autocomplete Daemon Server")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/apl_slm_best.pt", help="Model checkpoint path")
    args, _ = parser.parse_known_args()

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        ext_dir = os.path.dirname(__file__)
        candidate = os.path.join(ext_dir, checkpoint_path)
        if os.path.exists(candidate):
            checkpoint_path = candidate
        else:
            search_dirs = [Path(ext_dir) / "checkpoints", Path.cwd() / "checkpoints", Path.cwd()]
            found_candidates = []
            for s_dir in search_dirs:
                if s_dir.exists():
                    found_candidates.extend(list(s_dir.glob("**/*.pt")))
            if found_candidates:
                best_c = [c for c in found_candidates if "best" in c.name]
                checkpoint_path = str(best_c[0] if best_c else found_candidates[0])
            else:
                sys.stderr.write(f"Checkpoint not found at: {checkpoint_path}\n")
                sys.stderr.flush()

    try:
        completer = APLCompleter(checkpoint_path)
        sys.stderr.write(f"APL Completer loaded successfully from {checkpoint_path}\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"Failed to load model: {e}\n")
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
            top_k = int(req.get("top_k", 3))

            completion = completer.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_k=top_k,
            )

            res = {
                "id": req_id,
                "completion": completion,
            }
            sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Error processing request: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
