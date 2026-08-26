import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional
import torch

src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tokenizer import APLTokenizer
from model import AutoModel, APL_SLM
from v1.autocomplete_v1 import autocomplete_v1
from v2.autocomplete_v2 import autocomplete_v2
from v3.autocomplete_v3 import autocomplete_v3


def load_model(checkpoint_path: str = "checkpoints/apl_slm_best.pt", device_str: str = None) -> Tuple[APL_SLM, APLTokenizer, torch.device]:
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    path = Path(checkpoint_path)

    if not path.exists():
        path = Path("checkpoints/baseline/apl_slm_best.pt")
        if not path.exists():
            candidates = list(Path("checkpoints").glob("**/*.pt"))
            if candidates:
                best_candidates = [c for c in candidates if "best" in c.name]
                path = best_candidates[0] if best_candidates else candidates[0]
            else:
                raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} or in checkpoints/")

    tokenizer_path = path.parent / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = APLTokenizer.load(tokenizer_path)
    else:
        tokenizer = APLTokenizer()

    model, config, version = AutoModel.from_checkpoint(path, device=device)
    return model, tokenizer, device


def autocomplete(
    model: APL_SLM,
    tokenizer: APLTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 5,
    use_kv_cache: bool = True,
    guard_syntax: bool = True,
    verbose: bool = True,
) -> str:
    version = getattr(model, "version", 3)
    if version == 3:
        return autocomplete_v3(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_kv_cache=use_kv_cache,
            guard_syntax=guard_syntax,
            verbose=verbose,
        )
    elif version == 2:
        return autocomplete_v2(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_kv_cache=use_kv_cache,
            guard_syntax=guard_syntax,
            verbose=verbose,
        )
    else:
        return autocomplete_v1(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_kv_cache=use_kv_cache,
            guard_syntax=guard_syntax,
            verbose=verbose,
        )


class APLCompleter:
    """Backward-compatible wrapper class for autocomplete clients & extensions."""

    def __init__(self, checkpoint_path: str = "checkpoints/baseline/apl_slm_best.pt", device: str = None):
        self.model, self.tokenizer, self.device = load_model(checkpoint_path, device_str=device)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 5,
        stop_on_balanced_newline: bool = True,
    ) -> str:
        full = autocomplete(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            verbose=False,
        )
        if full.startswith(prompt):
            return full[len(prompt):]
        return full


def interactive_cli(model, tokenizer, device):
    print("=" * 60)
    print("📐 APL SLM Interactive Autocomplete Playground")
    ver = getattr(model, "version", "?")
    print(f"Model version: v{ver} | Device: {device}")
    print("Type APL prefix and press Enter to autocomplete (type 'exit' to quit)")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nAPL > ")
            if not prompt or prompt.strip().lower() in ("exit", "quit"):
                break
            result = autocomplete(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_new_tokens=64,
                temperature=0.7,
                top_k=5,
                verbose=False,
            )
            print(f"Completed:\n{result}")
        except (KeyboardInterrupt, EOFError):
            break


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="APL SLM Autocomplete CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline/apl_slm_best.pt")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.checkpoint)

    if args.prompt is not None:
        completed = autocomplete(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            verbose=True,
        )
        print(f"\n[Result]:\n{completed}")
    else:
        interactive_cli(model, tokenizer, device)


if __name__ == "__main__":
    main()
