"""
APL SLM Autocomplete CLI & Public Inference Engine.
Provides model loading, interactive terminal playground, and batch evaluation interfaces.
"""

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
from generator import APLGenerator
from v1.autocomplete_v1 import autocomplete_v1
from v2.autocomplete_v2 import autocomplete_v2
from v3.autocomplete_v3 import autocomplete_v3


def load_model(
    checkpoint_path: str = "checkpoints/apl_slm_best.pt",
    device_str: Optional[str] = None,
) -> Tuple[torch.nn.Module, APLTokenizer, torch.device]:
    """
    Finds and loads the model checkpoint and associated tokenizer.
    """
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    path = Path(checkpoint_path)

    if not path.exists():
        fallback_candidates = [
            Path("checkpoints/baseline/apl_slm_best.pt"),
            Path("checkpoints/Small-v3/apl_slm_best.pt"),
            Path("checkpoints/Small-v3.0/apl_slm_best.pt"),
        ]
        found = None
        for candidate in fallback_candidates:
            if candidate.exists():
                found = candidate
                break

        if not found:
            candidates = list(Path("checkpoints").glob("**/*.pt"))
            if candidates:
                best_candidates = [c for c in candidates if "best" in c.name]
                found = best_candidates[0] if best_candidates else candidates[0]

        if not found:
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}' or within 'checkpoints/'.")
        path = found

    tokenizer_path = path.parent / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = APLTokenizer.load(tokenizer_path)
    else:
        tokenizer = APLTokenizer()

    model, config, version = AutoModel.from_checkpoint(path, device=device)
    return model, tokenizer, device


def autocomplete(
    model: torch.nn.Module,
    tokenizer: APLTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 5,
    top_p: float = 0.9,
    use_kv_cache: bool = True,
    guard_syntax: bool = True,
    stop_on_balanced_newline: bool = True,
    verbose: bool = False,
) -> str:
    """Dispatches autocompletion via the unified generation engine."""
    return APLGenerator.generate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        use_kv_cache=use_kv_cache,
        guard_syntax=guard_syntax,
        stop_on_balanced_newline=stop_on_balanced_newline,
        verbose=verbose,
    )


class APLCompleter:
    """Backward-compatible wrapper class for autocomplete clients & extensions."""

    def __init__(self, checkpoint_path: str = "checkpoints/apl_slm_best.pt", device: Optional[str] = None):
        self.model, self.tokenizer, self.device = load_model(checkpoint_path, device_str=device)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 5,
        top_p: float = 0.9,
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
            top_p=top_p,
            stop_on_balanced_newline=stop_on_balanced_newline,
            verbose=False,
        )
        if full.startswith(prompt):
            return full[len(prompt) :]
        return full


def interactive_cli(model: torch.nn.Module, tokenizer: APLTokenizer, device: torch.device):
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="APL SLM Autocomplete CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/apl_slm_best.pt")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_kv", action="store_true", help="Disable single-token KV caching")
    args = parser.parse_args()

    try:
        model, tokenizer, device = load_model(args.checkpoint)
    except FileNotFoundError as e:
        print(f"[!] {e}")
        return

    if args.prompt is not None:
        completed = autocomplete(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            use_kv_cache=not args.no_kv,
            verbose=True,
        )
        print(f"\n[Result]:\n{completed}")
    else:
        interactive_cli(model, tokenizer, device)


if __name__ == "__main__":
    main()
