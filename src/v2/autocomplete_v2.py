"""
Architecture Version 2 Autocomplete Wrapper.
Provides load_model_v2 and autocomplete_v2 delegating to the unified APLGenerator.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional
import torch

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tokenizer import APLTokenizer
from generator import APLGenerator
from v2.model_v2 import APL_SLM_v2, APL_SLMConfig_v2


def load_model_v2(checkpoint_path: str = "checkpoints/apl_slm_best.pt", device_str: Optional[str] = None):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    path = Path(checkpoint_path)
    tokenizer_path = path.parent / "tokenizer.json"
    tokenizer = APLTokenizer.load(tokenizer_path) if tokenizer_path.exists() else APLTokenizer()

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, APL_SLMConfig_v2):
        saved_args = checkpoint.get("args", {})
        config = APL_SLMConfig_v2(
            vocab_size=checkpoint.get("vocab_size", tokenizer.vocab_size),
            max_seq_len=saved_args.get("seq_len", 512),
            n_layer=saved_args.get("n_layer", 4),
            n_head=saved_args.get("n_head", 4),
            n_embd=saved_args.get("n_embd", 64),
            max_depth=32,
            dropout=0.0,
            version=2,
        )
    model = APL_SLM_v2(config).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()
    return model, tokenizer, device


def autocomplete_v2(
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
