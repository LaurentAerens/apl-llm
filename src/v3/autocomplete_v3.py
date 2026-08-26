import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tokenizer import APLTokenizer
from v3.model_v3 import APL_SLM_v3, APL_SLMConfig_v3


def load_model_v3(checkpoint_path: str = "checkpoints/apl_slm_best.pt", device_str: str = None) -> Tuple[APL_SLM_v3, APLTokenizer, torch.device]:
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

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, APL_SLMConfig_v3):
        saved_args = checkpoint.get("args", {})
        config = APL_SLMConfig_v3(
            vocab_size=checkpoint.get("vocab_size", tokenizer.vocab_size),
            max_seq_len=saved_args.get("seq_len", 512),
            n_layer=saved_args.get("n_layer", 4),
            n_head=saved_args.get("n_head", 4),
            n_embd=saved_args.get("n_embd", 64),
            max_depth=32,
            dropout=0.0,
            version=3,
        )

    model = APL_SLM_v3(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return model, tokenizer, device


def autocomplete_v3(
    model: APL_SLM_v3,
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
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id]

    depth_seq = tokenizer.compute_depth_sequences(prompt_ids)
    cur_depth = depth_seq[-1] if depth_seq else 0

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    depths = torch.tensor([depth_seq], dtype=torch.long, device=device)

    if verbose:
        print(f"\n[+] Prompt: {repr(prompt)}")
        print("[+] Autocomplete v3: ", end="", flush=True)

    with torch.no_grad():
        if use_kv_cache:
            logits, depth_logits, kv_caches = model(idx, depth_ids=depths, use_cache=True)
            generated_ids = []

            for _ in range(max_new_tokens):
                next_token_logits = logits[:, -1, :]

                if guard_syntax and cur_depth == 0:
                    for ch in (")", "]", "}"):
                        if ch in tokenizer.char_to_id:
                            next_token_logits[:, tokenizer.char_to_id[ch]] = -float("inf")

                if temperature > 0:
                    scaled_logits = next_token_logits / max(temperature, 1e-5)
                    if top_k > 0:
                        v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                        scaled_logits[scaled_logits < v[:, [-1]]] = -float("inf")
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).item()

                if next_token in (tokenizer.eos_id, tokenizer.pad_id):
                    break

                generated_ids.append(next_token)
                info = tokenizer.get_token_info(next_token)
                cur_depth = max(0, cur_depth + info.paren_delta + info.bracket_delta + info.dfn_delta)

                if verbose:
                    token_str = tokenizer.decode([next_token], skip_special_tokens=True)
                    print(token_str, end="", flush=True)

                if cur_depth == 0 and info.text in ("\n", "\r", "⋄"):
                    break

                next_tok_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
                next_depth_tensor = torch.tensor([[cur_depth]], dtype=torch.long, device=device)

                logits, depth_logits, kv_caches = model(
                    next_tok_tensor,
                    depth_ids=next_depth_tensor,
                    kv_caches=kv_caches,
                    use_cache=True,
                )

            if verbose:
                print()
            return prompt + tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            curr_idx = idx
            curr_depths = depths
            generated_ids = []

            for _ in range(max_new_tokens):
                logits, depth_logits, _ = model(curr_idx, depth_ids=curr_depths)
                next_token_logits = logits[:, -1, :]

                if guard_syntax and cur_depth == 0:
                    for ch in (")", "]", "}"):
                        if ch in tokenizer.char_to_id:
                            next_token_logits[:, tokenizer.char_to_id[ch]] = -float("inf")

                if temperature > 0:
                    scaled_logits = next_token_logits / max(temperature, 1e-5)
                    if top_k > 0:
                        v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                        scaled_logits[scaled_logits < v[:, [-1]]] = -float("inf")
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).item()

                if next_token in (tokenizer.eos_id, tokenizer.pad_id):
                    break

                generated_ids.append(next_token)
                info = tokenizer.get_token_info(next_token)
                cur_depth = max(0, cur_depth + info.paren_delta + info.bracket_delta + info.dfn_delta)

                if cur_depth == 0 and info.text in ("\n", "\r", "⋄"):
                    break

                curr_idx = torch.cat([curr_idx, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
                curr_depths = torch.cat([curr_depths, torch.tensor([[cur_depth]], dtype=torch.long, device=device)], dim=1)

            return prompt + tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="APL SLM v3 Autocomplete")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/Small-v3/apl_slm_best.pt")
    parser.add_argument("--prompt", type=str, default="{+/⍵")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    model, tokenizer, device = load_model_v3(args.checkpoint)
    completed = autocomplete_v3(
        model, tokenizer, device, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"\n[Result]:\n{completed}")


if __name__ == "__main__":
    main()

