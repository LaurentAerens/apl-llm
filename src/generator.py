"""
Unified Autoregressive Generation & IntelliSense Completion Engine for APL SLMs.
Supports KV-cached decoding, top-k/top-p sampling, temperature control, and structural syntax guarding.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import APLTokenizer


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 5,
    top_p: float = 0.9,
) -> int:
    """
    Applies temperature scaling, top-k truncation, and top-p (nucleus) filtering,
    then samples or takes argmax.
    """
    if temperature <= 1e-5:
        return torch.argmax(logits, dim=-1).item()

    scaled_logits = logits / temperature

    # Top-K filtering
    if top_k > 0:
        val, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
        scaled_logits[scaled_logits < val[:, [-1]]] = -float("inf")

    # Top-P (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to keep the first token above threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scaled_logits[indices_to_remove] = -float("inf")

    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


class APLGenerator:
    """
    Unified text generation engine for all APL model architecture versions (v1, v2, v3).
    """

    @staticmethod
    def generate(
        model: nn.Module,
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
        """
        Autoregressively generates APL code completion from a given prompt.

        Args:
            model: PyTorch model (v1, v2, or v3)
            tokenizer: APLTokenizer instance
            device: Target torch device
            prompt: Input code prefix
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            top_k: Top-K candidate filtering
            top_p: Nucleus top-P filtering
            use_kv_cache: Whether to use fast single-token KV caching
            guard_syntax: Prevent invalid closing brackets when at base depth (0)
            stop_on_balanced_newline: Stop generation on newline or diamond when code is balanced
            verbose: Print tokens in real-time

        Returns:
            Full completed code string (prompt + newly generated tokens)
        """
        model.eval()
        version = getattr(model, "version", 3)
        supports_depth = version in (2, 3)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            prompt_ids = [tokenizer.bos_id]

        depth_seq = tokenizer.compute_depth_sequences(prompt_ids)
        cur_depth = depth_seq[-1] if depth_seq else 0

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        depths = torch.tensor([depth_seq], dtype=torch.long, device=device) if supports_depth else None

        if verbose:
            print(f"\n[+] Prompt: {repr(prompt)}")
            print(f"[+] Autocomplete (v{version}): ", end="", flush=True)

        closing_tokens = [tokenizer.char_to_id[c] for c in (")", "]", "}") if c in tokenizer.char_to_id]
        stop_tokens = {tokenizer.eos_id, tokenizer.pad_id}

        generated_ids: List[int] = []

        with torch.no_grad():
            if use_kv_cache:
                # Initial forward pass to seed KV cache
                if supports_depth:
                    logits, depth_logits, kv_caches = model(idx, depth_ids=depths, use_cache=True)
                else:
                    logits, depth_logits, kv_caches = model(idx, use_cache=True)

                for _ in range(max_new_tokens):
                    next_token_logits = logits[:, -1, :].clone()

                    if guard_syntax and cur_depth == 0:
                        for tid in closing_tokens:
                            next_token_logits[:, tid] = -float("inf")

                    next_token = sample_next_token(
                        next_token_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )

                    if next_token in stop_tokens:
                        break

                    generated_ids.append(next_token)
                    info = tokenizer.get_token_info(next_token)
                    cur_depth = max(0, cur_depth + info.composite_delta)

                    if verbose:
                        token_str = tokenizer.decode([next_token], skip_special_tokens=True)
                        print(token_str, end="", flush=True)

                    if stop_on_balanced_newline and cur_depth == 0 and info.text in ("\n", "\r", "⋄"):
                        break

                    next_tok_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
                    next_depth_tensor = (
                        torch.tensor([[cur_depth]], dtype=torch.long, device=device) if supports_depth else None
                    )

                    if supports_depth:
                        logits, depth_logits, kv_caches = model(
                            next_tok_tensor,
                            depth_ids=next_depth_tensor,
                            kv_caches=kv_caches,
                            use_cache=True,
                        )
                    else:
                        logits, depth_logits, kv_caches = model(
                            next_tok_tensor,
                            kv_caches=kv_caches,
                            use_cache=True,
                        )

            else:
                # Non-KV cache fallback decoding loop
                curr_idx = idx
                curr_depths = depths

                for _ in range(max_new_tokens):
                    if supports_depth:
                        logits, depth_logits, _ = model(curr_idx, depth_ids=curr_depths)
                    else:
                        logits, depth_logits, _ = model(curr_idx)

                    next_token_logits = logits[:, -1, :].clone()

                    if guard_syntax and cur_depth == 0:
                        for tid in closing_tokens:
                            next_token_logits[:, tid] = -float("inf")

                    next_token = sample_next_token(
                        next_token_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )

                    if next_token in stop_tokens:
                        break

                    generated_ids.append(next_token)
                    info = tokenizer.get_token_info(next_token)
                    cur_depth = max(0, cur_depth + info.composite_delta)

                    if verbose:
                        token_str = tokenizer.decode([next_token], skip_special_tokens=True)
                        print(token_str, end="", flush=True)

                    if stop_on_balanced_newline and cur_depth == 0 and info.text in ("\n", "\r", "⋄"):
                        break

                    next_tok_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
                    curr_idx = torch.cat([curr_idx, next_tok_tensor], dim=1)

                    if supports_depth:
                        next_depth_tensor = torch.tensor([[cur_depth]], dtype=torch.long, device=device)
                        curr_depths = torch.cat([curr_depths, next_depth_tensor], dim=1)

        if verbose:
            print()

        return prompt + tokenizer.decode(generated_ids, skip_special_tokens=True)

