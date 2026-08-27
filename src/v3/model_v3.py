"""
Architecture Version 3: Modern High-Performance Transformer for APL.
Built with Pre-RMSNorm, SwiGLU, Rotary Position Embeddings (RoPE), QK-Norm, and single-token KV caching.
"""

import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import APL_SLMConfig_v3, APLModelConfig

# Backward compatibility alias
APL_SLMConfig = APL_SLMConfig_v3


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Gated Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    d = q.shape[-1]
    q1, q2 = q[..., : d // 2], q[..., d // 2 :]
    k1, k2 = k[..., : d // 2], k[..., d // 2 :]

    rotated_q = torch.cat([-q2, q1], dim=-1)
    rotated_k = torch.cat([-k2, k1], dim=-1)

    q_out = (q * cos) + (rotated_q * sin)
    k_out = (k * cos) + (rotated_k * sin)
    return q_out, k_out


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:, :, offset : offset + seq_len, :],
            self.sin_cached[:, :, offset : offset + seq_len, :],
        )


class CausalSelfAttention_v3(nn.Module):
    def __init__(self, config: APL_SLMConfig_v3):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if config.use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if config.use_qk_norm else nn.Identity()

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        past_len = kv_cache[0].shape[2] if kv_cache is not None else 0
        cos, sin = self.rope(T, offset=past_len)
        cos, sin = cos.to(q.device), sin.to(q.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        att = F.scaled_dot_product_attention(
            q, k, v, is_causal=(T > 1 and kv_cache is None)
        )
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(att), new_kv_cache


class TransformerBlock_v3(nn.Module):
    def __init__(self, config: APL_SLMConfig_v3):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention_v3(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        norm_x = self.norm1(x)
        attn_out, new_cache = self.attn(norm_x, kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class APL_SLM_v3(nn.Module):
    """
    Architecture Version 3: Modern High-Performance Transformer for APL.
    Built with Pre-RMSNorm, SwiGLU, RoPE Rotary Embeddings, QK-Norm, and single-token KV caching.
    """

    def __init__(self, config: APL_SLMConfig_v3):
        super().__init__()
        self.config = config
        self.version = 3

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.depth_emb = nn.Embedding(config.max_depth, config.n_embd) if config.use_depth_node else None

        self.blocks = nn.ModuleList([
            TransformerBlock_v3(config) for _ in range(config.n_layer)
        ])

        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.depth_head = nn.Linear(config.n_embd, config.max_depth, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        token_ids: torch.Tensor,
        depth_ids: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = token_ids.size()
        x = self.tok_emb(token_ids)

        if self.config.use_depth_node and depth_ids is not None and self.depth_emb is not None:
            depth_clamped = torch.clamp(depth_ids, 0, self.config.max_depth - 1)
            x = x + self.depth_emb(depth_clamped)

        new_caches = [] if use_cache else None
        for idx, block in enumerate(self.blocks):
            layer_cache = kv_caches[idx] if kv_caches is not None else None
            x, next_cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(next_cache)

        x = self.norm_f(x)
        logits = self.lm_head(x)
        depth_logits = self.depth_head(x)

        return logits, depth_logits, new_caches
