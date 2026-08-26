import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*flash attention.*")
warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*")


@dataclass
class APL_SLMConfig_v1:
    vocab_size: int = 256
    max_seq_len: int = 1024
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    version: int = 1


class CausalSelfAttention_v1(nn.Module):
    def __init__(self, config: APL_SLMConfig_v1):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present = (k, v) if use_cache else None

        if hasattr(F, "scaled_dot_product_attention"):
            is_causal = (layer_past is None) and (T > 1)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=is_causal
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if layer_past is None:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, present


class MLP_v1(nn.Module):
    def __init__(self, config: APL_SLMConfig_v1):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block_v1(nn.Module):
    def __init__(self, config: APL_SLMConfig_v1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention_v1(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP_v1(config)

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present


class APL_SLM_v1(nn.Module):
    """
    Architecture Version 1: Baseline Causal Transformer for APL.
    Pure autoregressive sequence modeling with Post-LN, GELU, and learned positional embeddings.
    """

    def __init__(self, config: APL_SLMConfig_v1):
        super().__init__()
        self.config = config
        self.version = 1

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.max_seq_len, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block_v1(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        token_ids: torch.Tensor,
        depth_ids: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        device = token_ids.device
        B, T = token_ids.size()

        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0] is not None:
            past_len = kv_caches[0][0].size(-2)
            pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=device).unsqueeze(0)
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(token_ids)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            layer_past = kv_caches[i] if kv_caches is not None else None
            x, present = block(x, layer_past=layer_past, use_cache=use_cache)
            if use_cache:
                new_caches.append(present)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits, None, new_caches

