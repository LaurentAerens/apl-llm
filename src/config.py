"""
Model configuration definitions and unified hyperparameter presets for APL SLM.
Provides typed, serializable configuration classes for all architecture versions (v1, v2, v3).
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Tuple
import json
from pathlib import Path


@dataclass
class APLModelConfig:
    """Base configuration class for all APL Small Language Model generations."""
    vocab_size: int = 256
    max_seq_len: int = 1024
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    version: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, filepath: Path | str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path | str):
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class APL_SLMConfig_v1(APLModelConfig):
    """Architecture Version 1: Standard autoregressive Causal Transformer (LayerNorm, GELU, learned absolute wpe)."""
    version: int = 1


@dataclass
class APL_SLMConfig_v2(APLModelConfig):
    """Architecture Version 2: Structural Depth Conditioned Transformer with auxiliary depth head."""
    max_depth: int = 32
    use_depth_node: bool = True
    version: int = 2


@dataclass
class APL_SLMConfig_v3(APLModelConfig):
    """Architecture Version 3: Modern Transformer (Pre-RMSNorm, SwiGLU, RoPE, QK-Norm, depth conditioning, KV cache)."""
    max_depth: int = 32
    use_depth_node: bool = True
    use_rope: bool = True
    use_swiglu: bool = True
    use_qk_norm: bool = True
    version: int = 3


# Standard hyperparameter presets across the suite
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 32,
        "base_name": "Tiny",
        "description": "Ultra-fast smoke testing & CI baseline (~95k params)",
    },
    "small": {
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 64,
        "base_name": "Small",
        "description": "Lightweight & fast CPU baseline (~380k params)",
    },
    "medium": {
        "n_layer": 6,
        "n_head": 8,
        "n_embd": 256,
        "base_name": "Medium",
        "description": "Balanced capacity & training speed (~2.4M params)",
    },
    "large": {
        "n_layer": 8,
        "n_head": 12,
        "n_embd": 384,
        "base_name": "Large",
        "description": "High reasoning & syntax precision (~7.1M params)",
    },
    "deep": {
        "n_layer": 12,
        "n_head": 8,
        "n_embd": 256,
        "base_name": "Deep",
        "description": "Deep hierarchical sequence modeling (~4.7M params)",
    },
    "wide": {
        "n_layer": 6,
        "n_head": 16,
        "n_embd": 512,
        "base_name": "Wide",
        "description": "Broad multi-head representation capacity (~9.5M params)",
    },
    "xlarge": {
        "n_layer": 12,
        "n_head": 16,
        "n_embd": 512,
        "base_name": "XLarge",
        "description": "Large capacity for multi-line script synthesis (~18.8M params)",
    },
    "huge": {
        "n_layer": 16,
        "n_head": 16,
        "n_embd": 768,
        "base_name": "Huge",
        "description": "High capacity transformer (~57.4M params)",
    },
    "giant": {
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 1024,
        "base_name": "Giant",
        "description": "Maximum capacity preset (~152M params)",
    },
}


def resolve_preset_dimensions(
    preset_name: Optional[str],
    n_layer: Optional[int] = None,
    n_head: Optional[int] = None,
    n_embd: Optional[int] = None,
    default_preset: str = "small",
) -> Tuple[int, int, int]:
    """
    Resolves (n_layer, n_head, n_embd) using the specified preset name as fallback
    when explicit dimension arguments are not provided.
    """
    p_key = (preset_name or default_preset).lower().strip("()")
    preset = MODEL_PRESETS.get(p_key, MODEL_PRESETS.get(default_preset, MODEL_PRESETS["small"]))

    final_layer = n_layer if n_layer is not None else preset["n_layer"]
    final_head = n_head if n_head is not None else preset["n_head"]
    final_embd = n_embd if n_embd is not None else preset["n_embd"]

    return final_layer, final_head, final_embd


def create_config_for_version(
    version: int,
    vocab_size: int = 256,
    max_seq_len: int = 512,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 64,
    dropout: float = 0.0,
    max_depth: int = 32,
) -> APLModelConfig:
    """Factory helper to create the typed config corresponding to architecture version."""
    if version == 1:
        return APL_SLMConfig_v1(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            version=1,
        )
    elif version == 2:
        return APL_SLMConfig_v2(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            max_depth=max_depth,
            version=2,
        )
    elif version == 3:
        return APL_SLMConfig_v3(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            max_depth=max_depth,
            version=3,
        )
    else:
        raise ValueError(f"Unsupported architecture version: {version}. Expected 1, 2, or 3.")

