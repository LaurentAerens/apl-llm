"""
AutoModel factory and unified model dispatcher for APL SLM architectures.
Provides automatic checkpoint inspection, architecture version detection, and weight loading.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any

import torch
import torch.nn as nn

# Ensure root src is in sys.path
src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import (
    APLModelConfig,
    APL_SLMConfig_v1,
    APL_SLMConfig_v2,
    APL_SLMConfig_v3,
    MODEL_PRESETS,
    resolve_preset_dimensions,
    create_config_for_version,
)
from v1.model_v1 import APL_SLM_v1
from v2.model_v2 import APL_SLM_v2
from v3.model_v3 import (
    APL_SLM_v3,
    RMSNorm,
    SwiGLU,
    RotaryEmbedding,
    TransformerBlock_v3,
)

# Default alias points to the flagship architecture (v3)
APL_SLM = APL_SLM_v3
APL_SLMConfig = APL_SLMConfig_v3


class AutoModel:
    """
    Factory loader that inspects checkpoint metadata and architecture weights,
    automatically instantiating the correct model class (v1, v2, v3, etc.).
    """

    @staticmethod
    def detect_version(checkpoint: Dict[str, Any], state_dict: Dict[str, torch.Tensor]) -> int:
        """Determines model architecture version from metadata or weight keys."""
        if "model_version" in checkpoint:
            return int(checkpoint["model_version"])

        cfg = checkpoint.get("config")
        if cfg is not None:
            if hasattr(cfg, "version"):
                return int(cfg.version)
            if isinstance(cfg, dict) and "version" in cfg:
                return int(cfg["version"])

        # Inspect weight signatures
        keys = list(state_dict.keys())
        has_v3_weights = any("norm_f.weight" in k or "blocks." in k for k in keys)
        has_v2_weights = any("transformer.wde.weight" in k or "depth_head.weight" in k for k in keys)

        if has_v3_weights:
            return 3
        elif has_v2_weights:
            return 2
        return 1

    @staticmethod
    def from_checkpoint(
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        strict: bool = False,
    ) -> Tuple[nn.Module, APLModelConfig, int]:
        """
        Loads and initializes the appropriate version of APL_SLM from a saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (.pt)
            device: Target torch device (defaults to CUDA if available, else CPU)
            strict: Whether to enforce strict state dict matching

        Returns:
            Tuple of (model, config, version)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        config_raw = checkpoint.get("config")
        saved_args = checkpoint.get("args", {})

        version = AutoModel.detect_version(checkpoint, state_dict)

        # Extract base config hyperparameters
        vocab_size = checkpoint.get("vocab_size", getattr(config_raw, "vocab_size", saved_args.get("vocab_size", 256)))
        seq_len = getattr(config_raw, "max_seq_len", saved_args.get("seq_len", 512))
        n_layer = getattr(config_raw, "n_layer", saved_args.get("n_layer", 4))
        n_head = getattr(config_raw, "n_head", saved_args.get("n_head", 4))
        n_embd = getattr(config_raw, "n_embd", saved_args.get("n_embd", 64))
        dropout = getattr(config_raw, "dropout", saved_args.get("dropout", 0.0))
        max_depth = getattr(config_raw, "max_depth", saved_args.get("max_depth", 32))

        if version == 3:
            if isinstance(config_raw, APL_SLMConfig_v3):
                config = config_raw
            else:
                config = APL_SLMConfig_v3(
                    vocab_size=vocab_size,
                    max_seq_len=seq_len,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_embd=n_embd,
                    dropout=dropout,
                    max_depth=max_depth,
                    version=3,
                )
            model = APL_SLM_v3(config).to(device)

        elif version == 2:
            if isinstance(config_raw, APL_SLMConfig_v2):
                config = config_raw
            else:
                config = APL_SLMConfig_v2(
                    vocab_size=vocab_size,
                    max_seq_len=seq_len,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_embd=n_embd,
                    dropout=dropout,
                    max_depth=max_depth,
                    version=2,
                )
            model = APL_SLM_v2(config).to(device)

        else:
            if isinstance(config_raw, APL_SLMConfig_v1):
                config = config_raw
            else:
                config = APL_SLMConfig_v1(
                    vocab_size=vocab_size,
                    max_seq_len=seq_len,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_embd=n_embd,
                    dropout=dropout,
                    version=1,
                )
            model = APL_SLM_v1(config).to(device)

        model.load_state_dict(state_dict, strict=strict)
        model.eval()
        return model, config, version


__all__ = [
    "AutoModel",
    "APL_SLM",
    "APL_SLMConfig",
    "APLModelConfig",
    "APL_SLM_v1",
    "APL_SLMConfig_v1",
    "APL_SLM_v2",
    "APL_SLMConfig_v2",
    "APL_SLM_v3",
    "APL_SLMConfig_v3",
    "RMSNorm",
    "SwiGLU",
    "RotaryEmbedding",
    "TransformerBlock_v3",
]
