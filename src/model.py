import sys
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any

import torch
import torch.nn as nn

# Ensure root src is in sys.path
src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from v1.model_v1 import APL_SLM_v1, APL_SLMConfig_v1
from v2.model_v2 import APL_SLM_v2, APL_SLMConfig_v2
from v3.model_v3 import (
    APL_SLM_v3,
    APL_SLMConfig_v3,
    RMSNorm,
    SwiGLU,
    RotaryEmbedding,
    TransformerBlock_v3,
)

# Default alias
APL_SLM = APL_SLM_v3
APL_SLMConfig = APL_SLMConfig_v3


class AutoModel:
    """
    Factory loader that inspects checkpoint metadata and architecture weights,
    automatically instantiating the correct model class (v1, v2, v3, etc.).
    """

    @staticmethod
    def from_checkpoint(
        checkpoint_path: Union[str, Path],
        device: torch.device = None,
    ) -> Tuple[nn.Module, Any, int]:
        """
        Loads and initializes the appropriate version of APL_SLM from a saved checkpoint.

        Returns:
            model (nn.Module): Initialized model on target device with loaded weights in eval mode
            config: Model configuration (v1, v2, or v3)
            version (int): Architecture version (1, 2, 3)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint.get("config")
        state_dict = checkpoint.get("model_state_dict", {})

        # Determine architecture version reliably
        has_v3_weights = any("norm_f.weight" in k or "blocks." in k for k in state_dict.keys())
        has_v2_weights = any("transformer.wde.weight" in k or "depth_head.weight" in k for k in state_dict.keys())
        explicit_version = checkpoint.get("model_version")

        if explicit_version is not None:
            version = explicit_version
        elif has_v3_weights:
            version = 3
        elif has_v2_weights:
            version = 2
        else:
            version = 1

        saved_args = checkpoint.get("args", {})

        if version == 3:
            if not isinstance(config, APL_SLMConfig_v3):
                config = APL_SLMConfig_v3(
                    vocab_size=checkpoint.get("vocab_size", getattr(config, "vocab_size", 256)),
                    max_seq_len=saved_args.get("seq_len", getattr(config, "max_seq_len", 512)),
                    n_layer=saved_args.get("n_layer", getattr(config, "n_layer", 4)),
                    n_head=saved_args.get("n_head", getattr(config, "n_head", 4)),
                    n_embd=saved_args.get("n_embd", getattr(config, "n_embd", 64)),
                    max_depth=getattr(config, "max_depth", 32),
                    dropout=0.0,
                    version=3,
                )
            model = APL_SLM_v3(config).to(device)

        elif version == 2:
            if not isinstance(config, APL_SLMConfig_v2):
                config = APL_SLMConfig_v2(
                    vocab_size=checkpoint.get("vocab_size", getattr(config, "vocab_size", 256)),
                    max_seq_len=saved_args.get("seq_len", getattr(config, "max_seq_len", 512)),
                    n_layer=saved_args.get("n_layer", getattr(config, "n_layer", 4)),
                    n_head=saved_args.get("n_head", getattr(config, "n_head", 4)),
                    n_embd=saved_args.get("n_embd", getattr(config, "n_embd", 64)),
                    max_depth=getattr(config, "max_depth", 32),
                    dropout=0.0,
                    version=2,
                )
            model = APL_SLM_v2(config).to(device)

        else:
            if not isinstance(config, APL_SLMConfig_v1):
                config = APL_SLMConfig_v1(
                    vocab_size=checkpoint.get("vocab_size", getattr(config, "vocab_size", 256)),
                    max_seq_len=saved_args.get("seq_len", getattr(config, "max_seq_len", 512)),
                    n_layer=saved_args.get("n_layer", getattr(config, "n_layer", 4)),
                    n_head=saved_args.get("n_head", getattr(config, "n_head", 4)),
                    n_embd=saved_args.get("n_embd", getattr(config, "n_embd", 64)),
                    dropout=0.0,
                    version=1,
                )
            model = APL_SLM_v1(config).to(device)

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, config, version


__all__ = [
    "AutoModel",
    "APL_SLM",
    "APL_SLMConfig",
    "APL_SLM_v1",
    "APL_SLMConfig_v1",
    "APL_SLM_v2",
    "APL_SLMConfig_v2",
    "APL_SLM_v3",
    "APL_SLMConfig_v3",
    "RMSNorm",
    "SwiGLU",
    "RotaryEmbedding",
]
