"""
Model definitions and AutoModel factory for APL SLM (Extension Distribution).
"""

from pathlib import Path
import sys

parent_src = Path(__file__).resolve().parent.parent / "src"
if parent_src.exists() and str(parent_src) not in sys.path:
    sys.path.insert(0, str(parent_src))

from model import (
    AutoModel,
    APL_SLM,
    APL_SLMConfig,
    APLModelConfig,
    APL_SLM_v1,
    APL_SLMConfig_v1,
    APL_SLM_v2,
    APL_SLMConfig_v2,
    APL_SLM_v3,
    APL_SLMConfig_v3,
    RMSNorm,
    SwiGLU,
    RotaryEmbedding,
)

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
]
