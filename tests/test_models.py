"""Unit tests for APL model architecture versions (v1, v2, v3) and AutoModel."""

import unittest
import tempfile
from pathlib import Path
import sys
import torch

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import (
    APL_SLMConfig_v1,
    APL_SLMConfig_v2,
    APL_SLMConfig_v3,
    MODEL_PRESETS,
    resolve_preset_dimensions,
)
from v1.model_v1 import APL_SLM_v1
from v2.model_v2 import APL_SLM_v2
from v3.model_v3 import APL_SLM_v3
from model import AutoModel


class TestAPLModels(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 171
        self.seq_len = 32
        self.n_layer = 2
        self.n_head = 2
        self.n_embd = 32

    def test_preset_dimensions(self):
        l, h, e = resolve_preset_dimensions("small")
        self.assertEqual((l, h, e), (4, 4, 64))

        l, h, e = resolve_preset_dimensions("tiny")
        self.assertEqual((l, h, e), (2, 2, 32))

        # Explicit overrides
        l, h, e = resolve_preset_dimensions("small", n_layer=10)
        self.assertEqual(l, 10)

    def test_v1_model_forward(self):
        config = APL_SLMConfig_v1(
            vocab_size=self.vocab_size,
            max_seq_len=self.seq_len,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
        )
        model = APL_SLM_v1(config)
        self.assertGreater(model.count_parameters(), 0)

        idx = torch.randint(0, self.vocab_size, (2, 16))
        logits, depth_logits, caches = model(idx)
        self.assertEqual(logits.shape, (2, 16, self.vocab_size))
        self.assertIsNone(depth_logits)
        self.assertIsNone(caches)

        # KV cache step
        logits, _, caches = model(idx, use_cache=True)
        self.assertIsNotNone(caches)
        self.assertEqual(len(caches), self.n_layer)

        next_idx = torch.randint(0, self.vocab_size, (2, 1))
        next_logits, _, next_caches = model(next_idx, kv_caches=caches, use_cache=True)
        self.assertEqual(next_logits.shape, (2, 1, self.vocab_size))

    def test_v2_model_forward(self):
        config = APL_SLMConfig_v2(
            vocab_size=self.vocab_size,
            max_seq_len=self.seq_len,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            max_depth=16,
        )
        model = APL_SLM_v2(config)
        self.assertGreater(model.count_parameters(), 0)

        idx = torch.randint(0, self.vocab_size, (2, 16))
        depths = torch.randint(0, 16, (2, 16))
        logits, depth_logits, caches = model(idx, depth_ids=depths)

        self.assertEqual(logits.shape, (2, 16, self.vocab_size))
        self.assertIsNotNone(depth_logits)
        self.assertEqual(depth_logits.shape, (2, 16, 16))

    def test_v3_model_forward(self):
        config = APL_SLMConfig_v3(
            vocab_size=self.vocab_size,
            max_seq_len=self.seq_len,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            max_depth=16,
        )
        model = APL_SLM_v3(config)
        self.assertGreater(model.count_parameters(), 0)

        idx = torch.randint(0, self.vocab_size, (2, 16))
        depths = torch.randint(0, 16, (2, 16))
        logits, depth_logits, caches = model(idx, depth_ids=depths, use_cache=True)

        self.assertEqual(logits.shape, (2, 16, self.vocab_size))
        self.assertIsNotNone(depth_logits)
        self.assertEqual(depth_logits.shape, (2, 16, 16))
        self.assertIsNotNone(caches)

        # Step 2 with KV cache
        next_tok = torch.randint(0, self.vocab_size, (2, 1))
        next_depth = torch.randint(0, 16, (2, 1))
        step_logits, step_depth, step_cache = model(
            next_tok, depth_ids=next_depth, kv_caches=caches, use_cache=True
        )
        self.assertEqual(step_logits.shape, (2, 1, self.vocab_size))

    def test_automodel_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = APL_SLMConfig_v3(
                vocab_size=self.vocab_size,
                max_seq_len=self.seq_len,
                n_layer=self.n_layer,
                n_head=self.n_head,
                n_embd=self.n_embd,
            )
            model = APL_SLM_v3(config)
            ckpt_path = Path(tmpdir) / "test_v3.pt"

            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "model_version": 3,
                "val_loss": 1.234,
            }, ckpt_path)

            loaded_model, loaded_cfg, loaded_ver = AutoModel.from_checkpoint(
                ckpt_path, device=torch.device("cpu")
            )
            self.assertEqual(loaded_ver, 3)
            self.assertIsInstance(loaded_model, APL_SLM_v3)
            self.assertEqual(loaded_cfg.n_layer, self.n_layer)


if __name__ == "__main__":
    unittest.main()

