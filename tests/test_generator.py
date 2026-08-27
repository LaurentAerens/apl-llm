"""Unit tests for APLGenerator and autoregressive decoding."""

import unittest
from pathlib import Path
import sys
import torch

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tokenizer import APLTokenizer
from config import APL_SLMConfig_v1, APL_SLMConfig_v3
from v1.model_v1 import APL_SLM_v1
from v3.model_v3 import APL_SLM_v3
from generator import APLGenerator, sample_next_token


class TestAPLGenerator(unittest.TestCase):
    def setUp(self):
        self.tokenizer = APLTokenizer()
        self.device = torch.device("cpu")

        self.cfg_v3 = APL_SLMConfig_v3(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=64,
            n_layer=2,
            n_head=2,
            n_embd=32,
        )
        self.model_v3 = APL_SLM_v3(self.cfg_v3).to(self.device)

        self.cfg_v1 = APL_SLMConfig_v1(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=64,
            n_layer=2,
            n_head=2,
            n_embd=32,
        )
        self.model_v1 = APL_SLM_v1(self.cfg_v1).to(self.device)

    def test_sample_next_token_greedy(self):
        logits = torch.zeros((1, self.tokenizer.vocab_size))
        logits[0, 42] = 10.0
        token = sample_next_token(logits, temperature=0.0)
        self.assertEqual(token, 42)

    def test_sample_next_token_top_k(self):
        logits = torch.randn((1, self.tokenizer.vocab_size))
        token = sample_next_token(logits, temperature=0.7, top_k=5)
        self.assertIsInstance(token, int)
        self.assertTrue(0 <= token < self.tokenizer.vocab_size)

    def test_generator_v3_with_kv_cache(self):
        prompt = "{+/"
        result = APLGenerator.generate(
            model=self.model_v3,
            tokenizer=self.tokenizer,
            device=self.device,
            prompt=prompt,
            max_new_tokens=10,
            temperature=0.0,
            use_kv_cache=True,
        )
        self.assertTrue(result.startswith(prompt))
        self.assertGreaterEqual(len(result), len(prompt))

    def test_generator_v3_without_kv_cache(self):
        prompt = "{+/"
        result = APLGenerator.generate(
            model=self.model_v3,
            tokenizer=self.tokenizer,
            device=self.device,
            prompt=prompt,
            max_new_tokens=10,
            temperature=0.0,
            use_kv_cache=False,
        )
        self.assertTrue(result.startswith(prompt))

    def test_generator_v1_baseline(self):
        prompt = "1 2 3 +"
        result = APLGenerator.generate(
            model=self.model_v1,
            tokenizer=self.tokenizer,
            device=self.device,
            prompt=prompt,
            max_new_tokens=10,
            temperature=0.0,
            use_kv_cache=True,
        )
        self.assertTrue(result.startswith(prompt))


if __name__ == "__main__":
    unittest.main()

