"""Unit tests for APLSyntheticGenerator."""

import unittest
from pathlib import Path
import sys

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from synthetic_generator import APLSyntheticGenerator
from tokenizer import APLTokenizer


class TestSyntheticGenerator(unittest.TestCase):
    def setUp(self):
        self.tok = APLTokenizer()

    def test_reproducible_corpus(self):
        c1 = APLSyntheticGenerator.generate_synthetic_corpus(count=50, seed=42)
        c2 = APLSyntheticGenerator.generate_synthetic_corpus(count=50, seed=42)
        self.assertEqual(c1, c2)

    def test_generated_lines(self):
        corpus = APLSyntheticGenerator.generate_synthetic_corpus(count=20, seed=123)
        lines = [l for l in corpus.strip().split("\n") if l.strip()]
        self.assertGreaterEqual(len(lines), 20)

        # Ensure all generated characters can be encoded by the tokenizer
        for line in lines:
            encoded = self.tok.encode(line)
            self.assertNotIn(self.tok.unk_id, encoded, f"Unknown token found in line: {line}")


if __name__ == "__main__":
    unittest.main()

