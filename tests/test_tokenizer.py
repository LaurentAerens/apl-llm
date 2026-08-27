"""Unit tests for APLTokenizer."""

import unittest
import tempfile
from pathlib import Path
import sys

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tokenizer import APLTokenizer, TokenInfo


class TestAPLTokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = APLTokenizer()

    def test_vocab_initialization(self):
        self.assertGreater(self.tok.vocab_size, 100)
        self.assertEqual(self.tok.pad_id, self.tok.char_to_id["<pad>"])
        self.assertEqual(self.tok.bos_id, self.tok.char_to_id["<bos>"])
        self.assertEqual(self.tok.eos_id, self.tok.char_to_id["<eos>"])
        self.assertEqual(self.tok.unk_id, self.tok.char_to_id["<unk>"])

    def test_encode_decode_roundtrip(self):
        sample_code = "{+/⍵÷≢⍵} 1 2 3 4 5"
        encoded = self.tok.encode(sample_code, add_special_tokens=False)
        decoded = self.tok.decode(encoded, skip_special_tokens=True)
        self.assertEqual(sample_code, decoded)

    def test_special_tokens(self):
        sample_code = "⍳10"
        encoded = self.tok.encode(sample_code, add_special_tokens=True)
        self.assertEqual(encoded[0], self.tok.bos_id)
        self.assertEqual(encoded[-1], self.tok.eos_id)
        decoded = self.tok.decode(encoded, skip_special_tokens=True)
        self.assertEqual(sample_code, decoded)

    def test_depth_computation(self):
        code = "{(+/⍵)÷≢⍵}"
        tokens = self.tok.encode(code)
        depths = self.tok.compute_depth_sequences(tokens)
        self.assertEqual(len(tokens), len(depths))
        self.assertEqual(depths[0], 0)
        # Inside the dfn and paren, depth should increase
        self.assertGreater(max(depths), 1)

    def test_structural_balancing(self):
        # Balanced cases
        self.assertTrue(self.tok.is_balanced("{+/⍵}"))
        self.assertTrue(self.tok.is_balanced("{(+/⍵)÷≢⍵}"))
        self.assertTrue(self.tok.is_balanced("M[⍋M[;1];]"))
        self.assertTrue(self.tok.is_balanced("1+2×3"))

        # Unbalanced cases
        self.assertFalse(self.tok.is_balanced("{+/⍵"))
        self.assertFalse(self.tok.is_balanced("{(+/⍵÷≢⍵}"))
        self.assertFalse(self.tok.is_balanced("V[1;2"))
        self.assertFalse(self.tok.is_balanced(") ("))  # Underflow case
        self.assertFalse(self.tok.is_balanced("}{"))   # Underflow dfn case

    def test_unclosed_delimiters(self):
        unclosed = self.tok.get_unclosed_delimiters("{(+/⍵")
        self.assertEqual(unclosed, ["{", "("])

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "tokenizer.json"
            self.tok.save(save_path)
            self.assertTrue(save_path.exists())

            loaded_tok = APLTokenizer.load(save_path)
            self.assertEqual(self.tok.vocab_size, loaded_tok.vocab_size)
            self.assertEqual(self.tok.vocab, loaded_tok.vocab)

            code = "{+/⍵÷≢⍵}"
            self.assertEqual(self.tok.encode(code), loaded_tok.encode(code))


if __name__ == "__main__":
    unittest.main()

