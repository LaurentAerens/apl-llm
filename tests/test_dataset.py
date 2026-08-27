"""Unit tests for APLDataset and DataLoader splitting."""

import unittest
from pathlib import Path
import sys

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tokenizer import APLTokenizer
from dataset import APLDataset, create_dataloaders


class TestAPLDataset(unittest.TestCase):
    def setUp(self):
        self.tok = APLTokenizer()
        self.text = "{+/⍵÷≢⍵} 1 2 3 4 5\n" * 100

    def test_dataset_chunking(self):
        seq_len = 32
        dataset = APLDataset(
            text=self.text,
            tokenizer=self.tok,
            seq_len=seq_len,
        )
        self.assertGreater(len(dataset), 1)

        x_tok, y_tok, x_depth, y_depth = dataset[0]
        self.assertEqual(len(x_tok), seq_len)
        self.assertEqual(len(y_tok), seq_len)
        self.assertEqual(len(x_depth), seq_len)
        self.assertEqual(len(y_depth), seq_len)

    def test_dataloader_creation(self):
        dataset = APLDataset(
            text=self.text,
            tokenizer=self.tok,
            seq_len=16,
        )
        train_loader, val_loader = create_dataloaders(dataset, batch_size=4, val_split=0.2)
        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(val_loader), 0)

        for batch in train_loader:
            x_tok, y_tok, x_depth, y_depth = batch
            self.assertEqual(x_tok.shape[1], 16)
            break


if __name__ == "__main__":
    unittest.main()

