"""End-to-end smoke tests for training v1, v2, and v3 APL SLM architectures."""

import unittest
import tempfile
from pathlib import Path
import sys
import argparse

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from trainer import APLTrainer
from synthetic_generator import APLSyntheticGenerator


class TestTrainingSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir_obj = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls.tmpdir_obj.name)
        cls.corpus_file = cls.tmpdir / "synthetic_smoke_corpus.txt"

        # Generate small synthetic corpus for testing
        corpus = APLSyntheticGenerator.generate_synthetic_corpus(count=200, seed=42)
        with open(cls.corpus_file, "w", encoding="utf-8") as f:
            f.write(corpus)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir_obj.cleanup()

    def test_train_v1_smoke(self):
        args = argparse.Namespace(
            version=1,
            data_file=str(self.corpus_file),
            exp_name="Test-v1-Smoke",
            epochs=1,
            batch_size=4,
            seq_len=32,
            lr=1e-3,
            warmup_epochs=0,
            grad_accum_steps=1,
            early_stopping_patience=1,
            precision="fp32",
            model_preset="tiny",
            depth_loss_weight=0.2,
            device="cpu",
            finetune_from=None,
            resume=False,
            demo=False,
            dropout=0.0,
            n_layer=None,
            n_head=None,
            n_embd=None,
        )
        trainer = APLTrainer(args)
        model, best_val_loss = trainer.train()
        self.assertIsNotNone(model)
        self.assertLess(best_val_loss, 100.0)

    def test_train_v2_smoke(self):
        args = argparse.Namespace(
            version=2,
            data_file=str(self.corpus_file),
            exp_name="Test-v2-Smoke",
            epochs=1,
            batch_size=4,
            seq_len=32,
            lr=1e-3,
            warmup_epochs=0,
            grad_accum_steps=1,
            early_stopping_patience=1,
            precision="fp32",
            model_preset="tiny",
            depth_loss_weight=0.2,
            device="cpu",
            finetune_from=None,
            resume=False,
            demo=False,
            dropout=0.0,
            n_layer=None,
            n_head=None,
            n_embd=None,
        )
        trainer = APLTrainer(args)
        model, best_val_loss = trainer.train()
        self.assertIsNotNone(model)
        self.assertLess(best_val_loss, 100.0)

    def test_train_v3_smoke(self):
        args = argparse.Namespace(
            version=3,
            data_file=str(self.corpus_file),
            exp_name="Test-v3-Smoke",
            epochs=1,
            batch_size=4,
            seq_len=32,
            lr=1e-3,
            warmup_epochs=0,
            grad_accum_steps=1,
            early_stopping_patience=1,
            precision="fp32",
            model_preset="tiny",
            depth_loss_weight=0.2,
            device="cpu",
            finetune_from=None,
            resume=False,
            demo=False,
            dropout=0.0,
            n_layer=None,
            n_head=None,
            n_embd=None,
        )
        trainer = APLTrainer(args)
        model, best_val_loss = trainer.train()
        self.assertIsNotNone(model)
        self.assertLess(best_val_loss, 100.0)


if __name__ == "__main__":
    unittest.main()

