"""
Dataset pipeline and DataLoader builder for APL language models.
Handles sequence chunking, padding, structural depth tracking, and on-disk tensor caching.
"""

from pathlib import Path
from typing import Union, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from tokenizer import APLTokenizer


class APLDataset(Dataset):
    """
    Chunked sequence dataset for APL Causal Language Modeling.
    Yields (token_inputs, token_targets, depth_inputs, depth_targets) tensors.
    """

    def __init__(
        self,
        text: str,
        tokenizer: APLTokenizer,
        seq_len: int = 512,
        tokens_path: Optional[Union[str, Path]] = None,
        depths_path: Optional[Union[str, Path]] = None,
        force_recompute: bool = False,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        tokens_file = Path(tokens_path) if tokens_path else None
        depths_file = Path(depths_path) if depths_path else None

        loaded_from_cache = False
        if (
            tokens_file
            and depths_file
            and tokens_file.exists()
            and depths_file.exists()
            and not force_recompute
        ):
            try:
                self.tokens = torch.load(tokens_file, weights_only=False)
                self.depths = torch.load(depths_file, weights_only=False)
                if len(self.tokens) == len(self.depths):
                    loaded_from_cache = True
            except Exception:
                loaded_from_cache = False

        if not loaded_from_cache:
            raw_tokens = tokenizer.encode(text, add_special_tokens=False)
            self.tokens = torch.tensor(raw_tokens, dtype=torch.long)
            raw_depths = tokenizer.compute_depth_sequences(raw_tokens)
            self.depths = torch.tensor(raw_depths, dtype=torch.long)

            if tokens_file and depths_file:
                tokens_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.tokens, tokens_file)
                torch.save(self.depths, depths_file)

        num_samples = (len(self.tokens) - 1) // self.seq_len
        self.num_samples = max(1, num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len

        inputs = self.tokens[start_idx:end_idx]
        targets = self.tokens[start_idx + 1 : end_idx + 1]
        depth_inputs = self.depths[start_idx:end_idx]
        depth_targets = self.depths[start_idx + 1 : end_idx + 1]

        # Pad if short
        if len(inputs) < self.seq_len:
            pad_len = self.seq_len - len(inputs)
            inputs = torch.cat([inputs, torch.full((pad_len,), self.tokenizer.pad_id, dtype=torch.long)])
            depth_inputs = torch.cat([depth_inputs, torch.zeros((pad_len,), dtype=torch.long)])
        if len(targets) < self.seq_len:
            pad_len = self.seq_len - len(targets)
            targets = torch.cat([targets, torch.full((pad_len,), self.tokenizer.pad_id, dtype=torch.long)])
            depth_targets = torch.cat([depth_targets, torch.zeros((pad_len,), dtype=torch.long)])

        return inputs, targets, depth_inputs, depth_targets


def create_dataloaders(
    dataset: APLDataset,
    batch_size: int = 16,
    val_split: float = 0.1,
    pin_memory: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Splits an APLDataset into training and validation DataLoaders."""
    total_len = len(dataset)
    split_idx = int(total_len * (1.0 - val_split))

    train_indices = list(range(0, max(1, split_idx)))
    val_indices = list(range(max(1, split_idx), total_len))
    if not val_indices:
        val_indices = train_indices[: max(1, len(train_indices) // 5)]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    return train_loader, val_loader

