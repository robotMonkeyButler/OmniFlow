# dataloader/base.py
"""
Base classes and utilities for multimodal dataloaders.
All dataset-specific loaders inherit from BaseTriModalDataset.
"""

import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


# ============================================================
# Base Dataset Class
# ============================================================


class BaseTriModalDataset(Dataset, ABC):
    """
    Abstract base class for tri-modal (vision, audio, text) datasets.

    Subclasses must implement:
        - _load_data(): Load raw data from file
        - _process_label(raw_label): Convert raw label to tensor

    The __getitem__ method returns (vis, aud, txt, label) tuple,
    compatible with the shared collate_fn.
    """

    MODALITY_KEYS = {"vis": "vision", "aud": "audio", "txt": "text"}
    LABEL_KEY = "labels"

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        normalize: List[str] = None,
        normalize_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Args:
            data_path: Path to dataset file (.pkl or .npz)
            split: One of 'train', 'valid', 'test'
            normalize: List of modality keys to normalize (e.g., ["vis", "aud"])
            normalize_stats: Pre-computed (mean, std) for each modality to normalize.
                             If None and normalize is set, will be computed from data.
        """
        self.data_path = data_path
        self.split = split
        self.normalize_keys = normalize or []
        self.normalize_stats = normalize_stats or {}

        # Will be populated by _load_data
        self.data: Dict[str, Any] = {}
        self.n_samples: int = 0

        # Load data
        self._load_data()

    @abstractmethod
    def _load_data(self) -> None:
        """Load and preprocess data. Must set self.data and self.n_samples."""
        pass

    @abstractmethod
    def _process_label(self, raw_label: Any) -> torch.Tensor:
        """Convert raw label to torch.Tensor (long for classification)."""
        pass

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            vis: (T, D_vis) vision features
            aud: (T, D_aud) audio features
            txt: (T, D_txt) text features
            label: () scalar label
        """
        vis_key = self.MODALITY_KEYS["vis"]
        aud_key = self.MODALITY_KEYS["aud"]
        txt_key = self.MODALITY_KEYS["txt"]

        vis = torch.tensor(self.data[vis_key][idx], dtype=torch.float32)
        aud = torch.tensor(self.data[aud_key][idx], dtype=torch.float32)
        txt = torch.tensor(self.data[txt_key][idx], dtype=torch.float32)
        label = self._process_label(self.data[self.LABEL_KEY][idx])

        return vis, aud, txt, label

    def get_dims(self) -> Dict[str, int]:
        """Get feature dimensions for each modality."""
        vis_key = self.MODALITY_KEYS["vis"]
        aud_key = self.MODALITY_KEYS["aud"]
        txt_key = self.MODALITY_KEYS["txt"]

        return {
            "vis": int(self.data[vis_key][0].shape[-1]),
            "aud": int(self.data[aud_key][0].shape[-1]),
            "txt": int(self.data[txt_key][0].shape[-1]),
        }

    def compute_normalize_stats(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute normalization statistics (mean, std) for each modality."""
        stats = {}
        for key in self.normalize_keys:
            actual_key = self.MODALITY_KEYS[key]
            all_data = np.concatenate(self.data[actual_key], axis=0)
            mean = all_data.mean(axis=0)
            std = all_data.std(axis=0)
            std[std < 1e-6] = 1.0
            stats[key] = (mean, std)
        return stats

    def apply_normalization(
        self, stats: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Apply normalization using pre-computed statistics."""
        for key, (mean, std) in stats.items():
            actual_key = self.MODALITY_KEYS[key]
            for i in range(len(self.data[actual_key])):
                self.data[actual_key][i] = (self.data[actual_key][i] - mean) / std
        self.normalize_stats = stats


# ============================================================
# Collate Function (shared by all datasets)
# ============================================================


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function for tri-modal data with padding.
    Pads all modalities to the same max length across all modalities.

    Args:
        batch: List of (vis, aud, txt, label) tuples

    Returns:
        vis, aud, txt: Padded tensors (B, T, D)
        vis_pad, aud_pad, txt_pad: Padding masks (B, T), True = padded position
        labels: Label tensor (B,)
    """
    vis_list, aud_list, txt_list, y_list = zip(*batch)

    lens = {
        "vis": torch.tensor([v.size(0) for v in vis_list]),
        "aud": torch.tensor([a.size(0) for a in aud_list]),
        "txt": torch.tensor([t.size(0) for t in txt_list]),
    }

    vis = pad_sequence(vis_list, batch_first=True)
    aud = pad_sequence(aud_list, batch_first=True)
    txt = pad_sequence(txt_list, batch_first=True)

    # Pad to same max length across all modalities
    maxT = max(vis.size(1), aud.size(1), txt.size(1))
    vis = F.pad(vis, (0, 0, 0, maxT - vis.size(1)))
    aud = F.pad(aud, (0, 0, 0, maxT - aud.size(1)))
    txt = F.pad(txt, (0, 0, 0, maxT - txt.size(1)))

    B = vis.size(0)
    ar = torch.arange(maxT).unsqueeze(0).expand(B, maxT)
    vis_pad = ar >= lens["vis"].unsqueeze(1)
    aud_pad = ar >= lens["aud"].unsqueeze(1)
    txt_pad = ar >= lens["txt"].unsqueeze(1)

    return vis, aud, txt, vis_pad, aud_pad, txt_pad, torch.stack(y_list)


# ============================================================
# Data Loading Utilities
# ============================================================


def load_raw_data(path: str) -> Dict[str, Any]:
    """Load raw data from .pkl or .npz file."""
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif path.endswith(".npz"):
        data = dict(np.load(path, allow_pickle=True))
    else:
        raise ValueError(f"Unsupported file format: {path}. Use .pkl or .npz")
    return data


def split_data(
    data: Dict[str, Any],
    train_key: str = "train",
    valid_key: str = "valid",
    test_key: str = "test",
) -> Tuple[Dict, Dict, Dict]:
    """
    Split data dictionary into train/valid/test.
    Supports multiple naming conventions.
    """
    # Try different key patterns
    if train_key in data:
        return data[train_key], data[valid_key], data[test_key]
    if "train_data" in data:
        return data["train_data"], data["valid_data"], data["test_data"]
    if "trains" in data:
        return data["trains"], data["valid"], data["test"]

    # If no split keys found, split by ratio
    n = len(next(iter(data.values())))
    n_tr, n_val = int(n * 0.8), int(n * 0.1)
    return (
        {k: v[:n_tr] for k, v in data.items()},
        {k: v[n_tr : n_tr + n_val] for k, v in data.items()},
        {k: v[n_tr + n_val :] for k, v in data.items()},
    )
