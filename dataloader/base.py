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


class BaseMultiModalDataset(Dataset, ABC):
    """
    Abstract base class for multi-modal datasets.

    Subclasses must implement:
        - _load_data(): Load raw data from file
        - _process_label(raw_label): Convert raw label to tensor

    The __getitem__ method returns (*modalities, label) tuple,
    compatible with the shared collate_fn.
    
    By default supports tri-modal (vis/aud/txt) setup for backward compatibility.
    Subclasses can override MODALITY_KEYS to define custom modalities.
    """

    MODALITY_KEYS = {"vis": "vision", "aud": "audio", "txt": "text"}
    LABEL_KEY = "labels"
    num_classes: int
    label_names: List[str]

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            *modalities: One tensor per modality (T, D_modality)
            label: () scalar label
        """
        modality_tensors = []
        for modality_name in self.get_modality_names():
            storage_key = self.MODALITY_KEYS[modality_name]
            tensor = torch.tensor(self.data[storage_key][idx], dtype=torch.float32)
            modality_tensors.append(tensor)
        
        label = self._process_label(self.data[self.LABEL_KEY][idx])
        return tuple(modality_tensors + [label])

    def get_modality_names(self) -> List[str]:
        """Get list of modality names in order."""
        return list(self.MODALITY_KEYS.keys())
    
    def get_dims(self) -> Dict[str, int]:
        """Get feature dimensions for each modality."""
        dims = {}
        for modality_name, storage_key in self.MODALITY_KEYS.items():
            dims[modality_name] = int(self.data[storage_key][0].shape[-1])
        return dims

    def compute_normalize_stats(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute normalization statistics (mean, std) for each modality."""
        stats = {}
        for modality_name in self.normalize_keys:
            if modality_name not in self.MODALITY_KEYS:
                raise ValueError(f"Modality '{modality_name}' not found in MODALITY_KEYS")
            storage_key = self.MODALITY_KEYS[modality_name]
            all_data = np.concatenate(self.data[storage_key], axis=0)
            mean = all_data.mean(axis=0)
            std = all_data.std(axis=0)
            std[std < 1e-6] = 1.0
            stats[modality_name] = (mean, std)
        return stats

    def apply_normalization(
        self, stats: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Apply normalization using pre-computed statistics."""
        for modality_name, (mean, std) in stats.items():
            if modality_name not in self.MODALITY_KEYS:
                raise ValueError(f"Modality '{modality_name}' not found in MODALITY_KEYS")
            storage_key = self.MODALITY_KEYS[modality_name]
            for i in range(len(self.data[storage_key])):
                self.data[storage_key][i] = (self.data[storage_key][i] - mean) / std
        self.normalize_stats = stats

    def compute_clip_stats(
        self, margin: float = 0.1
    ) -> Dict[str, float]:
        """
        Compute clipping statistics (abs_max) for each modality.
        Handles inf/nan values by considering only finite values.

        Args:
            margin: Extra margin to add to abs_max (e.g., 0.1 for 10% buffer)

        Returns:
            Dictionary mapping modality names to clip values
        """
        stats = {}
        for modality_name, storage_key in self.MODALITY_KEYS.items():
            all_data = np.concatenate(self.data[storage_key], axis=0)

            # Keep only finite values (remove inf and nan)
            finite_mask = np.isfinite(all_data)
            finite_data = all_data[finite_mask]

            if len(finite_data) == 0:
                # If all values are inf/nan, use default
                abs_max = 1.0
            else:
                abs_max = np.max(np.abs(finite_data))

            # Add margin if specified
            if margin > 0:
                abs_max = abs_max * (1.0 + margin)

            stats[modality_name] = abs_max
        return stats

    def apply_clipping(self, stats: Dict[str, float]) -> None:
        """
        Apply value clipping to all modalities using pre-computed statistics.
        Replaces NaN values with 0.

        Args:
            stats: Dictionary mapping modality names to clip values
                   (output from compute_clip_stats)
        """
        for modality_name, abs_max in stats.items():
            if modality_name not in self.MODALITY_KEYS:
                raise ValueError(f"Modality '{modality_name}' not found in MODALITY_KEYS")
            storage_key = self.MODALITY_KEYS[modality_name]
            for i in range(len(self.data[storage_key])):
                data = self.data[storage_key][i]
                # Replace NaN with 0
                data = np.nan_to_num(data, nan=0.0)
                # Clip to [-abs_max, abs_max]
                self.data[storage_key][i] = np.clip(data, -abs_max, abs_max)

    def get_label_names(self) -> List[str]:
        """Get human-readable label names."""
        return self.label_names

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.num_classes


# ============================================================
# Collate Function (shared by all datasets)
# ============================================================


def collate_fn_with_names(batch: List[Tuple], modality_names: List[str]) -> Dict[str, torch.Tensor]:
    """
    Collate function for multi-modal data with padding.
    Returns dict with modality names as keys for semantic clarity.

    Args:
        batch: List of (*modalities, label) tuples
        modality_names: List of modality names in order (e.g., ["vis", "aud", "txt"])

    Returns:
        Dictionary with:
            - <modality_name>: Padded tensor (B, T, D) for each modality
            - <modality_name>_pad: Padding mask (B, T), True = padded
            - labels: Label tensor (B,)
    """
    num_modalities = len(modality_names)
    modality_lists = [[] for _ in range(num_modalities)]
    label_list = []
    
    for item in batch:
        for i in range(num_modalities):
            modality_lists[i].append(item[i])
        label_list.append(item[-1])
    
    # Get lengths for each modality
    lens = {}
    for i in range(num_modalities):
        lens[i] = torch.tensor([m.size(0) for m in modality_lists[i]])
    
    # Pad each modality
    padded_modalities = []
    for i in range(num_modalities):
        padded = pad_sequence(modality_lists[i], batch_first=True)
        padded_modalities.append(padded)
    
    # Find max length across all modalities
    maxT = max(m.size(1) for m in padded_modalities)
    
    # Pad all to same max length
    for i in range(num_modalities):
        if padded_modalities[i].size(1) < maxT:
            padded_modalities[i] = F.pad(
                padded_modalities[i], 
                (0, 0, 0, maxT - padded_modalities[i].size(1))
            )
    
    # Create padding masks and build result dict
    B = padded_modalities[0].size(0)
    ar = torch.arange(maxT).unsqueeze(0).expand(B, maxT)
    
    result = {}
    for i, name in enumerate(modality_names):
        result[name] = padded_modalities[i]
        result[f"{name}_pad"] = ar >= lens[i].unsqueeze(1)
    
    result["labels"] = torch.stack(label_list)
    return result


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Backward-compatible collate for tri-modal (vis/aud/txt) data.
    Returns tuple format for legacy code.
    """
    modality_names = ["vis", "aud", "txt"]
    batch_dict = collate_fn_with_names(batch, modality_names)
    return (
        batch_dict["vis"], batch_dict["aud"], batch_dict["txt"],
        batch_dict["vis_pad"], batch_dict["aud_pad"], batch_dict["txt_pad"],
        batch_dict["labels"]
    )


class CollateFunction:
    """
    Collate function wrapper that can be pickled for multiprocessing.
    This replaces the nested function approach which can't be pickled.
    """
    def __init__(self, modality_names: List[str]):
        self.modality_names = modality_names
    
    def __call__(self, batch: List[Tuple]) -> Dict[str, torch.Tensor]:
        return collate_fn_with_names(batch, self.modality_names)


def create_collate_fn(modality_names: List[str]):
    """
    Factory to create a collate function for specific modality names.
    Returns a dict-based collate function that can be pickled.
    
    Args:
        modality_names: List of modality names (e.g., ["timeseries", "static"])
    
    Returns:
        CollateFunction instance that can be pickled for multiprocessing
    """
    return CollateFunction(modality_names)


# Backward compatibility alias
BaseTriModalDataset = BaseMultiModalDataset


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
