# dataloader/mustard.py
"""
MUStARD Dataset Loader.

Supports:
    - SAR (Sarcasm): 2 class classification (sarcastic vs unsarcastic)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import BaseTriModalDataset, load_raw_data, split_data


class MUStARDDataset(BaseTriModalDataset):
    """
    MUStARD Dataset for multimodal sarcasm detection.

    Features:
        - Text: GloVe word embeddings (300-dim)
        - Audio: acoustic features (81-dim)
        - Video: visual features (371-dim)

    Tasks:
        - SAR (Sarcasm): sarcasm detection
          - Labels: -1 (unsarcastic), 1 (sarcastic)
    """

    MODALITY_KEYS = {"vis": "vision", "aud": "audio", "txt": "text"}
    LABEL_KEY = "labels"

    # Sarcasm labels
    SARCASM_LABELS = ["unsarcastic", "sarcastic"]

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        task: str = "SAR",
        normalize: List[str] = None,
        normalize_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Args:
            data_path: Path to MUStARD pickle file
            split: One of 'train', 'valid', 'test'
            task: 'SAR' for sarcasm detection
            normalize: List of modality keys to normalize
            normalize_stats: Pre-computed normalization statistics
        """
        if task not in ["SAR"]:
            raise ValueError(f"task must be 'SAR', got {task}")

        self.task = task
        self.num_classes = 2
        self.label_names = self.SARCASM_LABELS

        super().__init__(
            data_path=data_path,
            split=split,
            normalize=normalize,
            normalize_stats=normalize_stats,
        )

    def _load_data(self) -> None:
        """Load MUStARD data from pickle file."""
        raw_data = load_raw_data(self.data_path)

        # Get split data
        trains, valid, test = split_data(raw_data)

        if self.split == "train":
            self.data = trains
        elif self.split == "valid":
            self.data = valid
        else:
            self.data = test

        self.n_samples = len(self.data[self.MODALITY_KEYS["txt"]])

    def _process_label(self, raw_label: Any) -> torch.Tensor:
        """
        Convert raw label to discrete class.
        
        MUStARD labels: -1 (unsarcastic) -> 0, 1 (sarcastic) -> 1
        """
        label = int(raw_label)
        
        # Convert -1 to 0, 1 to 1
        if label == -1:
            cls = 0  # unsarcastic
        elif label == 1:
            cls = 1  # sarcastic
        else:
            raise ValueError(f"Invalid label value: {label}. Expected -1 or 1.")

        return torch.tensor(cls, dtype=torch.long)

    def get_label_names(self) -> List[str]:
        """Get human-readable label names."""
        return self.label_names

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.num_classes
