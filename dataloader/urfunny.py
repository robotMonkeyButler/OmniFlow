# dataloader/urfunny.py
"""
UR-FUNNY Dataset Loader.

Binary humor detection task.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import BaseTriModalDataset, load_raw_data, split_data


class URFunnyDataset(BaseTriModalDataset):
    """
    UR-FUNNY Dataset for humor detection.

    Features:
        - Text: GloVe word embeddings (300-dim)
        - Audio: COVAREP features (81-dim)
        - Video: OpenFace features (371-dim)

    Task:
        - Binary classification: humor (1) vs non-humor (0)
    """

    MODALITY_KEYS = {"vis": "vision", "aud": "audio", "txt": "text"}
    LABEL_KEY = "labels"

    LABEL_NAMES = ["non-humor", "humor"]

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        normalize: List[str] = None,
        normalize_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Args:
            data_path: Path to UR-FUNNY pickle file
            split: One of 'train', 'valid', 'test'
            normalize: List of modality keys to normalize
            normalize_stats: Pre-computed normalization statistics
        """
        self.num_classes = 2
        self.label_names = self.LABEL_NAMES

        super().__init__(
            data_path=data_path,
            split=split,
            normalize=normalize,
            normalize_stats=normalize_stats,
        )

    def _load_data(self) -> None:
        """Load UR-FUNNY data from pickle file."""
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
        """Convert raw label to binary class (0 or 1)."""
        label = np.asarray(raw_label)

        if label.ndim > 0 and label.size > 1:
            # One-hot or multi-label: take argmax
            cls = int(np.argmax(label))
        else:
            # Scalar label
            cls = int(label.flat[0])

        return torch.tensor(cls, dtype=torch.long)

    def get_label_names(self) -> List[str]:
        return self.label_names

    def get_num_classes(self) -> int:
        return self.num_classes
