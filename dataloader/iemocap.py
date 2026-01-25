# dataloader/iemocap.py
"""
IEMOCAP Dataset Loader.

Emotion recognition from conversational audio-visual data.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import BaseMultiModalDataset, load_raw_data, split_data


class IEMOCAPDataset(BaseMultiModalDataset):
    """
    IEMOCAP Dataset for emotion recognition.

    Task:
        - Emotion classification (typically 4 or 6 classes)
        - 4-class: happy, sad, angry, neutral
        - 6-class: + frustrated, excited
    """

    MODALITY_KEYS = {"vis": "vision", "aud": "audio", "txt": "text"}
    LABEL_KEY = "labels"

    EMOTION_LABELS_4 = ["happy", "sad", "angry", "neutral"]
    EMOTION_LABELS_6 = ["happy", "sad", "angry", "neutral", "frustrated", "excited"]

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        num_classes: int = 4,
        normalize: List[str] = None,
        normalize_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Args:
            data_path: Path to IEMOCAP pickle file
            split: One of 'train', 'valid', 'test'
            num_classes: Number of emotion classes (4 or 6)
            normalize: List of modality keys to normalize
            normalize_stats: Pre-computed normalization statistics
        """
        self.num_classes = num_classes
        self.label_names = (
            self.EMOTION_LABELS_4 if num_classes == 4 else self.EMOTION_LABELS_6
        )

        super().__init__(
            data_path=data_path,
            split=split,
            normalize=normalize,
            normalize_stats=normalize_stats,
        )

    def _load_data(self) -> None:
        """Load IEMOCAP data from pickle file."""
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
        """Convert raw label to class index."""
        label = np.asarray(raw_label)

        if label.ndim > 0 and label.size > 1:
            # One-hot: take argmax
            cls = int(np.argmax(label))
        else:
            # Scalar label
            cls = int(label.flat[0])

        return torch.tensor(cls, dtype=torch.long)

