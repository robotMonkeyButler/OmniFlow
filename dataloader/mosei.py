# dataloader/mosei.py
"""
CMU-MOSEI Dataset Loader.

Supports:
    - SEN (Sentiment): 2/3/5/7 class classification
    - EMO (Emotion): 6 class classification (Happiness, Sadness, Anger, Fear, Disgust, Surprise)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import BaseTriModalDataset, load_raw_data, split_data


class MOSEIDataset(BaseTriModalDataset):
    """
    CMU-MOSEI Dataset for multimodal sentiment and emotion analysis.

    Features:
        - Text: GloVe word embeddings (300-dim)
        - Audio: COVAREP acoustic features (74-dim)
        - Video: Facet visual features (35-dim)

    Tasks:
        - SEN (Sentiment): sentiment polarity classification
          - Labels: [-3, 3] continuous scale, discretized to 2/3/5/7 classes
        - EMO (Emotion): emotion classification
          - Labels: 6 emotions (Happiness, Sadness, Anger, Fear, Disgust, Surprise)
    """

    MODALITY_KEYS = {"vis": "vision", "aud": "audio", "txt": "text"}
    LABEL_KEY = "labels"

    # Sentiment labels
    SENTIMENT_LABELS = {
        7: [
            "highly negative",
            "negative",
            "weakly negative",
            "neutral",
            "weakly positive",
            "positive",
            "highly positive",
        ],
        5: ["highly negative", "negative", "neutral", "positive", "highly positive"],
        3: ["negative", "neutral", "positive"],
        2: ["negative", "positive"],
    }

    # Emotion labels (order in mosei_raw.pkl: [Sentiment, Happiness, Sadness, Anger, Fear, Disgust, Surprise])
    EMOTION_LABELS = ["Happiness", "Sadness", "Anger", "Fear", "Disgust", "Surprise"]

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        task: str = "SEN",
        num_classes: int = 2,
        normalize: List[str] = None,
        normalize_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
        clip_margin: float = 0.1,
    ):
        """
        Args:
            data_path: Path to MOSEI pickle file
            split: One of 'train', 'valid', 'test'
            task: 'SEN' for sentiment or 'EMO' for emotion
            num_classes: Number of classes (2/3/5/7 for SEN, 6 for EMO)
            normalize: List of modality keys to normalize
            normalize_stats: Pre-computed normalization statistics
            clip_margin: Margin to add to abs_max when clipping (e.g., 0.1 for 10%)
        """
        if task not in ["SEN", "EMO"]:
            raise ValueError(f"task must be 'SEN' or 'EMO', got {task}")

        self.task = task
        self.num_classes = 6 if task == "EMO" else num_classes
        self.clip_margin = clip_margin

        # Set label names
        if task == "EMO":
            self.label_names = self.EMOTION_LABELS
        else:
            self.label_names = self.SENTIMENT_LABELS.get(
                num_classes, self.SENTIMENT_LABELS[2]
            )

        super().__init__(
            data_path=data_path,
            split=split,
            normalize=normalize,
            normalize_stats=normalize_stats,
        )

    def _load_data(self) -> None:
        """Load MOSEI data from pickle file."""
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
        """Convert raw label to discrete class."""
        label = np.asarray(raw_label)

        # Handle different label shapes
        if label.ndim == 2:
            if label.shape[-1] == 7:
                # mosei_raw.pkl format: [Sentiment, H, Sa, A, F, D, Su]
                if self.task == "EMO":
                    # Take argmax of last 6 for emotion
                    emotion_scores = label[0, 1:]  # (6,)
                    cls = int(np.argmax(emotion_scores))
                else:
                    # Take first dimension for sentiment
                    label = label[0, 0]
                    cls = self._discretize_sentiment(label)
            else:
                label = label.squeeze()
                cls = self._discretize_sentiment(float(label))
        elif label.ndim == 1:
            if label.size == 7:
                if self.task == "EMO":
                    cls = int(np.argmax(label[1:]))
                else:
                    cls = self._discretize_sentiment(float(label[0]))
            elif label.size == 1:
                cls = self._discretize_sentiment(float(label[0]))
            else:
                cls = self._discretize_sentiment(float(label.mean()))
        else:
            cls = self._discretize_sentiment(float(label))

        return torch.tensor(cls, dtype=torch.long)

    def _discretize_sentiment(self, label: float) -> int:
        """Convert continuous sentiment value to discrete class."""
        if self.num_classes == 2:
            return 0 if label < 0 else 1
        elif self.num_classes == 3:
            if label <= -1:
                return 0  # negative
            elif label < 1:
                return 1  # neutral
            else:
                return 2  # positive
        elif self.num_classes == 5:
            if label < -2:
                return 0
            elif label < -1:
                return 1
            elif label < 1:
                return 2
            elif label < 2:
                return 3
            else:
                return 4
        elif self.num_classes == 7:
            return int(np.clip(np.round(label) + 3, 0, 6))
        else:
            # Default to binary
            return 0 if label < 0 else 1

    def get_label_names(self) -> List[str]:
        """Get human-readable label names."""
        return self.label_names

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.num_classes


class MOSIDataset(MOSEIDataset):
    """
    CMU-MOSI Dataset for multimodal sentiment analysis.

    Same format as MOSEI but smaller and only supports sentiment task.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        num_classes: int = 2,
        normalize: List[str] = None,
        normalize_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
    ):
        super().__init__(
            data_path=data_path,
            split=split,
            task="SEN",  # MOSI only supports sentiment
            num_classes=num_classes,
            normalize=normalize,
            normalize_stats=normalize_stats,
        )
