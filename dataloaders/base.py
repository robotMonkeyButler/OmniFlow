"""
Base classes for multimodal dataloaders.

Provides unified interface for text, audio, and video modalities across all datasets.
"""

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class MultimodalSample:
    """
    Unified container for a single multimodal sample.

    All modalities are optional to handle missing data gracefully.
    """
    sample_id: str
    text: Optional[torch.Tensor] = None           # (seq_len, text_dim) or raw text string
    audio: Optional[torch.Tensor] = None          # (seq_len, audio_dim)
    video: Optional[torch.Tensor] = None          # (seq_len, video_dim)
    label: Optional[torch.Tensor] = None          # task-specific labels
    label_text: Optional[str] = None              # label as text (e.g., "Positive", "Happiness")
    label_set: Optional[List[str]] = None         # full label set for this task
    raw_text: Optional[str] = None                # original text string
    metadata: Optional[Dict[str, Any]] = None     # additional info (speaker, show, etc.)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sample_id': self.sample_id,
            'text': self.text,
            'audio': self.audio,
            'video': self.video,
            'label': self.label,
            'label_text': self.label_text,
            'label_set': self.label_set,
            'raw_text': self.raw_text,
            'metadata': self.metadata,
        }


@dataclass
class DatasetInfo:
    """Metadata about a multimodal dataset."""
    name: str
    task: str                           # sentiment, emotion, sarcasm, humor
    num_classes: int
    text_dim: int
    audio_dim: int
    video_dim: int
    max_seq_len: int
    label_names: Optional[List[str]] = None
    target: Optional[List[str]] = None  # discrete label set (e.g., ['Positive', 'Negative'])
    language: str = "en"


class BaseMultimodalDataset(Dataset, ABC):
    """
    Abstract base class for all multimodal datasets.

    Provides a unified interface for:
    - Loading and preprocessing data
    - Accessing text, audio, video modalities
    - Train/val/test splits
    - Collation for batching
    """

    SPLIT_NAMES = ['train', 'valid', 'test']

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = 'train',
        aligned: bool = True,
        max_seq_len: Optional[int] = None,
        normalize: bool = True,
        modalities: List[str] = ['text', 'audio', 'video'],
        task: Optional[str] = None,
        require_all_modalities: bool = True,
    ):
        """
        Initialize base dataset.

        Args:
            data_path: Path to dataset root directory
            split: One of 'train', 'valid', 'test'
            aligned: Whether to use aligned features (if available)
            max_seq_len: Maximum sequence length (truncate/pad if needed)
            normalize: Whether to normalize features
            modalities: List of modalities to load ['text', 'audio', 'video']
            task: Optional task type (e.g., 'SEN', 'EMO' for MOSEI). Dataset-specific.
            require_all_modalities: If True, only keep samples with all requested modalities
        """
        self.data_path = Path(data_path)
        self.split = split
        self.aligned = aligned
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        self.modalities = modalities
        self.task = task
        self.require_all_modalities = require_all_modalities

        if split not in self.SPLIT_NAMES:
            raise ValueError(f"split must be one of {self.SPLIT_NAMES}, got {split}")

        # To be set by subclasses
        self.data: Dict[str, Any] = {}
        self.samples: List[MultimodalSample] = []
        self.info: DatasetInfo = None
        self.target: Optional[List[str]] = None  # discrete label set

        # Load data in subclass
        self._load_data()

        # Filter samples to ensure all required modalities are present
        if self.require_all_modalities:
            self._filter_complete_samples()

    @abstractmethod
    def _load_data(self) -> None:
        """Load and preprocess dataset. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_info(self) -> DatasetInfo:
        """Return dataset metadata."""
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultimodalSample:
        return self.samples[idx]

    def _filter_complete_samples(self) -> None:
        """
        Validate that all samples have all required modalities at dataset level.

        This is a validation check - if any sample is missing a modality,
        it indicates the dataset is incomplete and needs to be fixed.
        This ensures data quality and prevents silent failures.

        Raises:
            ValueError: If any sample is missing required modalities
        """
        incomplete_samples = []

        for idx, sample in enumerate(self.samples):
            missing_modalities = []

            # Check each requested modality
            for modality in self.modalities:
                if modality == 'text':
                    # Text can be raw_text or text tensor
                    if sample.text is None and sample.raw_text is None:
                        missing_modalities.append('text')
                elif modality == 'audio':
                    if sample.audio is None:
                        missing_modalities.append('audio')
                elif modality == 'video':
                    if sample.video is None:
                        missing_modalities.append('video')

            if missing_modalities:
                incomplete_samples.append({
                    'index': idx,
                    'sample_id': sample.sample_id,
                    'missing': missing_modalities
                })

        # If there are incomplete samples, raise an error
        if incomplete_samples:
            dataset_name = self.info.name if self.info else 'Dataset'
            total = len(self.samples)
            incomplete_count = len(incomplete_samples)

            # Show first few incomplete samples as examples
            examples = incomplete_samples[:5]
            example_str = '\n'.join([
                f"  - Sample {s['sample_id']} (index {s['index']}): missing {s['missing']}"
                for s in examples
            ])

            error_msg = (
                f"\n[{dataset_name}] Dataset validation failed!\n"
                f"Found {incomplete_count}/{total} samples with missing modalities.\n"
                f"Required modalities: {self.modalities}\n\n"
                f"Examples of incomplete samples:\n{example_str}\n"
            )

            if incomplete_count > 5:
                error_msg += f"\n... and {incomplete_count - 5} more incomplete samples.\n"

            error_msg += (
                f"\nThis indicates the dataset is incomplete or features are missing.\n"
                f"Please check your data files and feature extraction.\n"
                f"To disable this check, set require_all_modalities=False (not recommended)."
            )

            raise ValueError(error_msg)

    def _normalize_features(
        self,
        features: np.ndarray,
        eps: float = 1e-8
    ) -> np.ndarray:
        """Z-score normalization along feature dimension."""
        if features is None:
            return None
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        return (features - mean) / (std + eps)

    def _pad_or_truncate(
        self,
        features: np.ndarray,
        max_len: int,
        pad_value: float = 0.0
    ) -> np.ndarray:
        """Pad or truncate sequence to fixed length."""
        if features is None:
            return None

        seq_len = features.shape[0]

        if seq_len > max_len:
            return features[:max_len]
        elif seq_len < max_len:
            pad_shape = (max_len - seq_len,) + features.shape[1:]
            padding = np.full(pad_shape, pad_value, dtype=features.dtype)
            return np.concatenate([features, padding], axis=0)
        return features

    def _to_tensor(self, data: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        """Convert numpy array to torch tensor."""
        if data is None:
            return None
        return torch.from_numpy(data).float()


def multimodal_collate_fn(batch: List[MultimodalSample]) -> Dict[str, Any]:
    """
    Collate function for batching MultimodalSamples.

    Handles variable length sequences by padding and creating masks.
    """
    sample_ids = [s.sample_id for s in batch]

    # Collect non-None tensors for each modality
    texts = [s.text for s in batch if s.text is not None]
    audios = [s.audio for s in batch if s.audio is not None]
    videos = [s.video for s in batch if s.video is not None]
    labels = [s.label for s in batch if s.label is not None]
    raw_texts = [s.raw_text for s in batch]

    result = {
        'sample_ids': sample_ids,
        'raw_texts': raw_texts,
    }

    # Stack tensors if available
    if texts and len(texts) == len(batch):
        result['text'] = torch.stack(texts)
        result['text_mask'] = _create_mask(texts)

    if audios and len(audios) == len(batch):
        result['audio'] = torch.stack(audios)
        result['audio_mask'] = _create_mask(audios)

    if videos and len(videos) == len(batch):
        result['video'] = torch.stack(videos)
        result['video_mask'] = _create_mask(videos)

    if labels and len(labels) == len(batch):
        result['labels'] = torch.stack(labels)

    return result


def _create_mask(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Create attention mask (1 for valid positions, 0 for padding)."""
    # For pre-padded tensors, assume all positions are valid
    # Subclasses can override with actual mask if needed
    batch_size = len(tensors)
    seq_len = tensors[0].shape[0]
    return torch.ones(batch_size, seq_len)


def load_pickle(path: Union[str, Path], encoding: str = 'latin1') -> Any:
    """Load a pickle file with Python 2/3 compatibility."""
    if hasattr(path, 'read'):
        # Handle file-like objects (e.g., BytesIO)
        return pickle.load(path, encoding=encoding)
    with open(path, 'rb') as f:
        return pickle.load(f, encoding=encoding)


def get_dataloader(
    dataset: BaseMultimodalDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader with standard settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=multimodal_collate_fn,
    )
