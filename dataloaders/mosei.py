"""
CMU-MOSEI Dataset Loader.

CMU Multimodal Opinion Sentiment and Emotion Intensity dataset.
Contains 23,453 YouTube video clips with sentiment and emotion labels.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .base import (
    BaseMultimodalDataset,
    DatasetInfo,
    MultimodalSample,
    load_pickle,
)


class MOSEIDataset(BaseMultimodalDataset):
    """
    CMU-MOSEI Dataset for multimodal sentiment and emotion analysis.

    Features:
        - Text: GloVe word embeddings (300-dim)
        - Audio: COVAREP acoustic features
        - Video: Facet visual features (35-dim)

    Tasks:
        - SEN (Sentiment): sentiment polarity classification
          - Labels: [-3, 3] continuous scale, discretized to 2/3/5/7 classes
          - Data file: 'mosei_senti_data.pkl' or 'mosei_raw.pkl' (first dimension)
        - EMO (Emotion): emotion classification
          - Labels: 6 emotions (Happiness, Sadness, Anger, Fear, Disgust, Surprise)
          - Data file: 'mosei_raw.pkl' only (argmax of 6 emotions)

    Data format:
        Pickle file with train/valid/test splits.
        Each split contains: text, audio, vision, labels, id
    """

    # Sentiment labels
    SENTIMENT_LABELS_7 = [
        'highly negative', 'negative', 'weakly negative',
        'neutral',
        'weakly positive', 'positive', 'highly positive'
    ]
    SENTIMENT_LABELS_5 = [
        'highly negative', 'negative', 'neutral', 'positive', 'highly positive'
    ]
    SENTIMENT_LABELS_3 = ['negative', 'neutral', 'positive']
    SENTIMENT_LABELS_2 = ['negative', 'positive']
    
    # Emotion labels (order in mosei_raw.pkl: [Sentiment, Happiness, Sadness, Anger, Fear, Disgust, Surprise])
    EMOTION_LABELS = ['Happiness', 'Sadness', 'Anger', 'Fear', 'Disgust', 'Surprise']

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = 'train',
        aligned: bool = True,
        max_seq_len: int = 50,
        normalize: bool = True,
        modalities: List[str] = ['text', 'audio', 'video'],
        num_classes: int = 2,
        data_file: str = 'mosei_raw.pkl',
        task: str = 'SEN',
        require_all_modalities: bool = True,
    ):
        """
        Initialize MOSEI dataset.

        Args:
            data_path: Path to MOSEI directory containing pickle files
            split: 'train', 'valid', or 'test'
            aligned: Use aligned features (recommended for MOSEI)
            max_seq_len: Maximum sequence length (default 50)
            normalize: Normalize features
            modalities: Which modalities to load
            num_classes: Number of sentiment classes (2, 3, 5, or 7) - only for SEN task
            data_file: Name of pickle file to load ('mosei_senti_data.pkl' or 'mosei_raw.pkl')
            task: Task type - 'SEN' for sentiment or 'EMO' for emotion
        """
        if task not in ['SEN', 'EMO']:
            raise ValueError(f"task must be 'SEN' or 'EMO', got {task}")
        
        if task == 'EMO' and data_file == 'mosei_senti_data.pkl':
            raise ValueError("EMO task requires 'mosei_raw.pkl', cannot use 'mosei_senti_data.pkl'")
        
        self.num_classes = num_classes
        self.data_file = data_file
        super().__init__(
            data_path=data_path,
            split=split,
            aligned=aligned,
            max_seq_len=max_seq_len,
            normalize=normalize,
            modalities=modalities,
            task=task,
            require_all_modalities=require_all_modalities,
        )

    def _load_data(self) -> None:
        """Load MOSEI data from pickle file."""
        pkl_path = self.data_path / self.data_file
        if not pkl_path.exists():
            # Try MOSEI subdirectory
            pkl_path = self.data_path / 'MOSEI' / self.data_file

        if not pkl_path.exists():
            raise FileNotFoundError(f"MOSEI data file not found at {pkl_path}")

        data = load_pickle(pkl_path)
        split_data = data[self.split]

        # Extract features
        text_features = split_data['text']      # (N, seq_len, 300)
        audio_features = split_data.get('audio')  # (N, seq_len, audio_dim)
        video_features = split_data['vision']    # (N, seq_len, 35)
        labels = split_data['labels']            # (N, 1, 1) or (N,) or (N, 7)
        sample_ids = split_data.get('id', [f"{self.split}_{i}" for i in range(len(text_features))])

        # Process labels based on task
        if self.task == 'EMO':
            # EMO task: labels should be (N, 7), take argmax of last 6 for emotion
            if labels.ndim == 3:
                labels = labels.squeeze()
            if labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1):
                raise ValueError(f"EMO task requires multi-label data with 7 dimensions, got shape {labels.shape}")
            # Extract emotion labels (indices 1-6: Happiness, Sadness, Anger, Fear, Disgust, Surprise)
            emotion_labels = labels[:, 1:]  # (N, 6)
            labels = np.argmax(emotion_labels, axis=1)  # (N,) - get dominant emotion
            self.num_classes = 6
            label_names = self.EMOTION_LABELS
            self.target = self.EMOTION_LABELS
        else:
            # SEN task
            if labels.ndim == 3:
                labels = labels.squeeze()
            if labels.ndim == 2:
                if labels.shape[1] == 7:
                    # mosei_raw.pkl - take first dimension for sentiment
                    labels = labels[:, 0]
                else:
                    labels = labels.squeeze(-1)
            
            # Ensure labels is 1D array of floats
            labels = np.asarray(labels).flatten()
            
            # Get sentiment label names based on num_classes
            if self.num_classes == 7:
                label_names = self.SENTIMENT_LABELS_7
            elif self.num_classes == 5:
                label_names = self.SENTIMENT_LABELS_5
            elif self.num_classes == 3:
                label_names = self.SENTIMENT_LABELS_3
            else:
                label_names = self.SENTIMENT_LABELS_2
            self.target = label_names

        # Store dataset info
        self.info = DatasetInfo(
            name='MOSEI',
            task='emotion' if self.task == 'EMO' else 'sentiment',
            num_classes=self.num_classes,
            text_dim=300,
            audio_dim=audio_features.shape[-1] if audio_features is not None else 0,
            video_dim=35,
            max_seq_len=self.max_seq_len,
            label_names=label_names,
            target=self.target,
            language='en',
        )

        # Create samples
        for i in range(len(text_features)):
            sample_id = sample_ids[i] if isinstance(sample_ids, (list, np.ndarray)) else f"{self.split}_{i}"
            if isinstance(sample_id, np.ndarray):
                sample_id = str(sample_id.tolist())

            # Process text
            text = None
            if 'text' in self.modalities:
                text = self._process_features(text_features[i])

            # Process audio
            audio = None
            if 'audio' in self.modalities and audio_features is not None:
                audio = self._process_features(audio_features[i])

            # Process video
            video = None
            if 'video' in self.modalities:
                video = self._process_features(video_features[i])

            # Process label
            if self.task == 'EMO':
                label_idx = int(labels[i])
                label = torch.tensor(label_idx, dtype=torch.long)
                label_text = self.EMOTION_LABELS[label_idx]
            else:
                label = self._discretize_label(labels[i])
                label_idx = label.item()
                label_text = label_names[label_idx]

            self.samples.append(MultimodalSample(
                sample_id=str(sample_id),
                text=text,
                audio=audio,
                video=video,
                label=label,
                label_text=label_text,
                label_set=label_names,
                raw_text=None,
                metadata={'dataset': 'MOSEI', 'split': self.split},
            ))

    def _process_features(self, features: np.ndarray) -> torch.Tensor:
        """Process features with normalization and padding."""
        if self.normalize:
            features = self._normalize_features(features)
        if self.max_seq_len:
            features = self._pad_or_truncate(features, self.max_seq_len)
        return self._to_tensor(features)

    def _discretize_label(self, label: float) -> torch.Tensor:
        """Convert continuous sentiment to discrete classes."""
        if self.num_classes == 2:
            # Binary: negative (< 0) vs positive (>= 0)
            cls = 0 if label < 0 else 1
        elif self.num_classes == 3:
            # 3-class: negative, neutral, positive
            if label <= -1:
                cls = 0  # negative
            elif label < 1:
                cls = 1  # neutral
            else:
                cls = 2  # positive
        elif self.num_classes == 5:
            # 5-class: map [-3, 3] to [0, 4]
            if label < -2:
                cls = 0
            elif label < -1:
                cls = 1
            elif label < 1:
                cls = 2
            elif label < 2:
                cls = 3
            else:
                cls = 4
        elif self.num_classes == 7:
            # 7-class: [-3, 3] -> [0, 6]
            cls = int(np.clip(np.round(label) + 3, 0, 6))
        else:
            # Return raw value for regression
            return torch.tensor([label], dtype=torch.float32)

        return torch.tensor(cls, dtype=torch.long)

    def get_info(self) -> DatasetInfo:
        return self.info

    @classmethod
    def get_dataloaders(
        cls,
        data_path: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
    ) -> Dict[str, 'torch.utils.data.DataLoader']:
        """Create train/valid/test dataloaders."""
        from .base import get_dataloader

        dataloaders = {}
        for split in ['train', 'valid', 'test']:
            dataset = cls(data_path=data_path, split=split, **kwargs)
            dataloaders[split] = get_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
            )
        return dataloaders


class MOSEIRawDataset(MOSEIDataset):
    """
    MOSEI dataset loader for raw (unprocessed) data.

    Uses mosei_raw.pkl which contains more detailed features.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = 'train',
        **kwargs
    ):
        kwargs['data_file'] = 'mosei_raw.pkl'
        super().__init__(data_path=data_path, split=split, **kwargs)
