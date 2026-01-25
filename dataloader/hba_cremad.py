# dataloader/hba_cremad.py

from typing import Any, Dict, List, Optional, Tuple

from dataloader.hba_base import HBABaseDataset


class CremadDataset(HBABaseDataset):
    """CREMA-D subset loader with canonical label ordering and metadata."""

    DATASET_ID = "cremad"
    TEXT_POLICY = "problem"

    TASK = "emotion_classification"
    LABEL_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]

    def __init__(
        self,
        feature_root: str,
        preprocessed_data_root: str,
        split: str = "train",
        require_triple: bool = True,
        normalize: Optional[List[str]] = None,
        normalize_stats: Optional[Dict[str, Tuple[Any, Any]]] = None,
    ):
        super().__init__(
            feature_root=feature_root,
            preprocessed_data_root=preprocessed_data_root,
            split=split,
            require_triple=require_triple,
            normalize=normalize,
            normalize_stats=normalize_stats,
        )

        self.task = self.TASK

        expected = set(self.LABEL_NAMES)
        actual = set(self.label_map.keys())
        if expected != actual:
            missing = expected - actual
            extra = actual - expected
            raise ValueError(
                "CREMA-D label mismatch after preprocessing. "
                f"Missing: {sorted(missing)} | Unexpected: {sorted(extra)}"
            )

        # Enforce canonical label order for downstream heads.
        self.label_map = {label: idx for idx, label in enumerate(self.LABEL_NAMES)}
        self.label_names = self.LABEL_NAMES
        self.num_classes = len(self.LABEL_NAMES)
