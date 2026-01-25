# dataloader/mimic.py
"""
MIMIC Dataset Loader.

Adapts the MIMIC static + timeseries data into the multi-modal interface
defined by BaseMultiModalDataset so it plugs into the shared training code.
"""

from typing import Any, Dict, List, Tuple
import pickle

import numpy as np
import torch

from .base import BaseMultiModalDataset


class MIMICDataset(BaseMultiModalDataset):
    """
    MIMIC Dataset for clinical outcome prediction.

    Features:
        - Timeseries: Episode time-series data
        - Static: Admission features repeated across the episode length

    Tasks:
        - MOR (Mortality): predicting in-hospital mortality
        - ICD9-X (ICD9 codes): predicting 20 different ICD9 diagnosis codes
    """

    MODALITY_KEYS = {"timeseries": "timeseries", "static": "static"}
    LABEL_KEY = "labels"

    BINARY_LABELS = ["negative", "positive"]

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        task: str = "MOR",
        normalize: List[str] = None,
        normalize_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
    ):
        if task not in ["MOR"] and not task.startswith("ICD9-"):
            raise ValueError(f"task must be 'MOR' or 'ICD9-X', got {task}")

        self.task = task
        self.num_classes = 2
        self.label_names = self.BINARY_LABELS

        # Parse task
        if task == "MOR":
            self.task_id = -1
            self.task_name = "Mortality"
        elif task.startswith("ICD9-"):
            try:
                self.task_id = int(task.split("-")[1])
                if not (0 <= self.task_id <= 19):
                    raise ValueError(f"ICD9 task ID must be 0-19, got {self.task_id}")
                self.task_name = f"ICD9-{self.task_id}"
            except (ValueError, IndexError):
                raise ValueError(f"Invalid task format: {task}. Use 'MOR' or 'ICD9-X' (X=0-19)")
        else:
            raise ValueError(f"Unknown task: {task}. Use 'MOR' or 'ICD9-X'")

        super().__init__(
            data_path=data_path,
            split=split,
            normalize=normalize or ["timeseries", "static"],
            normalize_stats=normalize_stats,
        )

    def _load_data(self) -> None:
        with open(self.data_path, "rb") as f:
            raw_data = pickle.load(f)

        X_t = np.asarray(raw_data["ep_tdata"], dtype=np.float32)  # (N, T, D_t)
        X_s = np.asarray(raw_data["adm_features_all"], dtype=np.float32)  # (N, D_s)

        if self.task_id < 0:
            adm_labels = np.asarray(raw_data["adm_labels_all"], dtype=np.float32)
            y = (adm_labels[:, 1] > 0).astype(int)
        else:
            y = np.asarray(raw_data["y_icd9"], dtype=int)[:, self.task_id]

        total_samples = len(X_s)
        split_10 = total_samples // 10
        split_20 = split_10 * 2

        if self.split == "valid":
            indices = np.arange(0, split_10)
        elif self.split == "test":
            indices = np.arange(split_10, split_20)
        elif self.split == "train":
            indices = np.arange(split_20, total_samples)
        else:
            raise ValueError(f"Invalid split: {self.split}")

        X_s_split = np.nan_to_num(X_s[indices], nan=0.0, posinf=0.0, neginf=0.0)
        X_t_split = np.nan_to_num(X_t[indices], nan=0.0, posinf=0.0, neginf=0.0)
        y_split = y[indices]

        timeseries_data: List[np.ndarray] = []
        static_data: List[np.ndarray] = []

        for ts_sample, st_sample in zip(X_t_split, X_s_split):
            ts_clean = np.asarray(ts_sample, dtype=np.float32)
            T = ts_clean.shape[0]
            st_seq = np.repeat(st_sample[np.newaxis, :], T, axis=0).astype(np.float32)
            timeseries_data.append(ts_clean)
            static_data.append(st_seq)

        self.data = {
            "timeseries": timeseries_data,
            "static": static_data,
            "labels": y_split.astype(int),
        }
        self.n_samples = len(y_split)

    def _process_label(self, raw_label: Any) -> torch.Tensor:
        """Convert raw label to tensor."""
        label = int(raw_label)
        if label not in [0, 1]:
            raise ValueError(f"Invalid label value: {label}. Expected 0 or 1.")
        return torch.tensor(label, dtype=torch.long)

    def get_task_name(self) -> str:
        """Get task name."""
        return self.task_name