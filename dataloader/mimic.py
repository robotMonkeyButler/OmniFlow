# dataloader/mimic.py
"""
MIMIC Dataset Loader.

Adapts the MIMIC static + timeseries data into the tri-modal interface
defined by BaseTriModalDataset so it plugs into the shared training code.
"""

from typing import Any, Dict, List, Tuple
import pickle

import numpy as np
import torch

from .base import BaseTriModalDataset


class MIMICDataset(BaseTriModalDataset):
    """
    MIMIC Dataset for clinical outcome prediction.

    Features:
        - Timeseries: Episode time-series data (fed to vis branch)
        - Static: Admission features (fed to aud/txt branches as a length-1 sequence)

    Tasks:
        - MOR (Mortality): predicting in-hospital mortality
        - ICD9-X (ICD9 codes): predicting 20 different ICD9 diagnosis codes
    """

    MODALITY_KEYS = {"static": "static", "timeseries": "timeseries"}
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

        X_t = raw_data["ep_tdata"].copy()  # (N, T, D_t)
        X_s = raw_data["adm_features_all"].copy()  # (N, D_s)

        X_t[np.isinf(X_t)] = 0
        X_t[np.isnan(X_t)] = 0
        X_s[np.isinf(X_s)] = 0
        X_s[np.isnan(X_s)] = 0

        X_s_mean = np.average(X_s, axis=0)
        X_s_std = np.std(X_s, axis=0)
        X_s_std[X_s_std < 1e-6] = 1.0

        X_t_mean = np.average(X_t, axis=(0, 1))
        X_t_std = np.std(X_t, axis=(0, 1))
        X_t_std[X_t_std < 1e-6] = 1.0

        X_s = (X_s - X_s_mean) / X_s_std
        for i in range(len(X_t)):
            X_t[i] = (X_t[i] - X_t_mean) / X_t_std

        if self.task_id < 0:
            y = np.zeros(len(X_s), dtype=int)
            adm_labels = raw_data["adm_labels_all"]
            for i in range(len(adm_labels)):
                if adm_labels[i][1] > 0:
                    y[i] = 1
        else:
            y = raw_data["y_icd9"][:, self.task_id].copy().astype(int)

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

        X_s_split = X_s[indices]
        X_t_split = X_t[indices]
        y_split = y[indices]

        self.data = {
            "timeseries": [X_t_split[i] for i in range(len(X_t_split))],
            "static": [X_s_split[i] for i in range(len(X_s_split))],
            "labels": y_split,
        }
        self.n_samples = len(y_split)

        if self.normalize_stats:
            self.apply_normalization(self.normalize_stats)

    def _process_label(self, raw_label: Any) -> torch.Tensor:
        """Convert raw label to tensor."""
        label = int(raw_label)
        if label not in [0, 1]:
            raise ValueError(f"Invalid label value: {label}. Expected 0 or 1.")
        return torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (timeseries, static, label)."""
        ts = torch.tensor(self.data["timeseries"][idx], dtype=torch.float32)
        st = torch.tensor(self.data["static"][idx], dtype=torch.float32)
        label = self._process_label(self.data["labels"][idx])
        return ts, st, label

    def get_dims(self) -> Dict[str, int]:
        return {
            "timeseries": int(self.data["timeseries"][0].shape[-1]),
            "static": int(self.data["static"][0].shape[-1]),
        }

    def compute_normalize_stats(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        stats = {}
        for key in self.normalize_keys:
            if key == "timeseries":
                all_data = np.concatenate(self.data["timeseries"], axis=0)
            elif key == "static":
                all_data = np.stack(self.data["static"], axis=0)
            else:
                raise ValueError(f"Unknown modality: {key}")
            mean = all_data.mean(axis=0)
            std = all_data.std(axis=0)
            std[std < 1e-6] = 1.0
            stats[key] = (mean, std)
        return stats

    def apply_normalization(self, stats: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        for key, (mean, std) in stats.items():
            if key == "timeseries":
                for i in range(len(self.data["timeseries"])):
                    self.data["timeseries"][i] = (self.data["timeseries"][i] - mean) / std
            elif key == "static":
                for i in range(len(self.data["static"])):
                    self.data["static"][i] = (self.data["static"][i] - mean) / std
        self.normalize_stats = stats

    def get_label_names(self) -> List[str]:
        """Get human-readable label names."""
        return self.label_names

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.num_classes

    def get_task_name(self) -> str:
        """Get task name."""
        return self.task_name


def collate_fn_mimic(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Collate for MIMIC (only two real modalities):
    - vis := timeseries (padded)
    - aud := static (length-1 sequence)
    - txt := static (same tensor, placeholder to satisfy tri-modal interface)
    Returns vis, aud, txt, vis_pad, aud_pad, txt_pad, labels.
    """
    ts_list, st_list, y_list = zip(*batch)

    lens = torch.tensor([t.size(0) for t in ts_list])
    vis = torch.nn.utils.rnn.pad_sequence(ts_list, batch_first=True)
    maxT = vis.size(1)
    B = vis.size(0)
    ar = torch.arange(maxT).unsqueeze(0).expand(B, maxT)
    vis_pad = ar >= lens.unsqueeze(1)

    static = torch.stack(st_list, dim=0)  # (B, D_s)
    aud = static.unsqueeze(1)  # (B, 1, D_s)
    txt = static.unsqueeze(1)  # reuse static as text placeholder
    aud_pad = torch.zeros(B, 1, dtype=torch.bool)
    txt_pad = torch.zeros(B, 1, dtype=torch.bool)

    labels = torch.stack(y_list)
    return vis, aud, txt, vis_pad, aud_pad, txt_pad, labels