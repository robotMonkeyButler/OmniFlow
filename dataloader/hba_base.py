# dataloader/hba_base.py

import json
import os
import pickle
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch

from dataloader.base import BaseTriModalDataset


class HBABaseDataset(BaseTriModalDataset, ABC):
    """
    Base class for a single sub-dataset inside keentomato/human_behavior_atlas.

    Subclasses define:
      - DATASET_ID: str (e.g., "cremad")
      - TEXT_POLICY: {"problem", "texts", "both"}
    
    Preprocessed data pickle format:
    {
        "train": {
            "text_embeddings": [array(L1, 300), array(L2, 300), ...],
            "video_embed_paths": ["pose/...", "pose/...", ...],
            "audio_embed_paths": ["opensmile/...", "opensmile/...", ...],
            "labels": ["happy", "sad", ...]
        },
        "valid": {...},
        "test": {...}
    }
    """

    # ---- must be set by subclass ----
    DATASET_ID: str = ""  # e.g., "cremad"
    TEXT_POLICY: str = "problem"  # "problem" | "texts" | "both"

    MODALITY_KEYS = {"vis": "vision", "aud": "audio", "txt": "text"}
    LABEL_KEY = "labels"
    ALLOWED_TEXT_POLICIES = {"problem", "texts", "both"}

    def __init__(
        self,
        feature_root: str,
        preprocessed_data_root: str,
        split: str = "train",
        require_triple: bool = True,
        normalize: List[str] = None,
        normalize_stats: Dict[str, Tuple[Any, Any]] = None,
    ):
        self.feature_root = feature_root
        self.preprocessed_data_root = preprocessed_data_root
        self.require_triple = require_triple

        if not self.DATASET_ID:
            raise ValueError("Subclass must set DATASET_ID, e.g., 'cremad'.")
        if self.TEXT_POLICY not in self.ALLOWED_TEXT_POLICIES:
            raise ValueError(
                f"TEXT_POLICY must be one of {sorted(self.ALLOWED_TEXT_POLICIES)}, got {self.TEXT_POLICY}."
            )

        self._norm_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._clip_stats: Dict[str, float] = {}

        # placeholder data_path to satisfy BaseTriModalDataset
        super().__init__(
            data_path="hf://keentomato/human_behavior_atlas",
            split=split,
            normalize=normalize,
            normalize_stats=normalize_stats,
        )

    def _resolve_feature_path(self, rel_path: Optional[str]) -> Optional[str]:
        if not rel_path:
            return None
        if os.path.isabs(rel_path):
            return rel_path if os.path.isfile(rel_path) else None
        full_path = os.path.join(self.feature_root, rel_path)
        return full_path if os.path.isfile(full_path) else None

    def _load_feat_pt(self, rel_path: Optional[str]) -> torch.Tensor:
        """Lazy load .pt feature; fallback to zeros if missing."""
        if not rel_path:
            return torch.zeros(1, 1, dtype=torch.float32)
        full_path = self._resolve_feature_path(rel_path)
        if not full_path:
            return torch.zeros(1, 1, dtype=torch.float32)
        return torch.load(full_path, map_location="cpu")

    # ============================================================
    # Normalization / clipping overrides
    # ============================================================

    def apply_clipping(self, stats: Dict[str, float]) -> None:
        """Store per-modality clip limits for lazy application."""
        self._clip_stats = {}
        if not stats:
            return
        for key, clip in stats.items():
            try:
                clip_val = float(clip)
            except (TypeError, ValueError):
                continue
            if clip_val <= 0 or not torch.isfinite(torch.tensor(clip_val)):
                continue
            self._clip_stats[key] = clip_val

    def apply_normalization(
        self, stats: Dict[str, Tuple[Any, Any]]
    ) -> None:
        """Store normalization tensors for lazy application in __getitem__."""
        self._norm_stats = {}
        if not stats:
            return

        for key, pair in stats.items():
            if not pair or len(pair) != 2:
                continue
            mean, std = pair
            mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
            std_tensor = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)
            if mean_tensor.ndim != 1 or std_tensor.ndim != 1:
                continue
            if mean_tensor.shape != std_tensor.shape:
                continue
            self._norm_stats[key] = (mean_tensor, std_tensor)

    # ============================================================
    # Core loading
    # ============================================================

    def _load_data(self) -> None:
        """Load preprocessed data for this dataset/split from unified data.pkl."""
        dataset_dir = os.path.join(self.preprocessed_data_root, self.DATASET_ID)
        data_path = os.path.join(dataset_dir, "data.pkl")
        label_map_path = os.path.join(dataset_dir, "label_map.json")

        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(
                f"Preprocessed directory not found: {dataset_dir}. Run scripts/preprocess_hba.py first."
            )

        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"Missing preprocessed file: {data_path}. Run the preprocessing script first."
            )

        with open(data_path, "rb") as f:
            all_data = pickle.load(f)

        if self.split not in all_data:
            raise ValueError(
                f"Split '{self.split}' not found in {data_path}. Available: {list(all_data.keys())}"
            )

        split_data = all_data[self.split]

        if os.path.isfile(label_map_path):
            with open(label_map_path, "r", encoding="utf-8") as f:
                self.label_map: Dict[str, int] = json.load(f)
        else:
            # Fallback: rebuild from labels
            labels = set(split_data["labels"])
            self.label_map = {lb: i for i, lb in enumerate(sorted(labels))}

        # Store text embeddings directly
        self.text_embeddings = split_data["text_embeddings"]
        self.video_embed_paths = split_data["video_embed_paths"]
        self.audio_embed_paths = split_data["audio_embed_paths"]
        self.labels = split_data["labels"]

        # Apply require_triple filter if needed
        if self.require_triple:
            kept_indices = [
                i
                for i, (v, a) in enumerate(
                    zip(self.video_embed_paths, self.audio_embed_paths)
                )
                if v and a
            ]
            self.text_embeddings = [self.text_embeddings[i] for i in kept_indices]
            self.video_embed_paths = [self.video_embed_paths[i] for i in kept_indices]
            self.audio_embed_paths = [self.audio_embed_paths[i] for i in kept_indices]
            self.labels = [self.labels[i] for i in kept_indices]

        self.n_samples = len(self.labels)
        self.label_names = sorted(self.label_map.keys())
        self.num_classes = len(self.label_map)

        # Minimal data placeholder to satisfy BaseTriModalDataset APIs
        self.data = {"labels": self.labels}

    # ============================================================
    # Text encoding
    # ============================================================

    def _encode_text(self, idx: int) -> torch.Tensor:
        """Convert pre-computed text embedding to torch tensor."""
        embedding = self.text_embeddings[idx]
        if isinstance(embedding, torch.Tensor):
            return embedding.clone()
        return torch.from_numpy(embedding).float()

    def _postprocess_modality(self, tensor: torch.Tensor, modality: str) -> torch.Tensor:
        """Apply stored clipping and normalization lazily."""
        if modality in self._clip_stats:
            clip = self._clip_stats[modality]
            tensor = torch.nan_to_num(
                tensor, nan=0.0, posinf=clip, neginf=-clip
            )
            tensor = torch.clamp(tensor, min=-clip, max=clip)

        if modality in self._norm_stats:
            mean, std = self._norm_stats[modality]
            if tensor.shape[-1] == mean.numel():
                mean = mean.to(tensor.device)
                std = std.to(tensor.device)
                tensor = (tensor - mean) / std

        return tensor

    # ============================================================
    # Required overrides from BaseTriModalDataset
    # ============================================================

    def _process_label(self, raw_label: Any) -> torch.Tensor:
        return torch.tensor(self.label_map[str(raw_label)], dtype=torch.long)

    def __getitem__(self, idx: int):
        vis = self._load_feat_pt(self.video_embed_paths[idx])
        vis = self._postprocess_modality(vis, "vis")
        aud = self._load_feat_pt(self.audio_embed_paths[idx])
        aud = self._postprocess_modality(aud, "aud")
        txt = self._encode_text(idx)
        txt = self._postprocess_modality(txt, "txt")
        label = self._process_label(self.labels[idx])
        return vis, aud, txt, label

    def get_meta(self, idx: int) -> Dict[str, Any]:
        return {
            "label_str": self.labels[idx],
            "video_path": self.video_embed_paths[idx],
            "audio_path": self.audio_embed_paths[idx],
        }

    def get_dims(self) -> Dict[str, int]:
        """Return feature dimensions for all modalities."""
        txt_dim = int(self.text_embeddings[0].shape[-1]) if self.n_samples > 0 else 0

        def _feat_dim(rel_path: Optional[str]) -> int:
            full_path = self._resolve_feature_path(rel_path)
            if not full_path:
                return 1
            feat = torch.load(full_path, map_location="cpu")
            return int(feat.shape[-1]) if feat.ndim >= 2 else 1

        vis_dim = _feat_dim(self.video_embed_paths[0] if self.n_samples > 0 else None)
        aud_dim = _feat_dim(self.audio_embed_paths[0] if self.n_samples > 0 else None)

        return {"vis": vis_dim, "aud": aud_dim, "txt": txt_dim} 
