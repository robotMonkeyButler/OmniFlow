# dataloader/__init__.py
"""
Unified dataloader module for multimodal datasets.

Usage:
    from dataloader import get_dataloaders

    train_loader, val_loader, test_loader, dims = get_dataloaders(cfg, dataset_path)
"""

from typing import Dict, List, Optional, Tuple, Type
from torch.utils.data import DataLoader

from .base import BaseMultiModalDataset, BaseTriModalDataset, collate_fn, collate_fn_with_names, create_collate_fn, load_raw_data, split_data
from .mosei import MOSEIDataset, MOSIDataset
from .urfunny import URFunnyDataset
from .iemocap import IEMOCAPDataset
from .mustard import MUStARDDataset
from .mimic import MIMICDataset


# ============================================================
# Dataset Registry
# ============================================================

DATASET_REGISTRY: Dict[str, Type] = {
    "mosei": MOSEIDataset,
    "mosi": MOSIDataset,
    "urfunny": URFunnyDataset,
    "iemocap": IEMOCAPDataset,
    "mustard": MUStARDDataset,
    "mimic": MIMICDataset,
}


def register_dataset(name: str, dataset_class: Type[BaseMultiModalDataset]) -> None:
    """Register a new dataset class."""
    DATASET_REGISTRY[name.lower()] = dataset_class


# ============================================================
# Main Entry Point
# ============================================================


def get_dataloaders(
    cfg: Dict,
    dataset_path: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train, validation, and test dataloaders from config.

    Args:
        cfg: Configuration dictionary with keys:
            - dataset.name: Dataset name (mosei, mosi, urfunny, iemocap, generic)
            - dataset.task: Task type (for MOSEI: 'SEN' or 'EMO')
            - dataset.num_classes: Number of classes
            - dataset.normalize: List of modalities to normalize
            - dataset.num_workers: Number of data loading workers
            - dataset.pin_memory: Whether to pin memory
            - training.batch_size: Batch size
        dataset_path: Path to dataset file (.pkl or .npz)

    Returns:
        train_loader, val_loader, test_loader, dims
    """
    # Extract config
    ds_cfg = cfg.get("dataset", {})
    dataset_name = ds_cfg.get("name", "generic").lower()
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    num_workers = ds_cfg.get("num_workers", 0)
    pin_memory = ds_cfg.get("pin_memory", False)
    normalize = ds_cfg.get("normalize", ["vis", "aud", "text"])

    # Build dataset kwargs based on dataset type
    ds_kwargs = {
        "data_path": dataset_path,
        "normalize": normalize,
    }

    # Add dataset-specific kwargs
    if dataset_name in ["mosei"]:
        ds_kwargs["task"] = ds_cfg.get("task", "SEN")
        ds_kwargs["num_classes"] = ds_cfg.get("num_classes", 2)
    elif dataset_name in ["mosi"]:
        ds_kwargs["num_classes"] = ds_cfg.get("num_classes", 2)
    elif dataset_name in ["iemocap"]:
        ds_kwargs["num_classes"] = ds_cfg.get("num_classes", 4)
    elif dataset_name in ["mustard"]:
        ds_kwargs["task"] = ds_cfg.get("task", "SAR")
    elif dataset_name in ["mimic"]:
        # mimic input modalities are not default vis/aud/txt
        ds_kwargs["normalize"] = ds_cfg.get("normalize", ["timeseries", "static"])
        ds_kwargs["task"] = ds_cfg.get("task", "MOR")
    elif dataset_name == "generic":
        # Get modality keys from config
        modality_keys_cfg = ds_cfg.get("modality_keys", {})
        if dataset_name in modality_keys_cfg:
            ds_kwargs["modality_keys"] = modality_keys_cfg[dataset_name]
        ds_kwargs["label_key"] = ds_cfg.get("label_key", "labels")

    # Ensure normalize in kwargs reflects any dataset-specific override
    # (but don't override MIMIC's special normalize which was already set above)
    ds_kwargs["normalize"] = normalize

    # Get dataset class from registry
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")
    dataset_class = DATASET_REGISTRY[dataset_name]

    # Create train dataset first to compute normalization stats
    train_ds = dataset_class(split="train", **ds_kwargs)

    # Compute and apply clipping to handle inf values
    clip_stats = train_ds.compute_clip_stats()
    train_ds.apply_clipping(clip_stats)

    # Compute normalization stats from training data
    if ds_kwargs.get("normalize"):
        norm_stats = train_ds.compute_normalize_stats()
        train_ds.apply_normalization(norm_stats)
        print(f"Computed normalization stats from training data for: {ds_kwargs['normalize']}")
    else:
        norm_stats = {}

    # Create val and test datasets with same normalization stats
    ds_kwargs["normalize_stats"] = norm_stats
    val_ds = dataset_class(split="valid", **ds_kwargs)
    test_ds = dataset_class(split="test", **ds_kwargs)

    ds_kwargs["clip_stats"] = clip_stats 
    val_ds.apply_clipping(clip_stats)
    test_ds.apply_clipping(clip_stats)

    # Apply normalization to val and test
    if norm_stats:
        val_ds.apply_normalization(norm_stats)
        test_ds.apply_normalization(norm_stats)

    # Get feature dimensions
    dims = train_ds.get_dims()
    print(f"Feature dimensions: {dims}")
    print(
        f"Dataset: {dataset_name} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}"
    )

    # Create collate function with dataset's modality names
    modality_names = train_ds.get_modality_names()
    chosen_collate = create_collate_fn(modality_names)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=chosen_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=chosen_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=chosen_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, dims


# Alias for backwards compatibility
setup_data = get_dataloaders


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Main entry
    "get_dataloaders",
    "setup_data",
    # Base classes
    "BaseMultiModalDataset",
    "BaseTriModalDataset",  # Backward compatibility
    "collate_fn",
    "collate_fn_with_names",
    "create_collate_fn",
    # Dataset classes
    "MOSEIDataset",
    "MOSIDataset",
    "URFunnyDataset",
    "IEMOCAPDataset",
    "GenericDataset",
    "MUStARDDataset",
    "MIMICDataset",
    # Registry
    "DATASET_REGISTRY",
    "register_dataset",
    # Utils
    "load_raw_data",
    "split_data",
]
