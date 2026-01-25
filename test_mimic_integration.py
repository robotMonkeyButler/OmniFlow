#!/usr/bin/env python
"""
Test script to verify MIMIC dataset works with dict-based architecture.
"""
import torch
from dataloader.mimic import MIMICDataset
from dataloader import create_collate_fn
from torch.utils.data import DataLoader
from ContinuousFlow import ContinuousFlow
from DiscreteFlow import DiscreteFlow
from OmniFlow import FlowConfig, GeometryConfig


def test_mimic_loading():
    """Test MIMIC dataset loads correctly with semantic modality names."""
    print("=" * 70)
    print("Testing MIMIC Dataset Loading")
    print("=" * 70)
    
    # Load dataset
    dataset = MIMICDataset(
        data_path="./im.pk",
        split="train",
        normalize=["timeseries", "static"]
    )
    
    print(f"✓ Dataset loaded with {len(dataset)} samples")
    print(f"✓ Modality keys: {dataset.get_modality_names()}")
    print(f"✓ Dimensions: {dataset.get_dims()}")
    
    # Test __getitem__ - returns tuple (timeseries, static, label)
    sample = dataset[0]
    print(f"✓ Sample is tuple with {len(sample)} elements")
    assert len(sample) == 3  # timeseries, static, label
    timeseries, static, label = sample
    print(f"✓ Timeseries shape: {timeseries.shape}")
    print(f"✓ Static shape: {static.shape}")
    print(f"✓ Label type: {type(label)}")
    
    return dataset


def test_collate_fn(dataset):
    """Test dict-based collate function."""
    print("\n" + "=" * 70)
    print("Testing Dict-based Collate Function")
    print("=" * 70)
    
    collate_fn = create_collate_fn(dataset.get_modality_names())
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    batch = next(iter(loader))
    print(f"✓ Batch keys: {batch.keys()}")
    
    # Check required keys
    assert "timeseries" in batch
    assert "static" in batch
    assert "timeseries_pad" in batch
    assert "static_pad" in batch
    assert "labels" in batch
    
    print(f"✓ timeseries shape: {batch['timeseries'].shape}")
    print(f"✓ static shape: {batch['static'].shape}")
    print(f"✓ timeseries_pad shape: {batch['timeseries_pad'].shape}")
    print(f"✓ static_pad shape: {batch['static_pad'].shape}")
    print(f"✓ labels shape: {batch['labels'].shape}")
    
    return batch


def test_continuous_flow(batch, dims):
    """Test ContinuousFlow with dict batch."""
    print("\n" + "=" * 70)
    print("Testing ContinuousFlow with Dict Batch")
    print("=" * 70)
    
    geo_cfg = GeometryConfig(
        p_min=-1.0, p_max=1.0, eps=0.01,
        m_min=0.0, m_max=1.0, w_max=10.0,
        prior_scale=1.0, prior_type="gaussian"
    )
    
    cfg = FlowConfig(
        measure_dim=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        share_layers=False,
        adapter_hidden=128,
        mask_ratio={"timeseries": 0.5, "static": 0.5},
        span_len={"timeseries": 4, "static": 2},
        w_masked=1.0,
        w_visible=1.0,
        t_max=1.0,
        t_gamma=1.0,
        normalize_by_geometry=True,
        txt_usage_weight=0.0,
        geometry=geo_cfg
    )
    
    model = ContinuousFlow(dims, cfg)
    print(f"✓ Model created with modalities: {model.modality_names}")
    
    # Forward pass
    loss_dict = model.compute_loss(batch)
    print(f"✓ Loss computed: {loss_dict.keys()}")
    print(f"  - total: {loss_dict['total'].item():.4f}")
    print(f"  - loss_timeseries: {loss_dict.get('loss_timeseries', 0)}")
    print(f"  - loss_static: {loss_dict.get('loss_static', 0)}")
    
    # Backward pass
    loss_dict["total"].backward()
    print("✓ Backward pass successful")
    
    # Representation extraction
    model.eval()
    with torch.no_grad():
        rep = model.encode_representation(batch, t_star=1.0)
    print(f"✓ Representation shape: {rep.shape}")
    
    return model


def test_discrete_flow(batch, dims):
    """Test DiscreteFlow with dict batch."""
    print("\n" + "=" * 70)
    print("Testing DiscreteFlow with Dict Batch")
    print("=" * 70)
    
    geo_cfg = GeometryConfig(
        p_min=-1.0, p_max=1.0, eps=0.01,
        m_min=0.0, m_max=1.0, w_max=10.0,
        prior_scale=1.0, prior_type="gaussian"
    )
    
    cfg = FlowConfig(
        measure_dim=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        share_layers=False,
        adapter_hidden=128,
        mask_ratio={"timeseries": 0.5, "static": 0.5},
        span_len={"timeseries": 4, "static": 2},
        w_masked=1.0,
        w_visible=1.0,
        t_max=1.0,
        t_gamma=1.0,
        normalize_by_geometry=True,
        txt_usage_weight=0.0,
        geometry=geo_cfg
    )
    
    model = DiscreteFlow(dims, cfg, quantizer_k=256)
    print(f"✓ Model created with modalities: {model.modality_names}")
    
    # Initialize codebooks (need a small loader)
    from torch.utils.data import DataLoader
    from dataloader.mimic import MIMICDataset
    from dataloader import create_collate_fn
    
    mini_dataset = MIMICDataset(
        data_path="./im.pk",
        split="train",
        normalize=["timeseries", "static"]
    )
    collate_fn = create_collate_fn(mini_dataset.get_modality_names())
    mini_loader = DataLoader(mini_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    model.init_codebooks(mini_loader, "cpu")
    print("✓ Codebooks initialized")
    
    # Forward pass
    loss_dict = model.compute_loss(batch)
    print(f"✓ Loss computed: {loss_dict.keys()}")
    print(f"  - total: {loss_dict['total'].item():.4f}")
    
    # Backward pass
    loss_dict["total"].backward()
    print("✓ Backward pass successful")
    
    # Representation extraction
    model.eval()
    with torch.no_grad():
        rep = model.encode_representation(batch, t_star=1.0)
    print(f"✓ Representation shape: {rep.shape}")
    
    return model


def main():
    print("\n" + "=" * 70)
    print("MIMIC Integration Test Suite")
    print("=" * 70)
    
    try:
        # Test 1: Dataset loading
        dataset = test_mimic_loading()
        
        # Test 2: Collate function
        batch = test_collate_fn(dataset)
        
        # Test 3: ContinuousFlow
        dims = dataset.get_dims()
        test_continuous_flow(batch, dims)
        
        # Test 4: DiscreteFlow
        test_discrete_flow(batch, dims)
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ MIMIC dataset loads with semantic modality names (timeseries/static)")
        print("  ✓ Dict-based collate function works correctly")
        print("  ✓ ContinuousFlow accepts dict batches")
        print("  ✓ DiscreteFlow accepts dict batches")
        print("  ✓ Forward and backward passes work")
        print("  ✓ Representation extraction works")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
