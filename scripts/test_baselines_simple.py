import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path to import OmniFlow modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ContinuousFlow import ContinuousFlow
from DiscreteFlow import DiscreteFlow
from OmniFlow import FlowConfig, GeometryConfig


def get_dummy_data(batch_size=4, seq_len=20):
    # Dimensions from user request
    # use dims from humor.pkl
    dim_vis = 371
    dim_aud = 81
    dim_txt = 300

    vis = torch.randn(batch_size, seq_len, dim_vis)
    aud = torch.randn(batch_size, seq_len, dim_aud)
    txt = torch.randn(batch_size, seq_len, dim_txt)  # Continuous embedding

    # Pads (all valid for now)
    vis_pad = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    aud_pad = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    txt_pad = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # Dummy labels and others
    labels = torch.randint(0, 2, (batch_size, 1)).float()
    vm = torch.ones(batch_size, seq_len, dtype=torch.bool)
    am = torch.ones(batch_size, seq_len, dtype=torch.bool)
    tm = torch.ones(batch_size, seq_len, dtype=torch.bool)

    dataset = TensorDataset(vis, aud, txt, vm, am, tm, labels)
    loader = DataLoader(dataset, batch_size=batch_size)

    dims = {"vis": dim_vis, "aud": dim_aud, "txt": dim_txt}
    return loader, dims


def test_continuous(loader, dims, device):
    print("\n>>> Testing ContinuousFlow...")
    cfg = FlowConfig(
        measure_dim=256,
        d_model=32,  # Small d_model for speed
        n_layers=2,
        n_heads=4,
        t_max=1.0,
        t_gamma=1.0,
        normalize_by_geometry=True,
    )

    model = ContinuousFlow(dims, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    vis, aud, txt, _, _, _, _ = next(iter(loader))
    vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)

    # Forward & Loss
    loss_dict = model.compute_loss(vis, aud, txt)
    print("Loss keys:", loss_dict.keys())
    print("Total Loss:", loss_dict["total"].item())

    # Backward
    optimizer.zero_grad()
    loss_dict["total"].backward()
    optimizer.step()
    print("Backward pass successful.")

    # Representation
    rep = model.encode_representation(vis, aud, txt, t_star=1.0)
    print("Representation shape:", rep.shape)  # Should be (B, d_model * something)


def test_discrete(loader, dims, device):
    print("\n>>> Testing DiscreteFlow...")
    cfg = FlowConfig(
        measure_dim=None,  # Will be overridden by vocab_size
        d_model=32,
        n_layers=2,
        n_heads=4,
        t_max=1.0,
        t_gamma=1.0,
        normalize_by_geometry=True,
    )

    # Init model with small vocab
    vocab_size = 128
    model = DiscreteFlow(
        dims, cfg, quantizer_k=vocab_size, quantizer_mode="uniform"
    ).to(device)

    # Init codebooks (mock)
    model.init_codebooks(loader, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    vis, aud, txt, _, _, _, _ = next(iter(loader))
    vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)

    # Forward & Loss
    loss_dict = model.compute_loss(vis, aud, txt)
    print("Loss keys:", loss_dict.keys())
    print("Total Loss:", loss_dict["total"].item())

    # Backward
    optimizer.zero_grad()
    loss_dict["total"].backward()
    optimizer.step()
    print("Backward pass successful.")

    # Representation
    rep = model.encode_representation(vis, aud, txt, t_star=1.0)
    print("Representation shape:", rep.shape)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader, dims = get_dummy_data()

    try:
        test_continuous(loader, dims, device)
        test_discrete(loader, dims, device)
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback

        traceback.print_exc()
