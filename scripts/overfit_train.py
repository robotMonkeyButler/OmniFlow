#!/usr/bin/env python3
"""
Overfit Test Script
-------------------
用于验证模型是否有基本的学习能力。

核心思想：
- 使用非常小的数据集（前 N 个样本）
- 同一批数据同时作为 train 和 val
- 如果模型能够快速 overfit（loss → 0），说明模型架构没问题
- 如果不能，说明模型可能有 bug
"""

import os
import sys
import time
import argparse
import pickle

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from OmniFlow import OmniFlow, FlowConfig, GeometryConfig


# ============================================================
# Config Loading (from train.py)
# ============================================================


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_flow_config(cfg: dict) -> FlowConfig:
    """从 YAML 配置构建 FlowConfig"""
    geo_cfg = GeometryConfig(
        p_min=cfg["geometry"]["p_min"],
        p_max=cfg["geometry"]["p_max"],
        eps=cfg["geometry"]["eps"],
        m_min=cfg["geometry"]["m_min"],
        m_max=cfg["geometry"]["m_max"],
        w_max=cfg["geometry"]["w_max"],
        prior_scale=cfg["geometry"]["prior_scale"],
        prior_type=cfg["geometry"]["prior_type"],
    )
    return FlowConfig(
        measure_dim=cfg["model"]["measure_dim"],
        d_model=cfg["model"]["d_model"],
        n_layers=(
            cfg["model"]["n_layers"]
            if "n_layers" in cfg["model"]
            else cfg["model"]["num_layers"]
        ),
        n_heads=cfg["model"]["nhead"],
        dropout=cfg["model"]["dropout"],
        share_layers=cfg["model"]["share_layers"],
        adapter_hidden=cfg["model"]["adapter_hidden"],
        mask_ratio=cfg["training"]["stage1"]["mask_ratio"],
        span_len=cfg["masking"]["span_len"],
        w_masked=cfg["masking"]["w_masked"],
        w_visible=cfg["masking"]["w_visible"],
        t_max=cfg["flow"]["t_max"],
        t_gamma=cfg["flow"]["t_gamma"],
        normalize_by_geometry=cfg["flow"]["normalize_by_geometry"],
        txt_usage_weight=cfg["text_adapter"]["txt_usage_weight"],
        geometry=geo_cfg,
    )


# ============================================================
# Dataset & Collate
# ============================================================


class TriModalDataset(Dataset):
    def __init__(self, data_dict, vis_key, aud_key, txt_key, label_key="labels"):
        self.data = data_dict
        self.vis_key, self.aud_key, self.txt_key = vis_key, aud_key, txt_key
        self.label_key = label_key
        n = len(data_dict[txt_key])
        assert len(data_dict[vis_key]) == n and len(data_dict[aud_key]) == n

    def __len__(self):
        return len(self.data[self.txt_key])

    def __getitem__(self, idx):
        vis = torch.tensor(self.data[self.vis_key][idx], dtype=torch.float32)
        aud = torch.tensor(self.data[self.aud_key][idx], dtype=torch.float32)
        txt = torch.tensor(self.data[self.txt_key][idx], dtype=torch.float32)
        y = np.asarray(self.data[self.label_key][idx])
        label = (
            torch.from_numpy(y).argmax().long()
            if y.ndim > 0 and y.size > 1
            else torch.tensor(int(y.flat[0]), dtype=torch.long)
        )
        return vis, aud, txt, label


def collate_fn(batch):
    vis_list, aud_list, txt_list, y_list = zip(*batch)
    lens = {
        "vis": torch.tensor([v.size(0) for v in vis_list]),
        "aud": torch.tensor([a.size(0) for a in aud_list]),
        "txt": torch.tensor([t.size(0) for t in txt_list]),
    }
    vis = pad_sequence(vis_list, batch_first=True)
    aud = pad_sequence(aud_list, batch_first=True)
    txt = pad_sequence(txt_list, batch_first=True)

    maxT = max(vis.size(1), aud.size(1), txt.size(1))
    vis = F.pad(vis, (0, 0, 0, maxT - vis.size(1)))
    aud = F.pad(aud, (0, 0, 0, maxT - aud.size(1)))
    txt = F.pad(txt, (0, 0, 0, maxT - txt.size(1)))

    B = vis.size(0)
    ar = torch.arange(maxT).unsqueeze(0).expand(B, maxT)
    vis_pad = ar >= lens["vis"].unsqueeze(1)
    aud_pad = ar >= lens["aud"].unsqueeze(1)
    txt_pad = ar >= lens["txt"].unsqueeze(1)

    return vis, aud, txt, vis_pad, aud_pad, txt_pad, torch.stack(y_list)


# ============================================================
# Data Loading
# ============================================================


def load_data(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif path.endswith(".npz"):
        data = dict(np.load(path, allow_pickle=True))
    else:
        raise ValueError("Use .pkl or .npz")

    if "train" in data:
        return data["train"], data["valid"], data["test"]
    if "train_data" in data:
        return data["train_data"], data["valid_data"], data["test_data"]
    if "trains" in data:
        return data["trains"], data["valid"], data["test"]

    # Auto split
    n = len(next(iter(data.values())))
    n_tr, n_val = int(n * 0.8), int(n * 0.1)
    return (
        {k: v[:n_tr] for k, v in data.items()},
        {k: v[n_tr : n_tr + n_val] for k, v in data.items()},
        {k: v[n_tr + n_val :] for k, v in data.items()},
    )


def subset_data(data_dict, n_samples):
    """取前 n_samples 个样本"""
    return {k: v[:n_samples] for k, v in data_dict.items()}


def infer_dims(data_dict, vis_key, aud_key, txt_key):
    return {
        "vis": int(data_dict[vis_key][0].shape[-1]),
        "aud": int(data_dict[aud_key][0].shape[-1]),
        "txt": int(data_dict[txt_key][0].shape[-1]),
    }


# ============================================================
# Training Functions
# ============================================================


def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    metrics = {"total": 0, "vis": 0, "aud": 0, "txt": 0}
    n = 0

    for vis, aud, txt, vm, am, tm, _ in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)

        optimizer.zero_grad()
        loss_dict = model.compute_loss(vis, aud, txt, vm, am, tm)
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k in metrics:
            metrics[k] += loss_dict[f"loss_{k}" if k != "total" else k].item()
        n += 1

    return {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    metrics = {"total": 0, "vis": 0, "aud": 0, "txt": 0}
    n = 0

    for vis, aud, txt, vm, am, tm, _ in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)
        loss_dict = model.compute_loss(vis, aud, txt, vm, am, tm)
        for k in metrics:
            metrics[k] += loss_dict[f"loss_{k}" if k != "total" else k].item()
        n += 1

    return {k: v / n for k, v in metrics.items()}


def save_overfit_plot(history, save_path):
    """保存 overfit 测试的 loss 曲线"""
    plt.figure(figsize=(12, 4))

    # Total loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_total"], label="train", marker="o", markersize=2)
    plt.plot(history["val_total"], label="val", marker="x", markersize=2)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Overfit Test - Total Loss")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    # Per-modality loss
    plt.subplot(1, 2, 2)
    for k in ["vis", "aud", "txt"]:
        plt.plot(
            history[f"train_{k}"],
            label=f"train_{k}",
            linestyle="-",
            marker="o",
            markersize=2,
        )
    for k in ["vis", "aud", "txt"]:
        plt.plot(
            history[f"val_{k}"],
            label=f"val_{k}",
            linestyle="--",
            marker="x",
            markersize=2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Overfit Test - Per-Modality Loss")
    plt.legend(fontsize=8)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Plot saved to: {save_path}")


# ============================================================
# Main Overfit Test
# ============================================================


def run_overfit_test(args):
    print("=" * 70)
    print("🧪 OVERFIT TEST")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Num samples: {args.n_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    cfg = load_config(args.config)

    # Load data
    trains, valid, test = load_data(args.dataset_path)
    keys = cfg["dataset"]["modality_keys"][cfg["dataset"]["name"]]

    # Take subset
    print(f"\n📊 Original train size: {len(trains[keys['txt']])}")
    subset = subset_data(trains, args.n_samples)
    print(f"📊 Using subset size: {len(subset[keys['txt']])}")

    # Normalize (copy from train.py)
    for k in [keys["vis"], keys["aud"]]:
        all_data = np.concatenate(subset[k], axis=0)
        mean, std = all_data.mean(0), all_data.std(0)
        std[std < 1e-6] = 1.0
        for i in range(len(subset[k])):
            subset[k][i] = (subset[k][i] - mean) / std

    # Create dataloaders (same data for train and val)
    ds = TriModalDataset(subset, keys["vis"], keys["aud"], keys["txt"])
    train_loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    print(f"📊 Batches per epoch: {len(train_loader)}")

    # Infer dims and build model
    dims = infer_dims(subset, keys["vis"], keys["aud"], keys["txt"])
    print(f"📐 Dims: {dims}")

    flow_cfg = build_flow_config(cfg)

    # 对 overfit test 使用更激进的设置
    flow_cfg.mask_ratio = {
        "vis": args.mask_ratio,
        "aud": args.mask_ratio,
        "txt": args.mask_ratio,
    }
    flow_cfg.w_visible = 0.0  # 只关注 masked 部分
    flow_cfg.txt_usage_weight = 0.0  # 关闭 txt usage reg

    model = OmniFlow(dims, flow_cfg, cfg["geometry"]["alpha_init"]).to(device)
    model.set_cross_attention(args.cross_attention)

    if args.freeze_alpha:
        model.freeze_alpha()
        print("🔒 Alpha frozen")
    else:
        model.unfreeze_alpha()
        print("🔓 Alpha trainable")

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Total params: {total_params:,}")
    print(f"📈 Trainable params: {trainable_params:,}")

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training history
    history = {
        "train_total": [],
        "train_vis": [],
        "train_aud": [],
        "train_txt": [],
        "val_total": [],
        "val_vis": [],
        "val_aud": [],
        "val_txt": [],
    }

    print("\n" + "=" * 70)
    print("🚀 Starting Overfit Test...")
    print("=" * 70)

    best_loss = float("inf")

    for ep in range(args.epochs):
        t0 = time.time()

        tr = train_epoch(model, train_loader, optimizer, device, args.grad_clip)
        va = validate(model, val_loader, device)

        # Record history
        for k in ["total", "vis", "aud", "txt"]:
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(va[k])

        # Check if improving
        improved = va["total"] < best_loss
        if improved:
            best_loss = va["total"]

        # Get alphas
        alphas = model.alpha_geo.get_all_alphas()
        alpha_str = " | ".join([f"{k}={v.item():.3f}" for k, v in alphas.items()])

        print(
            f"Ep {ep+1:03d}/{args.epochs} | "
            f"Train {tr['total']:.6f} (V:{tr['vis']:.4f} A:{tr['aud']:.4f} T:{tr['txt']:.4f}) | "
            f"Val {va['total']:.6f} | "
            f"α: {alpha_str} | "
            f"{time.time()-t0:.1f}s {'✓' if improved else ''}"
        )

        # Early check: if loss is very low, likely overfitting well
        if va["total"] < 0.01:
            print("\n🎉 Loss < 0.01 - Model can overfit! Test PASSED early.")
            break

    # Final summary
    print("\n" + "=" * 70)
    print("📊 OVERFIT TEST SUMMARY")
    print("=" * 70)

    final_train = history["train_total"][-1]
    final_val = history["val_total"][-1]
    init_train = history["train_total"][0]

    print(f"Initial loss: {init_train:.6f}")
    print(f"Final train loss: {final_train:.6f}")
    print(f"Final val loss: {final_val:.6f}")
    print(f"Loss reduction: {(1 - final_train / init_train) * 100:.1f}%")

    # Verdict
    print("\n" + "-" * 70)
    if final_val < 0.1:
        print("✅ VERDICT: Model can OVERFIT well! Architecture seems OK.")
    elif final_val < 0.5:
        print("⚠️  VERDICT: Model is learning but not overfitting completely.")
        print("   → Consider: more epochs, higher lr, simpler config")
    elif final_val < init_train * 0.5:
        print("⚠️  VERDICT: Model is learning slowly.")
        print("   → Consider: check gradients, increase lr, reduce regularization")
    else:
        print("❌ VERDICT: Model NOT learning! Check for bugs.")
        print("   → Suggestions:")
        print("      1. Check if gradients are flowing (print grad norms)")
        print("      2. Check data preprocessing")
        print("      3. Check loss computation")
        print("      4. Try a simpler model first")
    print("-" * 70)

    # Save plot
    if args.save_plot:
        os.makedirs(
            os.path.dirname(args.save_plot) if os.path.dirname(args.save_plot) else ".",
            exist_ok=True,
        )
        save_overfit_plot(history, args.save_plot)

    return history, model


def main():
    parser = argparse.ArgumentParser(description="Overfit Test for OmniFlow")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "--n_samples", type=int, default=64, help="Number of samples to overfit on"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (0 for overfit test)",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument(
        "--mask_ratio", type=float, default=0.5, help="Mask ratio for all modalities"
    )
    parser.add_argument(
        "--cross_attention", action="store_true", help="Enable cross attention"
    )
    parser.add_argument(
        "--freeze_alpha", action="store_true", help="Freeze geometry alpha"
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="runs/overfit_test.png",
        help="Path to save plot",
    )

    args = parser.parse_args()

    # Handle relative config path
    if not os.path.isabs(args.config):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        args.config = os.path.join(parent_dir, args.config)

    run_overfit_test(args)


if __name__ == "__main__":
    main()
