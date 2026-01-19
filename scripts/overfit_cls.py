#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overfit Classification Sanity Check
-----------------------------------
目标：验证在极小数据集（前 N 个样本）上，模型是否能“把分类任务过拟合到接近 100%”。
如果做不到：通常说明（1）可训练模块位置太靠后；（2）表示链路丢信息；或（3）数据/标签管线有问题。

依赖：
- OmniFlow.py（需包含 encode_representation / extract_representation / freeze_all / freeze_alpha 等）
- config.yaml
- 数据集 pkl/npz

使用示例：
1) 从 best_joint.pt 开始，解冻 vf_net last 2 + vf in/out + projs（推荐）
python overfit_classification.py --config config.yaml --dataset_path /path/to/data.pkl --joint_ckpt ./runs/best_joint.pt --n_samples 64 --epochs 200 --unfreeze_projs --unfreeze_adapters_last

2) 更激进：解冻更多层
python overfit_classification.py --config config.yaml --dataset_path /path/to/data.pkl --joint_ckpt ./runs/best_joint.pt --n_samples 64 --unfreeze_last_k 4 --unfreeze_projs --unfreeze_adapters_last --backbone_lr 1e-4
"""

import os
import argparse
import pickle
import random
from typing import Dict, Any

import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from OmniFlow import OmniFlow, FlowConfig, GeometryConfig


# -----------------------------
# Reproducibility
# -----------------------------
def seed_all(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Config helpers
# -----------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_flow_config(cfg: dict) -> FlowConfig:
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
        n_layers=cfg["model"]["num_layers"],
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


# -----------------------------
# Data
# -----------------------------
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

    n = len(next(iter(data.values())))
    n_tr, n_val = int(n * 0.8), int(n * 0.1)
    return (
        {k: v[:n_tr] for k, v in data.items()},
        {k: v[n_tr : n_tr + n_val] for k, v in data.items()},
        {k: v[n_tr + n_val :] for k, v in data.items()},
    )


def subset_data(data_dict, n_samples: int):
    return {k: v[:n_samples] for k, v in data_dict.items()}


def infer_dims(data_dict, vis_key, aud_key, txt_key):
    return {
        "vis": int(data_dict[vis_key][0].shape[-1]),
        "aud": int(data_dict[aud_key][0].shape[-1]),
        "txt": int(data_dict[txt_key][0].shape[-1]),
    }


# -----------------------------
# Model head
# -----------------------------
class SmallMLPHead(nn.Module):
    def __init__(
        self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Freezing / unfreezing for overfit
# -----------------------------
def set_trainable_for_overfit(
    model: OmniFlow,
    unfreeze_last_k: int,
    unfreeze_vf_inout: bool,
    unfreeze_projs: bool,
    unfreeze_adapters_last: bool,
    freeze_alpha: bool,
):
    """
    策略：
    1) freeze all
    2) freeze alpha (optional)
    3) unfreeze: vf_net last_k + vf in/out proj (optional)
    4) unfreeze: model.projs (optional)
    5) unfreeze: adapters last Linear only (optional)
    """
    # freeze everything
    model.freeze_all()

    # keep geometry story fixed
    if freeze_alpha:
        model.freeze_alpha()
    else:
        model.unfreeze_alpha()

    # unfreeze backbone subset
    model.set_trainable_supervised_ft(
        last_k=unfreeze_last_k, train_vf_inout_proj=unfreeze_vf_inout
    )

    if unfreeze_projs:
        for p in model.projs.parameters():
            p.requires_grad = True

    if unfreeze_adapters_last:
        # only unfreeze the last Linear layer inside each adapter's net
        for k in ["vis", "aud", "txt"]:
            last_linear = model.adapters[k].net[-1]  # nn.Linear
            for p in last_linear.parameters():
                p.requires_grad = True


# -----------------------------
# Train/eval
# -----------------------------
def run_one_epoch(
    model, head, loader, device, opt, crit, t_star: float, grad_clip: float
):
    """
    注意：我们用 model.eval() 来关掉 dropout（但仍然允许梯度反传），
    这样 overfit 更稳定、更容易达到 100%。
    """
    model.eval()
    head.train()

    loss_sum, correct, total = 0.0, 0, 0
    for vis, aud, txt, vm, am, tm, y in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)
        y = y.to(device)

        rep = model.encode_representation(vis, aud, txt, vm, am, tm, t_star=t_star)
        logits = head(rep)
        loss = crit(logits, y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()), grad_clip
        )
        opt.step()

        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / max(1, len(loader)), correct / max(1, total)


@torch.no_grad()
def evaluate(model, head, loader, device, crit, t_star: float):
    model.eval()
    head.eval()

    loss_sum, correct, total = 0.0, 0, 0
    for vis, aud, txt, vm, am, tm, y in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)
        y = y.to(device)

        rep = model.extract_representation(vis, aud, txt, vm, am, tm, t_star=t_star)
        logits = head(rep)
        loss = crit(logits, y)

        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / max(1, len(loader)), correct / max(1, total)


@torch.no_grad()
def class_center_distance(model, loader, device, t_star: float):
    reps, ys = [], []
    model.eval()
    for vis, aud, txt, vm, am, tm, y in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)
        rep = model.extract_representation(vis, aud, txt, vm, am, tm, t_star=t_star)
        reps.append(rep.cpu())
        ys.append(y)
    reps = torch.cat(reps, dim=0)
    ys = torch.cat(ys, dim=0)

    num_classes = int(ys.max().item()) + 1
    if num_classes != 2:
        return None
    c0 = reps[ys == 0].mean(dim=0)
    c1 = reps[ys == 1].mean(dim=0)
    return (c0 - c1).norm().item()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument(
        "--joint_ckpt", type=str, default=None, help="Optional: path to best_joint.pt"
    )
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)

    # supervised overfit settings
    parser.add_argument("--t_star", type=float, default=1.0)
    parser.add_argument(
        "--freeze_alpha", action="store_true", help="Freeze alpha geometry"
    )
    parser.add_argument("--unfreeze_last_k", type=int, default=2)
    parser.add_argument(
        "--unfreeze_vf_inout",
        action="store_true",
        help="Unfreeze vf_net.in_proj/out_proj",
    )
    parser.add_argument(
        "--unfreeze_projs",
        action="store_true",
        help="Unfreeze OmniFlow.projs (raw -> d_model)",
    )
    parser.add_argument(
        "--unfreeze_adapters_last",
        action="store_true",
        help="Unfreeze adapters last Linear only",
    )

    # optimization
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    args = parser.parse_args()
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    trains, _, _ = load_data(args.dataset_path)
    keys = cfg["dataset"]["modality_keys"][cfg["dataset"]["name"]]

    subset = subset_data(trains, args.n_samples)
    print(f"Using subset size: {len(subset[keys['txt']])}")

    # Normalize vis/aud using subset stats (so overfit is easier / consistent)
    for k in [keys["vis"], keys["aud"]]:
        all_data = np.concatenate(subset[k], axis=0)
        mean, std = all_data.mean(0), all_data.std(0)
        std[std < 1e-6] = 1.0
        for i in range(len(subset[k])):
            subset[k][i] = (subset[k][i] - mean) / std

    ds = TriModalDataset(subset, keys["vis"], keys["aud"], keys["txt"])
    train_loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    dims = infer_dims(subset, keys["vis"], keys["aud"], keys["txt"])
    flow_cfg = build_flow_config(cfg)
    model = OmniFlow(dims, flow_cfg, cfg["geometry"]["alpha_init"]).to(device)

    # load pretrained joint if provided
    if args.joint_ckpt is not None:
        print(f"Loading joint ckpt: {args.joint_ckpt}")
        model.load_state_dict(torch.load(args.joint_ckpt, map_location=device))

    # critical: deterministic text adapter (no gumbel noise)
    model.adapters["txt"].set_gumbel(tau=1.0, hard=False, use_gumbel=False)

    # keep cross-attn ON (recommended for multimodal)
    model.set_cross_attention(True)

    # set trainable subset
    set_trainable_for_overfit(
        model=model,
        unfreeze_last_k=args.unfreeze_last_k,
        unfreeze_vf_inout=args.unfreeze_vf_inout,
        unfreeze_projs=args.unfreeze_projs,
        unfreeze_adapters_last=args.unfreeze_adapters_last,
        freeze_alpha=args.freeze_alpha,
    )

    # infer num classes
    y0 = None
    for *_, y in train_loader:
        y0 = y
        break
    num_classes = int(y0.max().item()) + 1
    rep_dim = model.cfg.d_model * 3
    head = SmallMLPHead(rep_dim, num_classes, hidden=256, dropout=0.0).to(device)

    backbone_params = [p for p in model.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    print(
        f"Trainable params: backbone={sum(p.numel() for p in backbone_params)}, head={sum(p.numel() for p in head_params)}"
    )
    print(
        f"LR: backbone={args.backbone_lr}, head={args.head_lr}, last_k={args.unfreeze_last_k}, "
        f"vf_inout={args.unfreeze_vf_inout}, projs={args.unfreeze_projs}, adapters_last={args.unfreeze_adapters_last}"
    )

    opt = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    crit = nn.CrossEntropyLoss()

    best_acc = -1.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_one_epoch(
            model, head, train_loader, device, opt, crit, args.t_star, args.grad_clip
        )
        va_loss, va_acc = evaluate(model, head, val_loader, device, crit, args.t_star)

        best_acc = max(best_acc, va_acc)
        print(
            f"Ep {ep:03d} | TrLoss {tr_loss:.4f} TrAcc {tr_acc:.3f} | ValLoss {va_loss:.4f} ValAcc {va_acc:.3f}"
        )

        if va_acc > 0.99:
            print("Overfit achieved (ValAcc > 0.99). Stop early.")
            break

    dist = class_center_distance(model, val_loader, device, args.t_star)
    if dist is not None:
        print(f"Class center distance (representation): {dist:.4f}")

    print(f"Best ValAcc: {best_acc:.3f}")


if __name__ == "__main__":
    main()
