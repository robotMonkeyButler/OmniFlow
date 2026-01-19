# train.py
import os
import time
import argparse
import pickle

import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from OmniFlow import OmniFlow, FlowConfig, GeometryConfig


# ============================================================
# Config Loading
# ============================================================


@torch.no_grad()
def _extract_reps_for_eval(model, loader, device, t_star=1.0):
    model.eval()
    reps, labels = [], []
    for vis, aud, txt, vm, am, tm, y in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)
        rep = model.extract_representation(vis, aud, txt, vm, am, tm, t_star=t_star)
        reps.append(rep.cpu())
        labels.append(y)
    return torch.cat(reps, dim=0), torch.cat(labels, dim=0)


def quick_downstream_valacc(
    model, train_loader, val_loader, device, clf_cfg, t_star=1.0
):
    """
    每个 Stage3 epoch 后，快速训练一个和最终 run_classification 一致的 MLP classifier，
    用它的 ValAcc 作为 Stage3 的 checkpoint 指标（对齐最终目标）。
    为了快：只训少量 epoch（例如 10），并用早停。
    """
    # 1) extract reps (CPU)
    rep_tr, y_tr = _extract_reps_for_eval(model, train_loader, device, t_star=t_star)
    rep_val, y_val = _extract_reps_for_eval(model, val_loader, device, t_star=t_star)

    # 2) normalize exactly like run_classification
    mu, std = rep_tr.mean(0), rep_tr.std(0).clamp_min(1e-6)
    rep_tr = (rep_tr - mu) / std
    rep_val = (rep_val - mu) / std

    # 3) dataloaders
    tr_loader = DataLoader(TensorDataset(rep_tr, y_tr), batch_size=256, shuffle=True)
    va_loader = DataLoader(TensorDataset(rep_val, y_val), batch_size=256, shuffle=False)

    num_classes = int(y_tr.max().item()) + 1
    clf = Classifier(
        rep_tr.shape[1], num_classes, clf_cfg["hidden_dims"], clf_cfg["dropout"]
    ).to(device)
    opt = optim.Adam(clf.parameters(), lr=clf_cfg["lr"])
    crit = nn.CrossEntropyLoss()

    # 4) train a few epochs
    best_va = 0.0
    patience = min(5, int(clf_cfg.get("patience", 10)))
    bad = 0
    max_ep = min(10, int(clf_cfg.get("max_epochs", 50)))

    for _ in range(max_ep):
        train_clf_epoch(clf, tr_loader, opt, crit, device)
        _, va_acc = eval_clf(clf, va_loader, crit, device)
        if va_acc > best_va + 1e-4:
            best_va = va_acc
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return float(best_va)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class ProbeMLP(nn.Module):
    def __init__(
        self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.1
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
# Classifier
# ============================================================


def multi_t_representation(
    model,
    vis,
    aud,
    txt,
    vm,
    am,
    tm,
    t_list,
    t_weights=None,
):
    """
    Compute rep(t) for multiple t_star values and aggregate.
    - t_list: list[float]
    - t_weights: list[float] or None (uniform if None)
    Returns: (B, D) aggregated representation
    """
    reps = []
    for t in t_list:
        reps.append(
            model.encode_representation(vis, aud, txt, vm, am, tm, t_star=float(t))
        )

    reps = torch.stack(reps, dim=0)  # (M, B, D)

    if t_weights is None:
        rep = reps.mean(dim=0)
    else:
        w = torch.tensor(t_weights, device=reps.device, dtype=reps.dtype)
        w = w / w.sum().clamp_min(1e-12)
        rep = (reps * w.view(-1, 1, 1)).sum(dim=0)

    return rep


class Classifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dims: list = [256, 128],
        dropout: float = 0.4,
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend(
                [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            )
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# >>> ADDED: simple linear probe for supervised finetune
class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# ============================================================
# Helpers
# ============================================================


def infer_dims(data_dict, vis_key, aud_key, txt_key):
    return {
        "vis": int(data_dict[vis_key][0].shape[-1]),
        "aud": int(data_dict[aud_key][0].shape[-1]),
        "txt": int(data_dict[txt_key][0].shape[-1]),
    }


class EarlyStopper:
    def __init__(self, mode="min", patience=10, min_delta=1e-4):
        self.mode, self.patience, self.min_delta = mode, patience, min_delta
        self.best, self.bad = None, 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False, True
        improved = (
            (self.best - value > self.min_delta)
            if self.mode == "min"
            else (value - self.best > self.min_delta)
        )
        if improved:
            self.best, self.bad = value, 0
            return False, True
        self.bad += 1
        return self.bad >= self.patience, False


def save_plot(train_losses, val_losses, alphas, path):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    for k in ["vis", "aud", "txt"]:
        plt.plot([a[k] for a in alphas], label=k)
    plt.xlabel("epoch")
    plt.ylabel("alpha")
    plt.legend()
    plt.title("Alpha")
    plt.ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def gumbel_tau(ep: int, max_ep: int, tau_start: float, tau_end: float) -> float:
    r = min(1.0, ep / max(1, max_ep - 1))
    return tau_start * (tau_end / tau_start) ** r


# ============================================================
# Training Loops
# ============================================================


def train_epoch(model, loader, optimizer, device, cfg, asym_mask=False):
    model.train()
    metrics = {"total": 0, "vis": 0, "aud": 0, "txt": 0, "txt_usage": 0}
    n = 0

    asym_cfg = cfg["training"]["stage2"]["asym_mask"]

    for vis, aud, txt, vm, am, tm, _ in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)

        # 非对称 masking
        if asym_mask and asym_cfg["enabled"]:
            target = np.random.choice(["vis", "aud", "txt"])
            low = asym_cfg["low_ratio"]
            ratios = {"vis": low, "aud": low, "txt": low}
            if target == "txt":
                ratios["txt"] = asym_cfg["high_ratio_txt"]
            else:
                ratios[target] = asym_cfg["high_ratio_vis_aud"]
            model.cfg.mask_ratio = ratios

        optimizer.zero_grad()
        loss_dict = model.compute_loss(vis, aud, txt, vm, am, tm)
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        optimizer.step()

        for k in metrics:
            metrics[k] += loss_dict[f"loss_{k}" if k != "total" else k].item()
        n += 1

    return {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    metrics = {"total": 0, "vis": 0, "aud": 0, "txt": 0, "txt_usage": 0}
    n = 0

    # >>> MODIFIED: your original code used mask_ratio_per_modality (not in FlowConfig)
    original_mask_ratios = model.cfg.mask_ratio
    model.cfg.mask_ratio = {"vis": 0.3, "aud": 0.3, "txt": 0.2}

    for vis, aud, txt, vm, am, tm, _ in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)
        loss_dict = model.compute_loss(vis, aud, txt, vm, am, tm)
        for k in metrics:
            metrics[k] += loss_dict[f"loss_{k}" if k != "total" else k].item()
        n += 1

    model.cfg.mask_ratio = original_mask_ratios
    return {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def extract_reps(model, loader, device, t_star=1.0):
    model.eval()
    reps, labels = [], []
    for vis, aud, txt, vm, am, tm, y in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)
        rep = model.extract_representation(vis, aud, txt, vm, am, tm, t_star)
        reps.append(rep.cpu())
        labels.append(y)
    return torch.cat(reps), torch.cat(labels)


def train_clf_epoch(clf, loader, opt, crit, device):
    clf.train()
    loss_sum, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = clf(x)
        loss = crit(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), correct / total


@torch.no_grad()
def eval_clf(clf, loader, crit, device):
    clf.eval()
    loss_sum, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = clf(x)
        loss_sum += crit(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), correct / total


def _grad_l2_norm(params):
    """L2 norm over gradients of a parameter iterable."""
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.norm(2).item() ** 2)
    return total**0.5


def supft_train_epoch(
    model,
    probe,
    loader,
    opt,
    crit,
    device,
    t_star=1.0,
    grad_clip=1.0,
    log_grads=False,
    cfg=None,
):
    """
    Supervised finetune one epoch.
    Returns: (avg_loss, acc, grad_stats_dict)
    grad_stats_dict: average grad norms over batches for vf_net / projs / adapters / probe
    """
    model.train()
    probe.train()

    loss_sum, correct, total = 0.0, 0, 0

    # accumulate grad stats
    g_vf, g_proj, g_adapt, g_head = 0.0, 0.0, 0.0, 0.0
    nb = 0

    for vis, aud, txt, vm, am, tm, y in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)
        y = y.to(device)

        # rep = model.encode_representation(vis, aud, txt, vm, am, tm, t_star=t_star)
        # t_star_batch = float(torch.empty(1, device=device).uniform_(0.7, 1.0).item())
        # rep = model.encode_representation(vis, aud, txt, vm, am, tm, t_star=t_star_batch)

        # t_list = [0.6, 0.8, 1.0]
        # t_w = [0.2, 0.3, 0.5]  # bias toward t=1.0
        combo = cfg["training"]["supervised_ft"].get("t_combo", {})
        t_list = combo.get("t_list", [0.7, 0.85, 1.0])
        t_w = combo.get("t_weights", [0.2, 0.3, 0.5])
        rep = multi_t_representation(model, vis, aud, txt, vm, am, tm, t_list, t_w)
        logits = probe(rep)
        loss = crit(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # ---- grad stats BEFORE clipping/step ----
        if log_grads:
            g_vf += _grad_l2_norm(model.vf_net.parameters())
            g_proj += (
                _grad_l2_norm(model.projs.parameters())
                if hasattr(model, "projs")
                else 0.0
            )
            g_adapt += (
                _grad_l2_norm(model.adapters.parameters())
                if hasattr(model, "adapters")
                else 0.0
            )
            g_head += _grad_l2_norm(probe.parameters())
            nb += 1

        # clip
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(probe.parameters()), grad_clip
        )
        opt.step()

        loss_sum += float(loss.item())
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(y.size(0))

    avg_loss = loss_sum / max(1, len(loader))
    acc = correct / max(1, total)

    grad_stats = {}
    if log_grads and nb > 0:
        grad_stats = {
            "grad_vf_net": g_vf / nb,
            "grad_projs": g_proj / nb,
            "grad_adapters": g_adapt / nb,
            "grad_probe": g_head / nb,
        }

    return avg_loss, acc, grad_stats


@torch.no_grad()
def supft_eval(model, probe, loader, crit, device, t_star=1.0, cfg=None):
    model.eval()
    probe.eval()
    loss_sum, correct, total = 0.0, 0, 0

    for vis, aud, txt, vm, am, tm, y in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)
        y = y.to(device)

        # rep = model.extract_representation(vis, aud, txt, vm, am, tm, t_star=t_star)
        combo = cfg["training"]["supervised_ft"].get("t_combo", {})
        t_list = combo.get("t_list", [0.7, 0.85, 1.0])
        t_w = combo.get("t_weights", [0.2, 0.3, 0.5])
        rep = multi_t_representation(model, vis, aud, txt, vm, am, tm, t_list, t_w)
        logits = probe(rep)
        loss = crit(logits, y)

        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / max(1, len(loader)), correct / max(1, total)


# ============================================================
# Stage Functions
# ============================================================


def run_stage1(model, train_loader, val_loader, device, cfg, outdir):
    """Stage 1: Geometry Learning"""
    print("\n" + "=" * 70)
    print(">>> Stage 1: GEOMETRY")
    print("=" * 70)

    s1 = cfg["training"]["stage1"]

    model.set_cross_attention(s1["cross_attention"])
    model.cfg.t_gamma = s1["t_gamma"]
    model.cfg.mask_ratio = s1["mask_ratio"]
    model.cfg.w_visible = s1["w_visible"]
    model.cfg.txt_usage_weight = s1["txt_usage_weight"]
    model.unfreeze_alpha()

    geo_ids = set(map(id, model.alpha_geo.parameters()))
    base_params = [p for p in model.parameters() if id(p) not in geo_ids]

    optimizer = optim.AdamW(
        [
            {"params": base_params, "lr": s1["base_lr"]},
            {"params": model.alpha_geo.parameters(), "lr": s1["alpha_lr"]},
        ],
        weight_decay=cfg["training"]["weight_decay"],
    )

    alphas = model.alpha_geo.get_all_alphas()
    print(f"Initial α: {' | '.join(f'{k}={v.item():.3f}' for k, v in alphas.items())}")

    es = EarlyStopper(patience=s1["patience"])
    hist_tr, hist_val, hist_alpha = [], [], []
    best_path = os.path.join(outdir, "best_geometry.pt")

    for ep in range(s1["max_epochs"]):
        tau = gumbel_tau(
            ep, s1["max_epochs"], s1["gumbel"]["tau_start"], s1["gumbel"]["tau_end"]
        )
        model.set_txt_gumbel(tau, hard=False)

        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, device, cfg, asym_mask=False)
        va = validate(model, val_loader, device)

        alphas = model.alpha_geo.get_all_alphas()
        alpha_vals = {k: v.item() for k, v in alphas.items()}
        hist_tr.append(tr["total"])
        hist_val.append(va["total"])
        hist_alpha.append(alpha_vals)

        stop, improved = es.step(va["total"])
        if improved:
            torch.save(model.state_dict(), best_path)

        print(
            f"[S1] Ep {ep+1:02d} | Tr {tr['total']:.4f} Val {va['total']:.4f} | "
            f"α: V {alpha_vals['vis']:.2f} A {alpha_vals['aud']:.2f} T {alpha_vals['txt']:.2f} | "
            f"tau {tau:.3f} | {time.time()-t0:.1f}s {'*' if improved else ''}"
        )

        if stop:
            break

    save_plot(hist_tr, hist_val, hist_alpha, os.path.join(outdir, "stage1_plot.png"))
    model.load_state_dict(torch.load(best_path, map_location=device))
    print(
        f"Stage 1 done. Final α: {' | '.join(f'{k}={v.item():.3f}' for k, v in model.alpha_geo.get_all_alphas().items())}"
    )
    return model


def run_stage2(model, train_loader, val_loader, device, cfg, outdir, geo_path=None):
    """Stage 2: Joint Training"""
    print("\n" + "=" * 70)
    print(">>> Stage 2: JOINT")
    print("=" * 70)

    if geo_path and os.path.exists(geo_path):
        print(f"Loading geometry from: {geo_path}")
        model.load_state_dict(torch.load(geo_path, map_location=device))

    s2 = cfg["training"]["stage2"]

    model.cfg.mask_ratio = s2["mask_ratio"]
    model.cfg.w_visible = s2["w_visible"]
    model.cfg.t_gamma = s2["t_gamma"]
    model.cfg.txt_usage_weight = s2["txt_usage_weight"]
    model.freeze_alpha()

    alphas = model.alpha_geo.get_all_alphas()
    print(f"Frozen α: {' | '.join(f'{k}={v.item():.3f}' for k, v in alphas.items())}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=s2["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    es = EarlyStopper(patience=s2["patience"])
    hist_tr, hist_val, hist_alpha = [], [], []
    best_path = os.path.join(outdir, "best_joint.pt")
    best_val = float("inf")

    for ep in range(s2["max_epochs"]):
        tau = gumbel_tau(
            ep, s2["max_epochs"], s2["gumbel"]["tau_start"], s2["gumbel"]["tau_end"]
        )
        model.set_txt_gumbel(tau, hard=False)

        use_cross = ep >= s2["cross_warmup"]
        model.set_cross_attention(use_cross)
        use_asym = use_cross and s2["asym_mask"]["enabled"]

        t0 = time.time()
        tr = train_epoch(
            model, train_loader, optimizer, device, cfg, asym_mask=use_asym
        )
        va = validate(model, val_loader, device)

        alphas = model.alpha_geo.get_all_alphas()
        alpha_vals = {k: v.item() for k, v in alphas.items()}
        hist_tr.append(tr["total"])
        hist_val.append(va["total"])
        hist_alpha.append(alpha_vals)

        improved = False
        if use_cross and va["total"] < best_val - 1e-4:
            best_val = va["total"]
            improved = True
            torch.save(model.state_dict(), best_path)

        stop, _ = es.step(va["total"]) if use_cross else (False, False)

        print(
            f"[S2] Ep {ep+1:02d} | Tr {tr['total']:.4f} Val {va['total']:.4f} | "
            f"cross {int(use_cross)} asym {int(use_asym)} | "
            f"V {tr['vis']:.3f} A {tr['aud']:.3f} T {tr['txt']:.3f} | "
            f"tau {tau:.3f} | {time.time()-t0:.1f}s {'*' if improved else ''}"
        )

        if stop:
            break

    save_plot(hist_tr, hist_val, hist_alpha, os.path.join(outdir, "stage2_plot.png"))
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    print("Stage 2 done.")
    return model


def run_supervised_finetune(
    model, train_loader, val_loader, device, cfg, outdir, joint_path=None
):
    print("\n" + "=" * 70)
    print(">>> Stage 3: SUPERVISED FINETUNE (partial backbone)")
    print("=" * 70)

    # -------------------------
    # Load joint checkpoint
    # -------------------------
    joint_path = joint_path or os.path.join(outdir, "best_joint.pt")
    if not os.path.exists(joint_path):
        raise FileNotFoundError(f"Joint checkpoint not found: {joint_path}.")
    print(f"Loading joint checkpoint from: {joint_path}")
    model.load_state_dict(torch.load(joint_path, map_location=device))

    # -------------------------
    # Config (safe defaults)
    # -------------------------
    sft_cfg = cfg["training"].get("supervised_ft", {})
    max_epochs = int(sft_cfg.get("max_epochs", 10))
    patience = int(sft_cfg.get("patience", 5))
    t_star = float(sft_cfg.get("t_star", 1.0))

    last_k = int(sft_cfg.get("unfreeze_last_k", 2))
    train_vf_inout_proj = bool(sft_cfg.get("train_vf_inout_proj", True))

    # IMPORTANT: set this FALSE to allow adapter training
    freeze_adapters = bool(sft_cfg.get("freeze_adapters", False))

    # recommended defaults
    unfreeze_projs = bool(sft_cfg.get("unfreeze_projs", True))
    unfreeze_adapters_partial = bool(sft_cfg.get("unfreeze_adapters_partial", True))

    # LR groups
    lr_vf = float(sft_cfg.get("lr_vf", 3e-5))
    lr_projs = float(sft_cfg.get("lr_projs", 1e-4))
    lr_adapters = float(sft_cfg.get("lr_adapters", 3e-5))
    lr_head = float(sft_cfg.get("lr_head", 1e-3))

    weight_decay = float(sft_cfg.get("weight_decay", 1e-4))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    log_grads = bool(sft_cfg.get("log_grads", True))

    # -------------------------
    # Deterministic settings for supervised FT
    # -------------------------
    model.adapters["txt"].set_gumbel(tau=1.0, hard=False, use_gumbel=False)
    model.set_cross_attention(True)

    # -------------------------
    # Freeze / unfreeze plan
    # -------------------------
    # 1) freeze everything
    model.freeze_all()

    # 2) keep geometry fixed
    model.freeze_alpha()

    # 3) unfreeze vf last_k + (optional) vf in/out proj
    model.set_trainable_supervised_ft(
        last_k=last_k, train_vf_inout_proj=train_vf_inout_proj
    )

    # 4) projs
    if unfreeze_projs:
        for p in model.projs.parameters():
            p.requires_grad = True

    # 5) adapters
    if freeze_adapters:
        if hasattr(model, "freeze_adapters"):
            model.freeze_adapters()
        print("Adapters: frozen (all)")
    else:
        # allow adapter training (partial recommended)
        if hasattr(model, "unfreeze_adapters"):
            model.unfreeze_adapters()

        if unfreeze_adapters_partial:
            # partial: LN + first Linear + last Linear
            for k in ["vis", "aud", "txt"]:
                for p in model.adapters[k].ln.parameters():
                    p.requires_grad = True
                for p in model.adapters[k].net[0].parameters():
                    p.requires_grad = True
                for p in model.adapters[k].net[-1].parameters():
                    p.requires_grad = True
            print("Adapters: trainable (partial: ln + net[0] + net[-1])")
        else:
            for p in model.adapters.parameters():
                p.requires_grad = True
            print("Adapters: trainable (full)")

    # -------------------------
    # Build probe head (MLP, aligns with downstream better than Linear)
    # -------------------------
    rep_dim = model.cfg.d_model * 3
    y0 = None
    for _, _, _, _, _, _, y in train_loader:
        y0 = y
        break
    if y0 is None:
        raise RuntimeError("Empty train_loader.")
    num_classes = int(y0.max().item()) + 1

    probe = ProbeMLP(rep_dim, num_classes, hidden=256, dropout=0.1).to(device)

    # -------------------------
    # Collect params into 4 groups: vf / projs / adapters / head
    # -------------------------
    params_vf, params_projs, params_adapters = [], [], []
    seen = set()

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)

        if n.startswith("vf_net."):
            params_vf.append(p)
        elif n.startswith("projs."):
            params_projs.append(p)
        elif n.startswith("adapters."):
            params_adapters.append(p)
        else:
            # should not happen if freeze_all() worked, but keep safe:
            params_projs.append(p)

    n_vf = sum(p.numel() for p in params_vf)
    n_projs = sum(p.numel() for p in params_projs)
    n_adapt = sum(p.numel() for p in params_adapters)
    n_head = sum(p.numel() for p in probe.parameters())

    print(
        f"Trainable params: vf_net={n_vf:,}, projs={n_projs:,}, adapters={n_adapt:,}, head={n_head:,}"
    )
    print(
        f"LR: vf={lr_vf}, projs={lr_projs}, adapters={lr_adapters}, head={lr_head} | "
        f"last_k={last_k} vf_inout={train_vf_inout_proj} | projs={unfreeze_projs} "
        f"| freeze_adapters={freeze_adapters} partial={unfreeze_adapters_partial}"
    )

    optimizer = optim.AdamW(
        [
            {"params": params_vf, "lr": lr_vf},
            {"params": params_projs, "lr": lr_projs},
            {"params": params_adapters, "lr": lr_adapters},
            {"params": probe.parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
    crit = nn.CrossEntropyLoss()

    # -------------------------
    # Train loop with ValAcc checkpointing
    # -------------------------
    es = EarlyStopper(mode="max", patience=patience, min_delta=1e-4)
    best_path = os.path.join(outdir, "best_task_ft.pt")
    best_probe_path = os.path.join(outdir, "best_task_ft_probe.pt")

    best_acc = -1.0
    for ep in range(max_epochs):
        t0 = time.time()

        tr_loss, tr_acc, grad_stats = supft_train_epoch(
            model,
            probe,
            train_loader,
            optimizer,
            crit,
            device,
            t_star=t_star,
            grad_clip=grad_clip,
            log_grads=log_grads,
            cfg=cfg,
        )
        va_loss, va_acc = supft_eval(
            model, probe, val_loader, crit, device, t_star=t_star, cfg=cfg
        )
        down_va = quick_downstream_valacc(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            clf_cfg=cfg["training"]["classification"],
            t_star=t_star,
        )

        stop, improved = es.step(down_va)
        if improved:
            best_acc = down_va
            torch.save(model.state_dict(), best_path)
            torch.save(probe.state_dict(), best_probe_path)

        grad_str = ""
        if log_grads and grad_stats:
            grad_str = " | grad(vf)={:.2e} grad(projs)={:.2e} grad(adapt)={:.2e} grad(head)={:.2e}".format(
                grad_stats["grad_vf_net"],
                grad_stats["grad_projs"],
                grad_stats["grad_adapters"],
                grad_stats["grad_probe"],
            )

        print(
            f"[SFT] Ep {ep+1:02d} | TrLoss {tr_loss:.4f} TrAcc {tr_acc:.3f} | "
            f"ValLoss {va_loss:.4f} ValAcc {va_acc:.3f} | {time.time()-t0:.1f}s {'*' if improved else ''}"
            f"{grad_str}"
        )

        if stop:
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    probe.load_state_dict(torch.load(best_probe_path, map_location=device))
    print(f"Stage 3 done. Best probe ValAcc = {best_acc:.3f}. Saved: {best_path}")

    return model


def run_classification(
    model, train_loader, val_loader, test_loader, device, cfg, outdir
):
    """Classification Evaluation"""
    print("\n" + "=" * 70)
    print(">>> Classification")
    print("=" * 70)

    clf_cfg = cfg["training"]["classification"]

    for p in model.parameters():
        p.requires_grad = False

    # ========== 诊断：检查表示质量 ==========
    rep_tr, y_tr = extract_reps(model, train_loader, device, t_star=1.0)

    print(f"\n[Diagnosis] Representation shape: {rep_tr.shape}")
    print(f"[Diagnosis] Rep mean: {rep_tr.mean():.4f}, std: {rep_tr.std():.4f}")
    print(f"[Diagnosis] Rep min: {rep_tr.min():.4f}, max: {rep_tr.max():.4f}")

    print(
        f"[Diagnosis] Has NaN: {torch.isnan(rep_tr).any()}, Has Inf: {torch.isinf(rep_tr).any()}"
    )

    d = rep_tr.shape[1] // 3
    print(f"[Diagnosis] Vis std: {rep_tr[:, :d].std():.4f}")
    print(f"[Diagnosis] Aud std: {rep_tr[:, d:2*d].std():.4f}")
    print(f"[Diagnosis] Txt std: {rep_tr[:, 2*d:].std():.4f}")

    num_classes = int(y_tr.max().item()) + 1
    for c in range(num_classes):
        mask = y_tr == c
        print(
            f"[Diagnosis] Class {c}: n={mask.sum().item()}, rep_mean={rep_tr[mask].mean():.4f}, rep_std={rep_tr[mask].std():.4f}"
        )

    if num_classes == 2:
        rep_0 = rep_tr[y_tr == 0].mean(dim=0)
        rep_1 = rep_tr[y_tr == 1].mean(dim=0)
        dist = (rep_0 - rep_1).norm()
        print(f"[Diagnosis] Class center distance: {dist:.4f}")

    rep_tr, y_tr = extract_reps(model, train_loader, device, clf_cfg["t_star"])
    rep_val, y_val = extract_reps(model, val_loader, device, clf_cfg["t_star"])
    rep_te, y_te = extract_reps(model, test_loader, device, clf_cfg["t_star"])

    # Normalize
    mu, std = rep_tr.mean(0), rep_tr.std(0).clamp_min(1e-6)
    rep_tr, rep_val, rep_te = (
        (rep_tr - mu) / std,
        (rep_val - mu) / std,
        (rep_te - mu) / std,
    )

    clf_loader_tr = DataLoader(
        TensorDataset(rep_tr, y_tr), batch_size=256, shuffle=True
    )
    clf_loader_val = DataLoader(
        TensorDataset(rep_val, y_val), batch_size=256, shuffle=False
    )
    clf_loader_te = DataLoader(
        TensorDataset(rep_te, y_te), batch_size=256, shuffle=False
    )

    num_classes = int(y_tr.max().item()) + 1
    clf = Classifier(
        rep_tr.shape[1], num_classes, clf_cfg["hidden_dims"], clf_cfg["dropout"]
    ).to(device)
    opt = optim.Adam(clf.parameters(), lr=clf_cfg["lr"])
    crit = nn.CrossEntropyLoss()

    es = EarlyStopper(mode="max", patience=clf_cfg["patience"])
    best_path = os.path.join(outdir, "best_clf.pt")

    for ep in range(clf_cfg["max_epochs"]):
        loss, acc_tr = train_clf_epoch(clf, clf_loader_tr, opt, crit, device)
        _, acc_val = eval_clf(clf, clf_loader_val, crit, device)

        stop, improved = es.step(acc_val)
        if improved:
            torch.save(clf.state_dict(), best_path)

        print(
            f"[CLF] Ep {ep+1:02d} | Loss {loss:.4f} TrAcc {acc_tr:.3f} | ValAcc {acc_val:.3f} {'*' if improved else ''}"
        )
        if stop:
            break

    clf.load_state_dict(torch.load(best_path, map_location=device))
    _, acc_te = eval_clf(clf, clf_loader_te, crit, device)
    print(f"\n>>> Test Accuracy: {acc_te:.4f}")
    return acc_te


# ============================================================
# Data Loading
# ============================================================


def load_data(path, dataset_name):
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


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./runs")
    # >>> MODIFIED: add supft stage option
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["geometry", "joint", "supft", "all", "classification"],
    )
    parser.add_argument("--geo_path", type=str, default=None)
    parser.add_argument(
        "--joint_path",
        type=str,
        default=None,
        help="Path to trained joint checkpoint for classification or supft stage",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Config: {args.config} | Device: {device} | Stage: {args.stage}")

    # Load data
    trains, valid, test = load_data(args.dataset_path, cfg["dataset"]["name"])
    keys = cfg["dataset"]["modality_keys"][cfg["dataset"]["name"]]

    # Normalize vis/aud
    for k in [keys["vis"], keys["aud"]]:
        all_data = np.concatenate(trains[k], axis=0)
        mean, std = all_data.mean(0), all_data.std(0)
        std[std < 1e-6] = 1.0
        print(f"Norm {k}: mean={mean.mean():.3f}, std={std.mean():.3f}")
        for split in [trains, valid, test]:
            for i in range(len(split[k])):
                split[k][i] = (split[k][i] - mean) / std

    def make_loader(data, shuffle):
        ds = TriModalDataset(data, keys["vis"], keys["aud"], keys["txt"])
        return DataLoader(
            ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    train_loader = make_loader(trains, True)
    val_loader = make_loader(valid, False)
    test_loader = make_loader(test, False)

    dims = infer_dims(trains, keys["vis"], keys["aud"], keys["txt"])
    print(f"Dims: {dims}")

    # Build model
    flow_cfg = build_flow_config(cfg)
    model = OmniFlow(dims, flow_cfg, cfg["geometry"]["alpha_init"]).to(device)

    # Run stages
    if args.stage == "geometry":
        run_stage1(model, train_loader, val_loader, device, cfg, args.outdir)

    elif args.stage == "joint":
        geo_path = args.geo_path or os.path.join(args.outdir, "best_geometry.pt")
        model = run_stage2(
            model, train_loader, val_loader, device, cfg, args.outdir, geo_path
        )
        run_classification(
            model, train_loader, val_loader, test_loader, device, cfg, args.outdir
        )

    elif args.stage == "supft":
        # Load joint, run supervised finetune, then classification
        joint_path = args.joint_path or os.path.join(args.outdir, "best_joint.pt")
        model.load_state_dict(torch.load(joint_path, map_location=device))
        model = run_supervised_finetune(
            model,
            train_loader,
            val_loader,
            device,
            cfg,
            args.outdir,
            joint_path=joint_path,
        )
        run_classification(
            model, train_loader, val_loader, test_loader, device, cfg, args.outdir
        )

    elif args.stage == "classification":
        # Load trained checkpoint and only do classification
        # >>> MODIFIED: prefer task-ft checkpoint if exists; else fallback to best_joint
        task_ft_path = os.path.join(args.outdir, "best_task_ft.pt")
        joint_path = args.joint_path or os.path.join(args.outdir, "best_joint.pt")
        load_path = task_ft_path if os.path.exists(task_ft_path) else joint_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {load_path}. Please run joint or supft first."
            )
        print(f"Loading checkpoint from: {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
        run_classification(
            model, train_loader, val_loader, test_loader, device, cfg, args.outdir
        )

    else:  # all
        model = run_stage1(model, train_loader, val_loader, device, cfg, args.outdir)
        geo_path = os.path.join(args.outdir, "best_geometry.pt")
        model = run_stage2(
            model, train_loader, val_loader, device, cfg, args.outdir, geo_path
        )

        # >>> ADDED: run supervised finetune after stage2
        joint_path = os.path.join(args.outdir, "best_joint.pt")
        model = run_supervised_finetune(
            model,
            train_loader,
            val_loader,
            device,
            cfg,
            args.outdir,
            joint_path=joint_path,
        )

        run_classification(
            model, train_loader, val_loader, test_loader, device, cfg, args.outdir
        )

    print("\n" + "=" * 70)
    print(">>> All Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
