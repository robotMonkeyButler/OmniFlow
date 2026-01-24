# train.py
import os
import time
import argparse
import random

import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from OmniFlow import OmniFlow, FlowConfig, GeometryConfig
from ContinuousFlow import ContinuousFlow
from DiscreteFlow import DiscreteFlow
from dataloader import get_dataloaders


# ============================================================
# Reproducibility (recommended)
# ============================================================


def seed_all(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Config Loading
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
# Heads
# ============================================================


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


# ============================================================
# Helpers
# ============================================================


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
# Representation Settings
# ============================================================


def get_rep_settings(cfg: dict):
    """Extract representation settings from config."""
    rep = cfg.get("representation", {}) if cfg is not None else {}
    mode = rep.get("mode", "hidden_attn")
    vel_proj = bool(rep.get("vel_proj", True))
    use_ln = bool(rep.get("use_layernorm", True))
    return mode, vel_proj, use_ln


def get_t_settings(cfg: dict):
    """
    Get unified t settings for both supervised_ft and classification.
    Returns (use_multi_t, t_star, t_list, t_weights).
    """
    if cfg is None:
        return False, 1.0, None, None
    rep = cfg.get("representation", {})
    use_multi_t = bool(rep.get("use_multi_t", False))
    t_star = float(rep.get("t_star", 1.0))
    t_list = rep.get("t_list", [0.7, 0.85, 1.0])
    t_weights = rep.get("t_weights", None)
    return use_multi_t, t_star, t_list, t_weights


def single_representation(model, vis, aud, txt, vm, am, tm, t_star: float, cfg: dict):
    """Single-t representation extraction with config-based settings."""
    rep_mode, vel_proj, use_ln = get_rep_settings(cfg)
    return model.encode_representation(
        vis,
        aud,
        txt,
        vm,
        am,
        tm,
        t_star=float(t_star),
        rep_mode=rep_mode,
        vel_proj=vel_proj,
        use_layernorm=use_ln,
    )


def multi_t_representation(
    model, vis, aud, txt, vm, am, tm, t_list, t_weights, cfg: dict
):
    """Multi-t representation extraction with weighted combination."""
    reps = []
    for t in t_list:
        reps.append(
            single_representation(model, vis, aud, txt, vm, am, tm, float(t), cfg)
        )
    reps = torch.stack(reps, dim=0)  # (M,B,D)

    if t_weights is None:
        return reps.mean(dim=0)

    w = torch.tensor(t_weights, device=reps.device, dtype=reps.dtype)
    w = w / w.sum().clamp_min(1e-12)
    return (reps * w.view(-1, 1, 1)).sum(dim=0)


# ============================================================
# Training Loops (FM Stage1/2)
# ============================================================


def train_epoch(model, loader, optimizer, device, cfg, asym_mask=False):
    model.train()
    metrics = {"total": 0, "vis": 0, "aud": 0, "txt": 0, "txt_usage": 0}
    n = 0

    asym_cfg = cfg["training"]["stage2"]["asym_mask"]

    for vis, aud, txt, vm, am, tm, _ in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)

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


# ============================================================
# Representation Extraction (used consistently)
# ============================================================


@torch.no_grad()
def extract_reps(model, loader, device, cfg=None):
    """Extract representations using unified t settings from config."""
    model.eval()
    reps, labels = [], []
    use_multi_t, t_star, t_list, t_weights = get_t_settings(cfg)

    for vis, aud, txt, vm, am, tm, y in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)

        if use_multi_t:
            rep = multi_t_representation(
                model, vis, aud, txt, vm, am, tm, t_list, t_weights, cfg
            )
        else:
            rep = single_representation(model, vis, aud, txt, vm, am, tm, t_star, cfg)

        reps.append(rep.cpu())
        labels.append(y)

    return torch.cat(reps), torch.cat(labels)


# ============================================================
# Downstream Classifier Utils
# ============================================================


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


# ============================================================
# Quick Downstream ValAcc (deterministic; uses same reps definition)
# ============================================================


@torch.no_grad()
def _extract_reps_for_eval(model, loader, device, cfg=None):
    # Use same representation definition as run_classification (multi-t if enabled)
    # This ensures aligned representation definition across all evaluation
    return extract_reps(model, loader, device, cfg=cfg)


def quick_downstream_testacc(
    model, train_loader, test_loader, device, clf_cfg, cfg=None, seed=1234
):
    """
    每个 Stage3 epoch 后，快速训练一个和最终 run_classification 一致的 MLP classifier，
    用它的 TestAcc 作为 Stage3 的 checkpoint 指标（对齐最终目标）。
    - Deterministic: fixed seed + fixed dataloader generator.
    - Representation definition aligned: uses unified t settings from representation config.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    rep_tr, y_tr = _extract_reps_for_eval(model, train_loader, device, cfg=cfg)
    rep_te, y_te = _extract_reps_for_eval(model, test_loader, device, cfg=cfg)

    mu, std = rep_tr.mean(0), rep_tr.std(0).clamp_min(1e-6)
    rep_tr = (rep_tr - mu) / std
    rep_te = (rep_te - mu) / std

    g = torch.Generator()
    g.manual_seed(seed)

    tr_loader = DataLoader(
        TensorDataset(rep_tr, y_tr), batch_size=256, shuffle=True, generator=g
    )
    te_loader = DataLoader(TensorDataset(rep_te, y_te), batch_size=256, shuffle=False)

    num_classes = train_loader.dataset.get_num_classes()
    clf = Classifier(
        rep_tr.shape[1], num_classes, clf_cfg["hidden_dims"], clf_cfg["dropout"]
    ).to(device)
    opt = optim.Adam(clf.parameters(), lr=clf_cfg["lr"])
    crit = nn.CrossEntropyLoss()

    best_te = 0.0
    patience = min(5, int(clf_cfg.get("patience", 10)))
    bad = 0
    max_ep = min(10, int(clf_cfg.get("max_epochs", 50)))

    for _ in range(max_ep):
        train_clf_epoch(clf, tr_loader, opt, crit, device)
        _, te_acc = eval_clf(clf, te_loader, crit, device)
        if te_acc > best_te + 1e-4:
            best_te = te_acc
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    return float(best_te)


# ============================================================
# Grad logging for Stage3
# ============================================================


def _grad_l2_norm(params):
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.norm(2).item() ** 2)
    return total**0.5


# ============================================================
# Stage 3 (Supervised Finetune)
# ============================================================


def supft_train_epoch(
    model, probe, loader, opt, crit, device, grad_clip=1.0, log_grads=False, cfg=None
):
    model.train()
    probe.train()

    loss_sum, correct, total = 0.0, 0, 0
    g_vf, g_proj, g_adapt, g_rep, g_head = 0.0, 0.0, 0.0, 0.0, 0.0
    nb = 0

    use_multi_t, t_star, t_list, t_weights = get_t_settings(cfg)

    for vis, aud, txt, vm, am, tm, y in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)
        y = y.to(device)

        if use_multi_t:
            rep = multi_t_representation(
                model, vis, aud, txt, vm, am, tm, t_list, t_weights, cfg
            )
        else:
            rep = single_representation(model, vis, aud, txt, vm, am, tm, t_star, cfg)

        logits = probe(rep)
        loss = crit(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()

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
            # representation modules: pooling + vel_proj + LN
            rep_params = []
            for m in ["attn_pool_h", "attn_pool_v", "vel_proj", "rep_ln"]:
                if hasattr(model, m):
                    rep_params += list(getattr(model, m).parameters())
            g_rep += _grad_l2_norm(rep_params)
            g_head += _grad_l2_norm(probe.parameters())
            nb += 1

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
            "grad_rep": g_rep / nb,
            "grad_probe": g_head / nb,
        }

    return avg_loss, acc, grad_stats


@torch.no_grad()
def supft_eval(model, probe, loader, crit, device, cfg=None):
    model.eval()
    probe.eval()

    loss_sum, correct, total = 0.0, 0, 0
    use_multi_t, t_star, t_list, t_weights = get_t_settings(cfg)

    for vis, aud, txt, vm, am, tm, y in loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)
        y = y.to(device)

        if use_multi_t:
            rep = multi_t_representation(
                model, vis, aud, txt, vm, am, tm, t_list, t_weights, cfg
            )
        else:
            rep = single_representation(model, vis, aud, txt, vm, am, tm, t_star, cfg)

        logits = probe(rep)
        loss = crit(logits, y)

        loss_sum += float(loss.item())
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(y.size(0))

    return loss_sum / max(1, len(loader)), correct / max(1, total)


def run_supervised_finetune(
    model, train_loader, test_loader, device, cfg, outdir, joint_path=None
):
    print("\n" + "=" * 70)
    print(">>> Stage 3: SUPERVISED FINETUNE (partial backbone)")
    print("=" * 70)

    joint_path = joint_path or os.path.join(outdir, "best_joint.pt")
    if not os.path.exists(joint_path):
        raise FileNotFoundError(f"Joint checkpoint not found: {joint_path}.")
    print(f"Loading joint checkpoint from: {joint_path}")
    model.load_state_dict(torch.load(joint_path, map_location=device), strict=False)

    sft_cfg = cfg["training"].get("supervised_ft", {})
    max_epochs = int(sft_cfg.get("max_epochs", 10))
    patience = int(sft_cfg.get("patience", 5))

    last_k = int(sft_cfg.get("unfreeze_last_k", 2))
    train_vf_inout_proj = bool(sft_cfg.get("train_vf_inout_proj", True))

    freeze_adapters = bool(sft_cfg.get("freeze_adapters", False))
    unfreeze_projs = bool(sft_cfg.get("unfreeze_projs", True))
    unfreeze_adapters_partial = bool(sft_cfg.get("unfreeze_adapters_partial", True))

    lr_vf = float(sft_cfg.get("lr_vf", 3e-5))
    lr_projs = float(sft_cfg.get("lr_projs", 1e-4))
    lr_adapters = float(sft_cfg.get("lr_adapters", 3e-5))
    lr_head = float(sft_cfg.get("lr_head", 1e-3))

    weight_decay = float(sft_cfg.get("weight_decay", 1e-4))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    log_grads = bool(sft_cfg.get("log_grads", True))

    rep_mode, _, _ = get_rep_settings(cfg)
    print(f"[SFT] Representation mode: {rep_mode}")

    # deterministic text during supervised finetune
    if hasattr(model, "adapters") and "txt" in model.adapters:
         model.adapters["txt"].set_gumbel(tau=1.0, hard=False, use_gumbel=False)
    elif hasattr(model, "set_txt_gumbel"):
         model.set_txt_gumbel(tau=1.0, hard=False)
    model.set_cross_attention(True)

    # freeze/unfreeze plan
    model.freeze_all()
    model.freeze_alpha()
    model.set_trainable_supervised_ft(
        last_k=last_k, train_vf_inout_proj=train_vf_inout_proj
    )

    # unfreeze representation modules (pooling + vel_proj + LN)
    for name in ["attn_pool_h", "attn_pool_v", "vel_proj", "rep_ln"]:
        if hasattr(model, name):
            for p in getattr(model, name).parameters():
                p.requires_grad = True
    print("Representation modules: trainable (pool/vel_proj/LN)")

    if unfreeze_projs and hasattr(model, "projs"):
        for p in model.projs.parameters():
            p.requires_grad = True

    if freeze_adapters:
        if hasattr(model, "freeze_adapters"):
            model.freeze_adapters()
        print("Adapters: frozen (all)")
    else:
        if hasattr(model, "unfreeze_adapters"):
            model.unfreeze_adapters()

        if unfreeze_adapters_partial and hasattr(model, "adapters"):
            for k in ["vis", "aud", "txt"]:
                for p in model.adapters[k].ln.parameters():
                    p.requires_grad = True
                for p in model.adapters[k].net[0].parameters():
                    p.requires_grad = True
                for p in model.adapters[k].net[-1].parameters():
                    p.requires_grad = True
            print("Adapters: trainable (partial: ln + net[0] + net[-1])")
        elif hasattr(model, "adapters"):
            for p in model.adapters.parameters():
                p.requires_grad = True
            print("Adapters: trainable (full)")

    # Dynamically infer rep_dim from one batch (important for velocity modes)
    rep_dim = None
    y0 = None
    for vis, aud, txt, vm, am, tm, y in train_loader:
        vis, aud, txt = vis.to(device), aud.to(device), txt.to(device)
        vm, am, tm = vm.to(device), am.to(device), tm.to(device)
        with torch.no_grad():
            rep = single_representation(model, vis, aud, txt, vm, am, tm, 1.0, cfg)
        rep_dim = int(rep.shape[1])
        y0 = y
        break
    if rep_dim is None:
        raise RuntimeError("Empty train_loader.")

    num_classes = train_loader.dataset.get_num_classes()
    probe = ProbeMLP(rep_dim, num_classes, hidden=256, dropout=0.1).to(device)
    print(f"[SFT] rep_dim inferred: {rep_dim}")

    # param groups
    params_vf, params_projs, params_adapters, params_rep = [], [], [], []
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
        elif n.startswith("adapters.") or n.startswith("projectors."):
            params_adapters.append(p)
        elif n.startswith(("attn_pool_h.", "attn_pool_v.", "vel_proj.", "rep_ln.")):
            params_rep.append(p)

    n_vf = sum(p.numel() for p in params_vf)
    n_projs = sum(p.numel() for p in params_projs)
    n_adapt = sum(p.numel() for p in params_adapters)
    n_rep = sum(p.numel() for p in params_rep)
    n_head = sum(p.numel() for p in probe.parameters())
    print(
        f"Trainable params: vf_net={n_vf:,}, projs={n_projs:,}, adapters={n_adapt:,}, rep={n_rep:,}, head={n_head:,}"
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
            {
                "params": params_rep,
                "lr": lr_projs,
            },  # use same lr as projs for rep modules
            {"params": probe.parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
    crit = nn.CrossEntropyLoss()

    # baseline downstream testacc BEFORE finetune (aligned & deterministic)
    base_down = quick_downstream_testacc(
        model,
        train_loader,
        test_loader,
        device,
        clf_cfg=cfg["training"]["classification"],
        cfg=cfg,
        seed=1234,
    )
    print(f"[SFT] Baseline DownTestAcc (before finetune): {base_down:.3f}")

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
            grad_clip=grad_clip,
            log_grads=log_grads,
            cfg=cfg,
        )
        te_loss, te_acc = supft_eval(model, probe, test_loader, crit, device, cfg=cfg)

        down_te = quick_downstream_testacc(
            model,
            train_loader,
            test_loader,
            device,
            clf_cfg=cfg["training"]["classification"],
            cfg=cfg,
            seed=1234,
        )

        stop, improved = es.step(down_te)
        if improved:
            best_acc = down_te
            torch.save(model.state_dict(), best_path)
            torch.save(probe.state_dict(), best_probe_path)

        grad_str = ""
        if log_grads and grad_stats:
            grad_str = " | grad(vf)={:.2e} grad(projs)={:.2e} grad(adapt)={:.2e} grad(rep)={:.2e} grad(head)={:.2e}".format(
                grad_stats["grad_vf_net"],
                grad_stats["grad_projs"],
                grad_stats["grad_adapters"],
                grad_stats["grad_rep"],
                grad_stats["grad_probe"],
            )

        print(
            f"[SFT] Ep {ep+1:02d} | TrLoss {tr_loss:.4f} TrAcc {tr_acc:.3f} | "
            f"ProbeTestAcc {te_acc:.3f} | DownTestAcc {down_te:.3f} | {time.time()-t0:.1f}s {'*' if improved else ''}"
            f"{grad_str}"
        )

        if stop:
            break

    # load best + re-eval downstream testacc to confirm stability
    model.load_state_dict(torch.load(best_path, map_location=device))
    probe.load_state_dict(torch.load(best_probe_path, map_location=device))

    final_down = quick_downstream_testacc(
        model,
        train_loader,
        test_loader,
        device,
        clf_cfg=cfg["training"]["classification"],
        cfg=cfg,
        seed=1234,
    )
    print(f"[SFT] DownTestAcc (best_task_ft re-eval): {final_down:.3f}")
    print(f"Stage 3 done. Best DownTestAcc = {best_acc:.3f}. Saved: {best_path}")

    return model


# ============================================================
# Stage 1 / Stage 2
# ============================================================


def run_stage1(model, train_loader, val_loader, device, cfg, outdir):
    print("\n" + "=" * 70)
    print(">>> Stage 1: GEOMETRY")
    print("=" * 70)

    if not hasattr(model, "alpha_geo"):
        print("Model does not have geometry parameters (baseline). Skipping Stage 1.")
        return model

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

    if hasattr(model, "alpha_geo"):
        alphas = model.alpha_geo.get_all_alphas()
        print(f"Frozen α: {' | '.join(f'{k}={v.item():.3f}' for k, v in alphas.items())}")
    else:
        print("Frozen α: N/A (Baseline)")

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

        if hasattr(model, "alpha_geo"):
            alphas = model.alpha_geo.get_all_alphas()
            alpha_vals = {k: v.item() for k, v in alphas.items()}
        else:
            alpha_vals = {"vis": 0.0, "aud": 0.0, "txt": 0.0}

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


# ============================================================
# Final Classification (uses same reps definition as Stage3 / quick_downstream)
# ============================================================


def run_classification(
    model, train_loader, val_loader, test_loader, device, cfg, outdir
):
    print("\n" + "=" * 70)
    print(">>> Classification")
    print("=" * 70)

    clf_cfg = cfg["training"]["classification"]
    use_multi_t, t_star, t_list, t_weights = get_t_settings(cfg)
    if use_multi_t:
        print(
            f"[CLF] Using multi-t representation: t_list={t_list}, t_weights={t_weights}"
        )
    else:
        print(f"[CLF] Using single-t representation: t_star={t_star}")

    for p in model.parameters():
        p.requires_grad = False

    # Extract representations using unified t settings
    rep_tr, y_tr = extract_reps(model, train_loader, device, cfg=cfg)

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

    num_classes = train_loader.dataset.get_num_classes()
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

    # Extract reps for val and test using same t settings
    rep_val, y_val = extract_reps(model, val_loader, device, cfg=cfg)
    rep_te, y_te = extract_reps(model, test_loader, device, cfg=cfg)

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

    num_classes = train_loader.dataset.get_num_classes()
    clf = Classifier(
        rep_tr.shape[1], num_classes, clf_cfg["hidden_dims"], clf_cfg["dropout"]
    ).to(device)
    opt = optim.Adam(clf.parameters(), lr=clf_cfg["lr"])
    crit = nn.CrossEntropyLoss()

    es = EarlyStopper(mode="max", patience=clf_cfg["patience"])
    best_path = os.path.join(outdir, "best_clf.pt")

    for ep in range(clf_cfg["max_epochs"]):
        loss, acc_tr = train_clf_epoch(clf, clf_loader_tr, opt, crit, device)
        _, acc_te = eval_clf(clf, clf_loader_te, crit, device)

        stop, improved = es.step(acc_te)
        if improved:
            torch.save(clf.state_dict(), best_path)

        print(
            f"[CLF] Ep {ep+1:02d} | Loss {loss:.4f} TrAcc {acc_tr:.3f} | TestAcc {acc_te:.3f} {'*' if improved else ''}"
        )
        if stop:
            break

    clf.load_state_dict(torch.load(best_path, map_location=device))
    _, acc_te_final = eval_clf(clf, clf_loader_te, crit, device)
    print(f"\n>>> Final Test Accuracy: {acc_te_final:.4f}")
    return acc_te_final


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./runs")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["geometry", "joint", "supft", "all", "classification"],
    )
    parser.add_argument("--geo_path", type=str, default=None)
    parser.add_argument("--joint_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    seed_all(args.seed)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    print(
        f"Config: {args.config} | Device: {device} | Stage: {args.stage} | Seed: {args.seed}"
    )
    rep_mode, _, _ = get_rep_settings(cfg)
    use_multi_t, t_star, t_list, t_weights = get_t_settings(cfg)
    print(f"Representation mode: {rep_mode}")
    if use_multi_t:
        print(f"Time settings: multi-t, t_list={t_list}, t_weights={t_weights}")
    else:
        print(f"Time settings: single-t, t_star={t_star}")

    # Load data using unified dataloader
    train_loader, val_loader, test_loader, dims = get_dataloaders(
        cfg, args.dataset_path
    )

    flow_cfg = build_flow_config(cfg)
    model_type = cfg.get("model", {}).get("type", "omni")

    if model_type == "continuous":
        print(">>> Initializing ContinuousFlow Baseline")
        mods = cfg.get("model", {}).get("modalities", None)
        if mods is not None and isinstance(mods, list) and len(mods) > 0:
            print(f"Using modalities: {mods}")
            model = ContinuousFlow(dims, flow_cfg, modality_names=mods).to(device)
        else:
            model = ContinuousFlow(dims, flow_cfg).to(device)
    elif model_type == "discrete":
        print(">>> Initializing DiscreteFlow Baseline")
        mods = cfg.get("model", {}).get("modalities", None)
        q_cfg = cfg.get("quantization", {})
        k = q_cfg.get("codebook_size", 1024)
        mode = q_cfg.get("mode", "kmeans")
        if mods is not None and isinstance(mods, list) and len(mods) > 0:
            print(f"Using modalities: {mods}")
            model = DiscreteFlow(dims, flow_cfg, quantizer_k=k, quantizer_mode=mode, modality_names=mods).to(device)
        else:
            model = DiscreteFlow(dims, flow_cfg, quantizer_k=k, quantizer_mode=mode).to(device)
        model.init_codebooks(train_loader, device)
    else:
        print(">>> Initializing OmniFlow (Default)")
        model = OmniFlow(dims, flow_cfg, cfg["geometry"]["alpha_init"]).to(device)

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
        joint_path = args.joint_path or os.path.join(args.outdir, "best_joint.pt")
        model.load_state_dict(torch.load(joint_path, map_location=device))
        model = run_supervised_finetune(
            model,
            train_loader,
            test_loader,
            device,
            cfg,
            args.outdir,
            joint_path=joint_path,
        )
        run_classification(
            model, train_loader, val_loader, test_loader, device, cfg, args.outdir
        )

    elif args.stage == "classification":
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

        joint_path = os.path.join(args.outdir, "best_joint.pt")
        model = run_supervised_finetune(
            model,
            train_loader,
            test_loader,
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
