# OmniFlow.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================


def masked_mean(
    x: torch.Tensor, mask_keep: Optional[torch.Tensor], dim: int
) -> torch.Tensor:
    """Compute mean over `dim` with optional boolean mask_keep (True = keep)."""
    if mask_keep is None:
        return x.mean(dim=dim)
    while mask_keep.ndim < x.ndim:
        mask_keep = mask_keep.unsqueeze(-1)
    mask_keep = mask_keep.to(dtype=x.dtype)
    num = (x * mask_keep).sum(dim=dim)
    den = mask_keep.sum(dim=dim).clamp_min(1e-8)
    return num / den


def generate_span_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float,
    span_len: int,
    device: torch.device,
    pad_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate span mask. True = mask/corrupt this position."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    span_len = max(int(span_len), 1)
    target = int(round(float(mask_ratio) * seq_len))

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    if target <= 0:
        return mask

    n_spans = max(1, math.ceil(target / span_len))
    n_spans = min(n_spans, seq_len)

    scores = torch.rand(batch_size, seq_len, device=device)
    _, starts = torch.topk(scores, k=n_spans, dim=1, largest=True, sorted=False)

    start_indicator = torch.zeros(
        batch_size, seq_len, device=device, dtype=torch.float32
    )
    start_indicator.scatter_add_(
        1, starts, torch.ones_like(starts, dtype=torch.float32)
    )

    kernel = torch.ones(1, 1, span_len, device=device, dtype=torch.float32)
    expanded = F.conv1d(start_indicator.unsqueeze(1), kernel, padding=span_len - 1)[
        :, :, :seq_len
    ]
    mask = expanded.squeeze(1) > 0

    if pad_mask is not None:
        mask = mask & (~pad_mask)

    return mask


# ============================================================
# Geometry: Per-modality learnable α ∈ (-1,1)
# ============================================================


@dataclass
class GeometryConfig:
    p_min: float = 1.001
    p_max: float = 50.0
    eps: float = 1e-8
    m_min: float = 1e-3
    m_max: float = 1e3
    w_max: float = 1e6
    prior_scale: float = 1.0
    prior_type: str = "lognormal"


class PerModalityAlphaGeometry(nn.Module):
    """Per-modality learnable α via tanh parameterization: α = tanh(raw) * 0.99"""

    def __init__(
        self,
        modality_names: List[str],
        cfg: GeometryConfig,
        learnable: bool = True,
        alpha_init: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.modality_names = list(modality_names)
        self.alpha_scale = 0.99

        alpha_init = alpha_init or {k: 0.0 for k in self.modality_names}

        if learnable:
            self.alpha_raw = nn.ParameterDict()
            for k in self.modality_names:
                target = max(-0.98, min(0.98, float(alpha_init.get(k, 0.0))))
                ratio = max(-0.99, min(0.99, target / self.alpha_scale))
                raw = 0.5 * math.log((1 + ratio) / (1 - ratio))  # atanh
                self.alpha_raw[k] = nn.Parameter(torch.tensor(raw))
        else:
            for k in self.modality_names:
                target = max(-0.98, min(0.98, float(alpha_init.get(k, 0.0))))
                ratio = max(-0.99, min(0.99, target / self.alpha_scale))
                raw = 0.5 * math.log((1 + ratio) / (1 - ratio))
                self.register_buffer(f"alpha_raw_{k}", torch.tensor(raw))
            self.alpha_raw = {
                k: getattr(self, f"alpha_raw_{k}") for k in self.modality_names
            }

    def get_alpha(self, modality: str) -> torch.Tensor:
        return torch.tanh(self.alpha_raw[modality]) * self.alpha_scale

    def get_p(self, modality: str) -> torch.Tensor:
        alpha = self.get_alpha(modality)
        p = 2.0 / (1.0 - alpha + 1e-8)
        return torch.clamp(p, min=self.cfg.p_min, max=self.cfg.p_max)

    def get_all_alphas(self) -> Dict[str, torch.Tensor]:
        return {k: self.get_alpha(k) for k in self.modality_names}

    def m_to_x(self, m: torch.Tensor, modality: str) -> torch.Tensor:
        p = self.get_p(modality).to(dtype=m.dtype, device=m.device)
        return torch.exp(torch.log(m.clamp_min(self.cfg.eps)) / p)

    def x_to_m(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        p = self.get_p(modality).to(dtype=x.dtype, device=x.device)
        return torch.exp(p * torch.log(x.clamp_min(self.cfg.eps)))

    def sample_prior_x(
        self, shape: torch.Size, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if self.cfg.prior_type == "lognormal":
            log_x = torch.randn(shape, device=device, dtype=dtype) * 0.5
            x0 = torch.exp(log_x).clamp_min(self.cfg.eps)
        elif self.cfg.prior_type == "softplus":
            x0 = (
                F.softplus(torch.randn(shape, device=device, dtype=dtype))
                + self.cfg.eps
            )
        else:
            raise ValueError(f"Unknown prior_type: {self.cfg.prior_type}")
        return self.cfg.prior_scale * x0

    def interpolate_x(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = t.view(-1, 1, 1).to(dtype=x0.dtype, device=x0.device)
        return (1.0 - t) * x0 + t * x1

    def target_u_x(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return x1 - x0

    def alpha_norm_weight(self, x_t: torch.Tensor, modality: str) -> torch.Tensor:
        p = self.get_p(modality).to(dtype=x_t.dtype, device=x_t.device)
        alpha = self.get_alpha(modality).to(dtype=x_t.dtype, device=x_t.device)

        x = x_t.clamp_min(self.cfg.eps)
        log_m = p * torch.log(x)
        log_m = torch.clamp(
            log_m, min=math.log(self.cfg.m_min), max=math.log(self.cfg.m_max)
        )

        log_w = torch.log(p * p) + alpha * log_m
        w = torch.exp(log_w)

        if self.cfg.w_max > 0:
            w = torch.clamp(w, max=self.cfg.w_max)
        return w


# ============================================================
# Modality Adapters
# ============================================================


class PositiveMeasureAdapter(nn.Module):
    """For continuous modalities (vis/aud): map to positive measure via softplus."""

    def __init__(
        self, in_dim: int, out_dim: int, hidden_dim: int = 512, eps: float = 1e-4
    ):
        super().__init__()
        self.eps = eps
        self.ln = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self, y: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        m = F.softplus(self.net(self.ln(y))) + self.eps
        if pad_mask is not None:
            m = m.masked_fill(pad_mask.unsqueeze(-1), 1.0)
        return m


class DiscreteSimplexAdapter(nn.Module):
    """For text: Gumbel-Softmax over V=measure_dim codes."""

    def __init__(
        self, in_dim: int, out_dim: int, hidden_dim: int = 512, eps: float = 1e-3
    ):
        super().__init__()
        self.eps = eps
        self.scale = float(out_dim)
        self.ln = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Gumbel settings (set externally)
        self.tau = 1.0
        self.hard = False
        self.use_gumbel = True

    def set_gumbel(self, tau: float, hard: bool = False, use_gumbel: bool = True):
        self.tau = tau
        self.hard = hard
        self.use_gumbel = use_gumbel

    def forward(
        self,
        y: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        logits = self.net(self.ln(y))

        if self.training and self.use_gumbel:
            probs = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)

        m = probs * self.scale + self.eps
        if pad_mask is not None:
            m = m.masked_fill(pad_mask.unsqueeze(-1), 1.0)

        return (m, probs) if return_probs else m


# ============================================================
# Transformer Components
# ============================================================


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return self.pe[:length].to(device=device, dtype=dtype)


class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 256), nn.SiLU(), nn.Linear(256, d_model))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t.view(-1, 1))


class TransformerLayer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 4
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_sa = nn.LayerNorm(d_model)
        self.norm_ca_q = nn.LayerNorm(d_model)
        self.norm_ca_kv = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def self_attention(
        self, h: torch.Tensor, pad_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        out, _ = self.self_attn(
            self.norm_sa(h), self.norm_sa(h), self.norm_sa(h), key_padding_mask=pad_mask
        )
        return h + self.drop(out)

    def cross_attention(
        self, h: torch.Tensor, ctx: torch.Tensor, ctx_pad: Optional[torch.Tensor]
    ) -> torch.Tensor:
        out, _ = self.cross_attn(
            self.norm_ca_q(h),
            self.norm_ca_kv(ctx),
            self.norm_ca_kv(ctx),
            key_padding_mask=ctx_pad,
        )
        return h + self.drop(out)

    def feed_forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.drop(self.ffn(self.norm_ffn(h)))


class MultiStreamTransformer(nn.Module):
    def __init__(
        self,
        modality_names: List[str],
        measure_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        share_layers: bool = False,
    ):
        super().__init__()
        self.modality_names = list(modality_names)
        self.n_layers = n_layers
        self.share_layers = share_layers
        self.use_cross_attention = True

        self.in_proj = nn.ModuleDict(
            {k: nn.Linear(measure_dim, d_model) for k in modality_names}
        )
        self.out_proj = nn.ModuleDict(
            {k: nn.Linear(d_model, measure_dim) for k in modality_names}
        )
        self.mod_emb = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros(d_model)) for k in modality_names}
        )
        self.time_emb = TimeEmbedding(d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.in_drop = nn.Dropout(dropout)

        if share_layers:
            self.layers = nn.ModuleList(
                [TransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
            )
        else:
            self.layers = nn.ModuleDict(
                {
                    k: nn.ModuleList(
                        [
                            TransformerLayer(d_model, n_heads, dropout)
                            for _ in range(n_layers)
                        ]
                    )
                    for k in modality_names
                }
            )

    def set_use_cross_attention(self, enabled: bool):
        self.use_cross_attention = enabled

    @staticmethod
    def _zero_pad(h: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        return h.masked_fill(pad_mask.unsqueeze(-1), 0.0) if pad_mask is not None else h

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        t: torch.Tensor,
        pad_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        pad_mask = pad_mask or {}
        te = self.time_emb(t)

        # Input projection
        h = {}
        for k in self.modality_names:
            hk = self.in_proj[k](x[k]) + te.unsqueeze(1) + self.mod_emb[k]
            pe = self.pos_enc(hk.size(1), hk.device, hk.dtype)
            hk = self.in_drop(hk + pe.unsqueeze(0))
            h[k] = self._zero_pad(hk, pad_mask.get(k))

        # Transformer layers
        for i in range(self.n_layers):
            # Self-attention
            h_sa = {}
            for k in self.modality_names:
                lyr = self.layers[i] if self.share_layers else self.layers[k][i]
                h_sa[k] = self._zero_pad(
                    lyr.self_attention(h[k], pad_mask.get(k)), pad_mask.get(k)
                )

            # Cross-attention
            h_ca = {}
            if not self.use_cross_attention:
                h_ca = h_sa
            else:
                for k in self.modality_names:
                    lyr = self.layers[i] if self.share_layers else self.layers[k][i]
                    ctx_list, ctx_pad_list = [], []
                    for j in self.modality_names:
                        if j != k:
                            ctx_list.append(h_sa[j])
                            pm = pad_mask.get(j)
                            if pm is None:
                                pm = torch.zeros(
                                    h_sa[j].size(0),
                                    h_sa[j].size(1),
                                    dtype=torch.bool,
                                    device=h_sa[j].device,
                                )
                            ctx_pad_list.append(pm)
                    if ctx_list:
                        ctx = torch.cat(ctx_list, dim=1)
                        ctx_pad = torch.cat(ctx_pad_list, dim=1)
                        h_ca[k] = self._zero_pad(
                            lyr.cross_attention(h_sa[k], ctx, ctx_pad), pad_mask.get(k)
                        )
                    else:
                        h_ca[k] = h_sa[k]

            # FFN
            h_out = {}
            for k in self.modality_names:
                lyr = self.layers[i] if self.share_layers else self.layers[k][i]
                h_out[k] = self._zero_pad(lyr.feed_forward(h_ca[k]), pad_mask.get(k))
            h = h_out

        u_hat = {k: self.out_proj[k](h[k]) for k in self.modality_names}
        return u_hat, h


# ============================================================
# OmniFlow Model
# ============================================================


@dataclass
class FlowConfig:
    # Model
    measure_dim: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    share_layers: bool = False
    adapter_hidden: int = 512

    # Masking
    mask_ratio: Dict[str, float] = field(
        default_factory=lambda: {"vis": 0.5, "aud": 0.5, "txt": 0.3}
    )
    span_len: Dict[str, int] = field(
        default_factory=lambda: {"vis": 6, "aud": 6, "txt": 3}
    )
    w_masked: float = 1.0
    w_visible: float = 0.05

    # Flow
    t_max: float = 1.0
    t_gamma: float = 1.0
    normalize_by_geometry: bool = True

    # Text regularizer
    txt_usage_weight: float = 0.05

    # Geometry
    geometry: GeometryConfig = field(default_factory=GeometryConfig)


class OmniFlow(nn.Module):
    MODALITIES = ["vis", "aud", "txt"]

    def __init__(
        self,
        dims: Dict[str, int],
        cfg: Optional[FlowConfig] = None,
        alpha_init: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.cfg = cfg or FlowConfig()
        self.modality_names = self.MODALITIES

        # Geometry
        self.alpha_geo = PerModalityAlphaGeometry(
            self.modality_names,
            self.cfg.geometry,
            learnable=True,
            alpha_init=alpha_init,
        )

        # Input projections
        self.projs = nn.ModuleDict(
            {
                "vis": nn.Linear(dims["vis"], self.cfg.d_model),
                "aud": nn.Linear(dims["aud"], self.cfg.d_model),
                "txt": nn.Linear(dims["txt"], self.cfg.d_model),
            }
        )

        # Adapters
        self.adapters = nn.ModuleDict(
            {
                "vis": PositiveMeasureAdapter(
                    self.cfg.d_model, self.cfg.measure_dim, self.cfg.adapter_hidden
                ),
                "aud": PositiveMeasureAdapter(
                    self.cfg.d_model, self.cfg.measure_dim, self.cfg.adapter_hidden
                ),
                "txt": DiscreteSimplexAdapter(
                    self.cfg.d_model, self.cfg.measure_dim, self.cfg.adapter_hidden
                ),
            }
        )

        # Vector field network
        self.vf_net = MultiStreamTransformer(
            self.modality_names,
            self.cfg.measure_dim,
            self.cfg.d_model,
            self.cfg.n_layers,
            self.cfg.n_heads,
            self.cfg.dropout,
            self.cfg.share_layers,
        )

    # ----------------------
    # Control methods
    # ----------------------
    def freeze_alpha(self):
        for p in self.alpha_geo.parameters():
            p.requires_grad = False

    def unfreeze_alpha(self):
        for p in self.alpha_geo.parameters():
            p.requires_grad = True

    def set_cross_attention(self, enabled: bool):
        self.vf_net.set_use_cross_attention(enabled)

    def set_txt_gumbel(self, tau: float, hard: bool = False):
        self.adapters["txt"].set_gumbel(tau, hard)

    # >>> ADDED: freeze adapters (optional for supervised finetune)
    def freeze_adapters(self):
        for p in self.adapters.parameters():
            p.requires_grad = False

    # >>> ADDED: unfreeze adapters (if you want extreme performance)
    def unfreeze_adapters(self):
        for p in self.adapters.parameters():
            p.requires_grad = True

    # >>> ADDED: utility to freeze all parameters
    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    # >>> ADDED: make only last K vf layers + vf in/out proj trainable (for supervised finetune)
    def set_trainable_supervised_ft(
        self, last_k: int = 2, train_vf_inout_proj: bool = True
    ):
        """
        Assumes you already called freeze_all() before this.
        Will unfreeze:
          - vf_net last_k transformer layers (per modality if not share_layers)
          - vf_net.in_proj / vf_net.out_proj (optional)
        """
        last_k = max(1, int(last_k))

        # Unfreeze vf_net in/out proj
        if train_vf_inout_proj:
            for p in self.vf_net.in_proj.parameters():
                p.requires_grad = True
            for p in self.vf_net.out_proj.parameters():
                p.requires_grad = True

        # Unfreeze last_k transformer layers
        if self.vf_net.share_layers:
            # shared layers: a single ModuleList
            n = len(self.vf_net.layers)
            start = max(0, n - last_k)
            for i in range(start, n):
                for p in self.vf_net.layers[i].parameters():
                    p.requires_grad = True
        else:
            # per-modality layers: ModuleDict[modality] -> ModuleList
            for m in self.vf_net.modality_names:
                n = len(self.vf_net.layers[m])
                start = max(0, n - last_k)
                for i in range(start, n):
                    for p in self.vf_net.layers[m][i].parameters():
                        p.requires_grad = True

    # ----------------------
    # Forward methods
    # ----------------------
    def _project(
        self, vis: torch.Tensor, aud: torch.Tensor, txt: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            "vis": self.projs["vis"](vis),
            "aud": self.projs["aud"](aud),
            "txt": self.projs["txt"](txt),
        }

    def _token_weights(
        self, span_mask: torch.Tensor, pad_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        w = torch.where(span_mask, self.cfg.w_masked, self.cfg.w_visible).float()
        if pad_mask is not None:
            w = w.masked_fill(pad_mask, 0.0)
        return w

    def compute_loss(
        self,
        vis: torch.Tensor,
        aud: torch.Tensor,
        txt: torch.Tensor,
        vis_pad: Optional[torch.Tensor] = None,
        aud_pad: Optional[torch.Tensor] = None,
        txt_pad: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, device = vis.size(0), vis.device
        pad_mask = {"vis": vis_pad, "aud": aud_pad, "txt": txt_pad}
        pad_mask = {k: v for k, v in pad_mask.items() if v is not None}

        y = self._project(vis, aud, txt)
        t = (torch.rand(B, device=device) ** self.cfg.t_gamma) * self.cfg.t_max

        x_tilde, u_target, w_tok, w_geo = {}, {}, {}, {}
        loss_txt_usage = torch.tensor(0.0, device=device)

        for k in self.modality_names:
            yk, pk = y[k], pad_mask.get(k)
            Tk = yk.size(1)

            # Adapter
            if k == "txt":
                m1, probs = self.adapters[k](yk, pk, return_probs=True)
                # Text usage regularizer
                if self.cfg.txt_usage_weight > 0:
                    keep = (
                        (~pk).float().unsqueeze(-1)
                        if pk is not None
                        else torch.ones(B, Tk, 1, device=device)
                    )
                    p_bar = (probs * keep).sum(dim=(0, 1)) / keep.sum().clamp_min(1.0)
                    p_bar = p_bar.clamp_min(1e-8)
                    uniform = torch.full_like(p_bar, 1.0 / p_bar.numel())
                    loss_txt_usage = (p_bar * (p_bar.log() - uniform.log())).sum()
            else:
                m1 = self.adapters[k](yk, pk)

            # Flow
            x1 = self.alpha_geo.m_to_x(m1, k)
            x0 = self.alpha_geo.sample_prior_x(x1.shape, device, x1.dtype)
            xt = self.alpha_geo.interpolate_x(x0, x1, t)
            ut = self.alpha_geo.target_u_x(x0, x1)

            # Mask
            mr = self.cfg.mask_ratio.get(k, 0.5)
            sl = self.cfg.span_len.get(k, 6)
            s = generate_span_mask(B, Tk, mr, sl, device, pk)
            xtilde = torch.where(s.unsqueeze(-1), x0, xt)

            # Weights
            w = self._token_weights(s, pk)
            omega = self.alpha_geo.alpha_norm_weight(xt, k)

            x_tilde[k], u_target[k], w_tok[k], w_geo[k] = xtilde, ut, w, omega

        # Network forward
        u_hat, _ = self.vf_net(x_tilde, t, pad_mask)

        # Loss computation
        losses = {}
        total_fm = torch.tensor(0.0, device=device)
        for k in self.modality_names:
            err2 = (u_hat[k] - u_target[k]).pow(2)
            wt = w_tok[k].unsqueeze(-1)
            weighted_err = err2 * w_geo[k] * wt
            if self.cfg.normalize_by_geometry:
                den = (w_geo[k] * wt).sum().clamp_min(1e-8)
            else:
                den = wt.sum().clamp_min(1e-8) * self.cfg.measure_dim
            lk = weighted_err.sum() / den
            losses[f"loss_{k}"] = lk
            total_fm = total_fm + lk

        total = total_fm + self.cfg.txt_usage_weight * loss_txt_usage

        alphas = self.alpha_geo.get_all_alphas()
        return {
            "total": total,
            "loss_vis": losses["loss_vis"],
            "loss_aud": losses["loss_aud"],
            "loss_txt": losses["loss_txt"],
            "loss_txt_usage": loss_txt_usage,
            **{f"alpha_{k}": alphas[k].detach() for k in self.modality_names},
        }

    # >>> ADDED: gradient-enabled representation encoding (for supervised finetune)
    def encode_representation(
        self,
        vis: torch.Tensor,
        aud: torch.Tensor,
        txt: torch.Tensor,
        vis_pad: Optional[torch.Tensor] = None,
        aud_pad: Optional[torch.Tensor] = None,
        txt_pad: Optional[torch.Tensor] = None,
        t_star: float = 1.0,
    ) -> torch.Tensor:
        """
        Same as extract_representation, but WITH gradients (no @torch.no_grad()).
        Used for supervised finetune where we update vf_net last layers + in/out proj.
        """
        B, device = vis.size(0), vis.device
        pad_mask = {"vis": vis_pad, "aud": aud_pad, "txt": txt_pad}
        pad_mask = {k: v for k, v in pad_mask.items() if v is not None}

        y = self._project(vis, aud, txt)
        t = torch.full((B,), float(t_star), device=device)

        x_in = {}
        for k in self.modality_names:
            if k == "txt":
                m1 = self.adapters[k](y[k], pad_mask.get(k), return_probs=False)
            else:
                m1 = self.adapters[k](y[k], pad_mask.get(k))
            x_in[k] = self.alpha_geo.m_to_x(m1, k)

        _, hidden = self.vf_net(x_in, t, pad_mask)

        r_parts = []
        for k in self.modality_names:
            keep = (~pad_mask[k]) if k in pad_mask else None
            r_parts.append(masked_mean(hidden[k], keep, dim=1))

        return torch.cat(r_parts, dim=-1)

    @torch.no_grad()
    def extract_representation(
        self,
        vis: torch.Tensor,
        aud: torch.Tensor,
        txt: torch.Tensor,
        vis_pad: Optional[torch.Tensor] = None,
        aud_pad: Optional[torch.Tensor] = None,
        txt_pad: Optional[torch.Tensor] = None,
        t_star: float = 1.0,
    ) -> torch.Tensor:
        # >>> MODIFIED: reuse encode_representation (but under no_grad)
        return self.encode_representation(
            vis, aud, txt, vis_pad, aud_pad, txt_pad, t_star
        )
