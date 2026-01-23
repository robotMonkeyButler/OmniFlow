from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from OmniFlow import (
    MultiStreamTransformer,
    AttentionPooling,
    masked_mean,
    generate_span_mask,
    FlowConfig,
)

class IdentityAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.net(x)
        if pad_mask is not None:
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return out

    def set_gumbel(self, tau: float, hard: bool = False, use_gumbel: bool = True):
        pass

class ContinuousFlow(nn.Module):
    MODALITIES = ["vis", "aud", "txt"]

    def __init__(self, dims: Dict[str, int], cfg: Optional[FlowConfig] = None):
        super().__init__()
        self.cfg = cfg or FlowConfig()
        self.modality_names = self.MODALITIES

        self.projs = nn.ModuleDict()
        for k in self.modality_names:
                # Use Linear for continuous features, 'txt' also inputs continuous embeddings
                self.projs[k] = nn.Linear(dims[k], self.cfg.d_model)

        self.adapters = nn.ModuleDict({
            k: IdentityAdapter(self.cfg.d_model, self.cfg.measure_dim) for k in self.modality_names
        })

        self.vf_net = MultiStreamTransformer(
            self.modality_names,
            self.cfg.measure_dim,
            self.cfg.d_model,
            self.cfg.n_layers,
            self.cfg.n_heads,
            self.cfg.dropout,
            self.cfg.share_layers,
        )

        self.attn_pool_h = nn.ModuleDict({
            k: AttentionPooling(self.cfg.d_model, hidden=128) for k in self.modality_names
        })
        self.vel_proj = nn.ModuleDict({
            k: nn.Linear(self.cfg.measure_dim, self.cfg.d_model) for k in self.modality_names
        })
        self.attn_pool_v = nn.ModuleDict({
            k: AttentionPooling(self.cfg.d_model, hidden=128) for k in self.modality_names
        })
        self.rep_ln = nn.ModuleDict({
            k: nn.LayerNorm(self.cfg.d_model * 2) for k in self.modality_names
        })

    # ----------------------
    # Control methods (Compatibility)
    # ----------------------
    def freeze_alpha(self): pass
    def unfreeze_alpha(self): pass
    
    def freeze_adapters(self):
        for p in self.adapters.parameters(): p.requires_grad = False
    
    def unfreeze_adapters(self):
        for p in self.adapters.parameters(): p.requires_grad = True
    
    def freeze_all(self):
        for p in self.parameters(): p.requires_grad = False
        
    def set_cross_attention(self, enabled: bool):
        self.vf_net.set_use_cross_attention(enabled)

    def set_txt_gumbel(self, tau: float, hard: bool = False):
        # IdentityAdapter has set_gumbel (noop)
        pass

    def set_trainable_supervised_ft(self, last_k: int = 2, train_vf_inout_proj: bool = True):
        last_k = max(1, int(last_k))
        if train_vf_inout_proj:
            for p in self.vf_net.in_proj.parameters(): p.requires_grad = True
            for p in self.vf_net.out_proj.parameters(): p.requires_grad = True
        
        if self.vf_net.share_layers:
            n = len(self.vf_net.layers)
            start = max(0, n - last_k)
            for i in range(start, n):
                for p in self.vf_net.layers[i].parameters(): p.requires_grad = True
        else:
            for m in self.vf_net.modality_names:
                n = len(self.vf_net.layers[m])
                start = max(0, n - last_k)
                for i in range(start, n):
                    for p in self.vf_net.layers[m][i].parameters(): p.requires_grad = True

    def _project(self, vis, aud, txt):
        return {
            "vis": self.projs["vis"](vis),
            "aud": self.projs["aud"](aud),
            "txt": self.projs["txt"](txt),
        }

    def _token_weights(self, span_mask, pad_mask):
        w = torch.where(span_mask, self.cfg.w_masked, self.cfg.w_visible).float()
        if pad_mask is not None:
            w = w.masked_fill(pad_mask, 0.0)
        return w

    def compute_loss(self, vis, aud, txt, vis_pad=None, aud_pad=None, txt_pad=None):
        B, device = vis.size(0), vis.device
        pad_mask = {"vis": vis_pad, "aud": aud_pad, "txt": txt_pad}
        
        y = self._project(vis, aud, txt)
        # Random time t, biased slightly towards 1 if t_gamma < 1
        t = torch.rand(B, device=device)
        if self.cfg.t_gamma != 1.0:
            t = (t ** self.cfg.t_gamma) * self.cfg.t_max
        else:
            t = t * self.cfg.t_max
        
        x_tilde, u_target, weights = {}, {}, {}
        
        for k in self.modality_names:
            pk = pad_mask.get(k)
            # Target x1
            x1 = self.adapters[k](y[k], pk)
            Tk = x1.size(1)

            # Prior x0 ~ Normal(0, 1)
            x0 = torch.randn_like(x1)

            # Interpolation
            t_reshaped = t.view(-1, 1, 1)
            xt = (1.0 - t_reshaped) * x0 + t_reshaped * x1
            ut = x1 - x0

            # Masking
            mr = self.cfg.mask_ratio.get(k, 0.5)
            sl = self.cfg.span_len.get(k, 6)
            s = generate_span_mask(B, Tk, mr, sl, device, pk)
            
            # Construct masked input: masked -> x0, visible -> xt
            xtilde_k = torch.where(s.unsqueeze(-1), x0, xt)
            x_tilde[k] = xtilde_k
            u_target[k] = ut
            
            # Loss weights
            weights[k] = self._token_weights(s, pk)

        # Forward
        u_hat, _ = self.vf_net(x_tilde, t, pad_mask)

        # Loss
        losses = {}
        total = 0.0
        for k in self.modality_names:
            err = (u_hat[k] - u_target[k]).pow(2)
            w = weights[k].unsqueeze(-1)
            # Euclidean: no geometry weight
            if self.cfg.normalize_by_geometry:
                 loss_k = (err * w).sum() / (w.sum().clamp_min(1e-8) * self.cfg.measure_dim)
            else:
                 loss_k = (err * w).sum() / (w.sum().clamp_min(1e-8) * self.cfg.measure_dim)
            
            losses[f"loss_{k}"] = loss_k
            total = total + loss_k

        return {
            "total": total,
            "loss_vis": losses["loss_vis"],
            "loss_aud": losses["loss_aud"],
            "loss_txt": losses["loss_txt"],
            "loss_txt_usage": torch.tensor(0.0, device=device), # No gumbel usage
            "alpha_vis": torch.tensor(-1.0, device=device), # Euclidean
            "alpha_aud": torch.tensor(-1.0, device=device),
            "alpha_txt": torch.tensor(-1.0, device=device),
        }

    def encode_representation(
        self,
        vis, aud, txt,
        vis_pad=None, aud_pad=None, txt_pad=None,
        t_star=1.0,
        rep_mode="hidden_attn",
        vel_proj=True,
        use_layernorm=True,
    ):
        B, device = vis.size(0), vis.device
        pad_mask = {"vis": vis_pad, "aud": aud_pad, "txt": txt_pad}
        
        y = self._project(vis, aud, txt)
        t = torch.full((B,), float(t_star), device=device)

        x1 = {}
        for k in self.modality_names:
            x1[k] = self.adapters[k](y[k], pad_mask.get(k))

        # Input to VF
        if rep_mode in ("hidden_mean", "hidden_attn", "hidden_attn_vel_x1"):
            x_in = x1
        elif rep_mode == "hidden_attn_vel_detprior":
            # x0 mean is 0 for Standard Normal
            x_in = {}
            for k in self.modality_names:
                x0 = torch.zeros_like(x1[k])
                x_in[k] = (1.0 - t_star) * x0 + t_star * x1[k]
        else:
             raise ValueError(f"Unknown mode {rep_mode}")

        u_hat, hidden = self.vf_net(x_in, t, pad_mask)

        parts = []
        if rep_mode == "hidden_mean":
             for k in self.modality_names:
                 keep = (~pad_mask[k]) if k in pad_mask else None
                 h = masked_mean(hidden[k], keep, dim=1)
                 parts.append(h)
             return torch.cat(parts, dim=-1)
        
        if rep_mode == "hidden_attn":
             for k in self.modality_names:
                 h = self.attn_pool_h[k](hidden[k], pad_mask.get(k))
                 parts.append(h)
             return torch.cat(parts, dim=-1)

        for k in self.modality_names:
            h_pool = self.attn_pool_h[k](hidden[k], pad_mask.get(k))
            v = u_hat[k]
            if vel_proj:
                v = self.vel_proj[k](v)
            v_pool = self.attn_pool_v[k](v, pad_mask.get(k))
            hv = torch.cat([h_pool, v_pool], dim=-1)
            if use_layernorm:
                hv = self.rep_ln[k](hv)
            parts.append(hv)
            
        return torch.cat(parts, dim=-1)
