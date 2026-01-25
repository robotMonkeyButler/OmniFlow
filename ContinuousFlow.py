from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn

from OmniFlow import (
    MultiStreamTransformer,
    AttentionPooling,
    masked_mean,
    generate_span_mask,
    FlowConfig,
)


class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(
        self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.net(x)
        if pad_mask is not None:
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return out

    def set_gumbel(self, tau: float, hard: bool = False, use_gumbel: bool = True):
        pass


class ContinuousFlow(nn.Module):
    MODALITIES = ["vis", "aud", "txt"]

    def __init__(
        self,
        dims: Dict[str, int],
        cfg: Optional[FlowConfig] = None,
        modality_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.cfg = cfg or FlowConfig()
        available = list(dims.keys())
        if modality_names is None or len(modality_names) == 0:
            self.modality_names = available
        else:
            missing = [m for m in modality_names if m not in dims]
            if missing:
                raise ValueError(f"Missing dims for modalities: {missing}")
            self.modality_names = modality_names

        self.projs = nn.ModuleDict()
        for k in self.modality_names:
            # Use Linear for continuous features, 'txt' also inputs continuous embeddings
            self.projs[k] = nn.Linear(dims[k], self.cfg.d_model)

        self.projectors = nn.ModuleDict(
            {
                k: Projector(self.cfg.d_model, self.cfg.measure_dim)
                for k in self.modality_names
            }
        )

        self.vf_net = MultiStreamTransformer(
            self.modality_names,
            self.cfg.measure_dim,
            self.cfg.d_model,
            self.cfg.n_layers,
            self.cfg.n_heads,
            self.cfg.dropout,
            self.cfg.share_layers,
        )

        self.attn_pool_h = nn.ModuleDict(
            {
                k: AttentionPooling(self.cfg.d_model, hidden=128)
                for k in self.modality_names
            }
        )
        self.vel_proj = nn.ModuleDict(
            {
                k: nn.Linear(self.cfg.measure_dim, self.cfg.d_model)
                for k in self.modality_names
            }
        )
        self.attn_pool_v = nn.ModuleDict(
            {
                k: AttentionPooling(self.cfg.d_model, hidden=128)
                for k in self.modality_names
            }
        )
        self.rep_ln = nn.ModuleDict(
            {k: nn.LayerNorm(self.cfg.d_model * 2) for k in self.modality_names}
        )

    # ----------------------
    # Control methods (Compatibility)
    # ----------------------
    def freeze_alpha(self):
        pass

    def unfreeze_alpha(self):
        pass

    def freeze_adapters(self):
        # Back-compat: treat projectors as former adapters
        for p in self.projectors.parameters():
            p.requires_grad = False

    def unfreeze_adapters(self):
        # Back-compat: treat projectors as former adapters
        for p in self.projectors.parameters():
            p.requires_grad = True

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def set_cross_attention(self, enabled: bool):
        self.vf_net.set_use_cross_attention(enabled)

    def set_txt_gumbel(self, tau: float, hard: bool = False):
        # IdentityAdapter has set_gumbel (noop)
        pass

    def set_trainable_supervised_ft(
        self, last_k: int = 2, train_vf_inout_proj: bool = True
    ):
        last_k = max(1, int(last_k))
        if train_vf_inout_proj:
            for p in self.vf_net.in_proj.parameters():
                p.requires_grad = True
            for p in self.vf_net.out_proj.parameters():
                p.requires_grad = True

        if self.vf_net.share_layers:
            n = len(self.vf_net.layers)
            start = max(0, n - last_k)
            for i in range(start, n):
                for p in self.vf_net.layers[i].parameters():
                    p.requires_grad = True
        else:
            for m in self.vf_net.modality_names:
                n = len(self.vf_net.layers[m])
                start = max(0, n - last_k)
                for i in range(start, n):
                    for p in self.vf_net.layers[m][i].parameters():
                        p.requires_grad = True

    def _project(self, inputs: Dict[str, torch.Tensor]):
        """Project input modalities to d_model."""
        out = {}
        for k in self.modality_names:
            if k in inputs:
                out[k] = self.projs[k](inputs[k])
        return out

    def _token_weights(self, span_mask, pad_mask):
        w = torch.where(span_mask, self.cfg.w_masked, self.cfg.w_visible).float()
        if pad_mask is not None:
            w = w.masked_fill(pad_mask, 0.0)
        return w

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        """
        Compute flow matching loss.
        
        Args:
            batch: Dict with keys:
                - <modality_name>: (B, T, D) tensors
                - <modality_name>_pad: (B, T) bool masks
                - labels: (B,) labels (not used in unsupervised loss)
        """
        device = batch["labels"].device
        B = batch["labels"].size(0)
        
        # Extract modalities and padding masks
        inputs = {}
        pad_mask = {}
        for k in self.modality_names:
            if k in batch:
                inputs[k] = batch[k]
                pad_mask[k] = batch.get(f"{k}_pad")
        
        if len(inputs) == 0:
            raise ValueError("No modality provided to ContinuousFlow.compute_loss")

        y = self._project(inputs)
        # Random time t, biased slightly towards 1 if t_gamma < 1
        t = torch.rand(B, device=device)
        if self.cfg.t_gamma != 1.0:
            t = (t**self.cfg.t_gamma) * self.cfg.t_max
        else:
            t = t * self.cfg.t_max

        x_tilde, u_target, weights = {}, {}, {}

        active = 0
        for k in self.modality_names:
            if k not in y:
                continue
            pk = pad_mask.get(k)
            # Target x1
            x1 = self.projectors[k](y[k], pk)
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
            active += 1

        # Forward
        if active == 0:
            raise ValueError("No active modalities for loss computation")

        u_hat, _ = self.vf_net(x_tilde, t, pad_mask)

        # Loss
        losses = {}
        total = 0.0
        for k in self.modality_names:
            if k not in u_hat:
                continue
            err = (u_hat[k] - u_target[k]).pow(2)
            w = weights[k].unsqueeze(-1)
            # Euclidean: no geometry weight
            loss_k = (err * w).sum() / (
                w.sum().clamp_min(1e-8) * self.cfg.measure_dim
            )

            losses[f"loss_{k}"] = loss_k
            total = total + loss_k

        # Build return dict with all possible modality keys for compatibility
        result = {"total": total, "loss_txt_usage": torch.tensor(0.0, device=device)}

        # Include actual modality-specific losses (e.g., loss_timeseries, loss_static)
        for k in self.modality_names:
            loss_key = f"loss_{k}"
            if loss_key in losses:
                result[loss_key] = losses[loss_key]
                # Euclidean geometry: alpha is fixed at -1.0 for all modalities
                result[f"alpha_{k}"] = torch.tensor(-1.0, device=device)

        # Legacy keys for compatibility with code expecting vis/aud/txt metrics
        for k in ["vis", "aud", "txt"]:
            result[f"loss_{k}"] = losses.get(f"loss_{k}", torch.tensor(0.0, device=device))
            result[f"alpha_{k}"] = torch.tensor(-1.0, device=device)  # Euclidean
        return result

    def encode_representation(
        self,
        batch: Dict[str, torch.Tensor],
        t_star=1.0,
        rep_mode="hidden_attn",
        vel_proj=True,
        use_layernorm=True,
    ):
        """
        Encode representation from batch.
        
        Args:
            batch: Dict with modality tensors and padding masks
            t_star: Time point for encoding
            rep_mode: Representation mode
            vel_proj: Whether to project velocity
            use_layernorm: Whether to use layer normalization
        """
        device = batch["labels"].device
        B = batch["labels"].size(0)
        
        # Extract modalities and padding masks
        inputs = {}
        pad_mask = {}
        for k in self.modality_names:
            if k in batch:
                inputs[k] = batch[k]
                pad_mask[k] = batch.get(f"{k}_pad")
        
        if len(inputs) == 0:
            raise ValueError(
                "No modality provided to ContinuousFlow.encode_representation"
            )

        y = self._project(inputs)
        t = torch.full((B,), float(t_star), device=device)

        x1 = {}
        for k in self.modality_names:
            if k not in y:
                continue
            x1[k] = self.projectors[k](y[k], pad_mask.get(k))

        if len(x1) == 0:
            raise ValueError("No active modalities for representation encoding")

        # Input to VF
        if rep_mode in ("hidden_mean", "hidden_attn", "hidden_attn_vel_x1"):
            x_in = x1
        elif rep_mode == "hidden_attn_vel_detprior":
            # x0 mean is 0 for Standard Normal
            x_in = {}
            for k in x1:
                x0 = torch.zeros_like(x1[k])
                x_in[k] = (1.0 - t_star) * x0 + t_star * x1[k]
        else:
            raise ValueError(f"Unknown mode {rep_mode}")

        u_hat, hidden = self.vf_net(x_in, t, pad_mask)

        parts = []
        if rep_mode == "hidden_mean":
            for k in self.modality_names:
                if k not in hidden:
                    continue
                mask_k = pad_mask.get(k)
                keep = (~mask_k) if mask_k is not None else None
                h = masked_mean(hidden[k], keep, dim=1)
                parts.append(h)
            if len(parts) == 0:
                raise ValueError("No hidden states available for pooling")
            return torch.cat(parts, dim=-1)

        if rep_mode == "hidden_attn":
            for k in self.modality_names:
                if k not in hidden:
                    continue
                h = self.attn_pool_h[k](hidden[k], pad_mask.get(k))
                parts.append(h)
            if len(parts) == 0:
                raise ValueError("No hidden states available for attention pooling")
            return torch.cat(parts, dim=-1)

        for k in self.modality_names:
            if k not in hidden or k not in u_hat:
                continue
            h_pool = self.attn_pool_h[k](hidden[k], pad_mask.get(k))
            v = u_hat[k]
            if vel_proj:
                v = self.vel_proj[k](v)
            v_pool = self.attn_pool_v[k](v, pad_mask.get(k))
            hv = torch.cat([h_pool, v_pool], dim=-1)
            if use_layernorm:
                hv = self.rep_ln[k](hv)
            parts.append(hv)

        if len(parts) == 0:
            raise ValueError("No representations available to concatenate")
        return torch.cat(parts, dim=-1)
