from __future__ import annotations
from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from OmniFlow import (
    MultiStreamTransformer,
    AttentionPooling,
    masked_mean,
    generate_span_mask,
    FlowConfig,
)


class VectorQuantizer(nn.Module):
    def __init__(self, dim: int, k: int = 1024, mode: str = "kmeans"):
        super().__init__()
        self.dim = dim
        self.k = k
        self.mode = mode
        self.register_buffer("codebook", torch.randn(k, dim))
        self.initialized = False

    def initialize(self, data: torch.Tensor):
        # data: (N, D)
        if self.initialized:
            return

        if self.mode == "kmeans":
            print(f"Initializing VQ with K-Means (K={self.k})...")
            # Simple PyTorch K-Means
            try:
                from sklearn.cluster import MiniBatchKMeans

                km = MiniBatchKMeans(n_clusters=self.k, n_init=3, batch_size=2048)
                km.fit(data.cpu().numpy())
                self.codebook.data.copy_(torch.from_numpy(km.cluster_centers_))
            except ImportError:
                print("sklearn not found. Using random init.")
                self._random_init(data)
        else:
            print("Initializing VQ with Uniform (Random) Codebook...")
            self._random_init(data)

        self.initialized = True

    def _random_init(self, data):
        # Randomly select samples as centroids
        n = data.size(0)
        indices = torch.randperm(n)[: self.k]
        if len(indices) < self.k:
            # duplicate if not enough data
            indices = torch.cat([indices, indices])[: self.k]
        self.codebook.data.copy_(data[indices])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        # returns: (B, T) indices
        B, T, D = x.shape
        flat_x = x.view(-1, D)

        # L2 distance: |x-c|^2 = |x|^2 + |c|^2 - 2xc
        dist = (
            torch.sum(flat_x**2, dim=1, keepdim=True)
            + torch.sum(self.codebook**2, dim=1)
            - 2 * torch.matmul(flat_x, self.codebook.t())
        )
        indices = torch.argmin(dist, dim=1).view(B, T)
        return indices


class DiscreteFlow(nn.Module):
    MODALITIES = ["vis", "aud", "txt"]

    def __init__(
        self,
        dims: Dict[str, int],
        cfg: Optional[FlowConfig] = None,
        quantizer_k: Optional[int] = None,
        quantizer_mode: str = "kmeans",
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

        # Use measure_dim as the default codebook/vocab size; allow optional override via quantizer_k
        self.vocab_size = (
            quantizer_k if quantizer_k is not None else self.cfg.measure_dim
        )
        self.cfg.measure_dim = self.vocab_size

        # Quantizers (Only for continuous inputs)
        self.quantizers = nn.ModuleDict(
            {
                k: VectorQuantizer(dims[k], k=self.vocab_size, mode=quantizer_mode)
                for k in self.modality_names
            }
        )

        self.vf_net = MultiStreamTransformer(
            self.modality_names,
            measure_dim=self.vocab_size,  # 输入/输出直接是 Logits/Probs
            d_model=self.cfg.d_model,
            n_layers=self.cfg.n_layers,
            n_heads=self.cfg.n_heads,
            dropout=self.cfg.dropout,
            share_layers=self.cfg.share_layers,
        )

        self.attn_pool_h = nn.ModuleDict(
            {
                k: AttentionPooling(self.cfg.d_model, hidden=128)
                for k in self.modality_names
            }
        )
        self.attn_pool_v = nn.ModuleDict(
            {
                k: AttentionPooling(self.cfg.d_model, hidden=128)
                for k in self.modality_names
            }
        )
        # Note: vel_proj now maps from vocab_size -> d_model
        self.vel_proj = nn.ModuleDict(
            {
                k: nn.Linear(self.vocab_size, self.cfg.d_model)
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
        pass  # No adapters to freeze

    def unfreeze_adapters(self):
        pass

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def set_cross_attention(self, enabled: bool):
        self.vf_net.set_use_cross_attention(enabled)

    def set_txt_gumbel(self, tau: float, hard: bool = False):
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

    def init_codebooks(self, dataloader, device):
        print("Initializing codebooks...")

        # Gather some data
        data_store = {k: [] for k in self.modality_names}
        max_samples = 4096 * 4

        total = 0
        for batch_dict in dataloader:
            if total >= max_samples:
                break
            for k in self.modality_names:
                if k in batch_dict:
                    data_store[k].append(batch_dict[k])
            # Count tokens from first modality
            first_mod = next(iter(self.modality_names))
            if first_mod in batch_dict:
                total += batch_dict[first_mod].size(0) * batch_dict[first_mod].size(1)

        for k in self.modality_names:
            if len(data_store[k]) > 0:
                all_data = torch.cat(data_store[k], dim=0).view(
                    -1, data_store[k][0].size(-1)
                )
                self.quantizers[k].initialize(all_data)

        print("Codebooks initialized.")

    def _get_indices(self, inputs: Dict[str, torch.Tensor]):
        """Get VQ indices for each modality."""
        out = {}
        with torch.no_grad():
            for k in self.modality_names:
                if k in inputs:
                    out[k] = self.quantizers[k](inputs[k])
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
                - labels: (B,) labels
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
            raise ValueError("No modality provided to DiscreteFlow.compute_loss")

        indices = self._get_indices(inputs)
        if len(indices) == 0:
            raise ValueError("No active modalities for loss computation")

        t = torch.rand(B, device=device)
        t = (t**self.cfg.t_gamma) * self.cfg.t_max
        t_view = t.view(-1, 1, 1)

        x_tilde, u_target, weights = {}, {}, {}
        active = 0

        for k in self.modality_names:
            if k not in indices:
                continue
            idx = indices[k]
            pk = pad_mask.get(k)

            # Fixed One-Hot Target (x1)
            x1 = F.one_hot(idx, num_classes=self.vocab_size).float()
            x0 = torch.full_like(x1, 1.0 / self.vocab_size)

            xt = (1.0 - t_view) * x0 + t_view * x1
            ut = x1 - x0

            # Masking
            mr = self.cfg.mask_ratio.get(k, 0.5)
            sl = self.cfg.span_len.get(k, 6)
            s = generate_span_mask(B, x1.size(1), mr, sl, device, pk)

            # Construct masked input: masked -> x0 (Uniform), visible -> xt
            xtilde_k = torch.where(s.unsqueeze(-1), x0, xt)

            x_tilde[k] = xtilde_k
            u_target[k] = ut
            weights[k] = self._token_weights(s, pk)
            active += 1

        if active == 0:
            raise ValueError("No active modalities for loss computation")

        # Forward
        u_hat, _ = self.vf_net(x_tilde, t, pad_mask)

        losses = {}
        total = 0.0
        for k in self.modality_names:
            if k not in u_hat:
                continue
            diff = u_hat[k] - u_target[k]
            err = (diff.pow(2)).mean(dim=-1) * self.vocab_size

            w = weights[k]
            loss_k = (err * w).sum() / w.sum().clamp_min(1e-8)
            losses[f"loss_{k}"] = loss_k
            total += loss_k

        # Build return dict with all possible modality keys for compatibility,
        # while also exposing actual modality-specific losses.
        result: Dict[str, torch.Tensor] = {
            "total": total,
            "loss_txt_usage": torch.tensor(0.0, device=device),
        }

        # Include losses and alphas for all active modalities (e.g., timeseries, static).
        for k in self.modality_names:
            loss_key = f"loss_{k}"
            if loss_key in losses:
                result[loss_key] = losses[loss_key]
                result[f"alpha_{k}"] = torch.tensor(1.0, device=device)

        # Backward-compatible vis/aud/txt keys expected by some training loops.
        for legacy_k in ["vis", "aud", "txt"]:
            loss_key = f"loss_{legacy_k}"
            if loss_key not in result:
                # Use zero when that modality is not present.
                result[loss_key] = torch.tensor(0.0, device=device)
            if f"alpha_{legacy_k}" not in result:
                result[f"alpha_{legacy_k}"] = torch.tensor(1.0, device=device)
        return result

    def encode_representation(
        self,
        batch: Dict[str, torch.Tensor],
        t_star: float = 1.0,
        rep_mode: str = "hidden_attn",
        vel_proj: bool = True,
        use_layernorm: bool = True,
    ) -> torch.Tensor:
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
                "No modality provided to DiscreteFlow.encode_representation"
            )

        indices = self._get_indices(inputs)
        if len(indices) == 0:
            raise ValueError("No active modalities for representation encoding")
        t = torch.full((B,), float(t_star), device=device)

        # quantization to one-hot
        x1_all = {}
        for k in self.modality_names:
            if k not in indices:
                continue
            x1_all[k] = F.one_hot(indices[k], num_classes=self.vocab_size).float()

        if rep_mode in ("hidden_mean", "hidden_attn", "hidden_attn_vel_x1"):
            x_in = x1_all
        elif rep_mode == "hidden_attn_vel_detprior":
            x_in = {}
            for k in x1_all:
                x0 = torch.full_like(x1_all[k], 1.0 / self.vocab_size)
                x_in[k] = (1.0 - t_star) * x0 + t_star * x1_all[k]
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

    @torch.no_grad()
    def extract_representation(
        self,
        batch: Dict[str, torch.Tensor],
        t_star: float = 1.0,
        rep_mode: str = "hidden_attn",
        vel_proj: bool = True,
        use_layernorm: bool = True,
    ) -> torch.Tensor:
        return self.encode_representation(
            batch,
            t_star=t_star,
            rep_mode=rep_mode,
            vel_proj=vel_proj,
            use_layernorm=use_layernorm,
        )
