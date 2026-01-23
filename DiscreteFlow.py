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
    DiscreteSimplexAdapter,
    FlowConfig
)

class VectorQuantizer(nn.Module):
    def __init__(self, dim: int, k: int = 1024, mode: str = 'kmeans'):
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
        
        if self.mode == 'kmeans':
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
        indices = torch.randperm(n)[:self.k]
        if len(indices) < self.k:
            # duplicate if not enough data
             indices = torch.cat([indices, indices])[:self.k]
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

    def __init__(self, dims: Dict[str, int], cfg: Optional[FlowConfig] = None, quantizer_k: int = 1024, quantizer_mode: str = 'kmeans'):
        super().__init__()
        self.cfg = cfg or FlowConfig()
        self.modality_names = ["vis", "aud", "txt"] # Explicit order
        
        self.vocab_size = quantizer_k
        self.cfg.measure_dim = self.vocab_size 

        # Quantizers (Only for continuous inputs)
        self.quantizers = nn.ModuleDict({
            k: VectorQuantizer(dims[k], k=quantizer_k, mode=quantizer_mode)
            for k in self.modality_names
        })

        
        self.vf_net = MultiStreamTransformer(
            self.modality_names,
            measure_dim=self.vocab_size, # 输入/输出直接是 Logits/Probs
            d_model=self.cfg.d_model,
            n_layers=self.cfg.n_layers,
            n_heads=self.cfg.n_heads,
            dropout=self.cfg.dropout,
            share_layers=self.cfg.share_layers,
        )
        
        self.attn_pool_h = nn.ModuleDict({
            k: AttentionPooling(self.cfg.d_model, hidden=128) for k in self.modality_names
        })
        self.attn_pool_v = nn.ModuleDict({
            k: AttentionPooling(self.cfg.d_model, hidden=128) for k in self.modality_names
        })
        # Note: vel_proj now maps from vocab_size -> d_model
        self.vel_proj = nn.ModuleDict({
            k: nn.Linear(self.vocab_size, self.cfg.d_model) for k in self.modality_names
        })
        self.rep_ln = nn.ModuleDict({
             k: nn.LayerNorm(self.cfg.d_model * 2) for k in self.modality_names
        })

    # ----------------------
    # Control methods (Compatibility)
    # ----------------------
    def freeze_alpha(self): pass
    def unfreeze_alpha(self): pass
    
    def freeze_adapters(self): pass # No adapters to freeze
    def unfreeze_adapters(self): pass
    
    def freeze_all(self):
        for p in self.parameters(): p.requires_grad = False
        
    def set_cross_attention(self, enabled: bool):
        self.vf_net.set_use_cross_attention(enabled)

    def set_txt_gumbel(self, tau: float, hard: bool = False): pass

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

    def init_codebooks(self, dataloader, device):
        print("Initializing codebooks...")
        
        # Gather some data
        data_store = {k: [] for k in self.modality_names} 
        max_samples = 4096 * 4
        
        total = 0
        for batch in dataloader:
            if total >= max_samples: break
            vis, aud, txt, _, _, _, _ = batch
            
            data_store["vis"].append(vis)
            data_store["aud"].append(aud)
            data_store["txt"].append(txt)
            total += vis.size(0) * vis.size(1) # num tokens
        
        for k in self.modality_names:
            if len(data_store[k]) > 0:
                all_data = torch.cat(data_store[k], dim=0).view(-1, data_store[k][0].size(-1))
                self.quantizers[k].initialize(all_data)
        
        print("Codebooks initialized.")

    def _get_indices(self, vis, aud, txt):
        with torch.no_grad():
            idx_vis = self.quantizers["vis"](vis)
            idx_aud = self.quantizers["aud"](aud)
            idx_txt = self.quantizers["txt"](txt)
        
        return {"vis": idx_vis, "aud": idx_aud, "txt": idx_txt}

    def _token_weights(self, span_mask, pad_mask):
        w = torch.where(span_mask, self.cfg.w_masked, self.cfg.w_visible).float()
        if pad_mask is not None:
            w = w.masked_fill(pad_mask, 0.0)
        return w

    def compute_loss(self, vis, aud, txt, vis_pad=None, aud_pad=None, txt_pad=None):
        B, device = vis.size(0), vis.device
        pad_mask = {"vis": vis_pad, "aud": aud_pad, "txt": txt_pad}
        
        indices = self._get_indices(vis, aud, txt)
        
        t = torch.rand(B, device=device)
        t = (t ** self.cfg.t_gamma) * self.cfg.t_max
        t_view = t.view(-1, 1, 1)

        x_tilde, u_target, weights = {}, {}, {}
        
        for k in self.modality_names:
            idx = indices[k]
            pk = pad_mask.get(k)
            
            #  Fixed One-Hot Target (x1)
            # x1 shape: (B, T, 1024)
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

        # Forward
        # vf_net 内部需要处理 (B, T, vocab_size) -> d_model 的输入投影 
        # (MultiStreamTransformer 已用 measure_dim 初始化了输入层)
        u_hat, _ = self.vf_net(x_tilde, t, pad_mask)
        
        losses = {}
        total = 0.0
        for k in self.modality_names:
            # Loss: MSE on Probability/Logits space
            diff = u_hat[k] - u_target[k]
            # Scale by dim to keep magnitude similar to other losses
            err = (diff.pow(2)).mean(dim=-1) * self.vocab_size 
            
            w = weights[k]
            loss_k = (err * w).sum() / w.sum().clamp_min(1e-8)
            losses[f"loss_{k}"] = loss_k
            total += loss_k
            
        return {
            "total": total,
            "loss_vis": losses["loss_vis"],
            "loss_aud": losses["loss_aud"],
            "loss_txt": losses["loss_txt"],
            "loss_txt_usage": torch.tensor(0.0, device=device),
            "alpha_vis": torch.tensor(1.0, device=device),
            "alpha_aud": torch.tensor(1.0, device=device),
            "alpha_txt": torch.tensor(1.0, device=device),
        }

    def encode_representation(self, vis, aud, txt, vis_pad=None, aud_pad=None, txt_pad=None, t_star=1.0, rep_mode="hidden_attn", vel_proj=True, use_layernorm=True):
        B, device = vis.size(0), vis.device
        pad_mask = {"vis": vis_pad, "aud": aud_pad, "txt": txt_pad}
        
        indices = self._get_indices(vis, aud, txt)
        t = torch.full((B,), float(t_star), device=device)

        # 构建输入 x_in
        x_in = {}
        
        # 预先计算所有 x1 (One-Hot)
        x1_all = {}
        for k in self.modality_names:
             x1_all[k] = F.one_hot(indices[k], num_classes=self.vocab_size).float()

        if rep_mode in ("hidden_mean", "hidden_attn", "hidden_attn_vel_x1"):
            x_in = x1_all
        elif rep_mode == "hidden_attn_vel_detprior":
             # x0 mean is uniform
             for k in self.modality_names:
                 x0 = torch.full_like(x1_all[k], 1.0 / self.vocab_size)
                 x_in[k] = (1.0 - t_star) * x0 + t_star * x1_all[k]
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

