"""Set encoder of REF-EvSet (class EvSetNet): set network over per-patch whitened order spectra.

token   = per-patch whitened order spectrum (+ its log-ratio to the healthy reference field)
encoder = 1-D conv over the order axis (shared by all patches)  -> d_model
RCN     = optional learned reference conditioning (the '+ FiLM' ablation, use_rcn): FiLM parameters from the set
          embedding of the reference field; the reported model uses use_rcn=False (physical reference ratio only)
set     = ISAB blocks (permutation invariant, inducing points) + PMA pooling -> classifier
gain    = low-rank head predicting a per-token scalar (pose gain / attention map, interpretable)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrderEncoder(nn.Module):
    def __init__(self, in_ch: int, d: int = 128, width: int = 48):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, w, 7, padding=3), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(w, w * 2, 5, padding=2), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(w * 2, w * 2, 5, padding=2), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(w * 2, d, 3, padding=1), nn.GELU(),
        )
        self.out = nn.Linear(2 * d, d)

    def forward(self, x):  # x: (B, P, O, C)
        B, P, O, C = x.shape
        h = self.net(x.reshape(B * P, O, C).transpose(1, 2))  # (BP, d, O/8)
        h = torch.cat([h.mean(-1), h.amax(-1)], -1)
        return self.out(h).view(B, P, -1)


class MAB(nn.Module):
    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, q, kv, need_weights=False):
        a, w = self.attn(q, kv, kv, need_weights=need_weights, average_attn_weights=True)
        h = self.ln1(q + a)
        h = self.ln2(h + self.ff(h))
        return (h, w) if need_weights else h


class ISAB(nn.Module):
    def __init__(self, d, m=16, heads=4, dropout=0.1):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, d) / math.sqrt(d))
        self.mab1 = MAB(d, heads, dropout)
        self.mab2 = MAB(d, heads, dropout)

    def forward(self, x):
        h = self.mab1(self.I.expand(x.shape[0], -1, -1), x)
        return self.mab2(x, h)


class PMA(nn.Module):
    def __init__(self, d, k=1, heads=4, dropout=0.1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, d) / math.sqrt(d))
        self.mab = MAB(d, heads, dropout)

    def forward(self, x, need_weights=False):
        return self.mab(self.S.expand(x.shape[0], -1, -1), x, need_weights=need_weights)


class SubPatchEncoder(nn.Module):
    """Encodes the 8x8 x (2 polarities x {target, target - reference}) sub-patch event image of every token."""

    def __init__(self, in_ch=4, d=128, w=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, 2 * w, 3, padding=1), nn.GELU(),
                                 nn.MaxPool2d(2), nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.GELU())
        self.out = nn.Linear(4 * w, d)

    def forward(self, s):  # (B, P, C, 8, 8)
        B, P = s.shape[:2]
        h = self.net(s.reshape(B * P, *s.shape[2:]))
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], -1)
        return self.out(h).view(B, P, -1)


class GridAggregator(nn.Module):
    """'Grid' ablation: the token embeddings are placed back on the (gh, gw) sensor grid and aggregated by a
    2-D CNN (translation-equivariant, position-dependent) instead of permutation-invariant set attention."""

    def __init__(self, d, grid=(30, 40), dropout=0.1):
        super().__init__()
        self.grid = grid
        self.net = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Conv2d(d, d, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Conv2d(d, d, 3, padding=1), nn.GELU(), nn.Dropout(dropout))

    def forward(self, h):  # (B, P, d) with P = gh*gw in row-major order (full token set, no subsampling)
        B, P, d = h.shape
        x = h.transpose(1, 2).reshape(B, d, *self.grid)
        f = self.net(x)
        return torch.cat([f.mean((2, 3)), f.amax((2, 3))], -1)


class EvSetNet(nn.Module):
    def __init__(self, n_classes=4, in_ch=4, n_attr=6, d=128, n_isab=2, m=16, heads=4, dropout=0.1,
                 use_rcn=True, use_attr=True, ref_ch=2, use_sub=False, sub_ch=4, agg="set", grid=(30, 40)):
        super().__init__()
        self.agg = agg
        if agg == "grid":
            self.grid_agg = GridAggregator(d, grid, dropout)
            self.grid_head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_classes))
        self.enc = OrderEncoder(in_ch, d)
        self.use_sub = use_sub
        self.sub_enc = SubPatchEncoder(sub_ch, d) if use_sub else None
        self.use_attr, self.use_rcn = use_attr, use_rcn
        self.attr = nn.Sequential(nn.Linear(n_attr, d), nn.GELU(), nn.Linear(d, d)) if use_attr else None
        if use_rcn:
            self.ref_enc = OrderEncoder(ref_ch, d)
            self.ref_pool = PMA(d, 1, heads, dropout)
            self.film = nn.Linear(d, 2 * d)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.ln_in = nn.LayerNorm(d)
        self.isabs = nn.ModuleList([ISAB(d, m, heads, dropout) for _ in range(n_isab)])
        self.pma = PMA(d, 1, heads, dropout)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_classes))
        self.gain = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def forward(self, x, a, r, return_maps=False, sub=None):
        h = self.enc(x)
        if self.use_attr:
            h = h + self.attr(a)
        if self.use_sub and sub is not None:
            h = h + self.sub_enc(sub)
        if self.use_rcn:
            rh = self.ref_enc(r)
            if self.use_attr:
                rh = rh + self.attr(a)
            e = self.ref_pool(rh)[:, 0]  # (B, d)
            g, b = self.film(e).chunk(2, -1)
            h = h * (1 + g[:, None]) + b[:, None]
        h = self.ln_in(h)
        if self.agg == "grid":
            z = self.grid_agg(h)
            logits = self.grid_head(z)
            if return_maps:
                return logits, z, None, None
            return logits, z
        for blk in self.isabs:
            h = blk(h)
        pooled, w = self.pma(h, need_weights=True)
        z = pooled[:, 0]
        logits = self.head(z)
        if return_maps:
            return logits, z, w[:, 0], self.gain(h)[..., 0]
        return logits, z
