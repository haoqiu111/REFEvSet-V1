"""Level-2 features: windowed, order-normalized spectral tokens per patch (the 2D / set representation).

For a window of the level-1 counts (2, P, L) we compute
  * global signed / unsigned rate -> shaft-frequency estimate f_r (harmonic-sum search around the nominal),
  * per-patch DFT on an exact order grid  X(p, o) = sum_t w(t) r(p, t) exp(-2 pi i o f_r t),
  * token channels: log|X_s|, log|X_u| (signed / unsigned = velocity-linear / rectified channel),
    cos, sin of the phase relative to the global signed spectrum (rigid rotation vs local impacts),
  * token attributes: grid position, log mean unsigned rate (local gain), polarity balance.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _dft_matrix(L: int, bin_us: int, freqs_hz: torch.Tensor, device) -> torch.Tensor:
    t = torch.arange(L, device=device, dtype=torch.float64) * (bin_us * 1e-6)
    w = torch.hann_window(L, periodic=False, device=device, dtype=torch.float64)
    ang = -2 * math.pi * t[:, None] * freqs_hz[None, :].to(torch.float64)
    M = torch.polar(w[:, None].expand_as(ang), ang)  # complex128 (L, F)
    return M.to(torch.complex64)


def estimate_shaft_freq(gs: torch.Tensor, gu: torch.Tensor, bin_us: int, f_nom: float, rel_range: float = 0.2,
                        step_hz: float = 0.01, n_harm: int = 8) -> tuple[float, float]:
    """Harmonic-sum search of f_r in [f_nom(1-rel), f_nom(1+rel)]. Returns (f_r, peak_ratio)."""
    L = gs.shape[-1]
    dev = gs.device
    cands = torch.arange(f_nom * (1 - rel_range), f_nom * (1 + rel_range), step_hz, device=dev)
    freqs = (cands[:, None] * torch.arange(1, n_harm + 1, device=dev)[None, :]).reshape(-1)
    M = _dft_matrix(L, bin_us, freqs, dev)
    sig = torch.stack([gs - gs.mean(), gu - gu.mean()]).to(torch.complex64)  # (2, L)
    X = (sig @ M).abs().view(2, len(cands), n_harm)  # (2, C, K)
    # normalise each harmonic row by its median across candidates so no single harmonic dominates
    X = X / (X.median(dim=1, keepdim=True).values + 1e-6)
    score = X.sum(dim=(0, 2))
    i = int(score.argmax())
    ratio = float(score[i] / (score.median() + 1e-9))
    return float(cands[i]), ratio


@torch.no_grad()
def window_tokens(counts_win: np.ndarray, bin_us: int, f_nom: float, orders: torch.Tensor, device="cuda",
                  f_r: float | None = None):
    """counts_win: (2, P, L) uint8. Returns dict of tensors (on CPU, float16 where large)."""
    dev = torch.device(device)
    c = torch.from_numpy(np.ascontiguousarray(counts_win)).to(dev).float()  # (2, P, L)
    pos, neg = c[1], c[0]  # Prophesee p=1 positive (ON)
    rs = pos - neg
    ru = pos + neg
    gs, gu = rs.sum(0), ru.sum(0)
    if f_r is None:
        f_r, ratio = estimate_shaft_freq(gs, gu, bin_us, f_nom)
    else:
        ratio = float("nan")
    L = c.shape[-1]
    M = _dft_matrix(L, bin_us, orders.to(dev) * f_r, dev)  # (L, O)
    Xs = (rs - rs.mean(-1, keepdim=True)).to(torch.complex64) @ M  # (P, O)
    Xu = (ru - ru.mean(-1, keepdim=True)).to(torch.complex64) @ M
    Gs = (gs - gs.mean()).to(torch.complex64) @ M  # (O,)
    Gu = (gu - gu.mean()).to(torch.complex64) @ M
    rel = Xs * torch.conj(Gs)[None, :]
    rel = rel / (rel.abs() + 1e-6)
    tok = torch.stack([torch.log(Xs.abs() + 1e-3), torch.log(Xu.abs() + 1e-3), rel.real, rel.imag], dim=-1)  # (P,O,4)
    mean_u = ru.mean(-1)
    attrs = torch.stack([torch.log(mean_u + 1e-4), (pos.sum(-1) - neg.sum(-1)) / (ru.sum(-1) + 1e-6)], dim=-1)
    glob = torch.stack([torch.log(Gs.abs() + 1e-3), torch.log(Gu.abs() + 1e-3)], dim=0)  # (2, O)
    return dict(tokens=tok.half().cpu(), attrs=attrs.float().cpu(), glob=glob.float().cpu(),
                f_r=f_r, f_ratio=ratio, rate=float(ru.sum() / (L * bin_us * 1e-6)))


def window_starts(valid_s: np.ndarray, bins_per_s: int, win_bins: int, hop_bins: int, T: int):
    """Window start bins fully inside valid seconds."""
    starts = []
    s = 0
    while s + win_bins <= T:
        secs = range(s // bins_per_s, (s + win_bins - 1) // bins_per_s + 1)
        if all(valid_s[k] for k in secs if k < len(valid_s)):
            starts.append(s)
        s += hop_bins
    return np.array(starts, dtype=np.int64)
