"""Level-1 cache: per-patch, per-polarity event counts on a fine time grid.

For a recording with sensor size (H, W), patch size s and time bin dt (us), the
cache holds counts[pol, patch, tbin] (uint8) with patch = (y // s) * (W // s) + x // s.
This is the "1D" representation of the paper: one signed rate signal per block.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from ..data.dat_reader import parse_header, read_raw, decode


def build_patch_counts(path: str, patch: int = 16, bin_us: int = 100, t_start_us: int | None = None,
                       dur_s: float | None = None, chunk: int = 20_000_000, device: str = "cuda",
                       min_events_per_s: float = 0.0):
    """Return dict with counts (2, P, T) uint8, grid shape, valid-second mask, meta."""
    h = parse_header(path)
    H, W = h.height, h.width
    gh, gw = H // patch, W // patch
    P = gh * gw
    # pass 1: time range
    if t_start_us is None:
        rec = read_raw(path, 0, min(2_000_000, h.n_events))
        t_start_us = int(np.asarray(rec["t"]).min())
    # find end time: scan last chunk (timestamps nearly monotonic)
    rec = read_raw(path, max(0, h.n_events - 2_000_000), min(2_000_000, h.n_events))
    t_end_us = int(np.asarray(rec["t"]).max())
    if dur_s is not None:
        t_end_us = min(t_end_us, t_start_us + int(dur_s * 1e6))
    T = int((t_end_us - t_start_us) // bin_us) + 1
    use_cpu = device == "cpu"
    if use_cpu:
        acc_np = np.zeros((2, P, T), dtype=np.uint16)
    else:
        dev = torch.device(device)
        acc = torch.zeros(2 * P * T, dtype=torch.int32, device=dev)
    n_used = 0
    t0 = time.time()
    for s in range(0, h.n_events, chunk):
        rec = np.asarray(read_raw(path, s, min(chunk, h.n_events - s)))
        t, x, y, p = decode(rec)
        tb = (t - t_start_us) // bin_us
        ok = (tb >= 0) & (tb < T) & (x < gw * patch) & (y < gh * patch)
        if not ok.all():
            t, x, y, p, tb = t[ok], x[ok], y[ok], p[ok], tb[ok]
        if len(tb) == 0:
            if t.size and (t - t_start_us).min() // bin_us >= T:
                break
            continue
        pid = (y.astype(np.int64) // patch) * gw + (x.astype(np.int64) // patch)
        if use_cpu:
            tb0, tb1 = int(tb.min()), int(tb.max()) + 1
            Tc = tb1 - tb0
            lin = (p.astype(np.int64) * P + pid) * Tc + (tb - tb0)
            bc = np.bincount(lin, minlength=2 * P * Tc).reshape(2, P, Tc)
            acc_np[:, :, tb0:tb1] += np.minimum(bc, 65535).astype(np.uint16)
        else:
            lin = (p.astype(np.int64) * P + pid) * T + tb
            lin_t = torch.from_numpy(lin).to(dev)
            acc.index_put_((lin_t,), torch.ones_like(lin_t, dtype=torch.int32), accumulate=True)
        n_used += len(lin)
    if use_cpu:
        counts = np.minimum(acc_np, 255).astype(np.uint8)
        del acc_np
    else:
        counts = acc.view(2, P, T).clamp_(max=255).to(torch.uint8).cpu().numpy()
        del acc
    # per-second event totals -> validity mask (acquisition holes)
    bins_per_s = int(1e6 // bin_us)
    n_s = int(np.ceil(T / bins_per_s))
    tot = counts.sum(axis=(0, 1), dtype=np.int64)
    sec_tot = np.add.reduceat(tot, np.arange(0, T, bins_per_s))
    valid_s = sec_tot >= min_events_per_s
    meta = dict(path=path, height=H, width=W, patch=patch, grid=(gh, gw), bin_us=bin_us,
                t_start_us=t_start_us, T=T, n_events_used=n_used, n_events_file=h.n_events,
                sec_events=sec_tot.tolist(), build_s=time.time() - t0)
    return dict(counts=counts, valid_s=valid_s, meta=meta)


def save_cache(out_path: str, res: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, counts=res["counts"], valid_s=res["valid_s"], meta=np.array(res["meta"], dtype=object))


def load_cache(path: str, mmap: bool = True):
    z = np.load(path, allow_pickle=True, mmap_mode="r" if mmap else None)
    return dict(counts=z["counts"], valid_s=z["valid_s"], meta=z["meta"].item())
