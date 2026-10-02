"""In-memory Rotor dataset of whitened per-patch order spectra (set tokens) + domain references."""
from __future__ import annotations

import os

import numpy as np
import torch

from ..eval.protocol import load_rotor_l2_meta, CLASSES, FileMeta

L_BINS_PER_S = 10000


class RotorTokens:
    """Holds log-SNR tokens for every window of every file.

    tokens : (N, P, O, 2) float16   log SNR (signed, unsigned) on a 0.25-order grid (O = 256, orders 0.25..64)
    attrs  : (N, P, 3) float32      log mean rate, polarity balance, (unused)
    """

    def __init__(self, cache_dir: str, win_s: float, order_pool: int = 2, max_order_bins: int | None = None,
                 bins_per_s: int = L_BINS_PER_S, classes: list[str] | None = None, frames_dir: str | None = None):
        self.files: list[FileMeta] = load_rotor_l2_meta(cache_dir)
        self.frames_dir = frames_dir
        self.classes = classes or CLASSES
        L = int(win_s * bins_per_s)
        sum_w2 = 0.375 * L
        toks, attrs, meta = [], [], []
        for fi, f in enumerate(self.files):
            z = np.load(f.path)
            t = z["tokens"][..., :2].astype(np.float32)  # (N,P,O,2)
            lm = z["attrs"][..., 0].astype(np.float32)
            s = 2 * t - lm[..., None, None] - np.log(sum_w2)
            N, P, O, C = s.shape
            s = s.reshape(N, P, O // order_pool, order_pool, C).mean(3)
            if max_order_bins:
                s = s[:, :, :max_order_bins]
            toks.append(s.astype(np.float16))
            attrs.append(z["attrs"].astype(np.float32))
            rm, em = f.ref_mask(), f.eval_mask()
            for i in range(N):
                meta.append((fi, self.classes.index(f.label), f.view, f.rpm, int(rm[i]), int(em[i])))
        self.tokens = np.concatenate(toks)
        self.attrs = np.concatenate(attrs)
        self.sub = None
        if frames_dir:  # sub-patch event images: (N, P, 2, 8, 8) log1p counts (2x2-binned pixels inside each 16x16 patch)
            subs = []
            for f in self.files:
                fr = np.load(os.path.join(frames_dir, f.name + ".npz"))["frames"].astype(np.float32)  # (N,2,240,320)
                n = len(fr)
                subs.append(np.log1p(fr.reshape(n, 2, 30, 8, 40, 8).transpose(0, 2, 4, 1, 3, 5).reshape(n, 1200, 2, 8, 8)).astype(np.float16))
            self.sub = np.concatenate(subs)
        m = np.array(meta, dtype=object)
        self.file_idx = m[:, 0].astype(int)
        self.y = m[:, 1].astype(int)
        self.view = m[:, 2].astype(str)
        self.rpm = m[:, 3].astype(int)
        self.is_ref = m[:, 4].astype(int).astype(bool)
        self.is_eval = m[:, 5].astype(int).astype(bool)
        self.domain = np.array([self.files[fi].domain for fi in self.file_idx])
        self.domains = sorted(set(self.domain))
        gh, gw = np.load(self.files[0].path, allow_pickle=True)["meta"].item()["grid"]
        yy, xx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        self.xy = np.stack([xx.reshape(-1) / (gw - 1), yy.reshape(-1) / (gh - 1)], 1).astype(np.float32)  # (P,2)
        self.grid = (gh, gw)
        self.orders = np.arange(1, self.tokens.shape[2] + 1) * 0.125 * order_pool
        # reference window indices per domain (healthy reference segment)
        self.ref_idx = {d: np.where((self.domain == d) & self.is_ref & (self.y == 0))[0] for d in self.domains}
        self.ref_field = {d: self.tokens[idx].astype(np.float32).mean(0) for d, idx in self.ref_idx.items()}
        self.ref_attr = {d: self.attrs[idx].mean(0) for d, idx in self.ref_idx.items()}
        self.ref_sub = {d: self.sub[idx].astype(np.float32).mean(0) for d, idx in self.ref_idx.items()} if self.sub is not None else None

    def reference(self, d: str, k: int | None = None, rng: np.random.Generator | None = None):
        """Reference field (P,O,2) and attrs (P,3): full mean or mean over k random reference windows."""
        if k is None or k >= len(self.ref_idx[d]):
            return self.ref_field[d], self.ref_attr[d]
        idx = rng.choice(self.ref_idx[d], k, replace=False)
        return self.tokens[idx].astype(np.float32).mean(0), self.attrs[idx].mean(0)


def make_item(ds: RotorTokens, i: int, ref_tok: np.ndarray, ref_attr: np.ndarray):
    """Build model input for window i given a reference field.

    x     : (P, O, 4)  [logSNR_s, logSNR_u, logSNR_s - ref_s, logSNR_u - ref_u]
    a     : (P, 6)     [x, y, log rate - ref log rate, polarity, ref logSNR_s at 1x, ref logSNR_u at 1x]
    r     : (P, O, 2)  reference field
    """
    t = ds.tokens[i].astype(np.float32)
    x = np.concatenate([t, t - ref_tok], -1)
    o1 = int(np.argmin(np.abs(ds.orders - 1.0)))
    a = np.concatenate([ds.xy, (ds.attrs[i, :, :1] - ref_attr[:, :1]), ds.attrs[i, :, 1:2],
                        ref_tok[:, o1, :]], 1).astype(np.float32)
    return x, a, ref_tok


class WindowSet(torch.utils.data.Dataset):
    def __init__(self, ds: RotorTokens, idx: np.ndarray, train: bool, ref_k: tuple[int, int] = (4, 12), seed: int = 0):
        self.ds, self.idx, self.train, self.ref_k = ds, idx, train, ref_k
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = int(self.idx[j])
        d = self.ds.domain[i]
        if self.train:
            k = int(self.rng.integers(self.ref_k[0], self.ref_k[1] + 1))
            rt, ra = self.ds.reference(d, k, self.rng)
        else:
            rt, ra = self.ds.reference(d)
        x, a, r = make_item(self.ds, i, rt, ra)
        return torch.from_numpy(x), torch.from_numpy(a), torch.from_numpy(r), int(self.ds.y[i]), i
