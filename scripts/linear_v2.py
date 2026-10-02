"""Linear baselines on shot-noise-whitened per-patch order spectra (deployable linear baseline and oracle).

Under the null (no periodic motion) the signed-rate DFT of a Poisson counting process has
E|X(o)|^2 = sum_t w(t)^2 lambda(t) ~= mean_u * sum w^2, so
    SNR(p, o) = |X(p, o)|^2 / (mean_u(p) * sum_w2)   ~ Exp(1) under the null.
log SNR is invariant to the patch event rate (the nuisance that dominates the un-whitened features of linear_baselines.py).
Feature sets: snr_mean, snr_max, snr_frac (fraction of patches with SNR > 8), snr_refratio, snr_glob.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evset.eval.protocol import load_rotor_l2_meta  # noqa: E402
from linear_baselines import pool2, run, L2, ROOT  # noqa: E402
from evset.data.registry import PUMP_CLASSES  # noqa: E402
from evset.eval.protocol import CLASSES  # noqa: E402

SUBSET = os.environ.get("SUBSET", "rotor")

FEAT = os.path.join(ROOT, "cache", os.environ.get("L2_NAME", "rotor_l2") + "_linear_feats_v2.npz")
L_BINS = int(float(os.environ.get("L2_WIN", 0.5)) * (5000 if SUBSET == "pump" else 10000))
SUM_W2 = 0.375 * L_BINS
KEYS = ["snr_mean", "snr_max", "snr_frac", "snr_refratio", "snr_glob", "snr_all"]


def log_snr(z):
    tok = z["tokens"].astype(np.float32)
    logmean = z["attrs"][..., 0].astype(np.float32)  # (N, P)
    s = 2 * tok[..., :2] - logmean[..., None, None] - np.log(SUM_W2)  # (N, P, O, 2)
    return s, tok


def build():
    files = load_rotor_l2_meta(L2)
    ref = {}
    for f in files:
        if f.label == "Healthy":
            z = np.load(f.path)
            s, _ = log_snr(z)
            ref[f.domain] = s[f.ref_mask()].mean(0)
    feats = {k: [] for k in KEYS}
    meta = []
    for f in files:
        z = np.load(f.path)
        s, tok = log_snr(z)
        N = len(s)
        m = pool2(s.mean(1).transpose(0, 2, 1)).reshape(N, -1)
        mx = pool2(s.max(1).transpose(0, 2, 1)).reshape(N, -1)
        frac = pool2((s > np.log(8.0)).mean(1).transpose(0, 2, 1)).reshape(N, -1)
        rr = s - ref[f.domain][None]
        rr_f = np.concatenate([pool2(rr.mean(1).transpose(0, 2, 1)).reshape(N, -1),
                               pool2(rr.max(1).transpose(0, 2, 1)).reshape(N, -1)], 1)
        # global whitened spectrum: rate-weighted sum of patches -> noise power = total rate * sum_w2
        tot = np.exp(z["attrs"][..., 0].astype(np.float32)).sum(1)  # (N,)
        g = 2 * z["glob"].astype(np.float32) - np.log(tot)[:, None, None] - np.log(SUM_W2)
        g = pool2(g).reshape(N, -1)
        feats["snr_mean"].append(m); feats["snr_max"].append(mx); feats["snr_frac"].append(frac)
        feats["snr_refratio"].append(rr_f); feats["snr_glob"].append(g)
        feats["snr_all"].append(np.concatenate([m, mx, frac], 1))
        for i in range(N):
            meta.append((f.name, f.label, f.view, f.rpm, int(f.ref_mask()[i]), int(f.eval_mask()[i]), f.domain))
        print(f.name, N, flush=True)
    out = {k: np.concatenate(v).astype(np.float32) for k, v in feats.items()}
    np.savez(FEAT, meta=np.array(meta, dtype=object), **out)
    return out, meta


if __name__ == "__main__":
    if os.path.exists(FEAT) and "--rebuild" not in sys.argv:
        z = np.load(FEAT, allow_pickle=True)
        feats = {k: z[k] for k in KEYS}
        meta = [tuple(m) for m in z["meta"]]
    else:
        feats, meta = build()
    if SUBSET == "pump":
        run(feats, meta, kinds=("pump_lovo", "pump_cs", "pump_lodo"), classes=PUMP_CLASSES, domains=[m[6] for m in meta],
            out_name="linear_pump.json")
    else:
        run(feats, meta)
