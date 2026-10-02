"""Per-pixel frequency-map baseline (FM-LR; EBFM / S-ROPE flavour): per patch, the dominant order and its whitened
amplitude -> histogram of dominant orders (64 one-order bins, plain and amplitude-weighted) and the fraction of active
patches, for the signed and the unsigned channel. Evaluated with logistic regression under raw / healthy-ref / oracle
normalisation, same protocols as the other linear baselines (0.5 s windows, cache/rotor_l2).
python scripts/linear_freqmap.py [--out linear_freqmap.json]   -> outputs/<out>"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evset.eval.protocol import load_rotor_l2_meta  # noqa: E402
from linear_v2 import log_snr  # noqa: E402
from linear_baselines import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="linear_freqmap.json"); args = ap.parse_args()
    files = load_rotor_l2_meta(os.path.join(ROOT, "cache", "rotor_l2")); feats, meta = [], []
    for f in files:
        s, _ = log_snr(np.load(f.path)); N = len(s); out = []          # (N, P, O, 2)
        for ch in range(2):
            a = s[..., ch]; dom = a.argmax(2); amp = a.max(2)          # dominant order index and its log SNR per patch
            ob = np.minimum(dom // 8, 63)                              # one-order bins
            H = np.zeros((N, 64)); Hw = np.zeros((N, 64))
            for i in range(N):
                H[i] = np.bincount(ob[i], minlength=64) / a.shape[1]
                Hw[i] = np.bincount(ob[i], weights=np.maximum(amp[i], 0), minlength=64) / a.shape[1]
            out += [H, Hw, (amp > 3).mean(1, keepdims=True)]
        feats.append(np.concatenate(out, 1).astype(np.float32))
        for i in range(N):
            meta.append((f.name, f.label, f.view, f.rpm, int(f.ref_mask()[i]), int(f.eval_mask()[i])))
    X = np.concatenate(feats); print("feature dim", X.shape)
    run({"freqmap": X}, meta, out_name=args.out)
