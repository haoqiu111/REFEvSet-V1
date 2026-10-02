"""Class-mean whitened log-SNR maps per Rotor domain, the input of scripts/make_fig_comb.py:
cache/rotor_class_snr_maps.npy = {(domain, class): mean log SNR (patches, orders, 2) over the evaluation windows,
(domain, 'ref'): mean over the reference windows of the healthy recording}.   python scripts/build_class_snr_maps.py"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evset.eval.protocol import load_rotor_l2_meta  # noqa: E402
from linear_v2 import log_snr, L2  # noqa: E402

if __name__ == "__main__":
    D = {}
    for f in load_rotor_l2_meta(L2):
        s, _ = log_snr(np.load(f.path))
        if f.label == "Healthy":
            D[(f.domain, "ref")] = s[f.ref_mask()].mean(0)
        D[(f.domain, f.label)] = s[f.eval_mask()].mean(0)
    out = os.path.join(ROOT, "cache", "rotor_class_snr_maps.npy"); np.save(out, D, allow_pickle=True); print("saved", out, len(D), "maps")
