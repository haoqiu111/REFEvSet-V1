"""Temporal decision smoothing from saved window probabilities: average log-probs over the last K windows
of the same file (deployment-legal, causal). usage: python scripts/smooth_eval.py <tag> [K ...]"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from evset.eval.protocol import load_rotor_l2_meta  # noqa: E402


def main():
    tag = sys.argv[1]
    Ks = [int(k) for k in sys.argv[2:]] or [1, 2, 4, 8]
    l2 = None
    res = {k: [] for k in Ks}
    for jp in sorted(glob.glob(os.path.join(ROOT, "outputs", "evset", tag, "*.json"))):
        r = json.load(open(jp))
        if l2 is None:
            l2 = r["args"]["l2"]
            files = load_rotor_l2_meta(os.path.join(ROOT, "cache", l2))
            file_of = np.concatenate([[i] * f.n_win for i, f in enumerate(files)])
        idx = np.array(r["idx"]); probs = np.array(r["probs"]); ys = np.array(r["ys"])
        fi = file_of[idx]
        lp = np.log(probs + 1e-9)
        for k in Ks:
            sm = np.zeros_like(lp)
            for f in np.unique(fi):
                m = np.where(fi == f)[0]  # windows are stored in time order per file
                for j, i in enumerate(m):
                    sm[i] = lp[m[max(0, j - k + 1):j + 1]].mean(0)
            res[k].append(float((sm.argmax(1) == ys).mean()))
    for k in Ks:
        print(f"{tag}: K={k:2d} windows ({0.25 * (k - 1) + 0.5:.2f} s context): mean acc {np.mean(res[k]):.3f} min {np.min(res[k]):.3f}")


if __name__ == "__main__":
    main()
