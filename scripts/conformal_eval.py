"""Reference-only conformal evaluation from saved split-reference runs (outputs/evset/<tag>/*.json).

For every target domain: calibration scores s_i = 1 - p_healthy on the held-out (odd) reference windows
(healthy, unlabelled by assumption), threshold q = ceil((n+1)(1-alpha))/n quantile. A test window is
flagged "not healthy" when 1 - p_healthy > q. Reported per alpha: false-alarm rate on healthy evaluation windows
(guarantee: <= alpha if exchangeable), detection rate on fault windows, and the class prediction-set size
(APS-style sets from the softmax with the same calibration windows, healthy-only calibration => sets are
calibrated for the healthy class only; reported for completeness).
usage: python scripts/conformal_eval.py <tag> [<tag> ...]
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
ALPHAS = [0.15, 0.20, 0.30, 0.40]


def main():
    for tag in sys.argv[1:]:
        rows = {a: [] for a in ALPHAS}
        for jp in sorted(glob.glob(os.path.join(ROOT, "outputs", "evset", tag, "*.json"))):
            r = json.load(open(jp))
            if not r.get("calib_probs"):
                print("no calibration probabilities in", jp); continue
            probs = np.array(r["probs"]); ys = np.array(r["ys"]); doms = np.array(r["domains"])
            for d, cp in r["calib_probs"].items():
                cp = np.array(cp); n = len(cp)
                s_cal = 1 - cp[:, 0]
                m = doms == d
                s_te = 1 - probs[m, 0]; y_te = ys[m]
                for a in ALPHAS:
                    k = int(np.ceil((n + 1) * (1 - a)))
                    q = np.sort(s_cal)[min(k, n) - 1] if k <= n else np.inf
                    flag = s_te > q
                    far = float(flag[y_te == 0].mean()) if (y_te == 0).any() else np.nan
                    det = float(flag[y_te != 0].mean()) if (y_te != 0).any() else np.nan
                    rows[a].append((r["task"], d, n, far, det))
        print(f"== {tag}")
        for a in ALPHAS:
            rr = rows[a]
            if not rr:
                continue
            far = np.nanmean([x[3] for x in rr]); det = np.nanmean([x[4] for x in rr])
            worst = max(x[3] for x in rr)
            print(f"  alpha {a:.2f}: n_cal {np.mean([x[2] for x in rr]):.0f}  false-alarm {far:.3f} (worst domain {worst:.3f})  detection {det:.3f}  over {len(rr)} target domains")


if __name__ == "__main__":
    main()
