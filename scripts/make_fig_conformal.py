"""Figure: reference-only conformal detection — empirical false-alarm rate vs nominal alpha per target domain
(Rotor LODO/LOVO/CS and Pump), plus detection rate. usage: python scripts/make_fig_conformal.py split_lodo_s0 split_lovo_s0 split_cs_s0 pumpsplit_pump_lodo_s0"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset import plot_style as ps  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
ALPHAS = np.linspace(0.01, 0.3, 30)


def curves(tag):
    far, det = [], []
    for jp in sorted(glob.glob(os.path.join(ROOT, "outputs", "evset", tag, "*.json"))):
        r = json.load(open(jp))
        if not r.get("calib_probs"):
            continue
        probs = np.array(r["probs"]); ys = np.array(r["ys"]); doms = np.array(r["domains"])
        for d, cp in r["calib_probs"].items():
            cp = np.array(cp); n = len(cp); s_cal = np.sort(1 - cp[:, 0])
            m = doms == d; s_te = 1 - probs[m, 0]; y_te = ys[m]
            f_, d_ = [], []
            for a in ALPHAS:
                k = int(np.ceil((n + 1) * (1 - a)))
                q = s_cal[min(k, n) - 1] if k <= n else np.inf
                flag = s_te > q
                f_.append(flag[y_te == 0].mean() if (y_te == 0).any() else np.nan); d_.append(flag[y_te != 0].mean() if (y_te != 0).any() else np.nan)
            far.append(f_); det.append(d_)
    return np.array(far), np.array(det)


def main():
    tags = sys.argv[1:]
    ps.apply(9)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    cols = [ps.BLUE, ps.GREYS[0], ps.GREYS[2], ps.BLUE_LIGHT, "#e69f00"]
    for t, col in zip(tags, cols):
        far, det = curves(t)
        if len(far) == 0:
            continue
        axs[0].plot(ALPHAS, np.nanmean(far, 0), color=col, lw=1.4, label=f"{t} ({len(far)} domains)")
        axs[0].fill_between(ALPHAS, np.nanmin(far, 0), np.nanmax(far, 0), color=col, alpha=0.12, lw=0)
        axs[1].plot(ALPHAS, np.nanmean(det, 0), color=col, lw=1.4, label=t)
    axs[0].plot(ALPHAS, ALPHAS, "k--", lw=0.8, label="nominal $\\alpha$")
    axs[0].set_xlabel("nominal false-alarm level $\\alpha$"); axs[0].set_ylabel("empirical false-alarm rate (healthy windows)")
    axs[0].set_title("(a) reference-only calibration: guarantee", loc="left"); axs[0].legend(fontsize=6.5, loc="upper left")
    axs[1].set_xlabel("nominal false-alarm level $\\alpha$"); axs[1].set_ylabel("detection rate (fault windows)"); axs[1].set_ylim(0, 1.02)
    axs[1].set_title("(b) detection at the same threshold", loc="left")
    fig.tight_layout(); out = os.path.join(ROOT, "figures", "fig_conformal.png"); fig.savefig(out); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)


if __name__ == "__main__":
    main()
