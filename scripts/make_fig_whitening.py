"""Figure: why shot-noise whitening is needed (viewpoint 1, 1000 rpm).
Row 1: (a) event-rate map (healthy); (b) raw log|X| at 1x (healthy); (c) whitened log-SNR at 1x (healthy);
       (d), (e) per-patch reference ratio at 3x for the ball and inner faults (the fault contrast the classifier sees).
Row 2: (f) mean spectral level vs patch event rate: the raw level tracks the rate, the whitened level does not;
       (g) the same linear classifier with un-whitened vs whitened per-patch reference-ratio features."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evset import plot_style as ps  # noqa: E402
from evset.eval.protocol import load_rotor_l2_meta  # noqa: E402
from linear_v2 import log_snr  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    files = {f.name: f for f in load_rotor_l2_meta(os.path.join(ROOT, "cache", "rotor_l2"))}
    orders = np.arange(0.125, 64.0001, 0.125); i1 = int(np.argmin(np.abs(orders - 1))); i3 = int(np.argmin(np.abs(orders - 3)))
    data = {}
    for c in ["Healthy", "Inner", "Outer", "Ball"]:
        f = files[f"angle1_{c}_1000"]; z = np.load(f.path); s, tok = log_snr(z)
        data[c] = dict(raw=tok[..., 0].astype(np.float32), snr=s[..., 0], rate=z["attrs"][..., 0].astype(np.float32), ref=f.ref_mask(), ev=f.eval_mask())
    ref_snr = data["Healthy"]["snr"][data["Healthy"]["ref"]].mean(0)
    ps.apply(9)
    fig = plt.figure(figsize=(7.4, 3.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[0.72, 1.0], left=0.07, right=0.975, top=0.95, bottom=0.12, wspace=0.62, hspace=0.3)

    def imshow(k, m, title, cmap="Blues", vmin=None, vmax=None, label=""):
        ax = fig.add_subplot(gs[0, k - 1])
        im = ax.imshow(m.reshape(30, 40), cmap=cmap, vmin=vmin, vmax=vmax); ax.set_title(title, loc="left", fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, aspect=12); cb.set_label(label, fontsize=6); cb.ax.tick_params(labelsize=6)
    H = data["Healthy"]
    imshow(1, H["rate"][H["ev"]].mean(0), "(a) event rate, healthy", label="log counts / 100 µs")
    imshow(2, H["raw"][H["ev"]].mean(0)[:, i1], "(b) raw log|X| at 1×", vmin=0, vmax=14, label="log amplitude")
    imshow(3, H["snr"][H["ev"]].mean(0)[:, i1], "(c) whitened log-SNR at 1×", vmin=-2, vmax=8, label="nats")
    imshow(4, data["Ball"]["snr"][data["Ball"]["ev"]].mean(0)[:, i3] - ref_snr[:, i3], "(d) ref. ratio 3×, ball", cmap="RdBu_r", vmin=-8, vmax=8, label="nats")
    imshow(5, data["Inner"]["snr"][data["Inner"]["ev"]].mean(0)[:, i3] - ref_snr[:, i3], "(e) ref. ratio 3×, inner", cmap="RdBu_r", vmin=-8, vmax=8, label="nats")
    # (f) level vs rate
    sel = orders <= 16
    ax = fig.add_subplot(gs[1, 0:2]); rate = H["rate"][H["ev"]].mean(0); lvl = H["raw"][H["ev"]].mean(0)[:, sel].mean(1); lvw = H["snr"][H["ev"]].mean(0)[:, sel].mean(1)
    ax.scatter(rate, lvl, s=4, color=ps.GREYS[2], alpha=0.6, label="raw log|X|"); ax.scatter(rate, lvw, s=4, color=ps.BLUE, alpha=0.6, label="whitened log-SNR")
    ax.set_xlabel("patch log event rate"); ax.set_ylabel("mean spectral level"); ax.set_title("(f) spectral level against event rate", loc="left", fontsize=8); ax.legend(fontsize=7, loc="upper left"); ps.grid(ax)
    # (g) linear accuracy with un-whitened vs whitened reference-ratio features (healthy-ref, 0.5-s windows)
    v1 = json.load(open(os.path.join(ROOT, "outputs", "linear_v1.json"))); v2 = json.load(open(os.path.join(ROOT, "outputs", "linear_baselines.json")))
    def acc(rows, feat, kind): return np.mean([r["acc"] for r in rows if r["feat"] == feat and r["norm"] == "healthy-ref" and r["task"].startswith(kind + ":")])
    ax = fig.add_subplot(gs[1, 2:5]); x = np.arange(3); w = 0.36; kinds = ["lovo", "cs", "lodo"]
    b1 = ax.bar(x - w / 2, [acc(v1, "refratio", k) for k in kinds], w, color=ps.GREYS[2], label="un-whitened log|X| ratio")
    b2 = ax.bar(x + w / 2, [acc(v2, "snr_refratio", k) for k in kinds], w, color=ps.BLUE, label="whitened log-SNR ratio")
    ps.annotate_bars(ax, b1, size=6.5); ps.annotate_bars(ax, b2, size=6.5)
    ax.set_xticks(x); ax.set_xticklabels(["LOVO", "CS", "LODO"]); ax.set_ylim(0.4, 0.85); ax.set_ylabel("linear accuracy (healthy-ref)"); ax.legend(fontsize=7, loc="upper left"); ps.grid(ax)
    ax.set_title("(g) same linear classifier, un-whitened vs whitened reference ratio", loc="left", fontsize=8)
    out = os.path.join(ROOT, "figures", "fig_whitening.png"); fig.savefig(out); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)


if __name__ == "__main__":
    main()
