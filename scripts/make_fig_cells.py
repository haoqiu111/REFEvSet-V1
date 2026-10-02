"""Figure: (a) per-cell accuracy heat-map on the six Rotor LODO cells for the deployable linear baseline, the
event-frame CNN, REF-EvSet and the class-balanced oracle (seed-averaged); (b) pooled row-normalised confusion matrix
of REF-EvSet over the six LODO cells and three seeds. usage: python scripts/make_fig_cells.py"""
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset import plot_style as ps  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CELLS = ["angle1@1000", "angle1@2000", "angle2@1000", "angle2@2000", "angle3@1000", "angle3@2000"]
LAB = ["Healthy", "Inner", "Outer", "Ball"]


def linear_cells(norm):
    """per-cell LODO accuracy of the linear model on 1 s windows (the run of Table 3), parsed from outputs/linear_v2_w1.log;
    the log lists the six cells in the order of CELLS"""
    for line in open(os.path.join(ROOT, "outputs", "linear_v2_w1.log"), encoding="utf-8", errors="ignore"):
        m = re.match(rf"snr_refratio\s+{norm}\s+lodo\s+acc=[\d.]+ \(min [\d.]+\) f1=[\d.]+\s+\[([\d. ]+)\]", line)
        if m:
            v = [float(x) for x in m.group(1).split()]; assert len(v) == len(CELLS); return v
    return [np.nan] * len(CELLS)


def run_cells(prefix, base="evset"):
    per = {c: [] for c in CELLS}; ys, pr = [], []
    for d in glob.glob(os.path.join(ROOT, "outputs", base, prefix + "_s*")):
        if not re.fullmatch(re.escape(prefix) + r"_s\d+", os.path.basename(d)):
            continue
        for p in glob.glob(os.path.join(d, "lodo_*.json")):
            r = json.load(open(p)); per[r["task"].split(":")[1]].append(r["acc"]); ys += r["ys"]; pr += r["preds"]
    return [np.mean(per[c]) if per[c] else np.nan for c in CELLS], np.array(ys), np.array(pr)


def main():
    ev, ys, pr = run_cells("main_w1mixnorcn_lodo"); cnn, _, _ = run_cells("frames_raw_lodo", "baselines")
    names = ["linear, healthy-ref", "event-frame CNN, raw", "REF-EvSet (ours)", "linear oracle (not deployable)"]
    M = np.array([linear_cells("healthy-ref"), cnn, ev, linear_cells("oracle")])
    ps.apply(9)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[2.1, 1]))
    ax = axs[0]; im = ps.heat(ax, M, vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(CELLS))); ax.set_xticklabels([c.replace("angle", "view ").replace("@", "\n") + " rpm" for c in CELLS], fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_title("(a) LODO: accuracy per held-out (viewpoint, speed) cell", loc="left", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.03, label="accuracy")
    cm = confusion_matrix(ys, pr, labels=list(range(4))).astype(float); cm /= cm.sum(1, keepdims=True)
    ax = axs[1]; ps.confusion(ax, cm, LAB, size=6.5); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("(b) REF-EvSet, pooled confusion", loc="left", fontsize=8); ax.tick_params(axis="x", labelrotation=45, labelsize=6.5); ax.tick_params(axis="y", labelsize=6.5)
    fig.tight_layout(); out = os.path.join(ROOT, "figures", "fig_cells_lodo.png"); fig.savefig(out); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)


if __name__ == "__main__":
    main()
