"""Figure: Pump hold-out (angle, speed): accuracy per held-out cell for the deployable linear baseline, REF-EvSet
(seed-averaged) and the class-balanced oracle. usage: python scripts/make_fig_pump.py"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset import plot_style as ps  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    cells = [f"{v}@{s}" for v in ["-30", "0", "30"] for s in [1200, 1800, 2400]]
    lin = {}
    for line in open(os.path.join(ROOT, "outputs", "linear_pump_w1.log"), encoding="utf-8", errors="ignore"):
        m = re.match(r"snr_all\s+(healthy-ref|oracle)\s+pump_lodo\s+acc=[\d.]+ \(min [\d.]+\) f1=[\d.]+\s+\[([\d. ]+)\]", line)
        if m:
            lin[m.group(1)] = [float(x) for x in m.group(2).split()]
    ev = defaultdict(list)
    for d in glob.glob(os.path.join(ROOT, "outputs", "evset", "pumpw1_mixnorcn_pump_lodo_s*")):
        for p in glob.glob(os.path.join(d, "*.json")):
            r = json.load(open(p)); ev[r["task"].split(":")[1]].append(r["acc"])
    M = np.array([lin["healthy-ref"], [np.mean(ev[c]) for c in cells], lin["oracle"]])
    ps.apply(9); fig, ax = plt.subplots(figsize=(7.2, 2.3))
    im = ps.heat(ax, M, vmin=0.5, vmax=1.0); ax.set_xticks(range(9)); ax.set_xticklabels([c.split("@")[0] + "°\n" + c.split("@")[1] + " rpm" for c in cells], fontsize=6.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(["linear, healthy-ref", "REF-EvSet (ours)", "linear oracle (not deployable)"], fontsize=8)
    ax.set_title("Pump hold-out (angle, speed): accuracy per held-out cell (camera angle / drive speed)", loc="left", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.03, label="accuracy"); fig.tight_layout()
    out = os.path.join(ROOT, "figures", "fig_pump.png"); fig.savefig(out); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)


if __name__ == "__main__":
    main()
