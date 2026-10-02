"""Figure: data and the Reference-Only protocol. (a)-(c) mean event-count images of the three Rotor viewpoints
(healthy, 1000 rpm) with the 16 x 16 patch grid; (d) time line of one recording: reference segment (healthy only),
guard, evaluation windows. The domain grids are given in Table 2 of the paper and are not repeated here."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset import plot_style as ps  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ps.apply(9); fig = plt.figure(figsize=(7.4, 3.3)); gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1.0])
    for k, v in enumerate(["angle1", "angle2", "angle3"]):
        z = np.load(os.path.join(ROOT, "cache", "rotor_frames", f"{v}_Healthy_1000.npz")); img = np.log1p(z["frames"].astype(np.float32).sum(1).mean(0))
        ax = fig.add_subplot(gs[0, k]); ax.imshow(img, cmap="gray_r"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"({'abc'[k]}) viewpoint {k + 1}, healthy, 1000 rpm", loc="left", fontsize=8)
        ax.add_patch(Rectangle((0, 0), 8, 8, fill=False, ec=ps.BLUE, lw=0.8)); ax.text(11, 8, "16 × 16 patch", fontsize=6, color=ps.BLUE, va="center")
    ax = fig.add_subplot(gs[1, :]); ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    ax.text(0, 2.8, "(d) one recording (10–19 s valid): the target domain exposes only the reference segment", fontsize=8, va="top")
    ax.add_patch(Rectangle((0, 1.0), 3, 0.8, fc=ps.BLUE_LIGHT, ec="none")); ax.text(1.5, 1.4, "healthy reference, 0–3 s\n(no label, no fault)", ha="center", va="center", fontsize=6.5)
    ax.add_patch(Rectangle((3, 1.0), 0.5, 0.8, fc="#ccc", ec="none")); ax.text(3.25, 0.85, "guard 0.5 s", ha="center", va="top", fontsize=6)
    ax.add_patch(Rectangle((3.5, 1.0), 8.3, 0.8, fc="#e8effc", ec=ps.BLUE, lw=0.5))
    ax.text(7.65, 1.4, "evaluation windows, 1 s with 0.25 s hop: test windows of the target domain (all classes),\ntraining windows of the source domains (labelled)", ha="center", va="center", fontsize=6.5)
    for x0 in np.arange(3.5, 11.6, 1.0):
        ax.add_patch(Rectangle((x0, 1.95), 1.0, 0.25, fc="none", ec=ps.BLUE, lw=0.5))
    ax.annotate("", xy=(12, 0.5), xytext=(0, 0.5), arrowprops=dict(arrowstyle="->", lw=0.7)); ax.text(6, 0.1, "time", ha="center", fontsize=6.5)
    fig.tight_layout(); out = os.path.join(ROOT, "figures", "fig_protocol.png"); fig.savefig(out); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)


if __name__ == "__main__":
    main()
