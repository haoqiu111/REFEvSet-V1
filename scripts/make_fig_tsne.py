"""Figure: t-SNE of (a) raw whitened set-mean tokens and (b) REF-EvSet embeddings for two held-out LODO cells
(hard cell view2@1000, good cell view3@2000): source windows (circles) vs held-out target windows (triangles),
coloured by class. usage: python scripts/make_fig_tsne.py  (CPU; uses outputs/evset/ckpt_lodo_s0)"""
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evset import plot_style as ps  # noqa: E402
from evset.data.rotor_dataset import RotorTokens  # noqa: E402
from evset.eval.protocol import CLASSES  # noqa: E402
from evset.models.evset_net import EvSetNet  # noqa: E402
from train_evset import GpuBank  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CELLS = [("lodo_angle2_1000", "angle2@1000", "hard cell, viewpoint 2, 1000 rpm"), ("lodo_angle3_2000", "angle3@2000", "viewpoint 3, 2000 rpm")]


def embed(model, bank, idx, no_refch):
    zs = []
    with torch.no_grad():
        for b in range(0, len(idx), 32):
            x, a, r = bank.batch(idx[b:b + 32], False)
            if no_refch:
                x = x[..., :2]
            _, z = model(x, a, r)
            zs.append(z.float().cpu().numpy())
    return np.concatenate(zs)


def main():
    tag = "ckpt_lodo_s0"
    ps.apply(9)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4))
    ds = None
    for row, (stem, dom, title) in enumerate(CELLS):
        r = json.load(open(os.path.join(ROOT, "outputs", "evset", tag, stem + ".json"))); a = r["args"]
        if ds is None:
            ds = RotorTokens(os.path.join(ROOT, "cache", a["l2"]), a["win"], max_order_bins=a["max_orders"]); bank = GpuBank(ds, "cpu")
        model = EvSetNet(len(CLASSES), in_ch=4 if not a["no_refch"] else 2, n_attr=6, d=a["d"], n_isab=a["n_isab"], m=a["m"],
                         use_rcn=not a["no_rcn"], use_attr=not a["no_attr"], dropout=a["dropout"]).eval()
        model.load_state_dict(torch.load(os.path.join(ROOT, "outputs", "evset", tag, stem + ".pt"), map_location="cpu"))
        tgt = np.array(r["idx"]); src = np.where((ds.domain != dom) & ds.is_eval)[0]
        rng = np.random.default_rng(0); src = rng.choice(src, min(400, len(src)), replace=False)
        idx = np.concatenate([src, tgt]); is_t = np.r_[np.zeros(len(src), bool), np.ones(len(tgt), bool)]; y = ds.y[idx]
        # (a) raw: set-mean of the reference-ratio tokens (P x O x 2 -> O x 2 mean over patches), PCA-50 -> t-SNE
        raw = []
        for i in idx:
            t = ds.tokens[i].astype(np.float32); ref = ds.ref_field[ds.domain[i]]
            raw.append((t - ref).mean(0).reshape(-1))
        raw = PCA(50, random_state=0).fit_transform(np.array(raw))
        emb = embed(model, bank, idx, a["no_refch"])
        for col, (feat, name) in enumerate([(raw, "whitened tokens (set mean)"), (emb, "REF-EvSet embedding")]):
            Y = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(feat)
            ax = axs[row, col]
            for c in range(4):
                m = (y == c) & ~is_t; ax.scatter(Y[m, 0], Y[m, 1], s=9, color=ps.TSNE_COLORS[c], alpha=0.45, lw=0, label=f"{CLASSES[c]} (source)" if row == 0 and col == 0 else None)
            for c in range(4):
                m = (y == c) & is_t; ax.scatter(Y[m, 0], Y[m, 1], s=22, color=ps.TSNE_COLORS[c], marker="^", edgecolor="k", lw=0.4, label=f"{CLASSES[c]} (held-out target)" if row == 0 and col == 0 else None)
            acc = r["acc"] if col == 1 else None
            ax.set_title(f"({'abcd'[row * 2 + col]}) {name}, {title}" + (f", target acc. {acc:.2f}" if acc else ""), loc="left", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("#999")
    axs[0, 0].legend(fontsize=6.5, loc="best", ncol=2, markerscale=1.0)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "fig_tsne.png"); fig.savefig(out); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)


if __name__ == "__main__":
    main()
