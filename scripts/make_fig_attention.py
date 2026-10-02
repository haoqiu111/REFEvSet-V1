"""Figure: interpretable pooling maps of REF-EvSet (PMA attention over patches) per class for a held-out domain.

usage: python scripts/make_fig_attention.py <ckpt_tag> <task_json_stem> [domain]
e.g.   python scripts/make_fig_attention.py ckpt_lodo_s0 lodo_angle3_2000
Draws, for the target domain: the reference 1x SNR field (healthy), and the class-averaged attention weight maps
(30 x 40 grid) of the evaluation windows of each class, plus the mean reference-ratio field at 1x for each class.
"""
import json
import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evset import plot_style as ps  # noqa: E402
from evset.data.rotor_dataset import RotorTokens  # noqa: E402
from evset.eval.protocol import CLASSES  # noqa: E402
from evset.models.evset_net import EvSetNet  # noqa: E402
from train_evset import GpuBank  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    tag, stem = sys.argv[1], sys.argv[2]
    jp = os.path.join(ROOT, "outputs", "evset", tag, stem + ".json")
    r = json.load(open(jp)); a = r["args"]
    ds = RotorTokens(os.path.join(ROOT, "cache", a["l2"]), a["win"], max_order_bins=a["max_orders"])
    DEV = os.environ.get("EVSET_DEVICE", "cuda")
    bank = GpuBank(ds, DEV)
    model = EvSetNet(len(CLASSES), in_ch=4 if not a["no_refch"] else 2, n_attr=6, d=a["d"], n_isab=a["n_isab"], m=a["m"],
                     use_rcn=not a["no_rcn"], use_attr=not a["no_attr"], dropout=a["dropout"]).to(DEV)
    model.load_state_dict(torch.load(os.path.join(ROOT, "outputs", "evset", tag, stem + ".pt"), map_location=DEV)); model.eval()
    te_idx = np.array(r["idx"]); doms = np.array(r["domains"])
    dom = sys.argv[3] if len(sys.argv) > 3 else sorted(set(doms))[0]
    sel = te_idx[doms == dom]
    gh, gw = ds.grid
    maps = {c: [] for c in range(4)}; ratio1 = {c: [] for c in range(4)}
    o1 = int(np.argmin(np.abs(ds.orders - 1.0)))
    with torch.no_grad():
        for b in range(0, len(sel), 16):
            idx = sel[b:b + 16]
            x, at, rt = bank.batch(idx, False)
            if a["no_refch"]:
                x = x[..., :2]
            logits, z, w, g = model(x, at, rt, return_maps=True)
            for j, i in enumerate(idx):
                maps[int(ds.y[i])].append(w[j].float().cpu().numpy()); ratio1[int(ds.y[i])].append(x[j, :, o1, 2].float().cpu().numpy())
    ps.apply(9)
    fig, axs = plt.subplots(2, 5, figsize=(10, 3.6))
    ref1 = ds.ref_field[dom][:, o1, 0].reshape(gh, gw)
    im = axs[0, 0].imshow(ref1, cmap="Blues"); axs[0, 0].set_title("reference log-SNR at 1×", loc="left"); plt.colorbar(im, ax=axs[0, 0], fraction=0.04)
    axs[1, 0].axis("off")
    for c in range(4):
        m = np.mean(maps[c], 0).reshape(gh, gw) * gh * gw
        im = axs[0, c + 1].imshow(m, cmap="Blues", vmin=0, vmax=max(3, m.max())); axs[0, c + 1].set_title(f"PMA attention: {CLASSES[c]}", loc="left")
        rr = np.mean(ratio1[c], 0).reshape(gh, gw)
        im2 = axs[1, c + 1].imshow(rr, cmap="RdBu_r", vmin=-6, vmax=6); axs[1, c + 1].set_title(f"reference ratio at 1×: {CLASSES[c]}", loc="left")
    plt.colorbar(im, ax=axs[0, 4], fraction=0.04, label="weight x P"); plt.colorbar(im2, ax=axs[1, 4], fraction=0.04, label="nats")
    for ax in axs.reshape(-1):
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"held-out cell: {dom.replace('angle', 'viewpoint ').replace('@', ', ')} rpm (LODO, seed-0 model)", fontsize=9)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", f"fig_attention_{stem}.png"); fig.savefig(out); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)


if __name__ == "__main__":
    main()
