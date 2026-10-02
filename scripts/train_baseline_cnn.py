"""Literature-style dense baselines on the same Reference-Only protocols.

--rep frames : polarity count images (2, 240, 320) -> log1p -> small ResNet-ish CNN  (event-frame CNN, TII-2023 style)
--rep voxel  : 8-bin voxel grid (8, 120, 160) -> log1p -> CNN                          (voxel-grid CNN)
--norm raw | ref : optional per-domain healthy-reference mean subtraction of the log image (deployable)
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset.eval.protocol import load_rotor_l2_meta, tasks, CLASSES, indomain_split  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_all(rep, l2_name, frame_name, classes=None):
    classes = classes or CLASSES
    files = load_rotor_l2_meta(os.path.join(ROOT, "cache", l2_name))
    X, meta = [], []
    for fi, f in enumerate(files):
        z = np.load(os.path.join(ROOT, "cache", frame_name, f.name + ".npz"))
        a = z[rep].astype(np.float32)
        X.append(np.log1p(a).astype(np.float16))
        rm, em = f.ref_mask(), f.eval_mask()
        for i in range(len(a)):
            meta.append((fi, classes.index(f.label), f.view, f.rpm, int(rm[i]), int(em[i]), f.domain))
    X = np.concatenate(X)
    m = np.array(meta, dtype=object)
    return X, dict(file=m[:, 0].astype(int), y=m[:, 1].astype(int), view=m[:, 2].astype(str), rpm=m[:, 3].astype(int),
                   is_ref=m[:, 4].astype(int).astype(bool), is_eval=m[:, 5].astype(int).astype(bool), domain=m[:, 6].astype(str))


class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, padding=1); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, padding=1); self.b2 = nn.BatchNorm2d(o)
        self.sk = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.b1(self.c1(x))); h = self.b2(self.c2(h))
        return F.max_pool2d(F.gelu(h + self.sk(x)), 2)


class CNN(nn.Module):
    def __init__(self, in_ch, n_classes=4, w=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, w, 5, stride=2, padding=2), nn.BatchNorm2d(w), nn.GELU())
        self.blocks = nn.Sequential(ConvBlock(w, w), ConvBlock(w, 2 * w), ConvBlock(2 * w, 4 * w), ConvBlock(4 * w, 4 * w))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * w, n_classes))

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.head(h.mean((2, 3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="frames"); ap.add_argument("--norm", default="raw")
    ap.add_argument("--kind", default="lodo"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", default="rotor_l2"); ap.add_argument("--frames", default="rotor_frames"); ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    dev = "cuda"
    X, M = load_all(args.rep, args.l2, args.frames)
    dom = np.array([f"{v}@{r}" for v, r in zip(M["view"], M["rpm"])])
    if args.norm == "ref":
        X = X.astype(np.float32)
        for d in np.unique(dom):
            ref = X[(dom == d) & M["is_ref"] & (M["y"] == 0)].mean(0)
            X[dom == d] -= ref
        X = X.astype(np.float16)
    print("loaded", X.shape, X.dtype, flush=True)
    tag = args.tag or f"{args.rep}_{args.norm}_{args.kind}_s{args.seed}"
    out_dir = os.path.join(ROOT, "outputs", "baselines", tag); os.makedirs(out_dir, exist_ok=True)
    results = []
    for tname, is_t in tasks(args.kind):
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        tmask = np.array([is_t(v, r) for v, r in zip(M["view"], M["rpm"])])
        tr = np.where(~tmask)[0]; te = np.where(tmask & M["is_eval"])[0]
        if args.kind == "indomain":
            tr, te = indomain_split(M["file"], M["is_eval"], args.seed)
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(M["y"])
        mu = Xt[tr].float().mean().item(); sd = Xt[tr].float().std().item() + 1e-6
        model = CNN(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        steps = args.epochs * (len(tr) // args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
        t0 = time.time()
        for ep in range(args.epochs):
            model.train(); perm = np.random.permutation(tr)
            for b in range(0, len(perm) - args.bs + 1, args.bs):
                idx = perm[b:b + args.bs]
                x = ((Xt[idx].float() - mu) / sd).to(dev); y = yt[idx].to(dev)
                # augmentation: random translation +-8 px, random horizontal flip
                dx, dy = np.random.randint(-8, 9, 2)
                x = torch.roll(x, (int(dy), int(dx)), (2, 3))
                if np.random.rand() < 0.5:
                    x = x.flip(3)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(x).float(), y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        model.eval(); preds = []
        with torch.no_grad():
            for b in range(0, len(te), 64):
                idx = te[b:b + 64]
                x = ((Xt[idx].float() - mu) / sd).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds.append(model(x).float().argmax(1).cpu())
        preds = torch.cat(preds).numpy(); ys = M["y"][te]
        acc = float((preds == ys).mean()); f1 = float(f1_score(ys, preds, average="macro"))
        per_class = [float((preds[ys == c] == c).mean()) for c in range(4)]
        print(f"[{tname}] acc {acc:.3f} f1 {f1:.3f} per-class {np.round(per_class, 2)} {time.time() - t0:.0f}s", flush=True)
        results.append(dict(task=tname, acc=acc, f1=f1, per_class=per_class, preds=preds.tolist(), ys=ys.tolist(), idx=te.tolist()))
        with open(os.path.join(out_dir, re.sub(r"[^\w.-]", "_", tname) + ".json"), "w") as fh:
            json.dump(dict(args=vars(args), **results[-1]), fh)
    print(f"== {tag}: mean acc {np.mean([r['acc'] for r in results]):.3f} min {np.min([r['acc'] for r in results]):.3f} "
          f"f1 {np.mean([r['f1'] for r in results]):.3f}")


if __name__ == "__main__":
    main()
