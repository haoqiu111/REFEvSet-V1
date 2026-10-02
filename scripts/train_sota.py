"""Same-protocol re-implementations of recent event-camera / domain-shift diagnosis paradigms (Reference-Only, Rotor).

All methods are trained on the labelled source domains of every task and evaluated on the evaluation windows of the
held-out domain(s); what differs is the representation, the network and how (or whether) the target's healthy
reference is used. Outputs follow outputs/baselines/<tag>/<task>.json (same format as train_baseline_cnn.py).

--method
  evit        bi-fovea event-frame Transformer (EViT paradigm, Jin et al., Sensors 2025): fine (16 px) and coarse
              (48 px) patch tokens, two encoders, fovea<-periphery cross-attention                     [norm raw|ref]
  snn         spiking CNN with LIF neurons and surrogate gradients over the 8-bin voxel grid
              (surrogate-gradient training, Neftci et al. 2019; Fang et al. 2021); the reported runs use
              --snn_readout mem --snn_thr 0.5 --lr 3e-4 (run_sota3.sh)                               [norm raw|ref]
  evstr       event voxel set Transformer (EVSTr paradigm, Xie et al. 2024): top-K active voxels as a point set with
              (t, y, x, value) features -> ISAB x2 -> PMA                                             [norm raw|ref]
  ssl         self-supervised pre-training + cross-supervision on unlabelled target events (camera-position
              robustness paradigm, Li et al. EAAI 2026): --variant strict (target = healthy reference only,
              our protocol) | relaxed (all unlabelled target windows incl. faults, their protocol)
  cnn         event-frame CNN (same network as train_baseline_cnn.py), used for the Pump runs   [norm raw|ref]
  tta         event-frame CNN [TII 2023] + test-time normalisation: AdaBN on the healthy reference (deployable),
              AdaBN on the test batch and TENT (transductive); one training, four evaluations
  spec        global event-rate signal -> log-STFT spectrogram -> 2-D CNN (event rate treated as a vibration
              signal: signal-alignment paradigm, Guang et al. EAAI 2025)                               [norm raw|ref]

usage: python scripts/train_sota.py --method evit --kind lodo --seed 0 [--norm raw|ref] [--variant strict|relaxed] [--subset rotor|pump]
"""
import argparse
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
sys.path.insert(0, os.path.dirname(__file__))
from evset.eval.protocol import load_rotor_l2_meta, tasks, CLASSES, indomain_split  # noqa: E402
from evset.models.evset_net import ISAB, PMA  # noqa: E402
from train_baseline_cnn import load_all, CNN  # noqa: E402
from evset.data.registry import PUMP_CLASSES  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEV = "cuda"


# ----------------------------------------------------------------------------------------------------------------- data
def spec_data(l2_name="rotor_l2_w1", win_s=1.0):
    """Global-rate log-STFT spectrograms per window: (N, 1, 129, 157) float16 + meta (same windows as the L2 cache)."""
    from scipy.signal import stft
    files = load_rotor_l2_meta(os.path.join(ROOT, "cache", l2_name))
    X, meta = [], []
    for fi, f in enumerate(files):
        g = np.load(os.path.join(ROOT, "cache", "rotor_global_rate", f.name + ".npz"), allow_pickle=True)
        rate = g["rate"].astype(np.float32); L = int(win_s * 10000)
        for i, ts in enumerate(f.t_start):
            s = int(round(float(ts) * 10000)); x = rate[s:s + L]
            _, _, S = stft(x - x.mean(), fs=10000, nperseg=256, noverlap=192)
            X.append(np.log1p(np.abs(S)).astype(np.float16)[None])
        rm, em = f.ref_mask(), f.eval_mask()
        for i in range(len(f.t_start)):
            meta.append((fi, CLASSES.index(f.label), f.view, f.rpm, int(rm[i]), int(em[i])))
    X = np.stack(X); m = np.array(meta, dtype=object)
    return X, dict(file=m[:, 0].astype(int), y=m[:, 1].astype(int), view=m[:, 2].astype(str), rpm=m[:, 3].astype(int),
                   is_ref=m[:, 4].astype(int).astype(bool), is_eval=m[:, 5].astype(int).astype(bool))


def ref_subtract(X, M):
    """Deployable healthy-reference normalisation of dense inputs: subtract the domain's mean reference image."""
    dom = M["domain"]; X = X.astype(np.float32)
    for d in np.unique(dom):
        X[dom == d] -= X[(dom == d) & M["is_ref"] & (M["y"] == 0)].mean(0)
    return X.astype(np.float16), dom


# --------------------------------------------------------------------------------------------------------------- models
class EViT(nn.Module):
    """Bi-fovea event Transformer: peripheral (coarse) and foveal (fine) token streams with fovea<-periphery cross-attention."""

    def __init__(self, in_ch=2, d=128, n_classes=4, heads=4, depth=2, drop=0.1):
        super().__init__()
        self.fine = nn.Conv2d(in_ch, d, 16, stride=16)      # 240x320 -> 15x20 = 300 tokens
        self.coarse = nn.Conv2d(in_ch, d, 48, stride=48)    # 240x320 -> 5x6 = 30 tokens
        self.pos_f = nn.Parameter(torch.zeros(1, 300, d)); self.pos_c = nn.Parameter(torch.zeros(1, 30, d))
        nn.init.trunc_normal_(self.pos_f, std=0.02); nn.init.trunc_normal_(self.pos_c, std=0.02)
        layer = lambda: nn.TransformerEncoderLayer(d, heads, 2 * d, drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc_f = nn.TransformerEncoder(layer(), depth); self.enc_c = nn.TransformerEncoder(layer(), depth)
        self.cross = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True); self.ln_q = nn.LayerNorm(d); self.ln_kv = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(drop), nn.Linear(2 * d, n_classes))

    def forward(self, x):
        f = self.fine(x).flatten(2).transpose(1, 2) + self.pos_f
        c = self.coarse(x).flatten(2).transpose(1, 2) + self.pos_c
        f = self.enc_f(f); c = self.enc_c(c)
        f = f + self.cross(self.ln_q(f), self.ln_kv(c), self.ln_kv(c))[0]
        return self.head(torch.cat([f.mean(1), c.mean(1)], -1))


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v); return (v > 0).float()

    @staticmethod
    def backward(ctx, g):
        v, = ctx.saved_tensors; return g / (1 + (np.pi * v) ** 2)   # arctan surrogate


class SpikingCNN(nn.Module):
    """LIF spiking CNN over the T = 8 voxel bins (input current = log1p counts of each bin), rate readout."""

    def __init__(self, n_classes=4, w=32, tau=0.5, thr=0.5, readout="spike"):
        super().__init__()
        self.tau, self.thr, self.readout = tau, thr, readout
        self.c1 = nn.Conv2d(1, w, 5, stride=2, padding=2); self.b1 = nn.BatchNorm2d(w)
        self.c2 = nn.Conv2d(w, 2 * w, 3, stride=2, padding=1); self.b2 = nn.BatchNorm2d(2 * w)
        self.c3 = nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1); self.b3 = nn.BatchNorm2d(4 * w)
        self.fc = nn.Linear(4 * w, n_classes)

    def forward(self, x):  # x: (B, T, H, W)
        B, T = x.shape[:2]; v1 = v2 = v3 = None; out = 0
        for t in range(T):
            i1 = self.b1(self.c1(x[:, t:t + 1])); v1 = i1 if v1 is None else self.tau * v1 + i1; s1 = SpikeFn.apply(v1 - self.thr); v1 = v1 * (1 - s1)
            i2 = self.b2(self.c2(s1)); v2 = i2 if v2 is None else self.tau * v2 + i2; s2 = SpikeFn.apply(v2 - self.thr); v2 = v2 * (1 - s2)
            i3 = self.b3(self.c3(s2)); v3 = i3 if v3 is None else self.tau * v3 + i3; s3 = SpikeFn.apply(v3 - self.thr)
            out = out + self.fc((v3 if self.readout == "mem" else s3).mean((2, 3))); v3 = v3 * (1 - s3)   # readout, then hard reset
        return out / T


class EVSTr(nn.Module):
    """Event voxel set Transformer: K most active voxels -> (t, y, x, value) point features -> set attention."""

    def __init__(self, n_classes=4, d=128, m=16, heads=4, drop=0.1):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.ln = nn.LayerNorm(d); self.isabs = nn.ModuleList([ISAB(d, m, heads, drop) for _ in range(2)]); self.pma = PMA(d, 1, heads, drop)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, n_classes))

    def forward(self, pts):  # (B, K, 4)
        h = self.ln(self.emb(pts))
        for blk in self.isabs:
            h = blk(h)
        return self.head(self.pma(h)[:, 0])


def voxel_to_set(x, K=1024, signed=False, pool=8):
    """x: (B, T, H, W) standardized voxel grid. Coarse voxels of pool x pool pixels (T x H/pool x W/pool), the K voxels with the
    largest value (or |value| if signed) form the set; features (t, y, x, value) -> (B, K, 4)."""
    B, T, H, W = x.shape
    v = F.avg_pool2d(x.reshape(B * T, 1, H, W), pool).reshape(B, T, H // pool, W // pool); Hc, Wc = v.shape[2:]
    flat = v.reshape(B, -1); score = flat.abs() if signed else flat
    idx = score.topk(min(K, flat.shape[1]), dim=1).indices; val = torch.gather(flat, 1, idx)
    t = (idx // (Hc * Wc)).float() / (T - 1); yx = idx % (Hc * Wc); yy = (yx // Wc).float() / (Hc - 1); xx = (yx % Wc).float() / (Wc - 1)
    return torch.stack([t, yy, xx, val], -1)


class Encoder(nn.Module):
    """Event-frame CNN encoder (same trunk as the TII-2023-style baseline) with a projection head for SimCLR."""

    def __init__(self, in_ch=2, w=32, n_classes=4):
        super().__init__()
        self.cnn = CNN(in_ch, n_classes, w); self.proj = nn.Sequential(nn.Linear(4 * w, 4 * w), nn.ReLU(), nn.Linear(4 * w, 64))

    def feat(self, x):
        return self.cnn.blocks(self.cnn.stem(x)).mean((2, 3))

    def forward(self, x):
        return self.cnn.head(self.feat(x))


# ------------------------------------------------------------------------------------------------------------ helpers
def augment_frames(x, strong=False):
    dx, dy = np.random.randint(-8, 9, 2); x = torch.roll(x, (int(dy), int(dx)), (2, 3))
    if np.random.rand() < 0.5:
        x = x.flip(3)
    if strong:
        B, C, H, W = x.shape; x = x.clone()
        for b in range(B):
            h, w = np.random.randint(H // 6, H // 3), np.random.randint(W // 6, W // 3); y0, x0 = np.random.randint(0, H - h), np.random.randint(0, W - w)
            x[b, :, y0:y0 + h, x0:x0 + w] = 0
        x = x + 0.1 * torch.randn_like(x)
    return x


def nt_xent(z1, z2, tau=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1); n = z1.shape[0]
    sim = z @ z.t() / tau; sim.fill_diagonal_(-1e9)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def predict(model, X, idx, mu, sd, fwd=None, bs=64):
    model.eval(); preds = []
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            x = ((X[idx[b:b + bs]].float() - mu) / sd)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = (fwd or model)(x)
            preds.append(out.float().argmax(1).cpu())
    return torch.cat(preds).numpy()


def train_supervised(model, X, y_all, tr, mu, sd, args, fwd=None, aug=augment_frames, extra_loss=None):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (len(tr) // args.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1)
    for ep in range(args.epochs):
        model.train(); perm = np.random.permutation(tr)
        for b in range(0, len(perm) - args.bs + 1, args.bs):
            idx = perm[b:b + args.bs]; x = aug((X[idx].float() - mu) / sd); y = y_all[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy((fwd or model)(x).float(), y)
                if extra_loss is not None:
                    loss = loss + extra_loss(model)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()


def evaluate(preds, ys, tname, t0, extra=None, n_classes=4):
    acc = float((preds == ys).mean()); f1 = float(f1_score(ys, preds, average="macro"))
    per_class = [float((preds[ys == c] == c).mean()) for c in range(n_classes)]
    print(f"[{tname}] acc {acc:.3f} f1 {f1:.3f} per-class {np.round(per_class, 2)} {time.time() - t0:.0f}s", flush=True)
    return dict(task=tname, acc=acc, f1=f1, per_class=per_class, preds=preds.tolist(), ys=ys.tolist(), **(extra or {}))


def save(out_dir, args, res):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, re.sub(r"[^\w.-]", "_", res["task"]) + ".json"), "w") as fh:
        json.dump(dict(args=vars(args), **res), fh)


# --------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["evit", "snn", "evstr", "ssl", "tta", "spec", "cnn"]); ap.add_argument("--subset", default="rotor", choices=["rotor", "pump"])
    ap.add_argument("--kind", default="lodo"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm", default="raw", choices=["raw", "ref"]); ap.add_argument("--variant", default="strict", choices=["strict", "relaxed"])
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--pre_epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--K", type=int, default=1024); ap.add_argument("--snn_readout", default="spike"); ap.add_argument("--snn_thr", type=float, default=0.5); ap.add_argument("--tag", default=None); ap.add_argument("--only", default=None)
    args = ap.parse_args()
    rep = {"evit": "frames", "snn": "voxel", "evstr": "voxel", "ssl": "frames", "tta": "frames", "spec": "spec", "cnn": "frames"}[args.method]
    classes = PUMP_CLASSES if args.subset == "pump" else CLASSES; NC = len(classes)
    if rep == "spec":
        X, M = spec_data()
    else:
        X, M = load_all(rep, "pump_l2_w1" if args.subset == "pump" else "rotor_l2_w1", "pump_frames_w1" if args.subset == "pump" else "rotor_frames_w1", classes)
    dom = M["domain"]
    if args.norm == "ref" and args.method in ("evit", "snn", "evstr", "spec", "cnn"):
        X, dom = ref_subtract(X, M)
    Xg = torch.from_numpy(X).to(DEV); y_all = torch.from_numpy(M["y"]).to(DEV)
    print(f"{args.method} {args.kind} s{args.seed} norm={args.norm}: loaded {tuple(X.shape)} {X.dtype} on GPU", flush=True)
    suffix = args.variant if args.method == "ssl" else args.norm
    tag = args.tag or f"{args.method}_{suffix}_{args.kind}_s{args.seed}"
    out_dir = os.path.join(ROOT, "outputs", "baselines", tag)
    results = []
    for tname, is_t in tasks(args.kind):
        if args.only and args.only not in tname:
            continue
        torch.manual_seed(args.seed); np.random.seed(args.seed); t0 = time.time()
        tmask = np.array([is_t(v, r) for v, r in zip(M["view"], M["rpm"])])
        tr = np.where(~tmask)[0]; te = np.where(tmask & M["is_eval"])[0]
        if args.kind == "indomain":
            tr, te = indomain_split(M["file"], M["is_eval"], args.seed)
        ys = M["y"][te]
        mu = Xg[tr].float().mean().item(); sd = Xg[tr].float().std().item() + 1e-6
        if args.method == "evit":
            model = EViT(X.shape[1], n_classes=NC).to(DEV)
            train_supervised(model, Xg, y_all, tr, mu, sd, args); preds = predict(model, Xg, te, mu, sd)
        elif args.method == "snn":
            model = SpikingCNN(n_classes=NC, readout=args.snn_readout, thr=args.snn_thr).to(DEV)
            train_supervised(model, Xg, y_all, tr, mu, sd, args); preds = predict(model, Xg, te, mu, sd)
        elif args.method == "evstr":
            model = EVSTr(n_classes=NC).to(DEV); signed = args.norm == "ref"
            fwd = lambda x: model(voxel_to_set(x, args.K, signed))
            train_supervised(model, Xg, y_all, tr, mu, sd, args, fwd=fwd); preds = predict(model, Xg, te, mu, sd, fwd=fwd)
        elif args.method == "cnn":
            model = CNN(X.shape[1], NC).to(DEV)
            train_supervised(model, Xg, y_all, tr, mu, sd, args); preds = predict(model, Xg, te, mu, sd)
        elif args.method == "spec":
            model = CNN(1, NC).to(DEV)
            aug = lambda x: torch.roll(x, int(np.random.randint(-4, 5)), 3)   # time shift only (frequency axis is physical)
            train_supervised(model, Xg, y_all, tr, mu, sd, args, aug=aug); preds = predict(model, Xg, te, mu, sd)
        elif args.method == "ssl":
            model = Encoder(X.shape[1], n_classes=NC).to(DEV)
            u_t = np.where(tmask & M["is_ref"] & (M["y"] == 0))[0] if args.variant == "strict" else np.where(tmask)[0]
            unl = np.concatenate([tr, u_t])
            # stage 1: SimCLR pre-training of the encoder on all unlabelled windows (source + allowed target windows)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05); bs = 64
            for ep in range(args.pre_epochs):
                model.train(); perm = np.random.permutation(unl)
                for b in range(0, len(perm) - bs + 1, bs):
                    x = (Xg[perm[b:b + bs]].float() - mu) / sd
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        z1 = model.proj(model.feat(augment_frames(x, True))); z2 = model.proj(model.feat(augment_frames(x, True)))
                        loss = nt_xent(z1.float(), z2.float())
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            # stage 2: supervised on source + cross-supervision (confidence-thresholded pseudo-labels) on the target windows
            ut = torch.as_tensor(u_t, device=DEV)

            def cross_sup(m):
                idx = ut[torch.randint(0, len(ut), (min(args.bs, len(ut)),), device=DEV)]; x = (Xg[idx].float() - mu) / sd
                with torch.no_grad():
                    p = m(augment_frames(x)).float().softmax(1); conf, pl = p.max(1)
                mask = (conf > 0.9).float()
                return (F.cross_entropy(m(augment_frames(x, True)).float(), pl, reduction="none") * mask).mean()
            train_supervised(model, Xg, y_all, tr, mu, sd, args, extra_loss=cross_sup if len(u_t) else None); preds = predict(model, Xg, te, mu, sd)
        elif args.method == "tta":
            model = CNN(X.shape[1], NC).to(DEV)
            train_supervised(model, Xg, y_all, tr, mu, sd, args); preds = predict(model, Xg, te, mu, sd)
            state = {k: v.clone() for k, v in model.state_dict().items()}
            variants = {}
            tdoms = sorted(set(dom[te]))

            def adabn(model, idx_by_dom):
                """re-estimate BN statistics per target domain on the given windows, predict that domain."""
                out = np.zeros(len(te), int)
                for d, idx in idx_by_dom.items():
                    model.load_state_dict(state)
                    for m in model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.reset_running_stats(); m.momentum = None
                    model.train()
                    with torch.no_grad():
                        for b in range(0, len(idx), 64):
                            model((Xg[idx[b:b + 64]].float() - mu) / sd)
                    sel = np.where(dom[te] == d)[0]; out[sel] = predict(model, Xg, te[sel], mu, sd)
                return out
            variants["adabnref"] = adabn(model, {d: np.where((dom == d) & M["is_ref"] & (M["y"] == 0))[0] for d in tdoms})
            variants["adabntest"] = adabn(model, {d: te[dom[te] == d] for d in tdoms})
            # TENT: BN statistics from the test batches, BN affine parameters updated by entropy minimisation (one pass)
            out = np.zeros(len(te), int)
            for d in tdoms:
                model.load_state_dict(state); sel = te[dom[te] == d]
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.reset_running_stats(); m.momentum = None
                params = [p for m in model.modules() if isinstance(m, nn.BatchNorm2d) for p in (m.weight, m.bias)]
                for p in model.parameters():
                    p.requires_grad_(False)
                for p in params:
                    p.requires_grad_(True)
                opt = torch.optim.Adam(params, lr=1e-3); model.train(); perm = np.random.permutation(sel)
                for b in range(0, len(perm), 64):
                    x = (Xg[perm[b:b + 64]].float() - mu) / sd; p = model(x).float().softmax(1)
                    loss = -(p * torch.log(p + 1e-8)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
                for p in model.parameters():
                    p.requires_grad_(True)
                s2 = np.where(dom[te] == d)[0]; out[s2] = predict(model, Xg, sel, mu, sd)
            variants["tent"] = out
            for name, pv in variants.items():
                r = evaluate(pv, ys, tname, t0, n_classes=NC); save(os.path.join(ROOT, "outputs", "baselines", f"frames_{name}_{args.kind}_s{args.seed}"), args, r)
        res = evaluate(preds, ys, tname, t0, dict(idx=te.tolist()), NC); results.append(res); save(out_dir, args, res)
        del model; torch.cuda.empty_cache()
    print(f"== {tag}: mean acc {np.mean([r['acc'] for r in results]):.3f} min {np.min([r['acc'] for r in results]):.3f} f1 {np.mean([r['f1'] for r in results]):.3f}", flush=True)


if __name__ == "__main__":
    main()
