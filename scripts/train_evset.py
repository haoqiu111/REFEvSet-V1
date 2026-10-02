"""Train / evaluate the REF-EvSet set encoder (EvSetNet) under the Reference-Only protocols (GPU-resident data):
Rotor lovo / cs / lodo / indomain, and Pump pump_lovo / pump_cs / pump_lodo with --subset pump.

usage: python scripts/train_evset.py --kind lodo --seed 0 [--l2 rotor_l2 --win 0.5] [--no-rcn --no-attr --no-refch]
Outputs: outputs/evset/<tag>/<task>.json with window predictions, accuracy, macro-F1, file-level vote accuracy.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset.data.rotor_dataset import RotorTokens  # noqa: E402
from evset.eval.protocol import tasks, CLASSES, indomain_split  # noqa: E402
from evset.data.registry import PUMP_CLASSES  # noqa: E402
from evset.models.evset_net import EvSetNet  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


class GpuBank:
    """All tokens on the GPU; builds batches (x, a, r) with domain references without CPU work."""

    def __init__(self, ds: RotorTokens, device, split_ref: bool = False):
        self.ds = ds
        self.tok = torch.from_numpy(ds.tokens).to(device)  # (N,P,O,2) fp16
        self.attr = torch.from_numpy(ds.attrs).to(device)  # (N,P,3)
        self.sub = torch.from_numpy(ds.sub).to(device) if ds.sub is not None else None  # (N,P,2,8,8) fp16
        self.last_sub = None
        self.xy = torch.from_numpy(ds.xy).to(device)  # (P,2)
        self.o1 = int(np.argmin(np.abs(ds.orders - 1.0)))
        self.dom_id = torch.tensor([ds.domains.index(d) for d in ds.domain], device=device)
        maxr = max(len(v) for v in ds.ref_idx.values())
        self.ref_tab = torch.full((len(ds.domains), maxr), -1, dtype=torch.long, device=device)
        self.ref_n = torch.zeros(len(ds.domains), dtype=torch.long, device=device)
        self.calib_idx = {}
        for di, d in enumerate(ds.domains):
            idx = ds.ref_idx[d]
            if split_ref:
                self.calib_idx[d] = idx[1::2]
                idx = idx[0::2]
            self.ref_tab[di, :len(idx)] = torch.from_numpy(idx).to(device)
            self.ref_n[di] = len(idx)
        self.device = device

    def reference(self, idx, k=None, gen=None):
        """Mean reference field/attrs for the domains of windows idx. k=None -> all reference windows."""
        di = self.dom_id[idx]
        n = self.ref_n[di]  # (B,)
        if k is None:
            k = int(n.min())
            sel = self.ref_tab[di][:, :k]
        else:
            u = torch.rand(len(idx), k, generator=gen, device=self.device)
            pos = (u * n[:, None]).long()  # sample with replacement (cheap, k <= n)
            sel = torch.gather(self.ref_tab[di], 1, pos)
        rt = self.tok[sel].float().mean(1)  # (B,P,O,2)
        ra = self.attr[sel].mean(1)  # (B,P,3)
        self._rs = self.sub[sel].float().mean(1) if self.sub is not None else None  # (B,P,2,8,8)
        return rt, ra

    def batch(self, idx, train, gen=None, ref_k=(4, 12)):
        idx = torch.as_tensor(idx, device=self.device)
        k = int(torch.randint(ref_k[0], ref_k[1] + 1, (1,), generator=gen, device=self.device)) if train else None
        rt, ra = self.reference(idx, k, gen)
        t = self.tok[idx].float()
        x = torch.cat([t, t - rt], -1)
        B, P = t.shape[:2]
        if getattr(self, "no_refattr", False):  # ablation: no reference information in the attributes either
            a = torch.cat([self.xy[None].expand(B, -1, -1), self.attr[idx][:, :, :1], self.attr[idx][:, :, 1:2],
                           torch.zeros_like(rt[:, :, self.o1, :])], -1)
        else:
            a = torch.cat([self.xy[None].expand(B, -1, -1), self.attr[idx][:, :, :1] - ra[:, :, :1],
                           self.attr[idx][:, :, 1:2], rt[:, :, self.o1, :]], -1)
        if self.sub is not None:
            st = self.sub[idx].float()
            self.last_sub = torch.cat([st, st - self._rs], 2)  # (B,P,4,8,8)
        return x, a, rt


def augment(x, a, r, args, gen, sub=None):
    B, P, O, C = x.shape
    if args.token_keep < 1:
        k = int(P * args.token_keep)
        perm = torch.rand(B, P, generator=gen, device=x.device).argsort(1)[:, :k]
        bi = torch.arange(B, device=x.device)[:, None]
        x, a, r = x[bi, perm], a[bi, perm], r[bi, perm]
        if sub is not None:
            sub = sub[bi, perm]
    if args.order_jitter > 0:
        sh = int(torch.randint(-args.order_jitter, args.order_jitter + 1, (1,), generator=gen, device=x.device))
        if sh != 0:
            x = torch.roll(x, sh, 2); r = torch.roll(r, sh, 2)
    if args.level_shift > 0:
        s = (torch.rand(B, 1, 1, 1, generator=gen, device=x.device) * 2 - 1) * args.level_shift
        x = x.clone(); x[..., :2] += s; r = r + s
    if args.contrast_scale > 1 and x.shape[-1] == 4:
        # random log-uniform scaling of the fault contrast (target - reference): the model must not rely on the
        # absolute contrast level, which varies with pose / speed
        la = (torch.rand(B, 1, 1, 1, generator=gen, device=x.device) * 2 - 1) * float(np.log(args.contrast_scale))
        al = torch.exp(la)
        x = x.clone()
        x[..., 2:4] = x[..., 2:4] * al
        x[..., :2] = r + x[..., 2:4]
    return (x, a, r, sub) if sub is not None else (x, a, r)


def align_loss(z, y, dom, n_classes):
    """Class-conditional cross-domain prototype alignment: variance of per-domain class means around the class mean."""
    loss, n = z.new_zeros(()), 0
    for c in range(n_classes):
        mc = y == c
        if mc.sum() < 2:
            continue
        doms = dom[mc].unique()
        if len(doms) < 2:
            continue
        mu_c = z[mc].mean(0)
        for d in doms:
            md = mc & (dom == d)
            loss = loss + ((z[md].mean(0) - mu_c) ** 2).sum(); n += 1
    return loss / max(n, 1)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def run_task(bank: GpuBank, tname, is_t, args, device):
    ds = bank.ds
    tmask = np.array([is_t(v, r) for v, r in zip(ds.view, ds.rpm)])
    tr_idx = np.where(~tmask)[0]
    te_idx = np.where(tmask & ds.is_eval)[0]
    if args.kind in ("indomain", "pump_indomain"):
        tr_idx, te_idx = indomain_split(ds.file_idx, ds.is_eval, args.seed)
    if args.exclude_class is not None:  # open-set: the excluded class is never seen in training (still in the test set)
        tr_idx = tr_idx[ds.y[tr_idx] != args.exclude_class]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    in_ch = 4 if not args.no_refch else 2
    model = EvSetNet(len(args.classes), in_ch=in_ch, n_attr=6, d=args.d, n_isab=args.n_isab, m=args.m,
                     use_rcn=not args.no_rcn, use_attr=not args.no_attr, dropout=args.dropout, use_sub=args.subpatch,
                     agg=args.agg, grid=tuple(ds.grid)).to(device)
    dom_head = torch.nn.Sequential(torch.nn.Linear(args.d, args.d), torch.nn.GELU(), torch.nn.Linear(args.d, len(ds.domains))).to(device)
    params = list(model.parameters()) + (list(dom_head.parameters()) if args.dann > 0 else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    n_batches = len(tr_idx) // args.bs
    steps = args.epochs * n_batches
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    y_all = torch.from_numpy(ds.y).to(device)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot, n, correct = 0.0, 0, 0
        perm = np.random.permutation(tr_idx)
        for b in range(n_batches):
            idx = perm[b * args.bs:(b + 1) * args.bs]
            x, a, r = bank.batch(idx, True, gen, (args.ref_kmin, args.ref_kmax))
            sub = bank.last_sub if args.subpatch else None
            y = y_all[idx]
            if sub is not None:
                x, a, r, sub = augment(x, a, r, args, gen, sub)
            else:
                x, a, r = augment(x, a, r, args, gen)
            if args.no_refch:
                x = x[..., :2]
            if args.mixup > 0:
                lam = float(np.random.beta(args.mixup, args.mixup))
                p2 = torch.randperm(x.shape[0], device=device)
                x = lam * x + (1 - lam) * x[p2]; a = lam * a + (1 - lam) * a[p2]; r = lam * r + (1 - lam) * r[p2]
                if sub is not None:
                    sub = lam * sub + (1 - lam) * sub[p2]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(x, a, r, sub=sub)
                loss = lam * F.cross_entropy(logits.float(), y) + (1 - lam) * F.cross_entropy(logits.float(), y[p2])
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, z = model(x, a, r, sub=sub)
                loss = F.cross_entropy(logits.float(), y, label_smoothing=args.ls)
                if args.align > 0:
                    loss = loss + args.align * align_loss(z.float(), y, bank.dom_id[torch.as_tensor(idx, device=device)], len(args.classes))
                if args.dann > 0:
                    dlog = dom_head(GradReverse.apply(z.float(), args.dann))
                    loss = loss + F.cross_entropy(dlog, bank.dom_id[torch.as_tensor(idx, device=device)])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(y); n += len(y); correct += (logits.argmax(1) == y).sum().item()
        if (ep + 1) % max(1, args.epochs // 5) == 0 or ep == args.epochs - 1:
            print(f"  [{tname}] ep {ep + 1:3d} loss {tot / n:.3f} train-acc {correct / n:.3f} {time.time() - t0:.0f}s", flush=True)
    model.eval(); preds, probs = [], []
    with torch.no_grad():
        for b in range(0, len(te_idx), args.bs):
            idx = te_idx[b:b + args.bs]
            x, a, r = bank.batch(idx, False)
            sub = bank.last_sub if args.subpatch else None
            if args.no_refch:
                x = x[..., :2]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x, a, r, sub=sub)
            p = logits.float().softmax(1)
            preds.append(p.argmax(1).cpu()); probs.append(p.cpu())
    preds = torch.cat(preds).numpy(); probs = torch.cat(probs).numpy(); ys = ds.y[te_idx]
    calib = {}
    if bank.calib_idx:
        tdoms = sorted(set(ds.domain[te_idx]))
        with torch.no_grad():
            for d in tdoms:
                ci = bank.calib_idx[d]
                x, a, r = bank.batch(ci, False)
                sub = bank.last_sub if args.subpatch else None
                if args.no_refch:
                    x = x[..., :2]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _ = model(x, a, r, sub=sub)
                calib[d] = lg.float().softmax(1).cpu().numpy().tolist()
    acc = float((preds == ys).mean()); f1 = float(f1_score(ys, preds, average="macro"))
    fidx = ds.file_idx[te_idx]; votes = []
    for fi in np.unique(fidx):
        m = fidx == fi
        votes.append(int(np.bincount(preds[m], minlength=len(args.classes)).argmax() == ys[m][0]))
    per_class = [float((preds[ys == c] == c).mean()) for c in range(len(args.classes))]
    print(f"[{tname}] acc {acc:.3f} f1 {f1:.3f} file-vote {np.mean(votes):.3f} per-class {np.round(per_class, 2)} ({time.time() - t0:.0f}s)", flush=True)
    res = dict(task=tname, acc=acc, f1=f1, file_vote=float(np.mean(votes)), per_class=per_class,
               preds=preds.tolist(), ys=ys.tolist(), idx=te_idx.tolist(), probs=probs.tolist(),
               domains=ds.domain[te_idx].tolist(), calib_probs=calib)
    if args.save_model:
        res["state_dict_path"] = os.path.join(args.out_dir, re.sub(r"[^\w.-]", "_", tname) + ".pt")
        torch.save(model.state_dict(), res["state_dict_path"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="lodo"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l2", default="rotor_l2"); ap.add_argument("--win", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--n_isab", type=int, default=2); ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--token_keep", type=float, default=0.5); ap.add_argument("--order_jitter", type=int, default=1)
    ap.add_argument("--level_shift", type=float, default=1.0); ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--ls", type=float, default=0.0); ap.add_argument("--max_orders", type=int, default=256)
    ap.add_argument("--ref_kmin", type=int, default=4); ap.add_argument("--ref_kmax", type=int, default=12)
    ap.add_argument("--contrast_scale", type=float, default=1.0); ap.add_argument("--align", type=float, default=0.0)
    ap.add_argument("--dann", type=float, default=0.0)
    ap.add_argument("--no-rcn", dest="no_rcn", action="store_true"); ap.add_argument("--no-attr", dest="no_attr", action="store_true")
    ap.add_argument("--no-refch", dest="no_refch", action="store_true"); ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--split_ref", action="store_true"); ap.add_argument("--exclude_class", type=int, default=None)
    ap.add_argument("--no-refattr", dest="no_refattr", action="store_true")
    ap.add_argument("--subpatch", action="store_true"); ap.add_argument("--frames", default="rotor_frames_w1")
    ap.add_argument("--agg", default="set", choices=["set", "grid"])
    ap.add_argument("--tag", default=None); ap.add_argument("--only", default=None)
    ap.add_argument("--subset", default="rotor")
    args = ap.parse_args()
    if args.agg == "grid":
        args.token_keep = 1.0
    device = "cuda"
    tag = args.tag or f"{args.l2}_{args.kind}_s{args.seed}"
    args.out_dir = os.path.join(ROOT, "outputs", "evset", tag); os.makedirs(args.out_dir, exist_ok=True)
    classes = PUMP_CLASSES if args.subset == "pump" else CLASSES
    bps = 5000 if args.subset == "pump" else 10000
    ds = RotorTokens(os.path.join(ROOT, "cache", args.l2), args.win, max_order_bins=args.max_orders, bins_per_s=bps, classes=classes,
                     frames_dir=os.path.join(ROOT, "cache", args.frames) if args.subpatch else None)
    args.classes = classes
    bank = GpuBank(ds, device, split_ref=args.split_ref)
    bank.no_refattr = args.no_refattr
    print(f"loaded {len(ds.y)} windows, tokens {ds.tokens.shape} ({ds.tokens.nbytes / 1e9:.1f} GB) on GPU", flush=True)
    results = []
    for tname, is_t in tasks(args.kind):
        if args.only and args.only not in tname:
            continue
        res = run_task(bank, tname, is_t, args, device)
        results.append(res)
        with open(os.path.join(args.out_dir, re.sub(r"[^\w.-]", "_", tname) + ".json"), "w") as fh:
            json.dump(dict(args=vars(args), **res), fh)
    accs = [r["acc"] for r in results]
    print(f"== {tag}: mean acc {np.mean(accs):.3f} min {np.min(accs):.3f} f1 {np.mean([r['f1'] for r in results]):.3f} "
          f"file-vote {np.mean([r['file_vote'] for r in results]):.3f}", flush=True)


if __name__ == "__main__":
    main()
