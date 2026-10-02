"""Cross-device Reference-Only transfer with REF-EvSet: train on ALL Rotor domains (4 classes), test on Pump
windows of the four shared classes (Healthy / Inner / Outer / Ball) with each Pump domain's own healthy reference.
Reports 4-class accuracy, healthy-vs-fault detection AUROC (1 - p_healthy) and per-class recall; and the reverse
direction (train on Pump shared classes, test on Rotor).
usage: python scripts/cross_device.py --seed 0 [--epochs 30]
"""
import argparse
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evset.data.rotor_dataset import RotorTokens  # noqa: E402
from evset.eval.protocol import CLASSES  # noqa: E402
from evset.data.registry import PUMP_CLASSES  # noqa: E402
from evset.models.evset_net import EvSetNet  # noqa: E402
from train_evset import GpuBank, augment  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SHARED = ["Healthy", "Inner", "Outer", "Ball"]


def train(bank, idx, args, device, n_classes):
    ds = bank.ds
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    model = EvSetNet(n_classes, in_ch=4, n_attr=6, d=128, n_isab=2, m=16, use_rcn=False, use_attr=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    nb = len(idx) // 16; sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=args.epochs * nb, pct_start=0.1)
    y_all = torch.from_numpy(ds.y).to(device)
    for ep in range(args.epochs):
        model.train(); perm = np.random.permutation(idx)
        for b in range(nb):
            bi = perm[b * 16:(b + 1) * 16]
            x, a, r = bank.batch(bi, True, gen, (4, 12)); y = y_all[bi]
            x, a, r = augment(x, a, r, args, gen)
            lam = float(np.random.beta(0.4, 0.4)); p2 = torch.randperm(x.shape[0], device=device)
            x = lam * x + (1 - lam) * x[p2]; a = lam * a + (1 - lam) * a[p2]; r = lam * r + (1 - lam) * r[p2]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x, a, r)
            loss = lam * F.cross_entropy(logits.float(), y) + (1 - lam) * F.cross_entropy(logits.float(), y[p2])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return model


def evaluate(model, bank, idx, device):
    model.eval(); probs = []
    with torch.no_grad():
        for b in range(0, len(idx), 16):
            x, a, r = bank.batch(idx[b:b + 16], False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x, a, r)
            probs.append(logits.float().softmax(1).cpu())
    return torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    args.token_keep, args.order_jitter, args.level_shift, args.contrast_scale = 0.5, 1, 1.0, 1.0
    device = "cuda"
    rot = RotorTokens(os.path.join(ROOT, "cache", "rotor_l2_w1"), 1.0, max_order_bins=256, bins_per_s=10000, classes=CLASSES)
    pum = RotorTokens(os.path.join(ROOT, "cache", "pump_l2_w1"), 1.0, max_order_bins=256, bins_per_s=5000, classes=PUMP_CLASSES)
    # remap pump labels to the shared index space (-1 for non-shared classes)
    pump_shared = np.array([SHARED.index(PUMP_CLASSES[c]) if PUMP_CLASSES[c] in SHARED else -1 for c in pum.y])
    out = {}
    for src_name, src, tgt_name, tgt, y_tgt in [("rotor", rot, "pump", pum, pump_shared), ("pump", pum, "rotor", rot, rot.y.copy())]:
        bank_s = GpuBank(src, device); bank_t = GpuBank(tgt, device)
        if src_name == "pump":
            keep = np.isin(src.y, [PUMP_CLASSES.index(c) for c in SHARED])
            src.y = np.array([SHARED.index(PUMP_CLASSES[c]) if PUMP_CLASSES[c] in SHARED else -1 for c in src.y])
            bank_s = GpuBank(src, device)
            tr_idx = np.where(keep)[0]
        else:
            tr_idx = np.arange(len(src.y))
        model = train(bank_s, tr_idx, args, device, 4)
        te = np.where(tgt.is_eval & (y_tgt >= 0))[0]
        probs = evaluate(model, bank_t, te, device); pred = probs.argmax(1); y = y_tgt[te]
        acc = float((pred == y).mean()); f1 = float(f1_score(y, pred, average="macro"))
        auc = float(roc_auc_score((y != 0).astype(int), 1 - probs[:, 0]))
        rec = [float((pred[y == c] == c).mean()) for c in range(4)]
        # per-domain detection AUROC
        doms = tgt.domain[te]; aucs = [roc_auc_score((y[doms == d] != 0).astype(int), 1 - probs[doms == d, 0]) for d in np.unique(doms) if len(np.unique(y[doms == d])) > 1]
        out[f"{src_name}->{tgt_name}"] = dict(acc=acc, f1=f1, det_auroc=auc, det_auroc_per_domain_mean=float(np.mean(aucs)), det_auroc_per_domain_min=float(np.min(aucs)), recall=rec, n=int(len(te)))
        print(f"{src_name}->{tgt_name}: 4-class acc {acc:.3f} f1 {f1:.3f} | detection AUROC pooled {auc:.3f}, per-domain mean {np.mean(aucs):.3f} min {np.min(aucs):.3f} | recall H/I/O/B {np.round(rec, 2)}", flush=True)
        del bank_s, bank_t, model; torch.cuda.empty_cache()
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "outputs", f"cross_device_s{args.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
