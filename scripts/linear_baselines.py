"""Linear baselines on the un-whitened Rotor level-2 tokens (log|X| features, including the per-patch
reference ratio); the whitened counterpart is scripts/linear_v2.py. Also provides pool2() and run() for linear_v2.py.

Feature sets (per 0.5-s window):
  glob     : global (spatially summed) signed+unsigned order spectra, pooled to 0.25 order  (512-D)
  setmean  : mean over patches of per-patch log order spectra (signed+unsigned)             (512-D)
  setstat  : setmean + std over patches                                                     (1024-D)
  coh      : spatial phase coherence per order (mean_p cos(phi_p - phi_glob))              (256-D)
  refratio : per-patch log-ratio to the domain's healthy reference field, pooled mean/max  (1024-D)
Normalisations: raw / healthy-ref (subtract domain reference mean, deployable) / oracle (per-domain z-score).
"""
import os
import sys
import time
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset.eval.protocol import load_rotor_l2_meta, tasks, split, CLASSES  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
L2 = os.path.join(ROOT, "cache", os.environ.get("L2_NAME", "rotor_l2"))
FEAT = os.path.join(ROOT, "cache", os.environ.get("L2_NAME", "rotor_l2") + "_linear_feats.npz")


def pool2(a):
    """pool the last axis by 2 (0.125 -> 0.25 order)."""
    return a.reshape(*a.shape[:-1], a.shape[-1] // 2, 2).mean(-1)


def build_features():
    files = load_rotor_l2_meta(L2)
    # reference fields per domain from healthy files
    ref_field = {}
    for f in files:
        if f.label != "Healthy":
            continue
        z = np.load(f.path)
        tok = z["tokens"][f.ref_mask()][..., :2].astype(np.float32)  # (Nr, P, O, 2)
        ref_field[f.domain] = tok.mean(0)
    feats = {k: [] for k in ["glob", "setmean", "setstat", "coh", "refratio"]}
    meta = []
    for f in files:
        z = np.load(f.path)
        tok = z["tokens"].astype(np.float32)  # (N, P, O, 4)
        glob = pool2(z["glob"]).reshape(len(tok), -1)
        amp = tok[..., :2]
        sm = pool2(amp.mean(1).transpose(0, 2, 1)).reshape(len(tok), -1)
        ss = pool2(amp.std(1).transpose(0, 2, 1)).reshape(len(tok), -1)
        coh = pool2(tok[..., 2].mean(1))
        rr = amp - ref_field[f.domain][None]
        rr_mean = pool2(rr.mean(1).transpose(0, 2, 1)).reshape(len(tok), -1)
        rr_max = pool2(rr.max(1).transpose(0, 2, 1)).reshape(len(tok), -1)
        feats["glob"].append(glob)
        feats["setmean"].append(sm)
        feats["setstat"].append(np.concatenate([sm, ss], 1))
        feats["coh"].append(coh)
        feats["refratio"].append(np.concatenate([rr_mean, rr_max], 1))
        for i in range(len(tok)):
            meta.append((f.name, f.label, f.view, f.rpm, int(f.ref_mask()[i]), int(f.eval_mask()[i])))
        print(f.name, len(tok), flush=True)
    out = {k: np.concatenate(v).astype(np.float32) for k, v in feats.items()}
    np.savez(FEAT, meta=np.array(meta, dtype=object), **out)
    return out, meta


def run(feats, meta, kinds=("lovo", "cs", "lodo"), classes=CLASSES, domains=None, out_name="linear_baselines.json"):
    name = np.array([m[0] for m in meta]); label = np.array([m[1] for m in meta])
    view = np.array([m[2] for m in meta]); rpm = np.array([m[3] for m in meta])
    is_ref = np.array([m[4] for m in meta]).astype(bool); is_eval = np.array([m[5] for m in meta]).astype(bool)
    dom = np.array([f"{v}@{r}" for v, r in zip(view, rpm)]) if domains is None else np.asarray(domains)
    y = np.array([classes.index(l) for l in label])
    results = []
    for fname, X in feats.items():
        for norm in ["raw", "healthy-ref", "oracle"]:
            Xn = X.copy()
            if norm == "healthy-ref":
                for d in np.unique(dom):
                    m = (dom == d) & is_ref & (label == "Healthy")
                    Xn[dom == d] -= X[m].mean(0)
            elif norm == "oracle":
                for d in np.unique(dom):
                    m = dom == d
                    Xn[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-6)
            for kind in kinds:
                accs, f1s = [], []
                for tname, is_t in tasks(kind):
                    tmask = np.array([is_t(v, r) for v, r in zip(view, rpm)])
                    tr = ~tmask
                    te = tmask & is_eval
                    mu, sd = Xn[tr].mean(0), Xn[tr].std(0) + 1e-6
                    clf = LogisticRegression(C=1.0, max_iter=2000)
                    clf.fit((Xn[tr] - mu) / sd, y[tr])
                    pred = clf.predict((Xn[te] - mu) / sd)
                    accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
                    results.append(dict(feat=fname, norm=norm, task=tname, acc=accs[-1], f1=f1s[-1], n_test=int(te.sum())))
                print(f"{fname:9s} {norm:12s} {kind:5s} acc={np.mean(accs):.3f} (min {np.min(accs):.3f}) f1={np.mean(f1s):.3f}  [{' '.join(f'{a:.2f}' for a in accs)}]", flush=True)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", out_name), "w") as fh:
        json.dump(results, fh, indent=1)


if __name__ == "__main__":
    if os.path.exists(FEAT) and "--rebuild" not in sys.argv:
        z = np.load(FEAT, allow_pickle=True)
        feats = {k: z[k] for k in ["glob", "setmean", "setstat", "coh", "refratio"]}
        meta = [tuple(m) for m in z["meta"]]
    else:
        feats, meta = build_features()
    run(feats, meta)
