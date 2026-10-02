"""Open-set (leave-one-class-out) evaluation from saved runs outputs/evset/openset_ex<c>_lodo_s0/*.json.

Scores for 'unknown' detection on target-domain evaluation windows:
  conf : 1 - max softmax probability of the closed-set model trained without class c
  ref  : zero-shot reference deviation, mean |z| of the whitened reference-ratio features against the target
         domain's healthy reference windows (no training at all)
  rank : average of the two rank-normalised scores
AUROC of (windows of the excluded class) vs (windows of known classes), averaged over the 6 LODO tasks.
Also reports closed-set accuracy on the known classes.
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
CLASSES = ["Healthy", "Inner", "Outer", "Ball"]


def main():
    z = np.load(os.path.join(ROOT, "cache", "rotor_l2_w1_linear_feats_v2.npz"), allow_pickle=True)
    meta = z["meta"]; label = meta[:, 1]; is_ref = meta[:, 4].astype(int).astype(bool)
    dom = np.array([f"{v}@{r}" for v, r in zip(meta[:, 2], meta[:, 3])])
    X = z["snr_refratio"].astype(np.float32)
    ref_score = np.zeros(len(X))
    for d in np.unique(dom):
        m = dom == d; r = m & is_ref & (label == "Healthy")
        mu, sd = X[r].mean(0), X[r].std(0) + 1e-3
        ref_score[m] = np.abs((X[m] - mu) / sd).mean(1)
    for c in (1, 2, 3):
        res = {"conf": [], "ref": [], "rank": [], "closed_acc": []}
        for jp in sorted(glob.glob(os.path.join(ROOT, "outputs", "evset", f"openset_ex{c}_lodo_s0", "*.json"))):
            r = json.load(open(jp))
            probs = np.array(r["probs"]); ys = np.array(r["ys"]); idx = np.array(r["idx"])
            known = [k for k in range(4) if k != c]
            pk = probs[:, known]
            conf = 1 - pk.max(1)
            rs = ref_score[idx]
            unk = (ys == c).astype(int)
            rank = (np.argsort(np.argsort(conf)) + np.argsort(np.argsort(rs))) / (2 * len(conf))
            res["conf"].append(roc_auc_score(unk, conf)); res["ref"].append(roc_auc_score(unk, rs)); res["rank"].append(roc_auc_score(unk, rank))
            kn = ys != c
            res["closed_acc"].append(float((np.array(known)[pk[kn].argmax(1)] == ys[kn]).mean()))
        if res["conf"]:
            print(f"unknown = {CLASSES[c]:8s}: AUROC conf {np.mean(res['conf']):.3f}  ref-deviation {np.mean(res['ref']):.3f}  rank-avg {np.mean(res['rank']):.3f}  "
                  f"| closed-set acc on known {np.mean(res['closed_acc']):.3f}  ({len(res['conf'])} tasks)")


if __name__ == "__main__":
    main()
