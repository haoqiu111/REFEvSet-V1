"""Nested configuration selection: pick the configuration on two protocols, report it on the third.

Candidates (seed 0 on every protocol): 0.5-s / 1-s / 2-s tokens x {mixup+FiLM, mixup+no-FiLM}.
For each held-out protocol R in {lodo, lovo, cs}: select argmax of the mean accuracy over the other two protocols
(seed 0), then report the selected configuration's accuracy on R (all available seeds).
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
CANDS = {"0.5 s + mixup (FiLM)": "v1mix", "0.5 s + mixup, no FiLM": "selA_w05mixnorcn", "1 s + mixup (FiLM)": "selA_w1mix",
         "1 s + mixup, no FiLM": "selA_w1mixnorcn", "2 s + mixup (FiLM)": "selA_w2mix", "2 s + mixup, no FiLM": "selA_w2mixnorcn"}
ALIAS = {"selA_w1mixnorcn": ["main_w1mixnorcn"], "selA_w1mix": ["main_w1mix"], "v1mix": ["main_w05mix"]}
KINDS = ["lodo", "lovo", "cs"]


def acc(prefix, kind, seeds="all"):
    vals = {}
    for pre in [prefix] + ALIAS.get(prefix, []):
        for d in glob.glob(os.path.join(ROOT, "outputs", "evset", f"{pre}_{kind}_s*")):
            s = int(re.search(r"_s(\d+)$", d).group(1))
            rs = [json.load(open(p))["acc"] for p in glob.glob(os.path.join(d, "*.json"))]
            if rs and (seeds == "all" or s in seeds):
                vals.setdefault(s, []).append(np.mean(rs))
    return {s: float(np.mean(v)) for s, v in vals.items()}


def main():
    table = {name: {k: acc(pre, k) for k in KINDS} for name, pre in CANDS.items()}
    print("candidate accuracies (seed 0 | mean over available seeds):")
    for name in CANDS:
        row = []
        for k in KINDS:
            v = table[name][k]
            row.append(f"{v.get(0, float('nan')):.3f} | {np.mean(list(v.values())):.3f} (n={len(v)})" if v else "—")
        print(f"  {name:26s} " + "  ".join(f"{k}: {r}" for k, r in zip(KINDS, row)))
    print("\nnested selection (select on the other two protocols, seed 0; report on the held-out protocol):")
    for R in KINDS:
        others = [k for k in KINDS if k != R]
        scores = {}
        for name in CANDS:
            vals = [table[name][k].get(0) for k in others]
            if all(v is not None for v in vals):
                scores[name] = np.mean(vals)
        if not scores:
            print(f"  {R}: no complete candidate"); continue
        best = max(scores, key=scores.get)
        rep = table[best][R]
        print(f"  report on {R:4s}: selected '{best}' (selection score {scores[best]:.3f}) -> {R} accuracy "
              f"{np.mean(list(rep.values())):.3f} ± {np.std(list(rep.values())):.3f} over {len(rep)} seed(s)")


if __name__ == "__main__":
    main()
