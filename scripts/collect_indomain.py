"""Aggregate the in-domain (random 70/30) runs of every paradigm and of REF-EvSet into outputs/sota_indomain.json:
{prefix: {mean, std, n}} over seeds. usage: python scripts/collect_indomain.py"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
out = {}
for base in ("baselines", "evset"):
    for d in glob.glob(os.path.join(ROOT, "outputs", base, "*_indomain_s*")):
        m = re.fullmatch(r"(.+)_indomain_s(\d+)", os.path.basename(d))
        if not m:
            continue
        rs = [json.load(open(p)) for p in glob.glob(os.path.join(d, "*.json"))]
        if rs:
            out.setdefault(m.group(1), []).append(float(np.mean([r["acc"] for r in rs])))
res = {k: dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v)) for k, v in out.items()}
json.dump(res, open(os.path.join(ROOT, "outputs", "sota_indomain.json"), "w"), indent=1)
for k, v in sorted(res.items()):
    print(f"{k:22s} {v['mean']:.3f} ± {v['std']:.3f} (n={v['n']})")
