"""Collect per-task JSON results from outputs/evset/<tag>/ and outputs/baselines/<tag>/ into one table.

usage: python scripts/collect_results.py [pattern]
Prints per tag: kind-wise mean / min accuracy, macro-F1 and (if present) file-vote accuracy; writes outputs/summary.csv.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    rows = []
    for base in ["evset", "baselines"]:
        for d in sorted(glob.glob(os.path.join(ROOT, "outputs", base, "*"))):
            tag = os.path.basename(d)
            if pat and pat not in tag:
                continue
            per_kind = defaultdict(list)
            for jp in sorted(glob.glob(os.path.join(d, "*.json"))):
                r = json.load(open(jp))
                kind = r["task"].split(":")[0]
                per_kind[kind].append(r)
            for kind, rs in per_kind.items():
                accs = [r["acc"] for r in rs]; f1s = [r["f1"] for r in rs]
                fv = [r.get("file_vote", np.nan) for r in rs]
                rows.append((base, tag, kind, len(rs), np.mean(accs), np.min(accs), np.mean(f1s), np.nanmean(fv)))
    rows.sort(key=lambda r: (r[2], -r[4]))
    print(f"{'base':9s} {'tag':40s} {'kind':5s} n  acc    min    f1     file")
    for b, t, k, n, a, mn, f, fv in rows:
        print(f"{b:9s} {t:40s} {k:5s} {n:d}  {a:.3f}  {mn:.3f}  {f:.3f}  {fv:.3f}")
    # seed aggregation: tag prefix without the trailing _s<seed>
    import re
    agg = defaultdict(list)
    for b, t, k, n, a, mn, f, fv in rows:
        agg[(b, re.sub(r"_s\d+$", "", t), k)].append((a, mn, f))
    print("\n== seed-aggregated (mean +- std over seeds)")
    for (b, t, k), v in sorted(agg.items(), key=lambda x: (x[0][2], -np.mean([y[0] for y in x[1]]))):
        a = np.array(v)
        print(f"{b:9s} {t:36s} {k:9s} seeds={len(v)}  acc {a[:, 0].mean():.3f} +- {a[:, 0].std():.3f}  min {a[:, 1].mean():.3f}  f1 {a[:, 2].mean():.3f}")
    with open(os.path.join(ROOT, "outputs", "summary.csv"), "w") as fh:
        fh.write("base,tag,kind,n_tasks,acc,min_acc,f1,file_vote\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
