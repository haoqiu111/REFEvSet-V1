"""Main result tables (markdown) from outputs/: linear brackets (healthy-ref / oracle) per window length and
REF-EvSet configurations aggregated over seeds. Writes outputs/main_table.md and prints it.

usage: python scripts/make_main_table.py
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
KINDS = ["lovo", "cs", "lodo"]
PKINDS = ["pump_lovo", "pump_cs", "pump_lodo"]


def linear_rows(log_path, feats=("snr_refratio", "snr_mean"), kinds=KINDS):
    """Parse the printed lines of linear_v2.py logs: '<feat> <norm> <kind> acc=... (min ...) f1=...'."""
    rows = {}
    if not os.path.exists(log_path):
        return rows
    for line in open(log_path, encoding="utf-8", errors="ignore"):
        m = re.match(r"(\S+)\s+(\S+)\s+(\S+)\s+acc=([\d.]+) \(min ([\d.]+)\) f1=([\d.]+)", line)
        if m and m.group(1) in feats and m.group(3) in kinds:
            rows[(m.group(1), m.group(2), m.group(3))] = (float(m.group(4)), float(m.group(5)), float(m.group(6)))
    return rows


def evset_rows():
    agg = defaultdict(lambda: defaultdict(list))
    for d in sorted(glob.glob(os.path.join(ROOT, "outputs", "evset", "*")) + glob.glob(os.path.join(ROOT, "outputs", "baselines", "*"))):
        tag = os.path.basename(d)
        cfg = re.sub(r"_s\d+$", "", tag)
        m = re.match(r"(.*?)_(pump_lovo|pump_cs|pump_lodo|lovo|cs|lodo)$", cfg)
        if not m:
            continue
        name, kind = m.group(1), m.group(2)
        rs = [json.load(open(p)) for p in glob.glob(os.path.join(d, "*.json"))]
        if not rs:
            continue
        accs = [r["acc"] for r in rs]
        agg[name][kind].append((np.mean(accs), np.min(accs), np.mean([r["f1"] for r in rs])))
    return agg


def fmt(vals):
    if not vals:
        return "—"
    a = np.array(vals)
    s = f"{a[:, 0].mean():.3f}"
    if len(vals) > 1:
        s += f" ± {a[:, 0].std():.3f}"
    return s + f" (min {a[:, 1].mean():.2f}, n={len(vals)})"


def main():
    out = []
    out.append("## Rotor: linear brackets (LR on whitened per-patch reference-ratio features)\n")
    out.append("| window | normalisation | LOVO | CS | LODO |\n|---|---|---|---|---|")
    for win, log in [("0.5 s", "linear_v2_rotor.log"), ("1 s", "linear_v2_w1.log"), ("2 s", "linear_v2_w2.log")]:
        rows = linear_rows(os.path.join(ROOT, "outputs", log))
        for norm in ["healthy-ref", "oracle"]:
            cells = [f"{rows[('snr_refratio', norm, k)][0]:.3f} (min {rows[('snr_refratio', norm, k)][1]:.2f})" if ("snr_refratio", norm, k) in rows else "—" for k in KINDS]
            out.append(f"| {win} | {norm} | " + " | ".join(cells) + " |")
    out.append("\n## Rotor: learned models (window accuracy, mean ± std over seeds; min = worst task)\n")
    out.append("| configuration | LOVO | CS | LODO |\n|---|---|---|---|")
    agg = evset_rows()
    for name in sorted(agg, key=lambda n: -np.mean([v[0] for v in agg[n].get("lodo", [(0, 0, 0)])])):
        if any(k in agg[name] for k in KINDS):
            out.append(f"| {name} | " + " | ".join(fmt(agg[name].get(k, [])) for k in KINDS) + " |")
    out.append("\n## Pump: linear brackets and learned models\n")
    out.append("| method | pump_lovo | pump_cs | pump_lodo |\n|---|---|---|---|")
    for win, log in [("0.5 s", "linear_pump.log"), ("1 s", "linear_pump_w1.log")]:
        rows = linear_rows(os.path.join(ROOT, "outputs", log), feats=("snr_mean", "snr_all"), kinds=PKINDS)
        for feat in ["snr_mean", "snr_all"]:
            for norm in ["healthy-ref", "oracle"]:
                cells = [f"{rows[(feat, norm, k)][0]:.3f} (min {rows[(feat, norm, k)][1]:.2f})" if (feat, norm, k) in rows else "—" for k in PKINDS]
                out.append(f"| linear {feat} {norm} ({win}) | " + " | ".join(cells) + " |")
    for name in sorted(agg):
        if any(k in agg[name] for k in PKINDS):
            out.append(f"| {name} | " + " | ".join(fmt(agg[name].get(k, [])) for k in PKINDS) + " |")
    text = "\n".join(out)
    open(os.path.join(ROOT, "outputs", "main_table.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
