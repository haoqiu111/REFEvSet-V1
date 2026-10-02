"""Figure + table: same-protocol comparison with re-implemented state-of-the-art paradigms on Rotor.
Reads outputs/baselines/<method>_<kind>_s*/*.json and outputs/evset/main_w1mixnorcn_<kind>_s*/*.json and the linear
brackets; writes figures/fig_sota.png|pdf (grouped bars: LOVO, CS, LODO, mean over the three protocols) and
outputs/sota_table.md (mean ± std over seeds, min task, mean over protocols). usage: python scripts/make_fig_sota.py"""
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset import plot_style as ps  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
E, B = os.path.join(ROOT, "outputs", "evset"), os.path.join(ROOT, "outputs", "baselines")
KINDS = ["lovo", "cs", "lodo"]

# (row label, tag prefix, base dir, target-domain input)
METHODS = [
    ("Per-pixel frequency map + LR [EBFM]", "freqmap", None, "healthy reference"),
    ("Event-frame CNN, raw [TII 2023]", "frames_raw", B, "none"),
    ("Event-frame CNN, reference-subtracted", "frames_ref", B, "healthy reference"),
    ("Event-frame CNN + AdaBN on reference", "frames_adabnref", B, "healthy reference"),
    ("Event-frame CNN + AdaBN on test batch", "frames_adabntest", B, "test batch (transductive)"),
    ("Event-frame CNN + TENT", "frames_tent", B, "test batch (transductive)"),
    ("Voxel-grid CNN, raw", "voxel_raw", B, "none"),
    ("Spiking CNN on voxel grids, raw [Vibration Vision]", "snn_raw", B, "none"),
    ("Spiking CNN, reference-subtracted", "snn_ref", B, "healthy reference"),
    ("Global-rate spectrogram CNN, raw [signal paradigm]", "spec_raw", B, "none"),
    ("Global-rate spectrogram CNN, reference-subtracted", "spec_ref", B, "healthy reference"),
    ("Bi-fovea event Transformer, raw [EViT]", "evit_raw", B, "none"),
    ("Bi-fovea event Transformer, reference-subtracted", "evit_ref", B, "healthy reference"),
    ("Event voxel set Transformer, raw [EVSTr]", "evstr_raw", B, "none"),
    ("Event voxel set Transformer, reference-subtracted", "evstr_ref", B, "healthy reference"),
    ("Self-supervised + cross-supervision, strict [EAAI 2026]", "ssl_strict", B, "healthy reference"),
    ("Self-supervised + cross-supervision, relaxed [EAAI 2026]", "ssl_relaxed", B, "unlabelled target incl. faults"),
    ("Linear LR, whitened reference ratio, healthy-ref", "linear_ref", None, "healthy reference"),
    ("EvSet-Net (ours)", "main_w1mixnorcn", E, "healthy reference"),
    ("Linear LR, class-balanced oracle (not deployable)", "linear_oracle", None, "all target classes"),
]

# manuscript labels (with reference numbers) of the METHODS rows
CITE = {"Per-pixel frequency map + LR [EBFM]": "FM-LR [6]", "Event-frame CNN, raw [TII 2023]": "EF-CNN [4]", "Event-frame CNN, reference-subtracted": "EF-CNN + ref.",
        "Event-frame CNN + AdaBN on reference": "EF-CNN + AdaBN-R [10]", "Event-frame CNN + AdaBN on test batch": "EF-CNN + AdaBN-T [10]", "Event-frame CNN + TENT": "EF-CNN + TENT [11]",
        "Voxel-grid CNN, raw": "Vox-CNN", "Spiking CNN on voxel grids, raw [Vibration Vision]": "SNN [12, 13]", "Spiking CNN, reference-subtracted": "SNN + ref.",
        "Global-rate spectrogram CNN, raw [signal paradigm]": "Spec-CNN [15]", "Global-rate spectrogram CNN, reference-subtracted": "Spec-CNN + ref.",
        "Bi-fovea event Transformer, raw [EViT]": "BiFovea-T [5]", "Bi-fovea event Transformer, reference-subtracted": "BiFovea-T + ref.",
        "Event voxel set Transformer, raw [EVSTr]": "EVS-T [14]", "Event voxel set Transformer, reference-subtracted": "EVS-T + ref.",
        "Self-supervised + cross-supervision, strict [EAAI 2026]": "SSL-CS [9], strict", "Self-supervised + cross-supervision, relaxed [EAAI 2026]": "SSL-CS [9], relaxed",
        "Linear LR, whitened reference ratio, healthy-ref": "Linear", "EvSet-Net (ours)": "**REF-EvSet (ours)**", "Linear LR, class-balanced oracle (not deployable)": "Oracle"}

def linear(feat, norm, kind, log="linear_v2_w1.log"):
    for line in open(os.path.join(ROOT, "outputs", log), encoding="utf-8", errors="ignore"):
        m = re.match(rf"{feat}\s+{norm}\s+{kind}\s+acc=([\d.]+) \(min ([\d.]+)\)", line)
        if m:
            return float(m.group(1)), float(m.group(2))
    return np.nan, np.nan


def seeds(prefix, kind, base):
    """returns (per-seed mean acc list, per-task mean over seeds dict)"""
    per_seed, per_task = [], {}
    for d in glob.glob(os.path.join(base, f"{prefix}_{kind}_s*")):
        if not re.fullmatch(re.escape(prefix) + f"_{kind}_s\\d+", os.path.basename(d)):
            continue
        rs = [json.load(open(p)) for p in glob.glob(os.path.join(d, "*.json"))]
        if not rs:
            continue
        per_seed.append(np.mean([r["acc"] for r in rs]))
        for r in rs:
            per_task.setdefault(r["task"], []).append(r["acc"])
    return per_seed, per_task


def stats(prefix, base, kind):
    """(mean, std, n_seeds, min task) for a method on a protocol"""
    if prefix == "freqmap":
        rows = json.load(open(os.path.join(ROOT, "outputs", "linear_freqmap.json")))
        accs = [r["acc"] for r in rows if r["norm"] == "healthy-ref" and r["task"].startswith(kind + ":")]
        return float(np.mean(accs)), 0.0, 1, float(np.min(accs))
    if prefix == "linear_ref":
        m, mn = linear("snr_refratio", "healthy-ref", kind); return m, 0.0, 1, mn
    if prefix == "linear_oracle":
        m, mn = linear("snr_refratio", "oracle", kind); return m, 0.0, 1, mn
    ps_, pt = seeds(prefix, kind, base)
    if not ps_:
        return np.nan, np.nan, 0, np.nan
    return float(np.mean(ps_)), float(np.std(ps_)), len(ps_), float(min(np.mean(v) for v in pt.values()))


def main():
    table = {}
    for name, prefix, base, tgt in METHODS:
        table[name] = {k: stats(prefix, base, k) for k in KINDS}
        vals = [table[name][k][0] for k in KINDS]; table[name]["mean"] = float(np.mean(vals)) if np.all(np.isfinite(vals)) else np.nan
    # markdown table
    lines = ["| Method | Target-domain input | LOVO (3) | CS (2) | LODO (6) | Mean |", "|---|---|---|---|---|---|"]
    for name, prefix, base, tgt in METHODS:
        cells = []
        for k in KINDS:
            m, s, n, mn = table[name][k]
            cells.append("n/a" if not np.isfinite(m) else (f"{m:.3f} ± {s:.3f}" if n > 1 else f"{m:.3f}"))
        mm = table[name]["mean"]; lines.append(f"| {name} | {tgt} | " + " | ".join(cells) + f" | {'n/a' if not np.isfinite(mm) else f'{mm:.3f}'} |")
    md = "\n".join(lines); open(os.path.join(ROOT, "outputs", "sota_table.md"), "w", encoding="utf-8").write(md); print(md)
    json.dump({n: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()} for n, d in table.items()}, open(os.path.join(ROOT, "outputs", "sota_table.json"), "w"), indent=1)
    # figure: a selected subset (best variant of each paradigm) as grouped bars over LOVO / CS / LODO / mean
    show = [("FM-LR", "Per-pixel frequency map + LR [EBFM]"), ("EF-CNN (best)", None), ("Spec-CNN (best)", None), ("SNN (best)", None),
            ("EVS-T (best)", None), ("BiFovea-T (best)", None), ("SSL-CS, relaxed", "Self-supervised + cross-supervision, relaxed [EAAI 2026]"),
            ("Linear", "Linear LR, whitened reference ratio, healthy-ref"), ("REF-EvSet (ours)", "EvSet-Net (ours)"), ("Oracle", "Linear LR, class-balanced oracle (not deployable)")]
    fam = {"EF-CNN (best)": [n for n in table if n.startswith("Event-frame CNN")], "Spec-CNN (best)": [n for n in table if n.startswith("Global-rate")],
           "SNN (best)": [n for n in table if n.startswith("Spiking")], "EVS-T (best)": [n for n in table if n.startswith("Event voxel")],
           "BiFovea-T (best)": [n for n in table if n.startswith("Bi-fovea")]}
    rows = []
    for lab, name in show:
        if name is None:
            cands = [n for n in fam[lab] if np.isfinite(table[n]["mean"]) and "transductive" not in dict((m[0], m[3]) for m in METHODS)[n]]
            name = max(cands, key=lambda n: table[n]["mean"]) if cands else None
        if name is None or not np.isfinite(table[name]["mean"]):
            continue
        rows.append((lab, name))
    ps.apply(9); plt.rcParams["hatch.linewidth"] = 0.7; fig, ax = plt.subplots(figsize=(7.4, 3.2)); n = len(rows); w = 0.8 / n; x = np.arange(4)
    # five solid greys, then hollow hatched bars in the same greys (no two bars share a look); ours blue, oracle light blue
    styles = [dict(color=g, edgecolor="w") for g in ps.GREYS[:4]] + [dict(color=ps.GREYS[4], edgecolor="#9a9a9a")] + [dict(color="w", edgecolor=g, hatch="////") for g in ps.GREYS[:3]]
    k = 0
    for i, (lab, name) in enumerate(rows):
        m = [table[name][kk][0] for kk in KINDS] + [table[name]["mean"]]; sd = [table[name][kk][1] for kk in KINDS] + [0]
        if "ours" in lab: st = dict(color=ps.BLUE, edgecolor="w")
        elif lab == "Oracle": st = dict(color="#7fb0ff", edgecolor="w")
        else: st = styles[k % len(styles)]; k += 1
        bars = ax.bar(x + (i - n / 2 + 0.5) * w, m, w, yerr=sd, label=lab, capsize=1.2, error_kw=dict(lw=0.5), lw=0.5, **st)
        for b, mv, sv in zip(bars, m, sd): ax.text(b.get_x() + b.get_width() / 2, mv + sv + 0.01, f"{mv:.2f}", ha="center", va="bottom", fontsize=4.6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(["LOVO", "CS", "LODO", "mean of the three protocols"]); ax.set_ylabel("window accuracy"); ax.set_ylim(0.2, 1.04); ps.grid(ax); ps.note(ax, "higher is better ↑", "upper left")
    ax.legend(fontsize=6.3, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.01)); fig.tight_layout()
    out = os.path.join(ROOT, "figures", "fig_sota.png"); fig.savefig(out); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)


if __name__ == "__main__":
    main()
