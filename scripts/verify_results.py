"""Recompute the reported numbers from the raw result files in outputs/ and compare them with the values of the paper.

The expected values are the hard-coded numbers below and, for the same-protocol comparison on Rotor (Table 3), every
cell of the table in results/expected_tables.md. No training and no GPU are needed.
Each check: (name, recomputed value, expected value, tolerance). Exit code 1 if any check fails.
usage: python scripts/verify_results.py
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
E = os.path.join(ROOT, "outputs", "evset")
B = os.path.join(ROOT, "outputs", "baselines")
checks = []


def chk(name, got, exp, tol=0.0015):
    ok = abs(got - exp) <= tol
    checks.append((name, got, exp, ok))


def runs(prefix, kind, base=E):
    per = defaultdict(list)
    for d in glob.glob(os.path.join(base, f"{prefix}_{kind}_s*")):
        if not re.fullmatch(re.escape(prefix) + f"_{kind}_s\\d+", os.path.basename(d)):
            continue
        rs = [json.load(open(p)) for p in glob.glob(os.path.join(d, "*.json"))]
        per[os.path.basename(d)] = rs
    return per


def mean_std(prefix, kind, base=E):
    per = runs(prefix, kind, base)
    accs = [np.mean([r["acc"] for r in rs]) for rs in per.values()]
    return float(np.mean(accs)), float(np.std(accs)), len(accs)


def linear(log, feat, norm, kind):
    for line in open(os.path.join(ROOT, "outputs", log), encoding="utf-8", errors="ignore"):
        m = re.match(rf"{feat}\s+{norm}\s+{kind}\s+acc=([\d.]+)", line)
        if m:
            return float(m.group(1))
    return float("nan")


# ---- Rotor: REF-EvSet, linear baseline / oracle and dense CNN baselines (Tables 3 and 7)
for kind, e_ref, e_or in [("lovo", 0.702, 0.904), ("cs", 0.745, 0.833), ("lodo", 0.688, 0.879)]:
    chk(f"linear healthy-ref 1s {kind}", linear("linear_v2_w1.log", "snr_refratio", "healthy-ref", kind), e_ref)
    chk(f"linear oracle 1s {kind}", linear("linear_v2_w1.log", "snr_refratio", "oracle", kind), e_or)
for kind, em, es in [("lovo", 0.786, 0.014), ("cs", 0.824, 0.029), ("lodo", 0.797, 0.025)]:
    m, s, n = mean_std("main_w1mixnorcn", kind); chk(f"EvSet main {kind} mean", m, em); chk(f"EvSet main {kind} std", s, es); chk(f"EvSet main {kind} n", n, 3, 0)
for kind, em in [("lovo", 0.787), ("cs", 0.827), ("lodo", 0.813)]:
    chk(f"EvSet sub-patch {kind}", mean_std("sub_w1mixnorcn", kind)[0], em)
for kind, em in [("lovo", 0.720), ("cs", 0.716), ("lodo", 0.776)]:
    chk(f"EvSet FiLM {kind}", mean_std("main_w1mix", kind)[0], em)
for kind, em in [("lovo", 0.348), ("cs", 0.598), ("lodo", 0.838)]:
    chk(f"frames raw CNN {kind}", mean_std("frames_raw", kind, B)[0], em)
for kind, em in [("lovo", 0.611), ("cs", 0.589), ("lodo", 0.688)]:
    chk(f"frames ref CNN {kind}", mean_std("frames_ref", kind, B)[0], em)
chk("voxel raw lodo", mean_std("voxel_raw", "lodo", B)[0], 0.700)
for kind, em in [("lovo", 0.415), ("cs", 0.556), ("lodo", 0.592)]:
    rows = json.load(open(os.path.join(ROOT, "outputs", "linear_freqmap.json")))
    chk(f"freqmap LR {kind}", float(np.mean([r["acc"] for r in rows if r["norm"] == "healthy-ref" and r["task"].startswith(kind + ":")])), em)
# ---- Rotor ablations (Table 7)
chk("0.5-s tokens 3 seeds", mean_std("v1mix", "lodo")[0], 0.776)
chk("no mixup 0.5 s (v1)", mean_std("v1", "lodo")[0], 0.722)
chk("no attributes", mean_std("dg_noattr", "lodo")[0], 0.774)
chk("no ratio channels", mean_std("abl_norefch", "lodo")[0], 0.791)
chk("no reference", mean_std("abl_noref", "lodo")[0], 0.458)
chk("no reference lovo", mean_std("abl_noref", "lovo")[0], 0.372); chk("no reference cs", mean_std("abl_noref", "cs")[0], 0.544)
chk("contrast aug", mean_std("dg_cs2", "lodo")[0], 0.801); chk("align", mean_std("dg_align", "lodo")[0], 0.756); chk("dann", mean_std("dg_dann", "lodo")[0], 0.763)
chk("grid lodo", mean_std("grid_w1mixnorcn", "lodo")[0], 0.585); chk("grid lovo", mean_std("grid_w1mixnorcn", "lovo")[0], 0.643); chk("grid cs", mean_std("grid_w1mixnorcn", "cs")[0], 0.602)
chk("fixhz lodo", mean_std("fixhz_w1mixnorcn", "lodo")[0], 0.771); chk("fixhz lovo", mean_std("fixhz_w1mixnorcn", "lovo")[0], 0.770); chk("fixhz cs", mean_std("fixhz_w1mixnorcn", "cs")[0], 0.785)
# ---- Pump (Table 4)
chk("pump lodo", mean_std("pumpw1_mixnorcn", "pump_lodo")[0], 0.994); chk("pump cs", mean_std("pumpw1_mixnorcn", "pump_cs")[0], 0.988); chk("pump lovo", mean_std("pumpw1_mixnorcn", "pump_lovo")[0], 0.510)
chk("pump linear snr_all ref 1s cs", linear("linear_pump_w1.log", "snr_all", "healthy-ref", "pump_cs"), 0.843)
chk("pump linear snr_all ref 1s lodo", linear("linear_pump_w1.log", "snr_all", "healthy-ref", "pump_lodo"), 0.856)
chk("pump linear snr_all oracle 1s lodo", linear("linear_pump_w1.log", "snr_all", "oracle", "pump_lodo"), 0.966)
# ---- detection / cross-device transfer (Table 5)
for s, ea, eb in [(0, 0.514, 0.398), (1, 0.547, 0.425)]:
    cd = json.load(open(os.path.join(ROOT, "outputs", f"cross_device_s{s}.json")))
    chk(f"cross rotor->pump acc s{s}", cd["rotor->pump"]["acc"], ea); chk(f"cross pump->rotor acc s{s}", cd["pump->rotor"]["acc"], eb)
    chk(f"cross det auroc s{s}", min(cd["rotor->pump"]["det_auroc"], cd["pump->rotor"]["det_auroc"]), 1.0)
# zero-shot detection AUROC (rotor, 0.5-s features, one-sided mean score)
# features: the 0.5-s linear-feature cache written by scripts/linear_v2.py if it exists (full reproduction), otherwise the
# copy of its two arrays (meta, snr_refratio) shipped in outputs/rotor_zero_shot_feats.npz
ZS = [os.path.join(ROOT, "cache", "rotor_l2_linear_feats_v2.npz"), os.path.join(ROOT, "cache", "rotor_linear_feats_v2.npz"),
      os.path.join(ROOT, "outputs", "rotor_zero_shot_feats.npz")]
z = np.load([p for p in ZS if os.path.exists(p)][0], allow_pickle=True)
meta = z["meta"]; label = meta[:, 1]; is_ref = meta[:, 4].astype(int).astype(bool); is_eval = meta[:, 5].astype(int).astype(bool)
dom = np.array([f"{v}@{r}" for v, r in zip(meta[:, 2], meta[:, 3])]); X = z["snr_refratio"].astype(np.float32)
aucs = []
for d in np.unique(dom):
    md = dom == d; ref = md & is_ref & (label == "Healthy"); te = md & is_eval
    mu, sd = X[ref].mean(0), X[ref].std(0) + 1e-3
    aucs.append(roc_auc_score((label[te] != "Healthy").astype(int), ((X[te][:, :512] - mu[:512]) / sd[:512]).mean(1)))
chk("zero-shot detection AUROC rotor", float(np.mean(aucs)), 0.928)
# ---- sensor-model calibration on Beam (Table 6)
bp = json.load(open(os.path.join(ROOT, "outputs", "beam_physics.json")))
chk("rectification ratio median", float(np.median([o["ratio_unsigned"] for o in bp])), 2.02, 0.01)
chk("flicker on median", float(np.median([o["flicker100"] for o in bp if o["light"] == "On"])), 78.1, 0.5)
bc = json.load(open(os.path.join(ROOT, "outputs", "beam_calibration.json")))
chk("spectral RMSE", float(np.mean([r["rmse"] for r in bc["reconstruction"]])), 0.052, 0.001)
chk("gamma set1", bc["saturation"]["Off/set1"]["gamma_lin"], 1.15, 0.01); chk("gamma set3", bc["saturation"]["Off/set3"]["gamma_lin"], 0.36, 0.01)

# ---- Table 3 of the paper (same-protocol comparison on Rotor): every cell against a recomputation from outputs/baselines
import io as _io
md = _io.open(os.path.join(ROOT, "results", "expected_tables.md"), encoding="utf-8").read()
if "## Table 3." in md:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import make_fig_sota as MS
    CITE = MS.CITE
    inv = {v.replace("**", ""): k for k, v in CITE.items()}; by_name = {m[0]: m for m in MS.METHODS}
    block = md.split("## Table 3.")[1].split("## Table 4.")[0]
    for ln in block.split(chr(10)):
        if not ln.startswith("|") or ln.startswith("|---") or ln.startswith("| Method"):
            continue
        r = [c.strip() for c in ln.strip().strip("|").split("|")]; label = r[0].replace("**", ""); key = inv.get(label, label)
        if key not in by_name:
            continue
        _, prefix, base, _ = by_name[key]
        for j, kind in enumerate(["lovo", "cs", "lodo"]):
            txt = r[3 + j].replace("**", "").split("±")[0].strip()
            if txt not in ("n/a", ""):
                chk(f"Table 3 {label[:40]} {kind}", MS.stats(prefix, base, kind)[0], float(txt))

n_ok = sum(c[3] for c in checks)
for name, got, exp, ok in checks:
    if not ok:
        print(f"FAIL {name}: got {got:.4f} expected {exp:.4f}")
print(f"{n_ok}/{len(checks)} checks pass")
sys.exit(0 if n_ok == len(checks) else 1)
