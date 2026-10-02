"""Generic level-2 token cache builder (Rotor or Pump).

usage: python scripts/build_l2.py --subset pump --win 0.5 --hop 0.25 --name pump_l2 [--fnom_scale 1.0]
Shaft frequency: harmonic-sum search in [0.8, 1.2] x (fnom_scale * nominal) on a subset of windows -> file median
-> per-window refinement (+-1 %). For Pump the nominal is the drive frequency (20/30/40 Hz); fnom_scale allows a
half-speed hypothesis. The chosen f_file and its harmonic-peak ratio are stored so the choice can be audited.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset.data.registry import rotor_files, pump_files  # noqa: E402
from evset.features.patch_rate import load_cache  # noqa: E402
from evset.features.order_tokens import window_tokens, window_starts, estimate_shaft_freq  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
ORDERS = torch.arange(0.125, 64.0001, 0.125)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="pump"); ap.add_argument("--win", type=float, default=0.5); ap.add_argument("--hop", type=float, default=0.25)
    ap.add_argument("--name", default=None); ap.add_argument("--fnom_scale", type=float, default=1.0); ap.add_argument("--l1", default=None); ap.add_argument("--device", default="cuda"); ap.add_argument("--fixed_hz", type=float, default=None, help="use a fixed Hz grid = ORDERS * fixed_hz instead of the tracked shaft frequency")
    args = ap.parse_args()
    name = args.name or f"{args.subset}_l2"
    l1_dir = os.path.join(ROOT, "cache", args.l1 or f"{args.subset}_l1")
    out_dir = os.path.join(ROOT, "cache", name); os.makedirs(out_dir, exist_ok=True)
    files = rotor_files() if args.subset == "rotor" else pump_files()
    for r in files:
        out = os.path.join(out_dir, r.name + ".npz")
        l1 = os.path.join(l1_dir, r.name + ".npz")
        if os.path.exists(out) or not os.path.exists(l1):
            continue
        t0 = time.time()
        z = load_cache(l1)
        c, valid, meta = z["counts"], z["valid_s"], z["meta"]
        bps = int(1e6 // meta["bin_us"])
        win, hop = int(args.win * bps), int(args.hop * bps)
        starts = window_starts(valid, bps, win, hop, meta["T"])
        if len(starts) == 0:
            print("no valid windows", r.name); continue
        f_nom = r.speed * args.fnom_scale
        est, ratios = [], []
        for s in starts[::2]:
            cw = torch.from_numpy(np.asarray(c[:, :, s:s + win])).to(args.device).float()
            gs = (cw[1] - cw[0]).sum(0); gu = (cw[1] + cw[0]).sum(0)
            f, ratio = estimate_shaft_freq(gs, gu, meta["bin_us"], f_nom)
            est.append(f); ratios.append(ratio)
        f_file = float(np.median(est))
        toks, attrs, globs, frs, rates = [], [], [], [], []
        for s in starts:
            cw = np.asarray(c[:, :, s:s + win])
            cwt = torch.from_numpy(cw).to(args.device).float()
            gs = (cwt[1] - cwt[0]).sum(0); gu = (cwt[1] + cwt[0]).sum(0)
            f_r, _ = estimate_shaft_freq(gs, gu, meta["bin_us"], f_file, rel_range=0.01, step_hz=0.002)
            if args.fixed_hz is not None:
                f_r = args.fixed_hz  # fixed frequency axis: bins at k * fixed_hz, identical for every speed
            o = window_tokens(cw, meta["bin_us"], f_nom, ORDERS, device=args.device, f_r=f_r)
            toks.append(o["tokens"][..., :2].numpy()); attrs.append(o["attrs"].numpy()); globs.append(o["glob"].numpy())
            frs.append(f_r); rates.append(o["rate"])
        np.savez(out, tokens=np.stack(toks), attrs=np.stack(attrs), glob=np.stack(globs),
                 f_r=np.array(frs, dtype=np.float32), rate=np.array(rates, dtype=np.float32),
                 t_start_s=(starts / bps).astype(np.float32), orders=ORDERS.numpy(),
                 meta=np.array(dict(name=r.name, label=r.label, view=r.view, speed=r.speed, grid=meta["grid"], extra=r.extra,
                                    win_s=args.win, hop_s=args.hop, f_file=f_file, f_nom=f_nom,
                                    harmonic_ratio=float(np.median(ratios)), subset=args.subset), dtype=object))
        print(f"{r.name:20s} N={len(starts):3d} f_nom={f_nom:6.2f} f_file={f_file:6.3f} ratio={np.median(ratios):5.1f} "
              f"f_r std={np.std(frs):.4f} {time.time() - t0:5.1f}s", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
