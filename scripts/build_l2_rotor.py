"""Build level-2 token cache for Rotor: 0.5-s windows, hop 0.25 s, order grid 0.125..64.

Per file -> cache/rotor_l2/<name>.npz with
  tokens (N, P, O, 4) float16, attrs (N, P, 2), glob (N, 2, O), f_r (N,), t_start_s (N,), meta
Shaft frequency: per-file median of the harmonic-sum estimate, then per-window refinement within +-1 %.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset.data.registry import rotor_files  # noqa: E402
from evset.features.patch_rate import load_cache  # noqa: E402
from evset.features.order_tokens import window_tokens, window_starts, estimate_shaft_freq  # noqa: E402

WIN_S = float(os.environ.get("L2_WIN", 0.5))
HOP_S = float(os.environ.get("L2_HOP", 0.25))
ORDERS = torch.arange(0.125, 64.0001, 0.125)
OUT = os.path.join(os.path.dirname(__file__), "..", "cache", os.environ.get("L2_NAME", "rotor_l2"))


def main():
    os.makedirs(OUT, exist_ok=True)
    for r in rotor_files():
        out = os.path.join(OUT, r.name + ".npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        z = load_cache(os.path.join(os.path.dirname(__file__), "..", "cache", "rotor_l1", r.name + ".npz"))
        c, valid, meta = z["counts"], z["valid_s"], z["meta"]
        bps = int(1e6 // meta["bin_us"])
        win, hop = int(WIN_S * bps), int(HOP_S * bps)
        starts = window_starts(valid, bps, win, hop, meta["T"])
        # pass 1: robust per-file shaft frequency
        est = []
        for s in starts[::2]:
            cw = torch.from_numpy(np.asarray(c[:, :, s:s + win])).cuda().float()
            gs = (cw[1] - cw[0]).sum(0)
            gu = (cw[1] + cw[0]).sum(0)
            est.append(estimate_shaft_freq(gs, gu, meta["bin_us"], r.speed)[0])
        f_file = float(np.median(est))
        toks, attrs, globs, frs, rates = [], [], [], [], []
        for s in starts:
            cw = np.asarray(c[:, :, s:s + win])
            cwt = torch.from_numpy(cw).cuda().float()
            gs = (cwt[1] - cwt[0]).sum(0)
            gu = (cwt[1] + cwt[0]).sum(0)
            f_r, _ = estimate_shaft_freq(gs, gu, meta["bin_us"], f_file, rel_range=0.01, step_hz=0.002)
            o = window_tokens(cw, meta["bin_us"], r.speed, ORDERS, f_r=f_r)
            toks.append(o["tokens"].numpy())
            attrs.append(o["attrs"].numpy())
            globs.append(o["glob"].numpy())
            frs.append(f_r)
            rates.append(o["rate"])
        np.savez(out, tokens=np.stack(toks), attrs=np.stack(attrs), glob=np.stack(globs),
                 f_r=np.array(frs, dtype=np.float32), rate=np.array(rates, dtype=np.float32),
                 t_start_s=(starts / bps).astype(np.float32), orders=ORDERS.numpy(),
                 meta=np.array(dict(name=r.name, label=r.label, view=r.view, speed=r.speed, grid=meta["grid"],
                                    win_s=WIN_S, hop_s=HOP_S, f_file=f_file), dtype=object))
        print(f"{r.name:22s} N={len(starts):3d} f_file={f_file:6.3f} f_r std={np.std(frs):.4f} {time.time() - t0:5.1f}s", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
