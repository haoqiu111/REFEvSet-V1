"""Parameter counts and per-window inference cost of the REF-EvSet set encoder (set / grid / FiLM / sub-patch variants) and the dense
CNN baselines, plus the front-end cost (level-1 binning + level-2 order DFT for one 1-s window).
usage: python scripts/cost_table.py [--device cuda|cpu]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evset.models.evset_net import EvSetNet  # noqa: E402
from evset.features.order_tokens import window_tokens  # noqa: E402
from train_baseline_cnn import CNN  # noqa: E402


def n_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def bench(fn, n=20, warm=3, sync=False):
    for _ in range(warm):
        fn()
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); args = ap.parse_args()
    dev = torch.device(args.device); sync = args.device == "cuda"
    P, O = 1200, 256
    x = torch.randn(1, P, O, 4, device=dev); a = torch.randn(1, P, 6, device=dev); r = torch.randn(1, P, O, 2, device=dev)
    sub = torch.randn(1, P, 4, 8, 8, device=dev)
    rows = []
    for name, kw, extra in [("EvSet-Net (set, main)", dict(use_rcn=False), {}), ("EvSet-Net + learned FiLM", dict(use_rcn=True), {}),
                            ("EvSet-Net + sub-patch", dict(use_rcn=False, use_sub=True), dict(sub=sub)), ("grid aggregator (ablation)", dict(use_rcn=False, agg="grid"), {})]:
        m = EvSetNet(4, in_ch=4, n_attr=6, **kw).to(dev).eval()
        with torch.no_grad():
            ms = bench(lambda: m(x, a, r, **extra), sync=sync)
        rows.append((name, n_params(m), ms))
    for name, shape in [("event-frame CNN (2x240x320)", (1, 2, 240, 320)), ("voxel CNN (8x120x160)", (1, 8, 120, 160))]:
        m = CNN(shape[1]).to(dev).eval(); xi = torch.randn(*shape, device=dev)
        with torch.no_grad():
            ms = bench(lambda: m(xi), sync=sync)
        rows.append((name, n_params(m), ms))
    # front-end: level-2 tokens for one 1-s window from level-1 counts (2, 1200, 10000)
    c = np.random.poisson(0.2, size=(2, P, 10000)).astype(np.uint8)
    orders = torch.arange(0.125, 64.0001, 0.125)
    ms = bench(lambda: window_tokens(c, 100, 16.67, orders, device=args.device, f_r=16.7), n=5, warm=1, sync=sync)
    rows.append(("front-end: order-DFT tokens, 1-s window", 0.0, ms))
    print(f"device = {args.device}")
    print(f"{'component':42s} {'params (M)':>10s} {'ms / window':>12s}")
    for n, p, t in rows:
        print(f"{n:42s} {p:10.2f} {t:12.2f}")
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "outputs"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "outputs", f"cost_table_{args.device}.md"), "w") as fh:
        fh.write(f"| component | params (M) | ms / window ({args.device}) |\n|---|---|---|\n")
        for n, p, t in rows:
            fh.write(f"| {n} | {p:.2f} | {t:.2f} |\n")


if __name__ == "__main__":
    main()
