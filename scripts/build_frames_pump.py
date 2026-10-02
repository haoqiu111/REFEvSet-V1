"""Pump: dense event-frame / voxel-grid cache for the literature baselines (event-frame CNN, voxel CNN).

Uses the same windows as the L2 cache (t_start_s from cache/<l2>/<name>.npz).
Per file -> cache/pump_frames_w1/<name>.npz:
  frames (N, 2, 240, 320) uint16  polarity count images (2x2 binned)
  voxel  (N, 8, 120, 160) uint16  8 time bins of unsigned counts (4x4 binned)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset.data.registry import pump_files  # noqa: E402
from evset.data.dat_reader import parse_header, read_raw, decode  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
L2 = os.path.join(ROOT, "cache", os.environ.get("L2_NAME", "pump_l2_w1"))
OUT = os.path.join(ROOT, "cache", os.environ.get("FRAME_NAME", "pump_frames_w1"))
WIN_S = float(os.environ.get("L2_WIN", 1.0))
NT = 8


def main():
    os.makedirs(OUT, exist_ok=True)
    for r in pump_files():
        out = os.path.join(OUT, r.name + ".npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        z = np.load(os.path.join(L2, r.name + ".npz"), allow_pickle=True)
        starts = z["t_start_s"].astype(np.float64)
        l1meta = np.load(os.path.join(ROOT, "cache", "pump_l1", r.name + ".npz"), allow_pickle=True)["meta"].item()
        t_start_us = l1meta["t_start_us"]
        h = parse_header(r.path)
        N = len(starts)
        frames = np.zeros((N, 2, 240, 320), np.uint32)
        voxel = np.zeros((N, NT, 120, 160), np.uint32)
        win_us = int(WIN_S * 1e6)
        s_us = (starts * 1e6).astype(np.int64) + t_start_us
        for s in range(0, h.n_events, 20_000_000):
            t, x, y, p = decode(np.asarray(read_raw(r.path, s, min(20_000_000, h.n_events - s))))
            # window index for each event (windows overlap: hop < win) -> loop over windows overlapping this chunk
            tmin, tmax = t.min(), t.max()
            for wi in range(N):
                a, b = s_us[wi], s_us[wi] + win_us
                if b < tmin or a > tmax:
                    continue
                m = (t >= a) & (t < b)
                if not m.any():
                    continue
                xx, yy, pp, tt = x[m] // 2, y[m] // 2, p[m], t[m]
                np.add.at(frames[wi], (pp.astype(np.int64), yy.astype(np.int64), xx.astype(np.int64)), 1)
                tb = ((tt - a) * NT // win_us).astype(np.int64)
                np.add.at(voxel[wi], (tb, (yy // 2).astype(np.int64), (xx // 2).astype(np.int64)), 1)
        np.savez(out, frames=np.minimum(frames, 65535).astype(np.uint16), voxel=np.minimum(voxel, 65535).astype(np.uint16),
                 t_start_s=z["t_start_s"])
        print(f"{r.name:22s} N={N:3d} {time.time() - t0:5.1f}s", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
