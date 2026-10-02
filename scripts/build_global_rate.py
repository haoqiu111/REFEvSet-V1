"""Global (whole-sensor) unsigned event-rate signal per Rotor recording at the level-1 bin (100 us = 10 kHz).

For the 'event rate as a vibration signal' paradigm (signal alignment / spectrogram CNN baselines):
cache/rotor_global_rate/<name>.npz  with  rate (T,) int32 = sum of counts over polarities and patches, plus the L1 meta.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset.data.registry import rotor_files  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "cache", "rotor_global_rate")


def main():
    os.makedirs(OUT, exist_ok=True)
    for r in rotor_files():
        out = os.path.join(OUT, r.name + ".npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        z = np.load(os.path.join(ROOT, "cache", "rotor_l1", r.name + ".npz"), allow_pickle=True, mmap_mode="r")
        c = z["counts"]; T = c.shape[-1]; rate = np.zeros(T, np.int64)
        for s in range(0, T, 20000):
            rate[s:s + 20000] = np.asarray(c[:, :, s:s + 20000]).astype(np.int64).sum(axis=(0, 1))
        np.savez(out, rate=rate.astype(np.int32), valid_s=z["valid_s"], meta=z["meta"])
        print(f"{r.name:22s} T={T} {time.time() - t0:.1f}s", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
