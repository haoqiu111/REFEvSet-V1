"""Pump: timestamp survey + L1 cache (CPU path, so it can run while the GPU trains).

L1 spec for Pump: 16x16 patches, 200 us bins, first PUMP_DUR s counted from the first second with
>= MIN_EV events (skips leading acquisition holes). Survey -> outputs/pump_survey.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset.data.registry import pump_files  # noqa: E402
from evset.data.dat_reader import timestamp_report  # noqa: E402
from evset.features.patch_rate import build_patch_counts, save_cache  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
LOG = os.path.join(ROOT, "extract_pump_log.txt")
PUMP_DUR = float(os.environ.get("PUMP_DUR", 20.0))
MIN_EV = 3e5


def wait_extraction():
    while True:
        if os.path.exists(LOG) and "DONE" in open(LOG).read():
            return
        time.sleep(30)


def main():
    wait_extraction()
    files = pump_files()
    print(len(files), "pump files", flush=True)
    surv_path = os.path.join(ROOT, "outputs", "pump_survey.json")
    survey = json.load(open(surv_path)) if os.path.exists(surv_path) else {}
    out_dir = os.path.join(ROOT, "cache", "pump_l1")
    os.makedirs(out_dir, exist_ok=True)
    for r in files:
        if r.name not in survey:
            rep = timestamp_report(r.path)
            secs = sorted(rep["sec_counts"])
            counts = [rep["sec_counts"][s] for s in secs]
            survey[r.name] = dict(n_events=int(rep["n_events"]), duration_s=float(rep["duration_s"]), n_backward=int(rep["n_backward"]),
                                  t_min=int(rep["t_min"]), secs=[int(x) for x in secs], counts=[int(x) for x in counts],
                                  height=int(rep["height"]), width=int(rep["width"]))
            json.dump(survey, open(surv_path, "w"))
        sv = survey[r.name]
        good = [s for s, c in zip(sv["secs"], sv["counts"]) if c >= MIN_EV]
        n_low = sum(1 for c in sv["counts"] if c < MIN_EV)
        print(f"{r.name:18s} n={sv['n_events'] / 1e6:6.1f}M dur={sv['duration_s']:6.1f}s back={sv['n_backward']:6d} "
              f"valid_secs={len(good)} low={n_low} {sv['height']}x{sv['width']}", flush=True)
        out = os.path.join(out_dir, r.name + ".npz")
        if os.path.exists(out) or not good:
            continue
        t_start = int(good[0] * 1e6)
        res = build_patch_counts(r.path, patch=16, bin_us=200, t_start_us=t_start, dur_s=PUMP_DUR, device="cpu",
                                 min_events_per_s=MIN_EV)
        save_cache(out, res)
        print(f"   L1 {res['counts'].shape} valid {int(res['valid_s'].sum())} s  {res['meta']['build_s']:.1f}s", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
