"""Reference-Only protocol on the Rotor level-2 cache.

Domain = (view, speed). Every file is split in time into a *reference segment* (first REF_S seconds of
valid windows) and an *evaluation segment* (after a guard). In the reference-only protocol the target
domain exposes ONLY the healthy reference windows (unlabelled healthy stream); all its evaluation windows
(all classes) are test data. Source domains contribute every window with labels.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

CLASSES = ["Healthy", "Inner", "Outer", "Ball"]
VIEWS = ["angle1", "angle2", "angle3"]
SPEEDS = [1000, 2000]
REF_S = 3.0      # reference segment length (seconds of window starts)
GUARD_S = 0.5    # guard between reference and evaluation windows (= one window length)


@dataclass
class FileMeta:
    name: str
    label: str
    view: str
    rpm: int
    n_win: int
    t_start: np.ndarray
    path: str
    setting: int | None = None

    @property
    def domain(self):
        return f"{self.view}@{self.rpm}" if self.setting is None else f"{self.view}_s{self.setting}@{self.rpm}"

    @property
    def y(self):
        return CLASSES.index(self.label)

    def ref_mask(self):
        return self.t_start < self.t_start[0] + REF_S

    def eval_mask(self):
        return self.t_start >= self.t_start[0] + REF_S + GUARD_S


def load_rotor_l2_meta(cache_dir: str) -> list[FileMeta]:
    out = []
    for p in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        z = np.load(p, allow_pickle=True)
        m = z["meta"].item()
        setting = (m.get("extra") or {}).get("setting") if isinstance(m.get("extra"), dict) else None
        out.append(FileMeta(m["name"], m["label"], m["view"], int(round(m["speed"] * 60)), len(z["t_start_s"]),
                            z["t_start_s"], p, setting))
    return out


PUMP_VIEWS = ["-30", "0", "30"]
PUMP_SPEEDS = [1200, 1800, 2400]   # 20/30/40 Hz drive x 60


def tasks(kind: str):
    """Yield (task_name, is_target(view, rpm)) pairs."""
    if kind == "pump_lovo":
        for v in PUMP_VIEWS:
            yield f"pump_lovo:{v}", (lambda view, rpm, v=v: view == v)
    elif kind == "pump_cs":
        for s in PUMP_SPEEDS:
            yield f"pump_cs:->{s}", (lambda view, rpm, s=s: rpm == s)
    elif kind == "pump_lodo":
        for v in PUMP_VIEWS:
            for s in PUMP_SPEEDS:
                yield f"pump_lodo:{v}@{s}", (lambda view, rpm, v=v, s=s: view == v and rpm == s)
    elif kind == "lovo":
        for v in VIEWS:
            yield f"lovo:{v}", (lambda view, rpm, v=v: view == v)
    elif kind == "cs":
        for s in SPEEDS:
            yield f"cs:->{s}", (lambda view, rpm, s=s: rpm == s)
    elif kind == "lodo":
        for v in VIEWS:
            for s in SPEEDS:
                yield f"lodo:{v}@{s}", (lambda view, rpm, v=v, s=s: view == v and rpm == s)
    elif kind in ("indomain", "pump_indomain"):
        # in-domain reference: no domain is held out; the split is a random 70/30 partition of the evaluation windows
        # of every recording (see indomain_split); is_target is never true
        yield f"{kind}:random70-30", (lambda view, rpm: False)
    else:
        raise ValueError(kind)


def split(files: list[FileMeta], is_target):
    """Return dict with source files, target files."""
    src = [f for f in files if not is_target(f.view, f.rpm)]
    tgt = [f for f in files if is_target(f.view, f.rpm)]
    return src, tgt


def indomain_split(file_idx, is_eval, seed: int = 0, frac_test: float = 0.3):
    """Random 70/30 split of the evaluation windows of every recording (in-domain protocol of the published methods).
    Returns (train_idx, test_idx)."""
    rng = np.random.default_rng(seed); tr, te = [], []
    for fi in np.unique(file_idx):
        idx = np.where((file_idx == fi) & is_eval)[0]; idx = rng.permutation(idx); k = int(round(len(idx) * frac_test))
        te.append(idx[:k]); tr.append(idx[k:])
    return np.concatenate(tr), np.concatenate(te)
