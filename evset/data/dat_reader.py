"""Fast reader for Prophesee .dat (Event2D) files used by XJTU-DV.

Format: ASCII header lines starting with "% ", then two bytes (event type, event
size) and a stream of records (t: uint32 microseconds, w: int32 packed x/y/p).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

REC_DTYPE = np.dtype([("t", "<u4"), ("w", "<i4")])


@dataclass
class DatHeader:
    body_offset: int
    ev_type: int
    ev_size: int
    height: int | None
    width: int | None
    n_events: int


def parse_header(path: str) -> DatHeader:
    height = width = None
    with open(path, "rb") as f:
        n_comment = 0
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b"% "):
                f.seek(pos)
                break
            n_comment += 1
            words = line.split()
            if len(words) > 2:
                if words[1] == b"Height":
                    height = int(words[2])
                elif words[1] == b"Width":
                    width = int(words[2])
        if n_comment > 0:
            ev_type = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
            ev_size = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
        else:
            ev_type, ev_size = 0, 8
        body = f.tell()
    total = os.path.getsize(path)
    return DatHeader(body, ev_type, ev_size, height, width, (total - body) // ev_size)


def read_raw(path: str, start: int = 0, count: int = -1) -> np.ndarray:
    """Return the raw (t, w) record array (memory-mapped, no copy)."""
    h = parse_header(path)
    assert h.ev_size == 8, f"unexpected event size {h.ev_size}"
    if count < 0:
        count = h.n_events - start
    count = max(0, min(count, h.n_events - start))
    return np.memmap(path, dtype=REC_DTYPE, mode="r", offset=h.body_offset + start * 8, shape=(count,))


def decode(rec: np.ndarray):
    """Decode packed records -> (t[us] int64, x int16, y int16, p int8 in {0,1})."""
    w = rec["w"]
    x = np.bitwise_and(w, 16383).astype(np.int16)
    y = np.right_shift(np.bitwise_and(w, 268419072), 14).astype(np.int16)
    p = np.right_shift(np.bitwise_and(w, 268435456), 28).astype(np.int8)
    t = rec["t"].astype(np.int64)
    return t, x, y, p


def iter_chunks(path: str, chunk: int = 20_000_000):
    """Yield decoded chunks (t, x, y, p) sequentially."""
    h = parse_header(path)
    for s in range(0, h.n_events, chunk):
        rec = read_raw(path, s, min(chunk, h.n_events - s))
        yield decode(np.asarray(rec))


def timestamp_report(path: str, chunk: int = 20_000_000) -> dict:
    """Quick diagnostics: duration, monotonicity violations, per-second counts."""
    h = parse_header(path)
    n_back = 0
    t_prev = None
    t_min, t_max = None, None
    sec_counts = {}
    for s in range(0, h.n_events, chunk):
        t = np.asarray(read_raw(path, s, min(chunk, h.n_events - s))["t"]).astype(np.int64)
        if t_prev is not None:
            n_back += int(t[0] < t_prev)
        n_back += int(np.count_nonzero(np.diff(t) < 0))
        t_prev = t[-1]
        t_min = t[0] if t_min is None else min(t_min, int(t.min()))
        t_max = int(t.max()) if t_max is None else max(t_max, int(t.max()))
        secs, cnt = np.unique(t // 1_000_000, return_counts=True)
        for a, b in zip(secs, cnt):
            sec_counts[int(a)] = sec_counts.get(int(a), 0) + int(b)
    return dict(n_events=h.n_events, height=h.height, width=h.width, t_min=t_min, t_max=t_max,
                duration_s=(t_max - t_min) / 1e6, n_backward=n_back, sec_counts=sec_counts)
