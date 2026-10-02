"""Beam subset: LDV-calibrated event-generation model (Section 3.1 and Table 6 of the paper).

Per trial (30 = On/Off x set1-3 x 5 trials):
  1. global unsigned / signed rate at 1 ms (from L1 cache, 16x16 patches);
  2. LDV velocity (256 Hz, 8 s), impact onset; event onset by rate burst -> time alignment;
  3. spectra: LDV fundamental f0 vs unsigned-rate main peak (expect 2 f0: rectification) and
     signed per-patch rate (expect f0 in patches with a consistent gradient sign);
  4. amplitude calibration: LDV velocity envelope (Hilbert) vs unsigned event rate envelope, over the
     decay -> rate = A * |v|^gamma fit (log-log) per trial, then pooled: gamma ~ 1 means linear regime.
  5. flicker: 100 Hz line in the signed global rate (lights On vs Off).
Outputs outputs/beam_physics.json and a summary print.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.signal import hilbert, butter, filtfilt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset.data.registry import beam_files  # noqa: E402
from evset.features.patch_rate import load_cache  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FS_EV = 1000.0  # 1 ms bins


def spectrum(x, fs, fmax=60.0):
    x = x - x.mean()
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    m = f <= fmax
    return f[m], X[m]


def peak(f, X, fmin=2.0):
    m = f >= fmin
    i = np.argmax(X[m])
    return f[m][i], X[m][i] / (np.median(X[m]) + 1e-9)


def main():
    out = []
    for r in beam_files():
        z = load_cache(os.path.join(ROOT, "cache", "beam_l1", r.name + ".npz"))
        c = np.asarray(z["counts"]).astype(np.float32)  # (2, P, T)
        gu = c.sum(axis=(0, 1)); gs = c[1].sum(0) - c[0].sum(0)
        t_ev = np.arange(len(gu)) / FS_EV
        ldv = np.loadtxt(r.extra["ldv"], skiprows=5)
        t_l, v = ldv[:, 0], ldv[:, 1]
        fs_l = 1 / np.median(np.diff(t_l))
        # onset: LDV -> first time |v| exceeds 20% of max ; events -> rate exceeds 5x pre-burst median
        on_l = t_l[np.argmax(np.abs(v) > 0.2 * np.abs(v).max())]
        base = np.median(gu[: int(0.5 * FS_EV)]) + 1
        on_e = t_ev[np.argmax(gu > max(5 * base, 0.2 * gu.max()))]
        # analysis segment: 6 s after onset in both signals
        seg_l = (t_l >= on_l) & (t_l < on_l + 6.0)
        seg_e = (t_ev >= on_e) & (t_ev < on_e + 6.0)
        f_l, X_l = spectrum(v[seg_l], fs_l); f0, snr0 = peak(f_l, X_l)
        f_u, X_u = spectrum(gu[seg_e], FS_EV); fu, snru = peak(f_u, X_u)
        f_s, X_s = spectrum(gs[seg_e], FS_EV); fsg, snrs = peak(f_s, X_s)
        # flicker: 100 Hz line in signed rate (relative to neighbours)
        f_hi, X_hi = spectrum(gs[seg_e], FS_EV, fmax=200)
        i100 = np.argmin(np.abs(f_hi - 100)); flick = X_hi[i100] / (np.median(X_hi[(f_hi > 80) & (f_hi < 120)]) + 1e-9)
        # per-patch signed rate: fraction of patches whose main peak is at f0 (+-0.3 Hz) vs 2 f0
        ps = (c[1] - c[0])[:, seg_e]
        act = ps.std(1) > 0.05
        n_f0 = n_2f0 = 0
        for p in np.where(act)[0]:
            fp, Xp = spectrum(ps[p], FS_EV); fpk, _ = peak(fp, Xp)
            n_f0 += abs(fpk - f0) < 0.3; n_2f0 += abs(fpk - 2 * f0) < 0.3
        # amplitude calibration on the decay: envelope of |v| (resampled to 1 ms) vs unsigned rate (low-passed)
        env_l = np.abs(hilbert(v[seg_l] - v[seg_l].mean()))
        tl = t_l[seg_l] - on_l
        te = t_ev[seg_e] - on_e
        b, a = butter(2, 2.0 / (FS_EV / 2)); gu_lp = filtfilt(b, a, gu[seg_e])
        env_i = np.interp(te, tl, env_l)
        m = (te > 0.3) & (te < 5.0) & (env_i > 1e-3) & (gu_lp > 1)
        A = np.polyfit(np.log(env_i[m]), np.log(gu_lp[m]), 1)
        corr = np.corrcoef(np.log(env_i[m]), np.log(gu_lp[m]))[0, 1]
        rec = dict(name=r.name, light=r.label, setup=r.view, trial=r.extra["trial"], f0_ldv=float(f0), f_unsigned=float(fu),
                   f_signed_global=float(fsg), ratio_unsigned=float(fu / f0), flicker100=float(flick),
                   n_active=int(act.sum()), n_patch_f0=int(n_f0), n_patch_2f0=int(n_2f0), gamma=float(A[0]),
                   logA=float(A[1]), corr_loglog=float(corr), v_rms=float(v[seg_l].std()), rate_mean=float(gu[seg_e].mean() * FS_EV),
                   onset_ldv=float(on_l), onset_ev=float(on_e))
        out.append(rec)
        print(f"{r.name:16s} f0={f0:5.2f} unsigned peak={fu:5.2f} ({fu / f0:.2f} f0) signed-global={fsg:5.2f} flick100={flick:5.1f} "
              f"patches f0/2f0={n_f0}/{n_2f0} of {act.sum()}  gamma={A[0]:.2f} corr={corr:.2f}", flush=True)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", "beam_physics.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    g = np.array([o["gamma"] for o in out]); ru = np.array([o["ratio_unsigned"] for o in out])
    fl_on = [o["flicker100"] for o in out if o["light"] == "On"]; fl_off = [o["flicker100"] for o in out if o["light"] == "Off"]
    print(f"gamma mean {g.mean():.2f} +- {g.std():.2f}; unsigned/f0 ratio median {np.median(ru):.2f}; "
          f"flicker100 On {np.median(fl_on):.1f} Off {np.median(fl_off):.1f}")


if __name__ == "__main__":
    main()
