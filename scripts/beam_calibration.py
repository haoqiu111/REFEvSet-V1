"""Beam: explicit event-sensor calibration and event -> LDV spectrum reconstruction metrics.

(1) Saturation law per camera set-up: unsigned global rate r vs LDV velocity envelope |v| pooled over the 5 trials
    of each (light, setup):  r = A |v| / (1 + |v| / v_sat)  (refractory-type compression), fitted in log domain.
    Reported: A, v_sat, R^2, and the linear-law R^2 for comparison.
(2) Spectrum reconstruction: LDV velocity spectrum (0-40 Hz) vs the spectrum of the best-K per-patch signed rates
    (K = 25 patches with the highest f0 SNR, sign-aligned by the sign of their correlation with the leading patch).
    Reported: normalised spectral RMSE, fundamental error (Hz), decay-rate (damping) error from log-envelope slopes.
Outputs outputs/beam_calibration.json.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import hilbert, butter, filtfilt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset.data.registry import beam_files  # noqa: E402
from evset.features.patch_rate import load_cache  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FS_EV = 1000.0


def spec(x, fs, fmax=40.0):
    x = x - x.mean()
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=8 * len(x)))
    f = np.fft.rfftfreq(8 * len(x), 1 / fs)
    m = (f >= 1.0) & (f <= fmax)
    return f[m], X[m] / X[m].max()


def main():
    groups = {}
    recon = []
    for r in beam_files():
        z = load_cache(os.path.join(ROOT, "cache", "beam_l1", r.name + ".npz"))
        c = np.asarray(z["counts"]).astype(np.float32)
        gu = c.sum(axis=(0, 1)); t_ev = np.arange(len(gu)) / FS_EV
        ldv = np.loadtxt(r.extra["ldv"], skiprows=5); t_l, v = ldv[:, 0], ldv[:, 1]
        fs_l = 1 / np.median(np.diff(t_l))
        on_l = t_l[np.argmax(np.abs(v) > 0.2 * np.abs(v).max())]
        base = np.median(gu[: int(0.5 * FS_EV)]) + 1
        on_e = t_ev[np.argmax(gu > max(5 * base, 0.2 * gu.max()))]
        seg_l = (t_l >= on_l) & (t_l < on_l + 6.0); seg_e = (t_ev >= on_e) & (t_ev < on_e + 6.0)
        env_l = np.abs(hilbert(v[seg_l] - v[seg_l].mean())); tl = t_l[seg_l] - on_l; te = t_ev[seg_e] - on_e
        b, a = butter(2, 2.0 / (FS_EV / 2)); gu_lp = filtfilt(b, a, gu[seg_e]) * FS_EV  # events / s
        env_i = np.interp(te, tl, env_l)
        m = (te > 0.3) & (te < 5.0) & (env_i > 1e-3) & (gu_lp > 100)
        key = (r.label, r.view)
        groups.setdefault(key, ([], []))
        groups[key][0].extend(env_i[m][::10].tolist()); groups[key][1].extend(gu_lp[m][::10].tolist())
        # ---- spectrum reconstruction from per-patch signed rates
        ps = (c[1] - c[0])[:, seg_e]
        f_l, S_l = spec(v[seg_l], fs_l)
        f0 = f_l[np.argmax(S_l)]
        # per-patch SNR at f0
        n = ps.shape[1]; w = np.hanning(n)
        k = np.exp(-2j * np.pi * f0 * np.arange(n) / FS_EV)
        Xf0 = np.abs((ps - ps.mean(1, keepdims=True)) * w @ k) ** 2
        noise = (c[0] + c[1])[:, seg_e].mean(1) * (w ** 2).sum() + 1e-6
        snr = Xf0 / noise
        top = np.argsort(snr)[-25:]
        lead = ps[top[-1]] - ps[top[-1]].mean()
        sig = np.zeros(n)
        for p in top:
            x = ps[p] - ps[p].mean(); s = np.sign(np.dot(x, lead)) or 1.0
            sig += s * x / (np.sqrt(snr[p]) + 1e-6) ** 0  # equal weights, sign aligned
        f_e, S_e = spec(sig, FS_EV)
        S_e_i = np.interp(f_l, f_e, S_e)
        rmse = float(np.sqrt(np.mean((S_e_i - S_l) ** 2)))
        f0_e = float(f_e[np.argmax(S_e)])
        # damping from the log-envelope slope of the fundamental (band-pass around f0)
        def decay(x, fs):
            bb, aa = butter(2, [max(0.5, f0 - 1.5) / (fs / 2), (f0 + 1.5) / (fs / 2)], btype="band")
            e = np.abs(hilbert(filtfilt(bb, aa, x - x.mean()))); t = np.arange(len(x)) / fs
            mm = (t > 0.5) & (t < 4.0) & (e > 1e-6)
            return float(np.polyfit(t[mm], np.log(e[mm]), 1)[0])
        d_l, d_e = decay(v[seg_l], fs_l), decay(sig, FS_EV)
        recon.append(dict(name=r.name, light=r.label, setup=r.view, f0_ldv=float(f0), f0_event=f0_e, rmse=rmse,
                          decay_ldv=d_l, decay_event=d_e, snr_top=float(np.log(snr[top]).mean())))
        print(f"{r.name:16s} f0 {f0:.3f} vs {f0_e:.3f}  spec-RMSE {rmse:.3f}  decay {d_l:.2f} vs {d_e:.2f} 1/s", flush=True)
    fits = {}
    for key, (vv, rr) in groups.items():
        vv, rr = np.array(vv), np.array(rr)
        def resid(th):
            A, vs = np.exp(th)
            return np.log(A * vv / (1 + vv / vs)) - np.log(rr)
        th0 = [np.log(rr.mean() / vv.mean()), np.log(vv.max())]
        sol = least_squares(resid, th0)
        A, vs = np.exp(sol.x)
        r2 = 1 - np.mean(sol.fun ** 2) / np.var(np.log(rr))
        lin = np.polyfit(np.log(vv), np.log(rr), 1); r2lin = 1 - np.mean((np.polyval(lin, np.log(vv)) - np.log(rr)) ** 2) / np.var(np.log(rr))
        fits["/".join(key)] = dict(A=float(A), v_sat=float(vs), r2_sat=float(r2), gamma_lin=float(lin[0]), r2_lin=float(r2lin),
                                  n=int(len(vv)), rate_max=float(rr.max()))
        print(f"{key}: A={A:.3g} ev/s per m/s, v_sat={vs * 1e3:.1f} mm/s, R2 sat {r2:.3f} | power law gamma {lin[0]:.2f} R2 {r2lin:.3f} | max rate {rr.max() / 1e6:.2f} Mev/s")
    json.dump(dict(saturation=fits, reconstruction=recon), open(os.path.join(ROOT, "outputs", "beam_calibration.json"), "w"), indent=1)
    rm = np.array([x["rmse"] for x in recon]); fe = np.abs(np.array([x["f0_ldv"] - x["f0_event"] for x in recon]))
    de = np.array([x["decay_event"] - x["decay_ldv"] for x in recon])
    print(f"spectral RMSE {rm.mean():.3f} +- {rm.std():.3f}; |f0 error| {fe.mean():.3f} Hz (max {fe.max():.3f}); decay error {de.mean():.2f} +- {de.std():.2f} 1/s")


if __name__ == "__main__":
    main()
