"""Falsifiable checks of the event-sensor model with the 1-D simulator, driven by REAL LDV velocity (Beam).

1. Rectification: unsigned rate peak at 2 f0, signed per-pixel rate at f0 (as observed on 30/30 real trials).
2. Amplitude law: rate vs |v| envelope -> power-law exponent gamma for three magnifications
   (far / mid / close) -> the observed gamma 1.1 -> 0.85 -> 0.3 trend should appear when the per-frame displacement
   exceeds the texture correlation length (refractory / threshold saturation).
3. Flicker: 100 Hz line in the signed global rate appears only with lighting modulation.
4. Whitening null: with background Poisson noise, SNR at non-harmonic frequencies ~ Exp(1).
Outputs outputs/sim_validation.json
"""
import json
import os
import sys

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evset.data.registry import beam_files  # noqa: E402
from evset.physics.event_sim import make_texture, simulate  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FS = 10000.0  # simulation rate


def spec(x, fs, fmax=60.0, fmin=1.0):
    x = x - x.mean(); X = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=4 * len(x)))
    f = np.fft.rfftfreq(4 * len(x), 1 / fs); m = (f >= fmin) & (f <= fmax)
    return f[m], X[m]


def main():
    r = [b for b in beam_files() if b.name == "Off_set2_trail1"][0]
    ldv = np.loadtxt(r.extra["ldv"], skiprows=5); t_l, v_l = ldv[:, 0], ldv[:, 1]
    on_l = t_l[np.argmax(np.abs(v_l) > 0.2 * np.abs(v_l).max())]
    seg = (t_l >= on_l) & (t_l < on_l + 6.0)
    t = np.arange(0, 6.0, 1 / FS)
    v = np.interp(t, t_l[seg] - on_l, v_l[seg])  # linear upsampling of the 256-Hz LDV velocity
    f_l, S_l = spec(v, FS); f0 = f_l[np.argmax(S_l)]
    tex = make_texture(256, corr_pix=4.0, contrast=0.5, seed=0)
    out = dict(f0_ldv=float(f0), cases=[])
    # magnification in pixels per metre: displacement amplitude ~ v_rms / (2 pi f0) ~ 0.3 / 42 ~ 7 mm
    for name, mag, flick, noise in [("far", 50.0, 0.0, 0.0), ("mid", 300.0, 0.0, 0.0), ("close", 3000.0, 0.0, 0.0),
                                    ("far+flicker", 50.0, 0.3, 0.0), ("far+noise", 50.0, 0.0, 200.0)]:
        tg, on, off = simulate(v, FS, mag, tex, theta=0.2, tau_r=1e-4, lambda_noise=noise, flicker=flick, dt_out=1e-3, seed=1)
        gu = (on + off).sum(0).astype(float); gs = (on - off).sum(0).astype(float)
        fu, Su = spec(gu, 1000.0); fs_, Ss = spec(gs, 1000.0)
        pk_u = fu[np.argmax(Su)]; pk_s = fs_[np.argmax(Ss)]
        # per-pixel signed rate: fraction of active pixels peaking at f0 vs 2f0
        ps = (on - off).astype(float); act = ps.std(1) > 0.02
        n_f0 = n_2f0 = 0
        for p in np.where(act)[0]:
            fp, Sp = spec(ps[p], 1000.0); pk = fp[np.argmax(Sp)]
            n_f0 += abs(pk - f0) < 0.4; n_2f0 += abs(pk - 2 * f0) < 0.4
        # amplitude law on the decay
        env = np.abs(hilbert(v - v.mean())); env_ms = np.interp(tg, t, env)
        b, a = butter(2, 2.0 / 500.0); gu_lp = filtfilt(b, a, gu) * 1000.0
        m = (tg > 0.3) & (tg < 5.0) & (env_ms > 1e-3) & (gu_lp > 10)
        gamma = float(np.polyfit(np.log(env_ms[m]), np.log(gu_lp[m]), 1)[0]) if m.sum() > 50 else float("nan")
        # flicker line
        fh, Sh = spec(gs, 1000.0, fmax=200, fmin=1); i100 = np.argmin(np.abs(fh - 100)); flick_ratio = float(Sh[i100] / (np.median(Sh[(fh > 80) & (fh < 120)]) + 1e-9))
        # whitening null on noise case: per-pixel SNR at non-harmonic frequencies
        ks = float("nan")
        if noise > 0:
            L = 5000; w = np.hanning(L)
            seg_u = (on + off)[:, 500:500 + L].astype(float); seg_s = (on - off)[:, 500:500 + L].astype(float)
            freqs = np.arange(20, 200, 0.7)  # avoid multiples of f0 loosely
            freqs = freqs[np.min(np.abs(freqs[:, None] / f0 - np.round(freqs[:, None] / f0)), 1) > 0.15]
            M = np.exp(-2j * np.pi * np.arange(L)[:, None] / 1000.0 * freqs[None, :]) * w[:, None]
            X = (seg_s - seg_s.mean(1, keepdims=True)) @ M
            snr = (np.abs(X) ** 2 / (seg_u.mean(1, keepdims=True) * (w ** 2).sum() + 1e-9)).reshape(-1)
            ks = float(stats.kstest(snr, "expon").statistic)
        rec = dict(case=name, magnification=mag, rate_mean=float(gu.mean() * 1000), peak_unsigned=float(pk_u), peak_signed_global=float(pk_s),
                   ratio_unsigned=float(pk_u / f0), n_active=int(act.sum()), n_pix_f0=int(n_f0), n_pix_2f0=int(n_2f0), gamma=gamma,
                   flicker100=flick_ratio, ks_exp1=ks, mean_snr=float(np.mean(snr)) if noise > 0 else float("nan"))
        out["cases"].append(rec)
        print(f"{name:12s} mag={mag:6.0f} rate={rec['rate_mean']:9.0f} ev/s  unsigned peak {pk_u:.2f} ({pk_u / f0:.2f} f0)  signed-global {pk_s:.2f}  "
              f"pixels f0/2f0 {n_f0}/{n_2f0} of {act.sum()}  gamma {gamma:.2f}  flicker100 {flick_ratio:.1f}  KS(Exp1) {ks:.3f}", flush=True)
    json.dump(out, open(os.path.join(ROOT, "outputs", "sim_validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
