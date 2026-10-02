"""Figure: LDV-calibrated event-sensor model (Fig. 6 of the paper).

(a) Beam trial: LDV velocity spectrum (f0) vs unsigned global event-rate spectrum (2 f0, rectification) vs
    sign-aligned sum of the 25 best per-patch signed rates (f0 recovered); lights-on trial shows the 100-Hz line.
(b) Amplitude law: low-passed unsigned rate vs LDV velocity envelope for the three camera set-ups (power-law fits).
(c) Whitening null: histogram of per-patch SNR at non-harmonic orders on a healthy Rotor file vs Exp(1);
    inset: the same for the simulator with Poisson noise.
(d) Simulator: amplitude-law exponent gamma vs magnification for three refractory periods (compressive regime).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evset import plot_style as ps  # noqa: E402
from evset.data.registry import beam_files  # noqa: E402
from evset.features.patch_rate import load_cache  # noqa: E402
from evset.physics.event_sim import make_texture, simulate  # noqa: E402
from linear_v2 import log_snr  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FS_EV = 1000.0


def spec(x, fs, fmax=60.0, fmin=1.0, pad=8):
    x = x - x.mean(); X = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=pad * len(x)))
    f = np.fft.rfftfreq(pad * len(x), 1 / fs); m = (f >= fmin) & (f <= fmax)
    return f[m], X[m] / X[m].max()


def beam_signals(name):
    r = [b for b in beam_files() if b.name == name][0]
    z = load_cache(os.path.join(ROOT, "cache", "beam_l1", r.name + ".npz")); c = np.asarray(z["counts"]).astype(np.float32)
    gu = c.sum(axis=(0, 1)); t_ev = np.arange(len(gu)) / FS_EV
    ldv = np.loadtxt(r.extra["ldv"], skiprows=5); t_l, v = ldv[:, 0], ldv[:, 1]; fs_l = 1 / np.median(np.diff(t_l))
    on_l = t_l[np.argmax(np.abs(v) > 0.2 * np.abs(v).max())]
    base = np.median(gu[: int(0.5 * FS_EV)]) + 1; on_e = t_ev[np.argmax(gu > max(5 * base, 0.2 * gu.max()))]
    seg_l = (t_l >= on_l) & (t_l < on_l + 6.0); seg_e = (t_ev >= on_e) & (t_ev < on_e + 6.0)
    ps_ = (c[1] - c[0])[:, seg_e]
    f_l, S_l = spec(v[seg_l], fs_l); f0 = f_l[np.argmax(S_l)]
    n = ps_.shape[1]; w = np.hanning(n); k = np.exp(-2j * np.pi * f0 * np.arange(n) / FS_EV)
    snr = np.abs((ps_ - ps_.mean(1, keepdims=True)) * w @ k) ** 2 / ((c[0] + c[1])[:, seg_e].mean(1) * (w ** 2).sum() + 1e-6)
    top = np.argsort(snr)[-25:]; lead = ps_[top[-1]] - ps_[top[-1]].mean()
    sig = sum((np.sign(np.dot(ps_[p] - ps_[p].mean(), lead)) or 1.0) * (ps_[p] - ps_[p].mean()) for p in top)
    return dict(v=v[seg_l], fs_l=fs_l, gu=gu[seg_e], sig=sig, f0=f0, t_l=t_l[seg_l] - on_l, t_e=t_ev[seg_e] - on_e, gs=(c[1] - c[0]).sum(0)[seg_e])


def main():
    ps.apply(9)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
    # (a) spectra
    ax = axs[0, 0]; b = beam_signals("Off_set2_trail1")
    f, S = spec(b["v"], b["fs_l"]); ax.plot(f, S, color=ps.GREYS[0], lw=1.2, label="LDV velocity")
    f, S = spec(b["gu"], FS_EV); ax.plot(f, S + 1.05, color=ps.GREYS[2], lw=1.2, label="unsigned event rate (global)")
    f, S = spec(b["sig"], FS_EV); ax.plot(f, S + 2.1, color=ps.BLUE, lw=1.2, label="signed rate, 25 best patches")
    bo = beam_signals("On_set2_trail1"); f, S = spec(bo["gs"], FS_EV, fmax=120, fmin=1); ax.plot(f, S * 0.9 + 3.15, color=ps.BLUE_LIGHT, lw=1.0, label="signed rate (global), lights on")
    for k, lab in [(1, "$f_0$"), (2, "$2f_0$")]:
        ax.axvline(k * b["f0"], color="#999", lw=0.6, ls=":"); ax.text(k * b["f0"] + 0.6, 4.35, lab, fontsize=8, color="#555")
    ax.axvline(100, color="#999", lw=0.6, ls=":"); ax.text(100.6, 4.35, "100 Hz", fontsize=8, color="#555")
    ax.set_xlim(0, 120); ax.set_ylim(0, 4.5); ax.set_yticks([]); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("normalised spectra (offset)")
    ax.legend(loc="upper right", fontsize=7); ax.set_title("(a) rectification, fundamental recovery, flicker", loc="left")
    # (b) amplitude law
    ax = axs[0, 1]
    cal = json.load(open(os.path.join(ROOT, "outputs", "beam_calibration.json")))["saturation"]
    cols = {"set1": ps.GREYS[0], "set2": ps.BLUE, "set3": ps.GREYS[2]}
    for setup, col in cols.items():
        vv, rr = [], []
        for name in [f"Off_{setup}_trail{i}" for i in (1, 3, 5)]:
            bb = beam_signals(name)
            env = np.abs(hilbert(bb["v"] - bb["v"].mean())); env_i = np.interp(bb["t_e"], bb["t_l"], env)
            fb, fa = butter(2, 2.0 / (FS_EV / 2)); lp = filtfilt(fb, fa, bb["gu"]) * FS_EV
            m = (bb["t_e"] > 0.3) & (bb["t_e"] < 5.0) & (env_i > 1e-3) & (lp > 100)
            vv.append(env_i[m][::20]); rr.append(lp[m][::20])
        vv, rr = np.concatenate(vv), np.concatenate(rr)
        g = cal[f"Off/{setup}"]["gamma_lin"]
        ax.scatter(vv * 1e3, rr / 1e6, s=3, color=col, alpha=0.35, lw=0)
        xx = np.linspace(vv.min(), vv.max(), 50); A = np.exp(np.mean(np.log(rr) - g * np.log(vv)))
        ax.plot(xx * 1e3, A * xx ** g / 1e6, color=col, lw=1.4, label=f"{setup}: $\\gamma$ = {g:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("LDV velocity envelope (mm/s)"); ax.set_ylabel("event rate (Mev/s)")
    ax.legend(loc="lower right", fontsize=7); ax.set_title("(b) amplitude law rate $\\propto |v|^{\\gamma}$ (far $\\to$ close)", loc="left")
    # (c) whitening null
    ax = axs[1, 0]
    z = np.load(os.path.join(ROOT, "cache", "rotor_l2", "angle2_Healthy_2000.npz")); s, _ = log_snr(z)
    orders = np.arange(0.125, 64.0001, 0.125); nonh = np.array([abs(o - round(o)) >= 0.375 for o in orders]) & (orders > 12)
    rate = np.exp(z["attrs"][..., 0]).mean(0); mlow = (rate >= 0.01) & (rate < 1.0); mhi = rate >= 1.0
    snr = np.exp(s[:, mlow][:, :, nonh, 0]).reshape(-1); snr = snr[np.isfinite(snr)]
    bins = np.linspace(0, 8, 65)
    ax.hist(snr, bins=bins, density=True, color=ps.BLUE_LIGHT, alpha=0.8, label="real, healthy Rotor (0.01-1 count/bin)")
    snr_hi = np.exp(s[:, mhi][:, :, nonh, 0]).reshape(-1); snr_hi = snr_hi[np.isfinite(snr_hi)]
    ax.hist(snr_hi, bins=bins, density=True, histtype="step", color=ps.GREYS[1], lw=1.0, label="real, rotating-disc patches (> 1 count/bin)")
    xx = np.linspace(0, 8, 200); ax.plot(xx, np.exp(-xx), color="k", lw=1.2, ls="--", label="Exp(1) (Poisson null)")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1.5); ax.set_xlabel("whitened SNR at non-harmonic orders"); ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=7); ax.set_title("(c) shot-noise whitening: null distribution", loc="left")
    # (d) simulator gamma vs magnification
    ax = axs[1, 1]
    r = [b for b in beam_files() if b.name == "Off_set2_trail1"][0]
    ldv = np.loadtxt(r.extra["ldv"], skiprows=5); t_l, v_l = ldv[:, 0], ldv[:, 1]
    on_l = t_l[np.argmax(np.abs(v_l) > 0.2 * np.abs(v_l).max())]; seg = (t_l >= on_l) & (t_l < on_l + 6.0)
    FS = 10000.0; t = np.arange(0, 6.0, 1 / FS); v = np.interp(t, t_l[seg] - on_l, v_l[seg]); tex = make_texture(256, 4.0, 0.5, 0)
    env = np.abs(hilbert(v - v.mean()))
    mags = [100, 300, 1000, 3000, 10000]
    for tau, col, lab in [(1e-4, ps.GREYS[2], "$\\tau_r$ = 0.1 ms"), (1e-3, ps.BLUE_LIGHT, "$\\tau_r$ = 1 ms"), (3e-3, ps.BLUE, "$\\tau_r$ = 3 ms")]:
        gs_ = []
        for mag in mags:
            tg, on, off = simulate(v, FS, mag, tex, theta=0.15, tau_r=tau, dt_out=1e-3, seed=1)
            gu = (on + off).sum(0).astype(float); env_ms = np.interp(tg, t, env)
            fb, fa = butter(2, 2.0 / 500.0); lp = filtfilt(fb, fa, gu) * 1000.0
            m = (tg > 0.3) & (tg < 5.0) & (env_ms > 1e-3) & (lp > 10)
            gs_.append(np.polyfit(np.log(env_ms[m]), np.log(lp[m]), 1)[0])
        ax.plot(mags, gs_, "o-", color=col, lw=1.3, ms=4, label=lab)
    for setup, g in [("set1", cal["Off/set1"]["gamma_lin"]), ("set2", cal["Off/set2"]["gamma_lin"]), ("set3", cal["Off/set3"]["gamma_lin"])]:
        ax.axhline(g, color="#bbb", lw=0.7, ls=":"); ax.text(105, g + 0.02, f"real {setup}", fontsize=7, color="#777")
    ax.set_xscale("log"); ax.set_xlabel("magnification (pixels per metre)"); ax.set_ylabel("exponent $\\gamma$")
    ax.legend(loc="upper right", fontsize=7); ax.set_title("(d) simulator: compressive regime", loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", f"fig_physics.{ext}"))
    print("saved figures/fig_physics.pdf")


if __name__ == "__main__":
    main()
