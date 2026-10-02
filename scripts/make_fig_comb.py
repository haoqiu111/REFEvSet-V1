"""Figure: what the camera sees. (a)(b) class-mean whitened order spectra (top-100 patches) at view 1 / 1000 rpm and
view 3 / 2000 rpm: shaft-harmonic comb, no bearing orders; (c) envelope spectrum of the 1-4 kHz band (global rate):
only shaft harmonics; (d) comb amplitude (mean log-SNR gain over orders 1-10, top-100 patches) per class x domain:
class ordering changes with the viewpoint (level ambiguity)."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evset import plot_style as ps  # noqa: E402
from evset.features.patch_rate import load_cache  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CLASSES = ["Healthy", "Inner", "Outer", "Ball"]
COL = {"Healthy": ps.GREYS[1], "Inner": ps.CLASS_BLUES[0], "Outer": ps.CLASS_BLUES[1], "Ball": ps.CLASS_BLUES[2]}


def main():
    D = np.load(os.path.join(ROOT, "cache", "rotor_class_snr_maps.npy"), allow_pickle=True).item()
    orders = np.arange(0.125, 64.0001, 0.125); K = [np.argmin(np.abs(orders - k)) for k in range(1, 11)]
    ps.apply(9)
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 4.6))
    for k, dom in enumerate(["angle1@1000", "angle3@2000"]):
        ax = axs[0, k]; ref = D[(dom, "ref")][..., 0]
        for c in CLASSES:
            A = D[(dom, c)][..., 0] - ref; top = np.argsort(A[:, K].mean(1))[-100:]
            prof = A[top].mean(0); m = orders <= 12
            ax.plot(orders[m], prof[m], color=COL[c], lw=1.1, label=c)
        for j in range(1, 13):
            ax.axvline(j, color="#ddd", lw=0.5, zorder=0)
        ax.set_xlabel("order (multiple of shaft frequency)"); ax.set_ylabel("log-SNR gain vs reference (nats)")
        ax.set_title(f"({'ab'[k]}) class-mean whitened spectra, {dom.replace('angle', 'viewpoint ').replace('@', ', ')} rpm", loc="left", fontsize=8); ps.grid(ax)
        if k == 0: ax.legend(fontsize=7, ncol=2)
    # (c) envelope spectrum of the 1-4 kHz band, global unsigned rate, view 1 / 1000 rpm, 4 s
    ax = axs[1, 0]; fs = 10000.0
    for c in CLASSES:
        z = load_cache(os.path.join(ROOT, "cache", "rotor_l1", f"angle1_{c}_1000.npz")); cnt = z["counts"]; valid = z["valid_s"]
        s0 = int(np.argmax(valid) * 10000); L = 40000; gu = np.asarray(cnt[:, :, s0:s0 + L]).astype(np.float32).sum(axis=(0, 1))
        b, a = butter(4, [1000 / (fs / 2), 4000 / (fs / 2)], btype="band"); env = np.abs(hilbert(filtfilt(b, a, gu - gu.mean())))
        E = np.abs(np.fft.rfft((env - env.mean()) * np.hanning(L))); fe = np.fft.rfftfreq(L, 1 / fs); o = fe / 16.69
        m = (o > 0.3) & (o < 12); ax.plot(o[m], E[m] / E[m].max() + {"Healthy": 0, "Inner": 1.1, "Outer": 2.2, "Ball": 3.3}[c], color=COL[c], lw=0.9, label=c)
    for j in range(1, 13):
        ax.axvline(j, color="#ddd", lw=0.5, zorder=0)
    ax.set_yticks([]); ax.set_xlabel("order"); ax.set_ylabel("envelope spectra, 1–4 kHz band (offset)"); ax.set_title("(c) envelope analysis: only shaft harmonics", loc="left", fontsize=8); ax.legend(fontsize=7, ncol=2)
    # (d) level ambiguity heat map: comb amplitude per class x domain
    ax = axs[1, 1]; doms = ["angle1@1000", "angle1@2000", "angle2@1000", "angle2@2000", "angle3@1000", "angle3@2000"]
    M = np.zeros((3, 6))
    for j, dom in enumerate(doms):
        ref = D[(dom, "ref")][..., 0]
        for i, c in enumerate(["Inner", "Outer", "Ball"]):
            A = D[(dom, c)][..., 0] - ref; top = np.argsort(A[:, K].mean(1))[-100:]; M[i, j] = A[top][:, K].mean()
    im = ps.heat(ax, M, vmin=0, vmax=14, fmt="{:.1f}", text_thresh=9); ax.set_xticks(range(6)); ax.set_xticklabels([d.replace("angle", "v").replace("@1000", "\n1000").replace("@2000", "\n2000") for d in doms], fontsize=7)
    ax.set_yticks(range(3)); ax.set_yticklabels(["Inner", "Outer", "Ball"], fontsize=7); ax.set_title("(d) comb amplitude per class and domain (nats)", loc="left", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "fig_comb.png"); fig.savefig(out); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)


if __name__ == "__main__":
    main()
