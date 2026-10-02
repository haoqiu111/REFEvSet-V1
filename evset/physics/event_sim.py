"""Minimal physically-parameterised event-camera simulator for a vibrating textured surface (1-D texture row).

Pixel model (Prophesee-like): log-intensity L(x,t) = log I(x - d(t)); an ON (OFF) event is emitted when
L rises (falls) by theta since the pixel's last event, with a refractory period tau_r after each event;
background noise events ~ Poisson(lambda_n) per pixel; optional multiplicative lighting flicker
(1 + a cos(2 pi 100 t)). d(t) = magnification * integral of the velocity v(t).
Returns per-pixel ON/OFF counts on a time grid (dt) -> the same 'patch rate' representation used for real data.
"""
from __future__ import annotations

import numpy as np


def make_texture(n_pix: int, corr_pix: float = 4.0, contrast: float = 0.5, seed: int = 0, oversample: int = 8):
    """Random smooth log-intensity texture sampled at 1/oversample pixel resolution."""
    rng = np.random.default_rng(seed)
    n = n_pix * oversample
    w = rng.standard_normal(n)
    k = np.arange(-4 * int(corr_pix * oversample), 4 * int(corr_pix * oversample) + 1)
    g = np.exp(-0.5 * (k / (corr_pix * oversample)) ** 2)
    tex = np.convolve(w, g, mode="same")
    tex = contrast * tex / tex.std()
    return tex  # log-intensity, index = sub-pixel position


def simulate(v: np.ndarray, fs: float, magnification: float, texture: np.ndarray, theta: float = 0.2,
             tau_r: float = 1e-4, lambda_noise: float = 0.0, flicker: float = 0.0, dt_out: float = 1e-3,
             oversample: int = 8, seed: int = 0):
    """v: velocity [m/s] sampled at fs; magnification: pixels per metre. Returns (t_grid, on, off) with
    on/off of shape (n_pix, T) counts per dt_out bin."""
    rng = np.random.default_rng(seed)
    n_pix = len(texture) // oversample
    t = np.arange(len(v)) / fs
    d = np.cumsum(v) / fs * magnification  # displacement in pixels
    d = d - d.mean()
    # log intensity of each pixel: texture sampled at (x - d(t)) with sub-pixel resolution
    x0 = np.arange(n_pix) * oversample
    L_prev = None
    T = int(np.ceil(t[-1] / dt_out)) + 1
    on = np.zeros((n_pix, T), np.int32); off = np.zeros((n_pix, T), np.int32)
    ref_level = None  # per-pixel log-intensity at last event
    t_last = np.full(n_pix, -np.inf)
    for i in range(len(t)):
        idx = np.round(x0 - d[i] * oversample).astype(np.int64) % len(texture)
        L = texture[idx]
        if flicker > 0:
            L = L + np.log1p(flicker * np.cos(2 * np.pi * 100.0 * t[i]))
        if ref_level is None:
            ref_level = L.copy(); continue
        dL = L - ref_level
        ready = (t[i] - t_last) >= tau_r
        k_on = np.floor(dL / theta).astype(np.int64); k_off = np.floor(-dL / theta).astype(np.int64)
        e_on = (k_on >= 1) & ready; e_off = (k_off >= 1) & ready
        b = min(int(t[i] / dt_out), T - 1)
        if e_on.any():
            on[e_on, b] += 1; ref_level[e_on] += theta * k_on[e_on]; t_last[e_on] = t[i]
        if e_off.any():
            off[e_off, b] += 1; ref_level[e_off] -= theta * k_off[e_off]; t_last[e_off] = t[i]
    if lambda_noise > 0:
        nz = rng.poisson(lambda_noise * dt_out, size=(2, n_pix, T))
        on += nz[0]; off += nz[1]
    return np.arange(T) * dt_out, on, off
