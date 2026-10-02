"""Figure 1: framework of REF-EvSet as an editable PowerPoint drawing (figures/fig_framework.pptx), rendered by PowerPoint
itself to figures/fig_framework.pdf (vector) and .png. Illustrations are computed from the Rotor caches: polarity event
frames of labelled source domains, event frames of the target healthy reference, the whitened log-SNR map of a fault
window, the reference field and the reference-ratio map. Elements: inputs, the fixed front end (patch rates, order-domain
DFT, shot-noise whitening), the reference ratio, the trainable set encoder, the loss with its gradient path, the outputs,
and the vibrometer-calibrated sensor model that anchors the whitening (calibration path).
python scripts/make_fig_framework.py"""
import os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evset.eval.protocol import load_rotor_l2_meta
from linear_v2 import log_snr
import pptx_draw as D
FIG = os.path.join(ROOT, 'figures'); TH = os.path.join(FIG, '_fw'); os.makedirs(TH, exist_ok=True)


def save(img, name, cmap, vmin=None, vmax=None, size=(1.6, 1.2)):
    fig = plt.figure(figsize=size); ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax); ax.axis('off')
    p = os.path.join(TH, name + '.png'); fig.savefig(p, dpi=200); plt.close(fig); return p


def thumbs():
    T = {}
    def frame(name, k=10, n=6):
        f = np.load(os.path.join(ROOT, 'cache', 'rotor_frames', name + '.npz'))['frames'][k:k + n].astype(np.float32).sum(0); return f[0], f[1]
    def polarity(pos, neg):       # event-camera rendering: positive events red, negative events blue, on white
        r, b = np.log1p(pos), np.log1p(neg); r, b = np.clip(r / (np.percentile(r, 99.5) + 1e-6), 0, 1), np.clip(b / (np.percentile(b, 99.5) + 1e-6), 0, 1)
        return np.clip(np.stack([1 - 0.9 * b, 1 - 0.75 * (r + b), 1 - 0.9 * r], -1), 0, 1)
    T['src'] = [save(polarity(*frame(n)), f'src{i}', None) for i, n in enumerate(['angle1_Inner_1000', 'angle2_Outer_2000', 'angle1_Ball_2000'])]   # labelled source domains
    T['ref'] = [save(np.log1p(sum(frame('angle3_Healthy_1000', k))), f'ref{i}', 'gray_r') for i, k in enumerate((2, 12, 22))]     # target viewpoint: healthy only
    files = {f.name: f for f in load_rotor_l2_meta(os.path.join(ROOT, 'cache', 'rotor_l2'))}
    orders = np.arange(0.125, 64.0001, 0.125); i1 = int(np.argmin(np.abs(orders - 1))); i3 = int(np.argmin(np.abs(orders - 3)))
    fh = files['angle1_Healthy_1000']; sh, _ = log_snr(np.load(fh.path)); ref = sh[..., 0][fh.ref_mask()].mean(0)
    fi = files['angle1_Inner_1000']; si, _ = log_snr(np.load(fi.path)); s = si[..., 0][fi.eval_mask()].mean(0)
    T['snr'] = save(s[:, i1].reshape(30, 40), 'snr', 'Blues', -2, 8, (1.4, 1.05)); T['snr_ref'] = save(ref[:, i1].reshape(30, 40), 'snr_ref', 'Greys', -2, 8, (1.4, 1.05))
    T['ratio'] = save((s[:, i3] - ref[:, i3]).reshape(30, 40), 'ratio', 'RdBu_r', -8, 8, (1.4, 1.05))
    return T


def draw(T, path):
    W, H = 13.33, 6.2; c = D.Canvas(W, H); G, B, O = D.GREEN, D.BLUE, D.ORANGE; RG = D.GREEN_D
    yU, yL = 1.5, 4.45
    # ---------------- regions
    c.frame(0.12, 0.15, 2.36, 5.5); c.frame(2.6, 0.15, 6.15, 5.5); c.frame(8.87, 0.15, 4.34, 3.0); c.frame(8.87, 3.25, 4.34, 2.4)
    # ---------------- inputs (illustrations)
    c.text(0.15, 0.2, 2.3, 0.3, 'Source events (labelled)', 12)
    c.stack(T['src'], 0.36, 0.95, 1.5, 1.12); c.text(0.15, 2.3, 2.3, 0.3, 'other viewpoints, speeds', 9.5, color=D.GREY)
    c.stack(T['ref'], 0.36, 3.9, 1.5, 1.12); c.text(0.15, 5.22, 2.3, 0.3, 'Target healthy reference (3 s)', 11.5)
    # ---------------- fixed front end: the same three operations for source windows and for the reference
    c.text(2.65, 0.2, 2.7, 0.3, 'Front end (no parameters)', 11.5)
    for x, lab in ((2.82, ['Patch', 'rates', '16 × 16']), (3.72, ['Order', 'DFT', 'Eq. (3)']), (4.62, ['Shot-noise', 'whitening', 'Eq. (4)'])):
        c.cube(x, 0.75, 0.78, 4.45, lab, fill='DEEBF7', line=D.BLUE_D, size=9.5, depth=0.22)
    c.arrow(2.04, yU, 2.8, yU, G); c.arrow(2.04, yL, 2.8, yL, RG, 2.0, dash='dash')
    for x in (3.42, 4.32):
        c.arrow(x, yU, x + 0.3, yU, G); c.arrow(x, yL, x + 0.3, yL, RG, 2.0, dash='dash')
    c.arrow(5.22, yU, 5.55, yU, G); c.arrow(5.22, yL, 5.55, yL, RG, 2.0, dash='dash')
    c.picture(T['snr'], 5.57, yU - 0.42, 1.1, 0.84, border='BFBFBF'); c.text(5.3, yU + 0.45, 1.65, 0.3, 'log SNR(p, o)', 9.5)
    c.picture(T['snr_ref'], 5.57, yL - 0.42, 1.1, 0.84, border='BFBFBF'); c.text(5.3, yL + 0.45, 1.65, 0.3, 'log SNR_{ref}(p, o)', 9.5)
    # ---------------- reference ratio (the pose-dependent gain cancels) and token set
    c.arrow(6.69, yU, 6.98, yU, G); c.node(7.2, yU, 0.44, '−'); c.text(6.6, yU - 0.62, 1.2, 0.3, 'Eq. (5)', 10, color=D.BLUE_D)
    c.path([(6.69, yL), (7.2, yL), (7.2, yU + 0.24)], RG, 2.0, dash='dash'); c.text(7.28, 2.6, 1.4, 0.45, ['gain G(p)', 'cancels, Eq. (2)'], 9.5, color=D.GREEN_D, align='l')
    c.arrow(7.42, yU, 7.6, yU, G); c.picture(T['ratio'], 7.62, yU - 0.42, 1.05, 0.84, border='BFBFBF'); c.text(7.3, yU + 0.45, 1.45, 0.42, ['Token set', 'R(p, o), attributes'], 9.5)
    # ---------------- calibrated sensor model (calibration / control of the front end)
    c.box(7.0, 4.98, 1.68, 0.6, ['Sensor model, Eq. (1)', 'LDV-calibrated'], fill='FFF2CC', line='BF9000', size=9.5)
    c.path([(7.0, 5.36), (5.0, 5.36), (5.0, 5.2)], 'BF9000', 2.0, dash='dash'); c.text(5.5, 5.38, 1.3, 0.24, 'Exp(1) null', 9.5, color='7F6000')
    # ---------------- set encoder (trainable), loss and gradient path
    c.text(8.95, 0.2, 2.9, 0.3, 'Set encoder (trainable)', 11.5)
    c.arrow(8.69, yU, 9.0, yU, G)
    for x, lab in ((9.02, ['Order', 'conv']), (9.92, ['ISAB', '× 2']), (10.82, ['PMA'])):
        c.cube(x, yU - 0.62, 0.8, 1.24, lab, fill=O, line=D.ORANGE_D, size=10, depth=0.22, alpha=0.9)
    c.arrow(9.74, yU, 9.92, yU, G, 3.5); c.arrow(10.64, yU, 10.82, yU, G, 3.5); c.arrow(11.56, yU, 11.98, yU, G)
    c.bar(12.0, yU - 0.7, 0.95, 1.4, ['Loss', 'L_{CE}'], 'F8CBAD', 11.5)
    c.path([(12.47, yU - 0.72), (12.47, yU - 0.9), (10.3, yU - 0.9), (10.3, yU - 0.64)], B); c.text(11.75, 0.24, 1.3, 0.26, 'gradient', 9.5, color=D.BLUE_D)
    c.text(8.98, yU + 0.75, 2.1, 0.5, ['mixup, token subsampling,', 'order and gain shifts'], 9.5, color=D.GREY, align='l')
    # ---------------- outputs
    c.path([(11.2, yU + 0.64), (11.2, 3.45)], G); c.text(11.3, 3.27, 1.1, 0.26, 'embedding z', 9.5, align='l')
    c.box(10.45, 3.55, 1.5, 0.6, ['Fault', 'class'], fill='FFFFFF', line=D.GREEN_D, size=11)
    c.box(9.05, 4.4, 1.9, 0.4, 'Detection score', fill='E2F0D9', line=D.GREEN_D, size=9.5); c.box(11.1, 4.4, 1.95, 0.4, 'Open-set score', fill='E2F0D9', line=D.GREEN_D, size=9.5)
    c.box(9.05, 4.95, 1.9, 0.4, 'Attention map', fill='E2F0D9', line=D.GREEN_D, size=9.5); c.box(11.1, 4.95, 1.95, 0.4, 'Cross-device transfer', fill='E2F0D9', line=D.GREEN_D, size=9.5)
    # ---------------- legend
    c.arrow(3.0, 5.92, 3.5, 5.92, G, 4.0); c.text(3.56, 5.8, 0.8, 0.25, 'forward', 9, align='l')
    c.arrow(4.6, 5.92, 5.1, 5.92, B, 4.0); c.text(5.16, 5.8, 0.8, 0.25, 'gradient', 9, align='l')
    c.arrow(6.2, 5.92, 6.7, 5.92, RG, 2.0, dash='dash'); c.text(6.76, 5.8, 1.2, 0.25, 'reference', 9, align='l')
    c.arrow(8.0, 5.92, 8.5, 5.92, 'BF9000', 2.0, dash='dash'); c.text(8.56, 5.8, 1.2, 0.25, 'calibration', 9, align='l')
    c.save(path)


if __name__ == '__main__':
    T = thumbs(); out = os.path.join(FIG, 'fig_framework.pptx'); draw(T, out); print('saved', out, *D.export(out))
