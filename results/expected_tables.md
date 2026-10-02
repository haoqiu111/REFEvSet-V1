# Expected tables

Tables 1 to 8 of "REF-EvSet: Whitened Set Tokens for Reference-Only Event-Camera Fault Diagnosis", with their notes.
`scripts/verify_results.py` parses Table 3 of this file and recomputes every cell from `outputs/`.
Bracketed numbers such as [4] refer to the reference list of the paper.

## Table 1. Published event-camera diagnosis methods and this work

| Work | Input | Model | Data | Protocol | Target data |
|---|---|---|---|---|---|
| Li et al. [4] | event frames | CNN | own benches | ID | L |
| Li et al. [19] | event frames | CNN, modality alignment | own benches | XP, XC | U, P |
| Guang et al. [15] | rate signal | alignment, clustering | own bench | ID | P |
| EViT [5] | event frames | bi-fovea Transformer | own bench | ID | L |
| MLLM [7] | event video | multimodal LLM | own benches | ID | L |
| Li et al. [9] | event frames | SSL, cross-supervision | XJTU-DV | XP | U |
| EBFM [6] | pixel frequency | none | own rigs | frequency estimation | none |
| **REF-EvSet** | whitened order tokens | set attention | XJTU-DV | RO | H, 3 s |

Protocol: ID in-domain random split, XP new camera position, XC new operating condition, RO Reference-Only. Target data: L labelled windows, U unlabelled windows with faults, P paired contact signals, H healthy reference.

## Table 2. Datasets and Reference-Only protocols

| Subset | Classes | Domains | Recordings | Length (s) | Protocols (tasks) |
|---|---|---|---|---|---|
| Rotor | 4 | 3 views, 2 speeds | 24 | 10–19 | LOVO (3), CS (2), LODO (6) |
| Pump | 6 | 3 angles, 3 speeds, 2 settings | 108 | 10 | HO-A (3), HO-S (3), HO-AS (9) |
| Beam | impact | 3 set-ups, 2 lightings | 30, with LDV | 11–16 | calibration |

Rotor: 1000 and 2000 rpm. Pump: angles −30°, 0°, +30°; drive frequencies 20, 30, 40 Hz. Reference: first 3 s of every recording; guard 0.5 s; evaluation windows 1 s with 0.25 s hop. Sensor $640 \times 480$ (Rotor, Pump) and $1280 \times 720$ (Beam); $16 \times 16$ patches; bins of 100 µs (Rotor), 200 µs (Pump) and 1 ms (Beam).

## Table 3. Rotor: same-protocol comparison with re-implemented state-of-the-art paradigms

| Method | Tgt. | In-domain | LOVO | CS | LODO | Mean |
|---|---|---|---|---|---|---|
| FM-LR [6] | R | n/a | 0.415 | 0.556 | 0.592 | 0.521 |
| EF-CNN [4] | none | 0.991 ± 0.004 | 0.348 ± 0.009 | 0.598 ± 0.042 | **0.838 ± 0.032** | 0.595 |
| EF-CNN + ref. | R | n/a | 0.611 ± 0.060 | 0.589 ± 0.032 | 0.688 ± 0.039 | 0.629 |
| EF-CNN + AdaBN-R [10] | R | n/a | 0.412 ± 0.019 | 0.424 ± 0.008 | 0.639 ± 0.016 | 0.492 |
| EF-CNN + AdaBN-T [10] | T | n/a | 0.440 ± 0.015 | 0.600 ± 0.004 | 0.883 ± 0.054 | 0.641 |
| EF-CNN + TENT [11] | T | n/a | 0.451 ± 0.023 | 0.628 ± 0.007 | 0.892 ± 0.048 | 0.657 |
| Vox-CNN | none | 0.991 ± 0.007 | 0.398 ± 0.029 | 0.531 ± 0.051 | 0.700 ± 0.040 | 0.543 |
| SNN [12, 13] | none | 0.385 ± 0.165 | 0.385 ± 0.011 | 0.365 ± 0.055 | 0.350 ± 0.078 | 0.367 |
| SNN + ref. | R | 0.892 ± 0.033 | **0.791 ± 0.006** | 0.782 ± 0.015 | 0.812 ± 0.007 | 0.795 |
| Spec-CNN [15] | none | 1.000 ± 0.000 | 0.242 ± 0.031 | 0.231 ± 0.042 | 0.220 ± 0.039 | 0.231 |
| Spec-CNN + ref. | R | 1.000 ± 0.000 | 0.360 ± 0.020 | 0.323 ± 0.046 | 0.405 ± 0.058 | 0.363 |
| BiFovea-T [5] | none | 0.927 ± 0.030 | 0.221 ± 0.052 | 0.599 ± 0.048 | 0.690 ± 0.031 | 0.503 |
| BiFovea-T + ref. | R | 0.980 ± 0.013 | 0.712 ± 0.007 | **0.819 ± 0.017** | 0.770 ± 0.008 | 0.767 |
| EVS-T [14] | none | 0.671 ± 0.080 | 0.353 ± 0.036 | 0.554 ± 0.018 | 0.483 ± 0.033 | 0.463 |
| EVS-T + ref. | R | 0.854 ± 0.019 | 0.730 ± 0.035 | 0.717 ± 0.026 | 0.711 ± 0.026 | 0.719 |
| SSL-CS [9], strict | R | 0.999 ± 0.002 | 0.346 ± 0.014 | 0.408 ± 0.008 | 0.411 ± 0.029 | 0.389 |
| SSL-CS [9], relaxed | F | n/a | 0.496 ± 0.069 | 0.653 ± 0.059 | 0.807 ± 0.032 | 0.652 |
| Linear | R | n/a | 0.702 | 0.745 | 0.688 | 0.712 |
| **REF-EvSet (ours)** | R | 0.976 ± 0.003 | 0.786 ± 0.014 | 0.824 ± 0.029 | 0.797 ± 0.025 | **0.802** |
| Oracle | O | n/a | 0.904 | 0.833 | 0.879 | 0.872 |

Window accuracy, mean ± std over 3 seeds. In-domain: random 70/30 split of the evaluation windows of every recording, the protocol of the published methods. Tgt.: target data; R healthy reference, T unlabelled test batch, F unlabelled target windows with faults, O all target classes. + ref.: reference-subtracted input. Rows with T, F or O are not deployable. Bold: best deployable value of a column. EF-CNN, EF-CNN + ref., Vox-CNN and FM-LR use 0.5 s windows; all other rows use 1 s windows.

## Table 4. Pump: window accuracy under the three Reference-Only protocols

| Method | HO-A | min | HO-S | min | HO-AS | min |
|---|---|---|---|---|---|---|
| Linear, 1 s | 0.317 | n/a | 0.843 | 0.82 | 0.856 | 0.58 |
| Linear, 0.5 s, set-mean | **0.556** | n/a | 0.759 | n/a | 0.854 | n/a |
| EF-CNN + ref. [4] | 0.490 ± 0.015 | 0.43 | **0.999 ± 0.000** | 1.00 | **1.000 ± 0.000** | 1.00 |
| SNN + ref. [12, 13] | 0.507 ± 0.013 | 0.42 | 0.948 ± 0.020 | 0.92 | 0.969 ± 0.002 | 0.88 |
| BiFovea-T + ref. [5] | 0.468 ± 0.036 | 0.37 | 0.970 ± 0.003 | 0.92 | 0.974 ± 0.003 | 0.84 |
| **REF-EvSet (ours)** | 0.510 ± 0.015 | 0.48 | 0.988 ± 0.005 | 0.98 | 0.994 ± 0.003 | 0.97 |
| REF-EvSet, 0.5 s, + FiLM | n/a | n/a | 0.933 | n/a | 0.985 | n/a |
| REF-EvSet, 0.5 s | n/a | n/a | 0.978 | n/a | 0.981 | n/a |
| Oracle | 0.497 | n/a | 0.961 | 0.94 | 0.966 | 0.91 |

Mean ± std over 3 seeds; min: worst task of the seed-averaged accuracy. Bold: best deployable value of a column. HO-A, HO-S, HO-AS: held-out angle, speed and (angle, speed) cell. Linear: deployable linear baseline on set-mean, max and fraction features; set-mean: set-mean features only. + ref.: reference-subtracted input.

## Table 5. Detection, open-set rejection and cross-device transfer

| Task | Setting | Score | AUROC | Accuracy |
|---|---|---|---|---|
| Detection | Rotor, 6 domains | ZS | 0.928 (0.84) | n/a |
| Detection | Pump, 18 domains | ZS | 0.907 (0.70) | n/a |
| Cross-device | Rotor to Pump | PH | 1.000 / 1.000 | 0.514 / 0.547 |
| Cross-device | Pump to Rotor | PH | 1.000 / 1.000 | 0.398 / 0.425 |
| Open-set, LODO | inner / outer / ball | MSP | 0.786 / 0.813 / 0.556 | 0.918 / 0.954 / 0.789 |
| Open-set, LODO | inner / outer / ball | ZS | 0.427 / 0.720 / 0.995 | n/a |
| Open-set, LODO | inner / outer / ball | RA | 0.636 / 0.846 / 0.823 | n/a |

Detection: training-free, mean (min over domains). Cross-device: two seeds, four shared classes; every Pump domain reaches an AUROC of 1.000; accuracy is the four-class accuracy. Open-set: the listed class is unknown; accuracy is the closed-set accuracy on the known classes.

## Table 6. Sensor-model predictions and measurements

| No. | Prediction of Equation (1) | Quantity | Measurement |
|---|---|---|---|
| P1 | $r_u$ is rectified | main line | $2.02 f_0$, 30 of 30 trials |
| P2 | $r_s$ keeps the sign | patches at $f_0$ | 110–130 per trial; at $2f_0$: 0–36 |
| P3 | flicker line at 100 Hz | relative amplitude | 78 lights on, 0.9 off |
| P4 | compression near $1/\tau_r$ | $\gamma$ in $r \propto \lvert v \rvert^{\gamma}$ | 1.15 far, 0.85 mid, 0.36 close; $R^2$ 0.91–0.99 |
| P5 | spectrum recoverable | RMSE; $f_0$ error; decay bias | 0.052 ± 0.007; 0.017 Hz; +0.6 ± 0.3 $\mathrm{s}^{-1}$ |
| P6 | $\mathrm{SNR} \sim \mathrm{Exp}(1)$ | KS distance | 0.005–0.02 below 1 count/bin |
| S | simulator | line; pixels at $f_0$; $\gamma$; KS | $2.00 f_0$; 174 of 256; 1.17 to 0.64; 0.004 |

P1 to P5: Beam subset, 30 trials. P6: three healthy Rotor recordings; patches above 1 count/bin are sub-Poisson. S: one-dimensional simulator driven by the real LDV velocity, with a flicker line at 5% depth. RMSE: normalised spectral RMSE; the largest $f_0$ error is 0.062 Hz.

## Table 7. Rotor ablations of REF-EvSet

| Variant | LODO | LOVO | CS | $\Delta$ LODO |
|---|---|---|---|---|
| Full | 0.797 ± 0.025 | 0.786 ± 0.014 | 0.824 ± 0.029 | n/a |
| w/o reference | 0.458 ± 0.006 | 0.372 † | 0.544 † | −33.9 |
| w/o ratio channels | 0.790 ± 0.029 | 0.737 † | 0.780 † | −0.7 |
| w/o attributes | 0.774 ± 0.013 | n/a | n/a | −2.3 |
| Grid CNN aggregator | 0.585 ± 0.042 | 0.643 ± 0.015 | 0.602 ± 0.019 | −21.2 ‡ |
| Fixed Hz axis | 0.771 ± 0.022 | 0.770 ± 0.027 | 0.785 ± 0.046 | −2.5 ¶ |
| + FiLM | 0.776 ± 0.019 | 0.720 ± 0.021 | 0.716 ± 0.009 | −2.1 |
| + sub-patch tokens | 0.813 ± 0.036 | 0.787 ± 0.016 | 0.827 ± 0.017 | +1.6 ¶ |
| 0.5 s tokens | 0.776 ± 0.014 | n/a | n/a | −2.1 |
| w/o mixup, 0.5 s, + FiLM | 0.722 † | 0.755 † | 0.619 † | −7.3 § |
| + contrast augmentation | 0.801 ± 0.026 | n/a | n/a | +0.4 ¶ |
| + prototype alignment | 0.756 ± 0.015 | n/a | n/a | −4.1 |
| + DANN | 0.763 ± 0.042 | n/a | n/a | −3.4 |
| + smoothing, 8 windows | n/a | n/a | n/a | +1 |

Window accuracy, mean ± std over 3 seeds; $\Delta$ in points against the full model. Full: 1 s tokens, mixup, physical reference ratio. Marker †: one seed. Marker ‡: $p = 0.0002$ in a Wilcoxon signed-rank test over tasks. Marker ¶: not significant. Marker §: against 0.5 s tokens with mixup. w/o reference: ratio channels and reference attributes removed. w/o ratio channels: the attributes keep the reference.

## Table 8. Parameters and inference cost per 1 s window

| Component | Parameters (M) | GPU (ms) | CPU (ms) |
|---|---|---|---|
| Front end | 0 | 8.8 | 320 |
| REF-EvSet set encoder | 0.85 | 2.2 | 128 |
| + FiLM | 1.16 | 3.3 | 263 |
| + sub-patch tokens | 0.92 | 2.6 | 167 |
| Grid CNN aggregator | 1.33 | 1.3 | 96 |
| EF-CNN | 0.60 | 0.9 | 16 |

Front end: level-1 binning to tokens. RTX 5090 GPU and desktop CPU, batch size 1.
