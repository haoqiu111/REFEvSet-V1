# Data preparation

## Dataset

All experiments use the public XJTU-DV dataset:

> X. Li, S. Yu, X. Chen, Y. Lei, B. Yang, N. Li. XJTU-DV: Open-source dynamic vision dataset for non-contact vibration
> measurement and fault diagnosis of mechanical systems. Chinese Journal of Mechanical Engineering 39:100163, 2026.
> <https://doi.org/10.1016/j.cjme.2025.100163>

Download the three archives through the data-availability statement of that article. Neither the recordings nor the
example code contained in the archives are redistributed in this repository.

| Archive | Size | Used content |
|---|---|---|
| `XJTU-DV-Rotor.zip` | 2.8 GB | `Data/<class>/*.dat`, 24 recordings |
| `XJTU-DV-Pump.zip` | 25.3 GB | `Data/<class>/*.dat`, 108 recordings |
| `XJTU-DV-Beam.zip` | 1.7 GB | `DV/*.dat`, 30 recordings, and `LDV/*.txt`, 30 laser-vibrometer records |

## Folder layout

The data root is `./data/XJTU-DV` inside the repository, or the folder named by the environment variable
`EVSET_DATA`. Place the three archives there and run stage 0:

```bash
export EVSET_DATA=/path/to/XJTU-DV    # optional
bash reproduce.sh 0 0
```

Stage 0 copies the event files into flat sub-folders and leaves the archives untouched:

```
$EVSET_DATA/
  XJTU-DV-Rotor.zip  XJTU-DV-Pump.zip  XJTU-DV-Beam.zip
  Rotor/      angle<1|2|3>_<Healthy|Inner|Outer|Ball>_<1000|2000>.dat
  Pump/       <20|30|40>HZ_<H|I|O|b|cav|MA>_<setting>_<-30|0|30>.dat
  Beam/       <On|Off>_set<1|2|3>_trail<1..5>.dat
  Beam_LDV/   <On|Off>_set<1|2|3>_trail<1..5>.txt
```

The same layout can be created by hand. `evset/data/registry.py` parses the labels from the file names:

- Rotor: viewpoint `angle1..3`, bearing state, shaft speed in rpm.
- Pump: drive frequency in Hz, class code (`H` healthy, `I` inner race, `O` outer race, `b` ball, `cav` cavitation,
  `MA` misalignment), operating setting, camera angle in degrees.
- Beam: lighting `On` / `Off`, camera set-up `set1..3`, trial number. Every trial is paired with the LDV text file of
  the same name. The LDV folder can be set independently with the environment variable `EVSET_LDV`
  (default `$EVSET_DATA/Beam_LDV`); the files are read with `numpy.loadtxt(..., skiprows=5)` (time in s, velocity in
  m/s).

The `.dat` files are Prophesee Event2D recordings (ASCII header, then records of a 32-bit time stamp in µs and a
32-bit word packing x, y and polarity); `evset/data/dat_reader.py` reads them through a memory map.

## Caches

The pipeline writes its intermediate representations to `./cache` (about 66 GB in total):

| Cache | Built by | Content |
|---|---|---|
| `rotor_l1`, `beam_l1`, `pump_l1` | stage 1 | per-patch, per-polarity counts (16 x 16 px patches; 100 µs, 1 ms, 200 µs bins) |
| `rotor_l2`, `rotor_l2_w1`, `rotor_l2_w2` | stage 2 | order-domain tokens for 0.5 / 1 / 2 s windows, hop 0.25 s |
| `pump_l2`, `pump_l2_w1` | stage 2 | order-domain tokens for 0.5 / 1 s windows |
| `rotor_l2_w1_fixhz` | stage 2 | tokens on a fixed frequency axis (ablation) |
| `rotor_frames`, `rotor_frames_w1`, `pump_frames_w1` | stage 2 | polarity count images and 8-bin voxel grids for the dense baselines |
| `*_linear_feats*.npz` | stage 3 | pooled features of the linear baselines |
| `rotor_global_rate` | stage 8 | whole-sensor event-rate signal for the spectrogram baseline |

Files named `cache/<name>_build.log` are written by `reproduce.sh`; the GPU queues wait for the line `DONE` in them.
`cache/`, `data/` and all run-time logs are excluded from version control by `.gitignore`.
