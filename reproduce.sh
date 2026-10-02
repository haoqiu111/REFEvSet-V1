#!/bin/bash
# REF-EvSet reproduction (strictly serial; one GPU job at a time). Stages are resumable: bash reproduce.sh <from> <to>
#   0 data extraction (.dat files of Rotor / Beam / Pump and the Beam LDV text files from the XJTU-DV zips)
#   1 level-1 caches (patch counts)
#   2 level-2 token caches (0.5 / 1 / 2 s Rotor, 0.5 / 1 s Pump, fixed-Hz Rotor) + dense frame caches
#   3 linear baselines (deployable baseline and oracle)
#   4 Beam physics + simulator validation + physics figure
#   5 REF-EvSet main runs (run_all.sh, phases A-G)
#   6 cross-device + tail queue (sub-patch tokens, CNN on LOVO / CS, no-reference ablations, Pump seeds) + set-vs-grid,
#     fixed-Hz ablations and the cost table
#   7 tables / figures / evaluations
#   8 same-protocol re-implementations of prior paradigms (run_sota*.sh), in-domain references, comparison table / figure,
#     verification of every reported number
# Environment: PY (Python interpreter, default "python"), EVSET_DATA (XJTU-DV root, default ./data/XJTU-DV),
#              EVSET_LDV (Beam LDV text files, default $EVSET_DATA/Beam_LDV). See docs/DATA.md.
# To re-verify the shipped results without any training: python scripts/verify_results.py
cd "$(dirname "$0")"
PY="${PY:-python}"
export PYTHONIOENCODING=utf-8
FROM=${1:-0}; TO=${2:-8}
mkdir -p cache outputs figures
stage() { [ "$1" -ge "$FROM" ] && [ "$1" -le "$TO" ]; }
if stage 0; then $PY - <<'PYEOF'
import zipfile, os
root = os.environ.get("EVSET_DATA", os.path.join("data", "XJTU-DV"))
for name, sub, out, ext in [("Rotor", "Data/", "Rotor", ".dat"), ("Beam", "DV/", "Beam", ".dat"), ("Pump", "Data/", "Pump", ".dat"),
                            ("Beam", "LDV/", "Beam_LDV", ".txt")]:
    z = zipfile.ZipFile(os.path.join(root, f"XJTU-DV-{name}.zip")); os.makedirs(os.path.join(root, out), exist_ok=True)
    for i in z.infolist():
        if i.filename.startswith(sub) and i.filename.endswith(ext):
            dst = os.path.join(root, out, os.path.basename(i.filename))
            if not os.path.exists(dst): open(dst, "wb").write(z.read(i))
PYEOF
fi
if stage 1; then
  $PY - <<'PYEOF'
import sys, os; sys.path.insert(0, os.getcwd())
from evset.data.registry import rotor_files, beam_files
from evset.features.patch_rate import build_patch_counts, save_cache
for r in rotor_files():
    out = os.path.join("cache", "rotor_l1", f"{r.name}.npz")
    if not os.path.exists(out): save_cache(out, build_patch_counts(r.path, patch=16, bin_us=100, min_events_per_s=3e5))
for r in beam_files():
    out = os.path.join("cache", "beam_l1", f"{r.name}.npz")
    if not os.path.exists(out): save_cache(out, build_patch_counts(r.path, patch=16, bin_us=1000, min_events_per_s=0))
PYEOF
  echo "DONE" > extract_pump_log.txt; $PY scripts/pump_survey_l1.py
fi
if stage 2; then
  # the "<name>_build.log" files are sentinels: the GPU queues wait for a line "DONE" in them
  $PY scripts/build_l2_rotor.py
  L2_WIN=1.0 L2_HOP=0.25 L2_NAME=rotor_l2_w1 $PY scripts/build_l2_rotor.py
  $PY scripts/build_l2.py --subset rotor --l1 rotor_l1 --win 2.0 --hop 0.25 --name rotor_l2_w2 | tee cache/rotor_l2_w2_build.log
  $PY scripts/build_l2.py --subset pump --win 0.5 --hop 0.25 --name pump_l2
  $PY scripts/build_l2.py --subset pump --win 1.0 --hop 0.25 --name pump_l2_w1 | tee cache/pump_l2_w1_build.log
  $PY scripts/build_l2.py --subset rotor --l1 rotor_l1 --win 1.0 --hop 0.25 --name rotor_l2_w1_fixhz --fixed_hz 16.6667 | tee cache/rotor_l2_w1_fixhz_build.log
  $PY scripts/build_frames_rotor.py
  L2_NAME=rotor_l2_w1 L2_WIN=1.0 FRAME_NAME=rotor_frames_w1 $PY scripts/build_frames_rotor.py
  $PY scripts/build_frames_pump.py | tee cache/pump_frames_w1_build.log
fi
if stage 3; then
  $PY scripts/linear_baselines.py > outputs/linear_v1.log; cp outputs/linear_baselines.json outputs/linear_v1.json
  $PY scripts/linear_freqmap.py > outputs/linear_freqmap.log     # FM-LR row of Table 3
  L2_WIN=1.0 L2_NAME=rotor_l2_w1 $PY scripts/linear_v2.py > outputs/linear_v2_w1.log
  L2_WIN=2.0 L2_NAME=rotor_l2_w2 $PY scripts/linear_v2.py > outputs/linear_v2_w2.log
  # the 0.5-s run is the last Rotor run: outputs/linear_baselines.json (read by make_fig_whitening.py and make_fig_cells.py) holds its results
  L2_NAME=rotor_l2 L2_WIN=0.5 $PY scripts/linear_v2.py > outputs/linear_v2_rotor.log
  SUBSET=pump L2_NAME=pump_l2 L2_WIN=0.5 $PY scripts/linear_v2.py > outputs/linear_pump.log
  SUBSET=pump L2_NAME=pump_l2_w1 L2_WIN=1.0 $PY scripts/linear_v2.py > outputs/linear_pump_w1.log
fi
if stage 4; then $PY scripts/beam_physics.py; $PY scripts/beam_calibration.py; $PY scripts/sim_validation.py; $PY scripts/make_fig_physics.py; fi
if stage 5; then
  for k in lodo lovo cs; do $PY scripts/train_evset.py --kind $k --seed 0 --tag v1_${k}_s0; done > outputs/evset_v1_s0.log
  { $PY scripts/train_evset.py --tag v1mix_lodo_s0 --kind lodo --seed 0 --mixup 0.4
    $PY scripts/train_evset.py --tag abl_norcn_lodo_s0 --kind lodo --seed 0 --no-rcn
    $PY scripts/train_evset.py --tag abl_noattr_lodo_s0 --kind lodo --seed 0 --no-attr
    $PY scripts/train_evset.py --tag v1w1_lodo_s0 --kind lodo --seed 0 --l2 rotor_l2_w1 --win 1.0
    echo "QUEUE1 DONE"; } > outputs/queue1.log
  bash run_all.sh > outputs/run_all.log
fi
if stage 6; then bash run_after.sh > outputs/run_after.log; bash run_after_b.sh > outputs/run_after_b.log; bash run_harden1.sh > outputs/harden1.log; bash run_harden2.sh > outputs/harden2.log; fi
if stage 7; then
  $PY scripts/collect_results.py; $PY scripts/make_main_table.py; $PY scripts/nested_selection.py; $PY scripts/verify_results.py
  $PY scripts/openset_eval.py; $PY scripts/conformal_eval.py split_lodo_s0 split_lovo_s0 split_cs_s0
  $PY scripts/make_fig_protocol.py; $PY scripts/make_fig_whitening.py; $PY scripts/build_class_snr_maps.py; $PY scripts/make_fig_comb.py; $PY scripts/make_fig_cells.py; $PY scripts/make_fig_pump.py
  EVSET_DEVICE=cpu $PY scripts/make_fig_attention.py ckpt_lodo_s0 lodo_angle3_2000; $PY scripts/make_fig_tsne.py
fi
if stage 8; then
  $PY scripts/build_global_rate.py | tee cache/rotor_global_rate_build.log
  bash run_sota.sh > outputs/sota.log; bash run_sota2.sh > outputs/sota2.log; bash run_sota3.sh > outputs/sota3.log
  bash run_sota4.sh > outputs/sota4.log; bash run_sota5.sh > outputs/sota5.log
  $PY scripts/collect_indomain.py
  # Fig. 1 is drawn through Microsoft PowerPoint (Windows only); opt in with EVSET_FRAMEWORK_FIG=1
  if [ "${EVSET_FRAMEWORK_FIG:-0}" = "1" ]; then $PY scripts/make_fig_framework.py; fi
  $PY scripts/make_fig_sota.py; $PY scripts/verify_results.py
fi
