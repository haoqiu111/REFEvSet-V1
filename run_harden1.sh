#!/bin/bash
# Hardening queue 1: seeds for CNN baselines / DG add-ons / Pump lovo; candidate configs on LOVO+CS for nested selection.
cd "$(dirname "$0")"
PY="${PY:-python}"
run() { tag=$1; shift; echo "### $tag $(date)"; $PY scripts/train_evset.py --tag $tag "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
runb() { tag=$1; shift; echo "### $tag $(date)"; $PY scripts/train_baseline_cnn.py --tag $tag --epochs 20 "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
W1="--l2 rotor_l2_w1 --win 1.0"; W2="--l2 rotor_l2_w2 --win 2.0"
# A. nested selection: candidate configurations on LOVO and CS (seed 0), LODO already exists for all of them
for k in lovo cs; do
  run v1mix_${k}_s0         --kind $k --seed 0 --mixup 0.4
  run selA_w05mixnorcn_${k}_s0 --kind $k --seed 0 --mixup 0.4 --no-rcn
  run selA_w2mix_${k}_s0    --kind $k --seed 0 $W2 --mixup 0.4
  run selA_w2mixnorcn_${k}_s0 --kind $k --seed 0 $W2 --mixup 0.4 --no-rcn
done
run selA_w05mixnorcn_lodo_s0 --kind lodo --seed 0 --mixup 0.4 --no-rcn
echo "HARDEN1A DONE $(date)"
# B. seeds 1-2 for the dense CNN baselines
for s in 1 2; do
  runb frames_raw_lodo_s$s --rep frames --norm raw --kind lodo --seed $s
  runb frames_raw_lovo_s$s --rep frames --norm raw --kind lovo --seed $s
  runb frames_raw_cs_s$s   --rep frames --norm raw --kind cs --seed $s
  runb frames_ref_lodo_s$s --rep frames --norm ref --kind lodo --seed $s
  runb frames_ref_lovo_s$s --rep frames --norm ref --kind lovo --seed $s
  runb frames_ref_cs_s$s   --rep frames --norm ref --kind cs --seed $s
  runb voxel_raw_lodo_s$s  --rep voxel --norm raw --kind lodo --seed $s
done
echo "HARDEN1B DONE $(date)"
# C. seeds 1-2 for DG add-ons and the attribute ablation (LODO)
for s in 1 2; do
  run dg_cs2_lodo_s$s   --kind lodo --seed $s $W1 --mixup 0.4 --no-rcn --contrast_scale 2.0
  run dg_align_lodo_s$s --kind lodo --seed $s $W1 --mixup 0.4 --no-rcn --align 0.1 --bs 32
  run dg_dann_lodo_s$s  --kind lodo --seed $s $W1 --mixup 0.4 --no-rcn --dann 0.3 --bs 32
  run dg_noattr_lodo_s$s --kind lodo --seed $s $W1 --mixup 0.4 --no-rcn --no-attr
done
# D. Pump leave-one-angle seeds
for s in 1 2; do run pumpw1_mixnorcn_pump_lovo_s$s --subset pump --l2 pump_l2_w1 --win 1.0 --kind pump_lovo --seed $s --epochs 20 --mixup 0.4 --no-rcn; done
echo "HARDEN1 DONE $(date)"
