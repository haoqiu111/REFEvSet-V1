#!/bin/bash
# Tail queue (after run_after.sh, cross-device): sub-patch tokens -> dense CNN on LOVO / CS and voxel CNN on LODO -> no-reference ablations -> Pump seeds.
cd "$(dirname "$0")"
PY="${PY:-python}"
until grep -q "RUN_AFTER DONE" outputs/run_after.log 2>/dev/null; do sleep 60; done
run() { tag=$1; shift; echo "### $tag $(date)"; $PY scripts/train_evset.py --tag $tag "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
runb() { tag=$1; shift; echo "### $tag $(date)"; $PY scripts/train_baseline_cnn.py --tag $tag --epochs 20 "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
W1="--l2 rotor_l2_w1 --win 1.0"
# 1. sub-patch spatial tokens
for s in 0 1 2; do
  for k in lodo lovo cs; do run sub_w1mixnorcn_${k}_s$s --kind $k --seed $s $W1 --mixup 0.4 --no-rcn --subpatch; done
done
echo "TAIL1 DONE $(date)"
# 2. dense CNN on LOVO / CS (raw) and voxel LODO
runb frames_raw_lovo_s0 --rep frames --norm raw --kind lovo --seed 0
runb frames_raw_cs_s0 --rep frames --norm raw --kind cs --seed 0
runb voxel_raw_lodo_s0 --rep voxel --norm raw --kind lodo --seed 0
echo "TAIL2 DONE $(date)"
# 3. reference-contribution ablations
for s in 0 1 2; do
  run abl_noref_lodo_s$s    --kind lodo --seed $s $W1 --mixup 0.4 --no-rcn --no-refch --no-refattr
  run abl_norefch_lodo_s$s  --kind lodo --seed $s $W1 --mixup 0.4 --no-rcn --no-refch
done
for k in lovo cs; do
  run abl_noref_${k}_s0   --kind $k --seed 0 $W1 --mixup 0.4 --no-rcn --no-refch --no-refattr
  run abl_norefch_${k}_s0 --kind $k --seed 0 $W1 --mixup 0.4 --no-rcn --no-refch
done
echo "TAIL3 DONE $(date)"
# 4. Pump seeds 1-2
for s in 1 2; do
  for k in pump_lodo pump_cs; do run pumpw1_mixnorcn_${k}_s$s --subset pump --l2 pump_l2_w1 --win 1.0 --kind $k --seed $s --epochs 20 --mixup 0.4 --no-rcn; done
done
echo "TAIL DONE $(date)"
