#!/bin/bash
# Main GPU queue (strictly serial): configuration selection, seeds, Pump runs, add-ons, split-reference / open-set runs and dense CNN baselines. Waits for the stage-5 sentinel (outputs/queue1.log) and for the 2-s cache.
cd "$(dirname "$0")"
PY="${PY:-python}"
until grep -q "QUEUE1 DONE" outputs/queue1.log 2>/dev/null; do sleep 60; done
until grep -q "^DONE" cache/rotor_l2_w2_build.log 2>/dev/null; do sleep 60; done
run() { tag=$1; shift; echo "### $tag $(date)"; $PY scripts/train_evset.py --tag $tag "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
W1="--l2 rotor_l2_w1 --win 1.0"; W2="--l2 rotor_l2_w2 --win 2.0"
# A. configuration selection on LODO (seed 0)
run selA_w1mix_lodo_s0      --kind lodo --seed 0 $W1 --mixup 0.4
run selA_w1mixnorcn_lodo_s0 --kind lodo --seed 0 $W1 --mixup 0.4 --no-rcn
run selA_w2mix_lodo_s0      --kind lodo --seed 0 $W2 --mixup 0.4
run selA_w2mixnorcn_lodo_s0 --kind lodo --seed 0 $W2 --mixup 0.4 --no-rcn
echo "PHASE A DONE $(date)"
# B. seeds for the two leading configurations on every Rotor protocol
for s in 0 1 2; do
  for k in lodo lovo cs; do
    run main_w1mixnorcn_${k}_s$s --kind $k --seed $s $W1 --mixup 0.4 --no-rcn
    run main_w1mix_${k}_s$s      --kind $k --seed $s $W1 --mixup 0.4
  done
done
echo "PHASE B DONE $(date)"
# C. Pump (0.5-s and 1-s tokens), mixup, with / without learned FiLM
for k in pump_lodo pump_cs; do
  run pump_mix_${k}_s0      --subset pump --l2 pump_l2 --kind $k --seed 0 --epochs 20 --mixup 0.4
  run pump_mixnorcn_${k}_s0 --subset pump --l2 pump_l2 --kind $k --seed 0 --epochs 20 --mixup 0.4 --no-rcn
done
until grep -q "^DONE" cache/pump_l2_w1_build.log 2>/dev/null; do sleep 60; done
for k in pump_lodo pump_cs; do
  run pumpw1_mixnorcn_${k}_s0 --subset pump --l2 pump_l2_w1 --win 1.0 --kind $k --seed 0 --epochs 20 --mixup 0.4 --no-rcn
done
run pumpw1_mixnorcn_pump_lovo_s0 --subset pump --l2 pump_l2_w1 --win 1.0 --kind pump_lovo --seed 0 --epochs 20 --mixup 0.4 --no-rcn
echo "PHASE C DONE $(date)"
# D. domain-generalisation add-ons on LODO (1-s, mixup, no-rcn)
run dg_cs2_lodo_s0   --kind lodo --seed 0 $W1 --mixup 0.4 --no-rcn --contrast_scale 2.0
run dg_align_lodo_s0 --kind lodo --seed 0 $W1 --mixup 0.4 --no-rcn --align 0.1 --bs 32
run dg_dann_lodo_s0  --kind lodo --seed 0 $W1 --mixup 0.4 --no-rcn --dann 0.3 --bs 32
run dg_noattr_lodo_s0 --kind lodo --seed 0 $W1 --mixup 0.4 --no-rcn --no-attr
run dg_norefch_lodo_s0 --kind lodo --seed 0 $W1 --mixup 0.4 --no-rcn --no-refch
echo "PHASE D DONE $(date)"
# E. split-reference runs (conformal) and open-set, checkpoints for figures
for k in lodo lovo cs; do run split_${k}_s0 --kind $k --seed 0 $W1 --mixup 0.4 --no-rcn --split_ref; done
for k in pump_lodo pump_cs; do run pumpsplit_${k}_s0 --subset pump --l2 pump_l2_w1 --win 1.0 --kind $k --seed 0 --epochs 20 --mixup 0.4 --no-rcn --split_ref; done
for c in 1 2 3; do run openset_ex${c}_lodo_s0 --kind lodo --seed 0 $W1 --mixup 0.4 --no-rcn --exclude_class $c --split_ref; done
run ckpt_lodo_s0 --kind lodo --seed 0 $W1 --mixup 0.4 --no-rcn --save_model
run ckpt_lovo_s0 --kind lovo --seed 0 $W1 --mixup 0.4 --no-rcn --save_model
echo "PHASE E DONE $(date)"
# F. extra seeds for the 0.5-s reference rows of the window-length ablation
for s in 1 2; do run v1mix_lodo_s$s --kind lodo --seed $s --mixup 0.4; done
echo "RUN_ALL DONE $(date)"
# G. dense literature baselines (event-frame / voxel CNN), 20 epochs
runb() { tag=$1; shift; echo "### $tag $(date)"; $PY scripts/train_baseline_cnn.py --tag $tag --epochs 20 "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
runb frames_ref_lodo_s0 --rep frames --norm ref --kind lodo --seed 0
runb frames_raw_lodo_s0 --rep frames --norm raw --kind lodo --seed 0
runb voxel_ref_lodo_s0 --rep voxel --norm ref --kind lodo --seed 0
runb frames_ref_lovo_s0 --rep frames --norm ref --kind lovo --seed 0
runb frames_ref_cs_s0 --rep frames --norm ref --kind cs --seed 0
echo "PHASE G DONE $(date)"
