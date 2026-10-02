#!/bin/bash
# In-domain (random 70/30 within every recording) reference numbers for all paradigms and REF-EvSet; waits for run_sota.sh.
cd "$(dirname "$0")"
PY="${PY:-python}"
export PYTHONIOENCODING=utf-8
until grep -q "SOTA DONE" outputs/sota.log 2>/dev/null; do sleep 60; done
run() { echo "### $* $(date)"; $PY scripts/train_sota.py "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
runb() { tag=$1; shift; echo "### $tag $(date)"; $PY scripts/train_baseline_cnn.py --tag $tag --epochs 20 "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
rune() { tag=$1; shift; echo "### $tag $(date)"; $PY scripts/train_evset.py --tag $tag "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
for s in 0 1 2; do
  runb frames_raw_indomain_s$s --rep frames --norm raw --kind indomain --seed $s
  runb voxel_raw_indomain_s$s --rep voxel --norm raw --kind indomain --seed $s
  run --method evit  --norm raw --kind indomain --seed $s
  run --method snn   --norm raw --kind indomain --seed $s
  run --method evstr --norm raw --kind indomain --seed $s
  run --method spec  --norm raw --kind indomain --seed $s
  run --method ssl   --variant strict --kind indomain --seed $s
  rune main_w1mixnorcn_indomain_s$s --kind indomain --seed $s --l2 rotor_l2_w1 --win 1.0 --mixup 0.4 --no-rcn
done
echo "SOTA2 DONE $(date)"
