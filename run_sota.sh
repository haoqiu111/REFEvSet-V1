#!/bin/bash
# Same-protocol SOTA re-implementations (Rotor, 3 protocols x 3 seeds), strictly serial on one GPU.
cd "$(dirname "$0")"
PY="${PY:-python}"
export PYTHONIOENCODING=utf-8
run() { echo "### $* $(date)"; $PY scripts/train_sota.py "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
runb() { tag=$1; shift; echo "### $tag $(date)"; $PY scripts/train_baseline_cnn.py --tag $tag --epochs 20 "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
until grep -q "^DONE" cache/rotor_global_rate_build.log 2>/dev/null; do sleep 30; done
for s in 0 1 2; do
  for k in lodo lovo cs; do
    run --method evit  --norm raw --kind $k --seed $s
    run --method evit  --norm ref --kind $k --seed $s
    run --method snn   --norm raw --kind $k --seed $s
    run --method snn   --norm ref --kind $k --seed $s
    run --method evstr --norm raw --kind $k --seed $s
    run --method evstr --norm ref --kind $k --seed $s
    run --method spec  --norm raw --kind $k --seed $s
    run --method spec  --norm ref --kind $k --seed $s
    run --method tta   --kind $k --seed $s
    run --method ssl   --variant strict  --kind $k --seed $s
    run --method ssl   --variant relaxed --kind $k --seed $s
  done
done
for s in 0 1 2; do for k in lovo cs; do runb voxel_raw_${k}_s$s --rep voxel --norm raw --kind $k --seed $s; done; done
echo "SOTA DONE $(date)"
