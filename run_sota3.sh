#!/bin/bash
# Final settings of the EVS-T (coarse voxel set, K = 1024) and spiking-CNN (membrane readout, threshold 0.5, lr 3e-4) baselines.
# These runs overwrite the EVS-T / SNN results of run_sota.sh and are the ones reported in the paper; the script also adds the
# in-domain references of the reference-subtracted variants. Strictly serial; run after run_sota.sh and run_sota2.sh.
cd "$(dirname "$0")"
PY="${PY:-python}"
export PYTHONIOENCODING=utf-8
run() { echo "### $* $(date)"; $PY scripts/train_sota.py "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
SNN="--snn_readout mem --snn_thr 0.5 --lr 3e-4"
for s in 0 1 2; do
  for k in lodo lovo cs; do
    run --method evstr --norm raw --kind $k --seed $s
    run --method evstr --norm ref --kind $k --seed $s
    run --method snn   --norm raw --kind $k --seed $s $SNN
    run --method snn   --norm ref --kind $k --seed $s $SNN
  done
  run --method evstr --norm raw --kind indomain --seed $s
  run --method evstr --norm ref --kind indomain --seed $s
  run --method snn   --norm raw --kind indomain --seed $s $SNN
  run --method snn   --norm ref --kind indomain --seed $s $SNN
  run --method evit  --norm ref --kind indomain --seed $s
  run --method spec  --norm ref --kind indomain --seed $s
done
echo "SOTA3 DONE $(date)"
