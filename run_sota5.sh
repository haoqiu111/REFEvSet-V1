#!/bin/bash
# Pump hold-out angle (HO-A): the three reference-subtracted prior paradigms, 3 seeds. Run after run_sota4.sh.
cd "$(dirname "$0")"
PY="${PY:-python}"; export PYTHONIOENCODING=utf-8
run() { echo "### $* $(date)"; $PY scripts/train_sota.py --subset pump "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
for s in 0 1 2; do
  run --method cnn  --norm ref --kind pump_lovo --seed $s
  run --method snn  --norm ref --kind pump_lovo --seed $s --snn_readout mem --snn_thr 0.5 --lr 3e-4
  run --method evit --norm ref --kind pump_lovo --seed $s
done
echo "SOTA5 DONE $(date)"
