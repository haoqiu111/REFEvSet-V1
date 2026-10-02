#!/bin/bash
# Pump (18 domains, 6 classes): the three strongest deployable prior paradigms with reference subtraction, hold-out (angle, speed) and hold-out speed, 3 seeds.
cd "$(dirname "$0")"
PY="${PY:-python}"
export PYTHONIOENCODING=utf-8
until grep -q "^DONE" cache/pump_frames_w1_build.log 2>/dev/null; do sleep 60; done
run() { echo "### $* $(date)"; $PY scripts/train_sota.py --subset pump "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
SNN="--snn_readout mem --snn_thr 0.5 --lr 3e-4"
for s in 0 1 2; do
  for k in pump_lodo pump_cs; do
    run --method snn  --norm ref --kind $k --seed $s $SNN
    run --method evit --norm ref --kind $k --seed $s
    run --method cnn  --norm ref --kind $k --seed $s
  done
done
echo "SOTA4 DONE $(date)"
