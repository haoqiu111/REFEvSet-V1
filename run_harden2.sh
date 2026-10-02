#!/bin/bash
# Hardening queue 2 (after harden1 + fixed-Hz cache): set-vs-grid and order-vs-fixed-frequency ablations, 3 seeds x 3 protocols; GPU cost table.
cd "$(dirname "$0")"
PY="${PY:-python}"
until grep -q "HARDEN1 DONE" outputs/harden1.log 2>/dev/null; do sleep 60; done
until grep -q "^DONE" cache/rotor_l2_w1_fixhz_build.log 2>/dev/null; do sleep 60; done
run() { tag=$1; shift; echo "### $tag $(date)"; $PY scripts/train_evset.py --tag $tag "$@" 2>&1 | grep -E "^\[|^==|Traceback|Error"; }
W1="--l2 rotor_l2_w1 --win 1.0"
for s in 0 1 2; do
  for k in lodo lovo cs; do
    run grid_w1mixnorcn_${k}_s$s --kind $k --seed $s $W1 --mixup 0.4 --no-rcn --agg grid
    run fixhz_w1mixnorcn_${k}_s$s --kind $k --seed $s --l2 rotor_l2_w1_fixhz --win 1.0 --mixup 0.4 --no-rcn
  done
done
$PY scripts/cost_table.py --device cuda 2>&1 | tail -9
echo "HARDEN2 DONE $(date)"
