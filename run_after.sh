#!/bin/bash
# After run_all: cross-device (Rotor <-> Pump) with REF-EvSet, 2 seeds.
cd "$(dirname "$0")"
PY="${PY:-python}"
until grep -q "PHASE G DONE" outputs/run_all.log 2>/dev/null; do sleep 60; done
for s in 0 1; do echo "### cross_device s$s $(date)"; $PY scripts/cross_device.py --seed $s 2>&1 | grep -E "^rotor|^pump|Traceback|Error"; done
echo "RUN_AFTER DONE $(date)"
