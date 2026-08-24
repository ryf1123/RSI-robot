#!/bin/bash
set -e
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONPATH=.
until grep -q DENSE2_DONE runs_dense/log2; do sleep 30; done
for s in 0 1 2 3 4 5 6 7; do
  d=runs/contrast_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm contrast --seed $s --budget 32 --gen 8
  python -m rsi.loop run --run $d --procs 6
done
RSI_PROCS=6 python scripts/reeval.py > runs/reeval4.log 2>&1
echo CONTRAST_DONE
