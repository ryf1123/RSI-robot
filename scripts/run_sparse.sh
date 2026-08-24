#!/bin/bash
set -e
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONPATH=.
for s in 0 1 2 3 4 5 6 7; do
  for k in 2 3 5 8 12 14; do
    d=runs/sparse${k}_s${s}
    [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm sparse${k} --seed $s --budget 32 --gen 8
    python -m rsi.loop run --run $d --procs 6
  done
done
RSI_PROCS=6 python scripts/reeval.py > runs/reeval5.log 2>&1
echo SPARSE_DONE
