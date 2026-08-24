#!/bin/bash
set -e
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONPATH=.
for s in 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23; do
  d=runs/sparse5_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm sparse5 --seed $s --budget 32 --gen 8
  python -m rsi.loop run --run $d --procs 6
done
RSI_PROCS=6 python scripts/reeval.py > runs/reeval9.log 2>&1
echo SPARSE5X_DONE
