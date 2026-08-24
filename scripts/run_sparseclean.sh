#!/bin/bash
set -e
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONPATH=.
until grep -q SPARSE_DONE runs/sparse.log; do sleep 30; done
for s in 0 1 2 3 4 5 6 7; do
  for k in 3 5 8 12; do
    d=runs/sparseclean${k}_s${s}
    [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm sparseclean${k} --seed $s --budget 32 --gen 8
    python -m rsi.loop run --run $d --procs 6
  done
done
RSI_PROCS=6 python scripts/reeval.py > runs/reeval6.log 2>&1
echo SPARSECLEAN_DONE
