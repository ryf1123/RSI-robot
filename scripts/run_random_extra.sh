#!/bin/bash
set -e
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONPATH=.
until grep -q MIDHP_DONE runs/midhp.log; do sleep 30; done
for s in 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23; do
  d=runs/random_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm random --seed $s --budget 32 --gen 8
  python -m rsi.loop run --run $d --procs 6
done
RSI_PROCS=6 python scripts/reeval.py > runs/reeval8.log 2>&1
echo RANDOM_EXTRA_DONE
