#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate; export PYTHONPATH=.
until grep -q ALL_NOFB_DONE runs/nofb.log; do sleep 20; done
for s in 4 5 6 7; do for arm in random evo; do
  d=runs/${arm}_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm $arm --seed $s --budget 32 --gen 8
  python -m rsi.loop run --run $d --procs 5
done; done
echo AUTO2_DONE
