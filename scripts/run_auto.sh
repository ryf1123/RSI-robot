#!/bin/bash
# automatic arms: random + evo, 4 seeds each
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=.
for s in 0 1 2 3; do
  for arm in random evo; do
    d=runs/${arm}_s${s}
    [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm $arm --seed $s --budget 32 --gen 8
    python -m rsi.loop run --run $d --procs 6
  done
done
echo ALL_AUTO_DONE
