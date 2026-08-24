#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate; export PYTHONPATH=. RSI_SPACE=dense
for s in 0 1 2 3; do
  for arm in random evo; do
    d=runs_dense/${arm}_s${s}
    [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm $arm --seed $s --budget 32 --gen 8
    python -m rsi.loop run --run $d --procs 6
  done
  d=runs_dense/llm_named_nofb_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm llm --seed $s --budget 32 --gen 8 --nofb
  python -m rsi.loop request --run $d >/dev/null
done
python scripts/answer_nofb_dense.py
for s in 0 1 2 3; do python -m rsi.loop step --run runs_dense/llm_named_nofb_s${s} --procs 6; done
echo DENSE_DONE
