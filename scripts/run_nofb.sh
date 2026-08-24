#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=.
until grep -q ALL_AUTO_DONE runs/auto.log; do sleep 20; done
for s in 0 1 2 3; do
  for arm in llm_named_nofb llm_anon_nofb; do
    python -m rsi.loop step --run runs/${arm}_s${s} --procs 6
  done
done
echo ALL_NOFB_DONE
