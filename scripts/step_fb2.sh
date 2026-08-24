#!/bin/bash
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONPATH=.
for s in 4 5 6 7; do for a in llm_named_fb llm_anon_fb; do
  python -m rsi.loop step --run runs/${a}_s${s} --procs 2 || exit 1
  python -m rsi.loop request --run runs/${a}_s${s} >/dev/null
done; done
echo FB2_GEN_DONE
