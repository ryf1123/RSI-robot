#!/bin/bash
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONPATH=.
for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
  python -m rsi.loop step --run runs_k4/llm_named_fb_s${s} --procs 6 || exit 1
  python -m rsi.loop step --run runs_k4/llm_anon_fb_s${s} --procs 6 || exit 1
done
RSI_ROOT=runs_k4 RSI_PROCS=6 python scripts/reeval.py > runs_k4/reeval2.log 2>&1
python -m rsi.auto sweep > research/sweep_after_k4.txt 2>&1
echo K4FB_DONE
