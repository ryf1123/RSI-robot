#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=.
python -m rsi.loop init --run runs/llm_anon_nofb_s8 --arm llm --seed 8 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s8 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s8 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s9 --arm llm --seed 9 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s9 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s9 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s10 --arm llm --seed 10 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s10 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s10 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s11 --arm llm --seed 11 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s11 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s11 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s12 --arm llm --seed 12 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s12 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s12 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s13 --arm llm --seed 13 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s13 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s13 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s14 --arm llm --seed 14 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s14 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s14 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s15 --arm llm --seed 15 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s15 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s15 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s16 --arm llm --seed 16 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s16 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s16 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s17 --arm llm --seed 17 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s17 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s17 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s18 --arm llm --seed 18 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s18 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s18 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s19 --arm llm --seed 19 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s19 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s19 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s20 --arm llm --seed 20 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s20 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s20 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s21 --arm llm --seed 21 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s21 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s21 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s22 --arm llm --seed 22 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s22 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s22 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s23 --arm llm --seed 23 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s23 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s23 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s24 --arm llm --seed 24 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s24 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s24 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s25 --arm llm --seed 25 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s25 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s25 --procs 6
python -m rsi.loop init --run runs/llm_anon_nofb_s26 --arm llm --seed 26 --budget 32 --gen 8 --nofb --anon >/dev/null
python -m rsi.loop request --run runs/llm_anon_nofb_s26 >/dev/null
python scripts/answer_prior.py >/dev/null
python -m rsi.loop step --run runs/llm_anon_nofb_s26 --procs 6
RSI_ROOT=runs python scripts/reeval.py >> research/reeval.log 2>&1
python -m rsi.auto verify-batch
python -m rsi.auto sweep
echo BATCH_DONE
