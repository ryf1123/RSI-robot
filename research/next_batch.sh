#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=.
python -m rsi.loop init --run runs/sparse12_s8 --arm sparse12 --seed 8 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s8 --procs 6
python -m rsi.loop init --run runs/sparse12_s9 --arm sparse12 --seed 9 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s9 --procs 6
python -m rsi.loop init --run runs/sparse12_s10 --arm sparse12 --seed 10 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s10 --procs 6
python -m rsi.loop init --run runs/sparse12_s11 --arm sparse12 --seed 11 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s11 --procs 6
python -m rsi.loop init --run runs/sparse12_s12 --arm sparse12 --seed 12 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s12 --procs 6
python -m rsi.loop init --run runs/sparse12_s13 --arm sparse12 --seed 13 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s13 --procs 6
python -m rsi.loop init --run runs/sparse12_s14 --arm sparse12 --seed 14 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s14 --procs 6
python -m rsi.loop init --run runs/sparse12_s15 --arm sparse12 --seed 15 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s15 --procs 6
python -m rsi.loop init --run runs/sparse12_s16 --arm sparse12 --seed 16 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s16 --procs 6
python -m rsi.loop init --run runs/sparse12_s17 --arm sparse12 --seed 17 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s17 --procs 6
python -m rsi.loop init --run runs/sparse12_s18 --arm sparse12 --seed 18 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s18 --procs 6
python -m rsi.loop init --run runs/sparse12_s19 --arm sparse12 --seed 19 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s19 --procs 6
python -m rsi.loop init --run runs/sparse12_s20 --arm sparse12 --seed 20 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s20 --procs 6
python -m rsi.loop init --run runs/sparse12_s21 --arm sparse12 --seed 21 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s21 --procs 6
python -m rsi.loop init --run runs/sparse12_s22 --arm sparse12 --seed 22 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s22 --procs 6
python -m rsi.loop init --run runs/sparse12_s23 --arm sparse12 --seed 23 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s23 --procs 6
python -m rsi.loop init --run runs/sparse12_s24 --arm sparse12 --seed 24 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s24 --procs 6
python -m rsi.loop init --run runs/sparse12_s25 --arm sparse12 --seed 25 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s25 --procs 6
python -m rsi.loop init --run runs/sparse12_s26 --arm sparse12 --seed 26 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s26 --procs 6
python -m rsi.loop init --run runs/sparse12_s27 --arm sparse12 --seed 27 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s27 --procs 6
python -m rsi.loop init --run runs/sparse12_s28 --arm sparse12 --seed 28 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s28 --procs 6
python -m rsi.loop init --run runs/sparse12_s29 --arm sparse12 --seed 29 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s29 --procs 6
python -m rsi.loop init --run runs/sparse12_s30 --arm sparse12 --seed 30 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse12_s30 --procs 6
python -m rsi.loop init --run runs/sparse3_s8 --arm sparse3 --seed 8 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse3_s8 --procs 6
python -m rsi.loop init --run runs/sparse3_s9 --arm sparse3 --seed 9 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse3_s9 --procs 6
python -m rsi.loop init --run runs/sparse3_s10 --arm sparse3 --seed 10 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse3_s10 --procs 6
python -m rsi.loop init --run runs/sparse3_s11 --arm sparse3 --seed 11 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse3_s11 --procs 6
python -m rsi.loop init --run runs/sparse3_s12 --arm sparse3 --seed 12 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse3_s12 --procs 6
python -m rsi.loop init --run runs/sparse3_s13 --arm sparse3 --seed 13 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse3_s13 --procs 6
python -m rsi.loop init --run runs/sparse3_s14 --arm sparse3 --seed 14 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse3_s14 --procs 6
python -m rsi.loop init --run runs/sparse3_s15 --arm sparse3 --seed 15 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse3_s15 --procs 6
python -m rsi.loop init --run runs/sparse3_s16 --arm sparse3 --seed 16 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse3_s16 --procs 6
python -m rsi.loop init --run runs/sparse3_s17 --arm sparse3 --seed 17 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse3_s17 --procs 6
python -m rsi.loop init --run runs/sparse3_s18 --arm sparse3 --seed 18 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse3_s18 --procs 6
python -m rsi.loop init --run runs/sparse3_s19 --arm sparse3 --seed 19 --budget 32 --gen 8 >/dev/null
python -m rsi.loop run --run runs/sparse3_s19 --procs 6
python scripts/reeval.py > research/reeval.log 2>&1
python -m rsi.auto sweep
echo BATCH_DONE
