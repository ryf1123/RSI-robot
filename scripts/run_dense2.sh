#!/bin/bash
set -e
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONPATH=. RSI_SPACE=dense
until grep -q "rep 13" runs/budget_split.log; do sleep 30; done
for s in 4 5 6 7; do
  for arm in random evo; do
    d=runs_dense/${arm}_s${s}
    [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm $arm --seed $s --budget 32 --gen 8
    python -m rsi.loop run --run $d --procs 5
  done
  d=runs_dense/llm_named_nofb_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm llm --seed $s --budget 32 --gen 8 --nofb
  python -m rsi.loop request --run $d >/dev/null
done
python - <<'PY'
import json, numpy as np
from rsi.proposers.prior import semantic_prior
for seed in range(4,8):
    p=f"runs_dense/llm_named_nofb_s{seed}"; req=json.load(open(f"{p}/req_00.json"))
    rng=np.random.default_rng(hash(("dense",seed))%2**32)
    json.dump({"designs":[dict(semantic_prior(rng)) for _ in range(req["k"])]},
              open(f"{p}/resp_00.json","w"), indent=1)
PY
for s in 4 5 6 7; do python -m rsi.loop step --run runs_dense/llm_named_nofb_s${s} --procs 5; done
RSI_ROOT=runs_dense RSI_PROCS=5 python scripts/reeval.py > runs_dense/reeval2.log 2>&1
echo DENSE2_DONE
