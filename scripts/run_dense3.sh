#!/bin/bash
set -e
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONPATH=. RSI_SPACE=dense
for s in 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23; do
  d=runs_dense/random_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm random --seed $s --budget 32 --gen 8
  python -m rsi.loop run --run $d --procs 6
done
for s in 8 9 10 11 12 13 14 15; do
  d=runs_dense/llm_named_nofb_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm llm --seed $s --budget 32 --gen 8 --nofb
  python -m rsi.loop request --run $d >/dev/null
done
python - <<'PY'
import json, numpy as np
from rsi.proposers.prior import semantic_prior
for seed in range(8,16):
    p=f"runs_dense/llm_named_nofb_s{seed}"; req=json.load(open(f"{p}/req_00.json"))
    rng=np.random.default_rng(hash(("dense",seed))%2**32)
    json.dump({"designs":[dict(semantic_prior(rng)) for _ in range(req["k"])]},
              open(f"{p}/resp_00.json","w"), indent=1)
PY
for s in 8 9 10 11 12 13 14 15; do python -m rsi.loop step --run runs_dense/llm_named_nofb_s${s} --procs 6; done
RSI_ROOT=runs_dense RSI_PROCS=6 python scripts/reeval.py > runs_dense/reeval3.log 2>&1
echo DENSE3_DONE
