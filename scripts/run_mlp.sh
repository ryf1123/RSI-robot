#!/bin/bash
set -e
cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONPATH=. RSI_POLICY=mlp
for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
  d=runs_mlp/random_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm random --seed $s --budget 8 --gen 4 --inner-seeds 4
  python -m rsi.loop run --run $d --procs 6
  d=runs_mlp/llm_named_nofb_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm llm --seed $s --budget 8 --gen 4 --nofb --inner-seeds 4
  python -m rsi.loop request --run $d >/dev/null
done
python - <<'PY'
import json, numpy as np
from rsi.proposers.prior import semantic_prior
for seed in range(12):
    p=f"runs_mlp/llm_named_nofb_s{seed}"; req=json.load(open(f"{p}/req_00.json"))
    rng=np.random.default_rng(hash(("mlp",seed))%2**32)
    json.dump({"designs":[dict(semantic_prior(rng)) for _ in range(req["k"])]},
              open(f"{p}/resp_00.json","w"), indent=1)
PY
for s in 0 1 2 3 4 5 6 7 8 9 10 11; do python -m rsi.loop step --run runs_mlp/llm_named_nofb_s${s} --procs 6; done
RSI_ROOT=runs_mlp RSI_PROCS=6 python scripts/reeval.py > runs_mlp/reeval.log 2>&1
echo MLP_DONE
