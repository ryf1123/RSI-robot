#!/bin/bash
set -e
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONPATH=.
until grep -q DENSE_DONE runs_dense/log; do sleep 20; done
for s in 4 5 6 7; do for arm in llm_named_nofb llm_anon_nofb; do
  d=runs/${arm}_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm llm --seed $s --budget 32 --gen 8 --nofb $([ $arm = llm_anon_nofb ] && echo --anon)
  python -m rsi.loop request --run $d >/dev/null
done; done
python - <<'PY'
import json, numpy as np
from rsi.proposers.prior import semantic_prior, structural_prior
from rsi.design import anonymise
for seed in range(4,8):
    for arm, gen in (("llm_named_nofb", semantic_prior), ("llm_anon_nofb", structural_prior)):
        p=f"runs/{arm}_s{seed}"; req=json.load(open(f"{p}/req_00.json"))
        rng=np.random.default_rng(hash((arm,seed))%2**32)
        ds=[gen(rng) for _ in range(req["k"])]
        json.dump({"designs":[anonymise(d,seed) if req["anon"] else dict(d) for d in ds]},
                  open(f"{p}/resp_00.json","w"), indent=1)
PY
for s in 4 5 6 7; do for arm in llm_named_nofb llm_anon_nofb; do
  python -m rsi.loop step --run runs/${arm}_s${s} --procs 6
done; done
python scripts/reeval.py > runs/reeval2.log 2>&1
echo NOFB2_DONE
