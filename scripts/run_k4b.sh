#!/bin/bash
set -e
cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONPATH=.
for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
  d=runs_k4/evo_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm evo --seed $s --budget 8 --gen 4 --inner-seeds 4
  python -m rsi.loop run --run $d --procs 6
  d=runs_k4/llm_anon_nofb_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm llm --seed $s --budget 8 --gen 4 --nofb --anon --inner-seeds 4
  python -m rsi.loop request --run $d >/dev/null
done
python - <<'PY'
import json, numpy as np
from rsi.proposers.prior import structural_prior
from rsi.design import anonymise
for seed in range(12):
    p=f"runs_k4/llm_anon_nofb_s{seed}"; req=json.load(open(f"{p}/req_00.json"))
    rng=np.random.default_rng(hash(("k4a",seed))%2**32)
    ds=[structural_prior(rng) for _ in range(req["k"])]
    json.dump({"designs":[anonymise(d,seed) for d in ds]}, open(f"{p}/resp_00.json","w"), indent=1)
PY
for s in 0 1 2 3 4 5 6 7 8 9 10 11; do python -m rsi.loop step --run runs_k4/llm_anon_nofb_s${s} --procs 6; done
# feedback arms: generation 0 uses the same priors
for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
  d=runs_k4/llm_named_fb_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm llm --seed $s --budget 8 --gen 4 --inner-seeds 4
  python -m rsi.loop request --run $d >/dev/null
  d=runs_k4/llm_anon_fb_s${s}
  [ -f $d/config.json ] || python -m rsi.loop init --run $d --arm llm --seed $s --budget 8 --gen 4 --anon --inner-seeds 4
  python -m rsi.loop request --run $d >/dev/null
done
python - <<'PY'
import json, numpy as np
from rsi.proposers.prior import semantic_prior, structural_prior
from rsi.design import anonymise
for seed in range(12):
    for arm, gen in (("llm_named_fb", semantic_prior), ("llm_anon_fb", structural_prior)):
        p=f"runs_k4/{arm}_s{seed}"; req=json.load(open(f"{p}/req_00.json"))
        rng=np.random.default_rng(hash((arm,seed,"k4g0"))%2**32)
        ds=[gen(rng) for _ in range(req["k"])]
        json.dump({"designs":[anonymise(d,seed) if req["anon"] else dict(d) for d in ds]},
                  open(f"{p}/resp_00.json","w"), indent=1)
PY
for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
  python -m rsi.loop step --run runs_k4/llm_named_fb_s${s} --procs 6
  python -m rsi.loop request --run runs_k4/llm_named_fb_s${s} >/dev/null
  python -m rsi.loop step --run runs_k4/llm_anon_fb_s${s} --procs 6
  python -m rsi.loop request --run runs_k4/llm_anon_fb_s${s} >/dev/null
done
echo K4B_GEN0_DONE
