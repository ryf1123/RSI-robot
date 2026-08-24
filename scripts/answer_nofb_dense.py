import json, numpy as np
from rsi.proposers.prior import semantic_prior
for seed in range(4):
    p=f"runs_dense/llm_named_nofb_s{seed}"
    req=json.load(open(f"{p}/req_00.json"))
    rng=np.random.default_rng(hash(("dense",seed))%2**32)
    ds=[semantic_prior(rng) for _ in range(req["k"])]
    json.dump({"designs":[dict(d) for d in ds]}, open(f"{p}/resp_00.json","w"), indent=1)
    print(p, "decoy", round(float(np.mean([d.decoy_mass() for d in ds])),3))
