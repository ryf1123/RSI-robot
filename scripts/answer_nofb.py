"""Fill in resp_00.json for the two zero-feedback arms."""
import json, sys, numpy as np
from rsi.proposers.prior import semantic_prior, structural_prior
from rsi.design import Design, anonymise
for seed in range(4):
    for arm, gen in (("llm_named_nofb", semantic_prior), ("llm_anon_nofb", structural_prior)):
        p = f"runs/{arm}_s{seed}"
        req = json.load(open(f"{p}/req_00.json"))
        rng = np.random.default_rng(hash((arm, seed)) % 2**32)
        ds = [gen(rng) for _ in range(req["k"])]
        out = [anonymise(d, seed) if req["anon"] else dict(d) for d in ds]
        json.dump({"designs": out}, open(f"{p}/resp_00.json", "w"), indent=1)
        print(p, len(out), "mean decoy mass", round(float(np.mean([d.decoy_mass() for d in ds])), 3))
