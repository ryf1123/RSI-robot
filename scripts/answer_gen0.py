"""Generation 0 of the feedback arms has no feedback yet, so the honest answer
is the same prior the zero-feedback arms use. This also makes gen 0 shared
between fb/nofb arms, which is what we want for the 2x2."""
import json, numpy as np
from rsi.proposers.prior import semantic_prior, structural_prior
from rsi.design import anonymise
for seed in range(4):
    for arm, gen in (("llm_named_fb", semantic_prior), ("llm_anon_fb", structural_prior)):
        p = f"runs/{arm}_s{seed}"
        req = json.load(open(f"{p}/req_00.json"))
        rng = np.random.default_rng(hash((arm, seed, "g0")) % 2**32)
        ds = [gen(rng) for _ in range(req["k"])]
        json.dump({"designs": [anonymise(d, seed) if req["anon"] else dict(d) for d in ds]},
                  open(f"{p}/resp_00.json", "w"), indent=1)
print("gen0 written")
