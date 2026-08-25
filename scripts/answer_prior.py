"""Answer every unanswered zero-feedback request using the code-written priors.

The two nofb arms are deliberately implemented as auditable code rather than
per-request model calls, which is what lets the autoresearch driver close its own
loop without a human in it."""
import json, glob, os, numpy as np
from rsi.proposers.prior import semantic_prior, structural_prior
from rsi.design import anonymise

n = 0
for req in sorted(glob.glob("runs*/llm_*nofb_s*/req_*.json")):
    d = os.path.dirname(req)
    resp = req.replace("req_", "resp_")
    if os.path.exists(resp):
        continue
    r = json.load(open(req))
    if r.get("feedback", True):
        continue
    seed = json.load(open(f"{d}/config.json"))["seed"]
    gen = structural_prior if r["anon"] else semantic_prior
    tag = "k4a" if ("runs_k4" in d and r["anon"]) else ("k4" if "runs_k4" in d else
          ("dense" if "runs_dense" in d else ("llm_anon_nofb" if r["anon"] else "llm_named_nofb")))
    rng = np.random.default_rng(hash((tag, seed)) % 2**32)
    ds = [gen(rng) for _ in range(r["k"])]
    json.dump({"designs": [anonymise(x, seed) if r["anon"] else dict(x) for x in ds]},
              open(resp, "w"), indent=1)
    n += 1
print(f"answered {n} zero-feedback requests from code priors")
