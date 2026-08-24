"""Ring 3: separate PROPOSAL quality from SELECTION pressure.

best-so-far mixes the two: even a blind proposer's best-so-far climbs, because
max() of more samples is bigger. The proposal-quality signal is the MEAN fitness
of a generation, which selection cannot inflate."""
import json, glob, numpy as np
from rsi.report import ARMS, boot

print(f"{'arm':<16}{'gen':>4} {'mean fitness':>22} {'mean decoy mass':>20}  {'best-so-far':>12}")
print("-"*78)
for arm in ARMS:
    per_gen, per_gen_d, bsf = {}, {}, {}
    for r in sorted(glob.glob(f"runs/{arm}_s*")):
        try: h=[json.loads(l) for l in open(f"{r}/history.jsonl")]
        except FileNotFoundError: continue
        cfg=json.load(open(f"{r}/config.json")); g=cfg["gen"]
        if not cfg.get("feedback",True): g=8   # zero-feedback arms: chop into pseudo-gens
        m=-1e9
        for x in h:
            k=x["i"]//g
            per_gen.setdefault(k,[]).append(x["fitness"])
            if x["decoy_mass"]==x["decoy_mass"]: per_gen_d.setdefault(k,[]).append(x["decoy_mass"])
            m=max(m,x["fitness"]); bsf[k]=bsf.get(k,[])+[m] if x["i"]%g==g-1 else bsf.get(k,[])
    for k in sorted(per_gen):
        f=boot(per_gen[k]); d=boot(per_gen_d[k]) if per_gen_d.get(k) else (float('nan'),)*3
        b=np.mean(bsf[k]) if bsf.get(k) else float('nan')
        print(f"{arm:<16}{k:>4} {f[0]:7.2f} [{f[1]:5.2f},{f[2]:5.2f}] {d[0]:7.3f} [{d[1]:.3f},{d[2]:.3f}]  {b:12.2f}")
    print()
