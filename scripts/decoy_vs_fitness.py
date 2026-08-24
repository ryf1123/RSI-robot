"""Is decoy mass predictive of task fitness at all? If not, the whole
'the agent understands better' story is decoration."""
import json, glob, numpy as np
from scipy import stats
from rsi.design import Design
from rsi.rewards import DECOYS, HARMFUL, KIND

rows=[]
for r in sorted(glob.glob("runs/*_s*")):
    try: h=[json.loads(l) for l in open(f"{r}/history.jsonl")]
    except FileNotFoundError: continue
    for x in h:
        d=Design(x["design"]); dm=d.decoy_mass()
        if dm!=dm: continue
        hm=sum(abs(d["w"][n]) for n in HARMFUL)/max(1e-9,sum(abs(v) for v in d["w"].values()))
        rows.append((dm, hm, x["fitness"], d["w"]["fwd_vel"], d["w"]["height"], sum(1 for v in d["w"].values() if v>0)))
A=np.array([r[:6] for r in rows],float)
print(f"n = {len(A)} evaluations across all arms")
for i,name in [(0,"decoy mass"),(1,"harmful-only mass"),(5,"n active terms")]:
    rho,p=stats.spearmanr(A[:,i],A[:,2])
    print(f"  spearman(fitness, {name:18s}) = {rho:+.3f}   p={p:.2e}")
# binned
print("\nfitness by decoy-mass bin:")
bins=[0,1e-9,0.1,0.2,0.3,0.45,1.01]
for lo,hi in zip(bins[:-1],bins[1:]):
    m=(A[:,0]>=lo)&(A[:,0]<hi)
    if m.sum()<5: continue
    f=A[m,2]
    bs=np.random.default_rng(0).choice(f,(4000,len(f))).mean(1)
    print(f"  [{lo:.2f},{hi:.2f})  n={m.sum():4d}  mean={f.mean():5.2f} [{np.percentile(bs,2.5):5.2f},{np.percentile(bs,97.5):5.2f}]  P(fit>2)={np.mean(f>2):.3f}")
print("\nharmful-mass = 0 vs > 0:")
for lab,m in [("=0",A[:,1]==0),(">0",A[:,1]>0)]:
    f=A[m,2]; bs=np.random.default_rng(0).choice(f,(4000,len(f))).mean(1)
    print(f"  {lab}: n={m.sum():4d} mean={f.mean():5.2f} [{np.percentile(bs,2.5):5.2f},{np.percentile(bs,97.5):5.2f}]  P(fit>2)={np.mean(f>2):.3f}")
