"""How much of a fitness difference is just the inner ARS seed?
Without this number no outer-loop comparison can be read."""
import numpy as np, multiprocessing as mp, json, time
from rsi.inner import train_and_eval
from rsi.design import Design
def mk(**kw):
    d=Design.zeros()
    for k,v in kw.items():
        if k in d["w"]: d["w"][k]=v
        elif k in d["hp"]: d["hp"][k]=v
    return d
DES = {"B_mid": mk(fwd_vel=2, alive=1, height=1),
       "F_good": mk(fwd_vel=2, alive=1, height=1, step_size=0.05),
       "J_decoy": mk(fwd_vel=2, alive=1, height=1, stand_still=2, step_size=0.05)}
SEEDS = list(range(6))
def run(a):
    n,d,s=a; return n,s,train_and_eval(d,seed=s,n_iters=120)["fitness"]
if __name__=="__main__":
    jobs=[(n,d,s) for n,d in DES.items() for s in SEEDS]
    out={}
    with mp.Pool(6) as p:
        for n,s,f in p.imap_unordered(run, jobs):
            out.setdefault(n,[]).append(f); print(n,s,f,flush=True)
    print()
    for n,v in out.items():
        v=np.array(v); print(f"{n:8s} mean={v.mean():.3f} std={v.std(ddof=1):.3f} min={v.min():.3f} max={v.max():.3f} n={len(v)}")
    json.dump(out, open("runs/noise_floor.json","w"), indent=1)
