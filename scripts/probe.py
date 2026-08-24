import time, sys, numpy as np, multiprocessing as mp
from rsi.inner import train_and_eval
from rsi.design import Design
def mk(**kw):
    d=Design.zeros()
    for k,v in kw.items():
        if k in d["w"]: d["w"][k]=v
        elif k in d["hp"]: d["hp"][k]=v
    return d
CFGS = {
 "A fwd2+alive1":        mk(fwd_vel=2, alive=1),
 "B +height":            mk(fwd_vel=2, alive=1, height=1),
 "C +ctrl+upright":      mk(fwd_vel=2, alive=1, height=1, ctrl_cost=0.5, upright=0.5),
 "D fwd4 only":          mk(fwd_vel=4),
 "E B noise.02":         mk(fwd_vel=2, alive=1, height=1, noise_std=0.02),
 "F B step.05":          mk(fwd_vel=2, alive=1, height=1, step_size=0.05),
 "G progress":           mk(progress=2, alive=1, height=1),
 "H B+hop+clear":        mk(fwd_vel=2, alive=1, height=1, hop=0.5, foot_clear=0.5),
 "I B, th0.6":           mk(fwd_vel=2, alive=1, height=1, term_height=0.6),
 "J decoy: stand_still": mk(fwd_vel=2, alive=1, height=1, stand_still=2),
 "K decoy: back_vel":    mk(fwd_vel=2, alive=1, height=1, back_vel=2),
 "L all-zero":           mk(),
}
NI=int(sys.argv[1]) if len(sys.argv)>1 else 120
def run(it):
    n,d=it; t=time.time(); r=train_and_eval(d,0,n_iters=NI); return n,r,time.time()-t
if __name__=="__main__":
    with mp.Pool(6) as p:
        for n,r,dt in p.imap_unordered(run, list(CFGS.items())):
            print(f"{n:22s} fit={r['fitness']:6.2f} fall={r['fall_rate']:.2f} ep={r['ep_seconds']:.2f}s "
                  f"air={r['airborne_frac']:.2f} curve={r['task_curve']} ({dt:.0f}s)", flush=True)
