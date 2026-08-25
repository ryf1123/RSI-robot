"""Feasibility: does ARS with a small tanh MLP learn this task in a comparable
inner budget? Threat `external_validity_inner` -- everything so far rests on one
policy class."""
import time, numpy as np, multiprocessing as mp, sys
from rsi.env import Hopper
from rsi.inner import rollout, Normalizer, MAX_STEPS, act
from rsi.rewards import TERM_NAMES
from rsi.design import Design

def train_mlp(design, seed, n_iters=120, n_dirs=6, hidden=8):
    d = Design(design); w = d.weight_vec; hp = d["hp"]
    rng = np.random.default_rng(seed)
    env = Hopper(term_height=hp["term_height"], max_steps=MAX_STEPS, seed=seed)
    W1 = rng.standard_normal((hidden, Hopper.obs_dim)) / np.sqrt(Hopper.obs_dim)
    W2 = np.zeros((Hopper.act_dim, hidden))
    norm = Normalizer(Hopper.obs_dim)
    top = max(1, int(round(hp["top_frac"] * n_dirs)))
    for it in range(n_iters):
        ds = [(rng.standard_normal(W1.shape), rng.standard_normal(W2.shape)) for _ in range(n_dirs)]
        rp = np.array([rollout(env, (W1 + hp["noise_std"]*a, W2 + hp["noise_std"]*b), norm, w)[0] for a, b in ds])
        rm = np.array([rollout(env, (W1 - hp["noise_std"]*a, W2 - hp["noise_std"]*b), norm, w)[0] for a, b in ds])
        o = np.argsort(-np.maximum(rp, rm))[:top]
        sr = np.concatenate([rp[o], rm[o]]).std() + 1e-6
        step = hp["step_size"] / (top * sr)
        W1 = W1 + step * sum((rp[i]-rm[i]) * ds[i][0] for i in o)
        W2 = W2 + step * sum((rp[i]-rm[i]) * ds[i][1] for i in o)
    ee = Hopper(term_height=0.7, max_steps=MAX_STEPS, seed=seed+10_000)
    xs = [rollout(ee, (W1, W2), norm, w, train=False)[1]["x"] for _ in range(8)]
    return float(np.mean(xs))

def mk(**kw):
    d = Design.zeros()
    for k, v in kw.items():
        (d["hp"] if k in d["hp"] else d["w"])[k] = v
    return d

CFG = {"good(linear-tuned)": mk(fwd_vel=2, alive=1, height=1, step_size=0.05),
       "fwd+alive+height":   mk(fwd_vel=2, alive=1, height=1),
       "decoyed":            mk(fwd_vel=2, alive=1, height=1, stand_still=2, step_size=0.05)}
def work(a):
    name, d, it, s = a; t = time.time()
    return name, it, s, train_mlp(d, s, n_iters=it), time.time()-t
if __name__ == "__main__":
    jobs = [(n, d, it, s) for n, d in CFG.items() for it in (120, 300) for s in range(3)]
    with mp.Pool(6) as p:
        res = {}
        for n, it, s, f, dt in p.map(work, jobs):
            res.setdefault((n, it), []).append((f, dt))
    for (n, it), v in sorted(res.items()):
        f = np.array([x[0] for x in v]); dt = np.mean([x[1] for x in v])
        print(f"{n:<22} iters={it:<5} fit={f.mean():6.2f} ± {f.std(ddof=1):.2f}   {dt:5.1f}s/run")
