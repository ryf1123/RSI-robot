"""Mandatory probes.

The system refuses to run comparisons on a design space until these have been
executed, because each of them is a thing this project got wrong by skipping it.

  noise_floor   -- how much of a fitness difference is just the inner seed?
                   Without this, no outer curve is readable.
  instrument    -- does the ground-truth instrument (decoy mass) actually have
                   the floor and chance level it claims?
  protocol      -- does the information-equivalent-to-random control arm in fact
                   score at random? If not, information is leaking.
"""
import json, os, numpy as np, multiprocessing as mp
from ..design import Design
from ..rewards import TERM_NAMES, DECOYS, CHANCE_DECOY_MASS
from ..inner import train_and_eval
from . import data, stats as S
from .spec import RESEARCH_DIR

PROBE_DIR = f"{RESEARCH_DIR}/probes"


def _w(a):
    d, s = a
    return train_and_eval(d, seed=s, n_iters=120)["fitness"]


def noise_floor(designs=None, seeds=range(6), procs=6, out=f"{PROBE_DIR}/noise_floor.json"):
    """Same design, different inner seed. Returns per-design mean/std."""
    if designs is None:
        mk = lambda **kw: (lambda d: (d["w"].update({k: v for k, v in kw.items() if k in d["w"]}),
                                      d["hp"].update({k: v for k, v in kw.items() if k in d["hp"]}), d)[-1])(Design.zeros())
        designs = {"mid": mk(fwd_vel=2, alive=1, height=1),
                   "good": mk(fwd_vel=2, alive=1, height=1, step_size=0.05),
                   "decoyed": mk(fwd_vel=2, alive=1, height=1, stand_still=2, step_size=0.05)}
    jobs = [(dict(d), s) for d in designs.values() for s in seeds]
    with mp.Pool(procs) as p:
        vals = p.map(_w, jobs)
    k = len(list(seeds)); res = {}
    for i, name in enumerate(designs):
        v = np.array(vals[i * k:(i + 1) * k])
        res[name] = dict(mean=round(float(v.mean()), 3), std=round(float(v.std(ddof=1)), 3),
                         min=round(float(v.min()), 3), max=round(float(v.max()), 3), vals=v.round(3).tolist())
    res["verdict"] = ("noise dominates: the best design's seed-std exceeds its own mean"
                      if res[max(res, key=lambda n: res[n]["mean"] if isinstance(res[n], dict) else -9)]["std"] >
                      res[max(res, key=lambda n: res[n]["mean"] if isinstance(res[n], dict) else -9)]["mean"]
                      else "noise is smaller than the between-design signal")
    os.makedirs(PROBE_DIR, exist_ok=True); json.dump(res, open(out, "w"), indent=1)
    return res


def instrument(n=4000, seed=0):
    """Does decoy mass have the floor and chance level it advertises?"""
    rng = np.random.default_rng(seed)
    ds = [Design.random(rng) for _ in range(n)]
    dm = np.array([d.decoy_mass() for d in ds]); dm = dm[dm == dm]
    res = dict(n_terms=len(TERM_NAMES), n_decoys=len(DECOYS),
               claimed_chance=round(CHANCE_DECOY_MASS, 4),
               measured_chance=round(float(dm.mean()), 4),
               floor_reachable=bool((dm == 0).any()))
    res["ok"] = abs(res["measured_chance"] - res["claimed_chance"]) < 0.02 and res["floor_reachable"]
    os.makedirs(PROBE_DIR, exist_ok=True)
    json.dump(res, open(f"{PROBE_DIR}/instrument.json", "w"), indent=1)
    return res


def protocol(root="runs", control="llm_anon_nofb", baseline="random", metric="decoy"):
    """The control arm must be indistinguishable from random. If it beats random,
    the anonymisation leaked and every downstream comparison is void."""
    a, b = data.series(root, control, metric), data.series(root, baseline, metric)
    c = S.compare(a, b, alt="less")
    c["ok"] = bool(c.get("intervals_overlap", False))
    c["control"], c["baseline"], c["metric"] = control, baseline, metric
    os.makedirs(PROBE_DIR, exist_ok=True)
    json.dump(c, open(f"{PROBE_DIR}/protocol.json", "w"), indent=1)
    return c


def gate():
    """What the runner calls before allowing a comparison."""
    missing = [n for n, f in (("noise_floor", "noise_floor.json"), ("instrument", "instrument.json"),
                              ("protocol", "protocol.json")) if not os.path.exists(f"{PROBE_DIR}/{f}")]
    return dict(ok=not missing, missing=missing)
