"""Outer loop. A run is a directory holding history.jsonl plus, for LLM arms,
a request/response file pair per generation so that a human-or-model proposer
can answer out of band."""
import json, os, sys, time, argparse
import numpy as np
import multiprocessing as mp
from .design import Design, anonymise, deanonymise, anon_maps, W_LEVELS, HP_LEVELS
from .rewards import TERM_NAMES, KIND, DECOYS
from .inner import train_and_eval

N_ITERS_DEFAULT = 120


def _work(a):
    design, seed, n_iters = a
    t = time.time()
    r = train_and_eval(design, seed=seed, n_iters=n_iters)
    r["wall_s"] = round(time.time() - t, 1)
    return r


class Run:
    def __init__(self, path):
        self.path = path
        self.cfg = json.load(open(f"{path}/config.json"))
        self.hist = [json.loads(l) for l in open(f"{path}/history.jsonl")] \
            if os.path.exists(f"{path}/history.jsonl") else []

    @staticmethod
    def init(path, arm, seed, budget=48, gen=4, n_iters=N_ITERS_DEFAULT):
        os.makedirs(path, exist_ok=True)
        json.dump(dict(arm=arm, seed=seed, budget=budget, gen=gen, n_iters=n_iters),
                  open(f"{path}/config.json", "w"), indent=1)
        open(f"{path}/history.jsonl", "a").close()
        return Run(path)

    # ---------------- proposals ----------------
    def propose(self, k, rng):
        arm = self.cfg["arm"]
        if arm == "random":
            return [Design.random(rng) for _ in range(k)]
        if arm == "contrast":
            return self._contrast(k, rng)
        if arm == "evo":
            if not self.hist: return [Design.random(rng) for _ in range(k)]
            best = Design(self.best()["design"])
            return [best.mutate(rng, n_changes=int(rng.choice([1, 2, 3]))) for _ in range(k)]
        raise RuntimeError(f"arm {arm} is answered out of band; use request/respond")

    def _contrast(self, k, rng):
        """Coordinate-wise controlled contrasts: change exactly one term against a
        fixed baseline, with the inner seed held constant, so that an inert term
        shows up as a byte-identical fitness. See notes/00_preregistration.md."""
        if not self.hist:
            return [Design.random(rng) for _ in range(k)]
        base_rec = self.best()
        base = Design(base_rec["design"])
        base_fit = base_rec["fitness"]
        # digest what earlier contrasts already told us
        killed = set(self.cfg.get("killed", []))
        for h in self.hist:
            d = Design(h["design"])
            diff = [n for n in TERM_NAMES if d["w"][n] != base["w"][n]]
            if len(diff) == 1 and d["w"][diff[0]] == 0.0 and base["w"][diff[0]] > 0:
                delta = h["fitness"] - base_fit
                if abs(delta) < 1e-9 or delta > 0.3:
                    killed.add(diff[0])
        self.cfg["killed"] = sorted(killed)
        json.dump(self.cfg, open(f"{self.path}/config.json", "w"), indent=1)
        for n in killed:
            base["w"][n] = 0.0
        active = [n for n in TERM_NAMES if base["w"][n] > 0 and n not in killed]
        inactive = [n for n in TERM_NAMES if base["w"][n] == 0 and n not in killed]
        out = []
        rng.shuffle(active); rng.shuffle(inactive)
        for n in active[:k]:                       # knock one active term out
            d = Design(w=dict(base["w"]), hp=dict(base["hp"])); d["w"][n] = 0.0; out.append(d)
        for n in inactive:                         # or switch one dormant term on
            if len(out) >= k: break
            d = Design(w=dict(base["w"]), hp=dict(base["hp"])); d["w"][n] = 1.0; out.append(d)
        while len(out) < k:
            out.append(base.mutate(rng, n_changes=1))
        return out

    def best(self):
        return max(self.hist, key=lambda h: h["fitness"]) if self.hist else None

    def evaluate(self, designs, pool=None):
        seed, ni = self.cfg["seed"], self.cfg["n_iters"]
        args = [(dict(d), seed, ni) for d in designs]
        res = pool.map(_work, args) if pool else [_work(a) for a in args]
        with open(f"{self.path}/history.jsonl", "a") as f:
            for d, r in zip(designs, res):
                rec = dict(i=len(self.hist), design=dict(d), **r)
                self.hist.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return res

    def curve(self):
        b, out = -1e9, []
        for h in self.hist:
            b = max(b, h["fitness"]); out.append(round(b, 3))
        return out


# ---------------- request / response for out-of-band proposers ----------------
DIAG_KEYS = ["fitness", "fall_rate", "ep_seconds", "speed_mps", "airborne_frac",
             "action_sat", "mean_height", "mean_abs_pitch", "task_curve"]


def write_request(run):
    """Emit the JSON the out-of-band proposer answers. `anon` hides what the
    reward terms mean; `feedback=False` hides every result so far."""
    anon, fb = run.cfg.get("anon", False), run.cfg.get("feedback", True)
    seed = run.cfg["seed"]
    gen = len(run.hist) // run.cfg["gen"]
    k = min(run.cfg["gen"], run.cfg["budget"] - len(run.hist)) if fb else run.cfg["budget"]
    to_anon, _ = anon_maps(seed)
    hist = []
    if fb:
        for h in run.hist:
            d = Design(h["design"])
            rec = dict(i=h["i"], design=anonymise(d, seed) if anon else dict(d),
                       **{key: h[key] for key in DIAG_KEYS})
            rec["term_contrib"] = ({to_anon[kk]: vv for kk, vv in h["term_contrib"].items()}
                                   if anon else h["term_contrib"])
            if not anon:
                rec["failure_mode"] = h["failure_mode"]
            hist.append(rec)
    terms = sorted(to_anon.values()) if anon else TERM_NAMES
    req = dict(run=run.path, gen=gen, k=k, anon=anon, feedback=fb, terms=terms,
               weight_levels=W_LEVELS, hp_levels=HP_LEVELS,
               task=("unknown control task" if anon else
                     "planar one-legged hopper; fitness = metres travelled in 2.4 s before falling"),
               history=hist)
    p = f"{run.path}/req_{gen:02d}.json"
    json.dump(req, open(p, "w"), ensure_ascii=False, indent=1)
    return p


def read_response(run, gen):
    p = f"{run.path}/resp_{gen:02d}.json"
    if not os.path.exists(p): return None
    raw = json.load(open(p))
    seed = run.cfg["seed"]
    return [(deanonymise(d, seed) if run.cfg.get("anon") else Design(d)).sanitize()
            for d in raw["designs"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "run", "request", "step", "show"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--gen", type=int, default=4)
    ap.add_argument("--iters", type=int, default=N_ITERS_DEFAULT)
    ap.add_argument("--anon", action="store_true")
    ap.add_argument("--nofb", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()

    if a.cmd == "init":
        r = Run.init(a.run, a.arm, a.seed, a.budget, a.gen, a.iters)
        r.cfg["anon"] = a.anon; r.cfg["feedback"] = not a.nofb
        json.dump(r.cfg, open(f"{a.run}/config.json", "w"), indent=1)
        print("init", a.run, r.cfg)

    elif a.cmd == "run":                      # automatic arms only
        r = Run(a.run); rng = np.random.default_rng(r.cfg["seed"] + 777)
        with mp.Pool(a.procs) as pool:
            while len(r.hist) < r.cfg["budget"]:
                k = min(r.cfg["gen"], r.cfg["budget"] - len(r.hist))
                r.evaluate(r.propose(k, rng), pool)
                print(f"{a.run} {len(r.hist)}/{r.cfg['budget']} best={r.best()['fitness']:.3f}", flush=True)

    elif a.cmd == "request":
        r = Run(a.run)
        print(write_request(r))

    elif a.cmd == "step":
        r = Run(a.run)
        gen = len(r.hist) // r.cfg["gen"]
        ds = read_response(r, gen)
        if ds is None: sys.exit(f"missing {a.run}/resp_{gen:02d}.json")
        with mp.Pool(a.procs) as pool:
            r.evaluate(ds, pool)
        print(f"{a.run} {len(r.hist)}/{r.cfg['budget']} best={r.best()['fitness']:.3f}")

    elif a.cmd == "show":
        r = Run(a.run); b = r.best()
        print(json.dumps(dict(n=len(r.hist), curve=r.curve(), best_fit=b["fitness"],
                              best_decoy=b["decoy_mass"], best=Design(b["design"]).pretty()),
                         ensure_ascii=False, indent=1))
