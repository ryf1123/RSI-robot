"""The design space the outer loop searches over: 17 reward weights on a fixed
6-level grid, plus 4 training/environment hyper-parameters. Everything is
discrete so that "uniform random" is a well-defined, fair lower-bound baseline."""
import json, hashlib
import numpy as np
from .rewards import TERM_NAMES, KIND, DECOYS

W_LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
HP_LEVELS = {
    "step_size":   [0.005, 0.01, 0.02, 0.05],
    "noise_std":   [0.01, 0.02, 0.05, 0.1],
    "top_frac":    [0.25, 0.5, 1.0],
    "term_height": [0.6, 0.7, 0.8],
}
HP_NAMES = list(HP_LEVELS)
SPACE_SIZE = len(W_LEVELS) ** len(TERM_NAMES) * int(np.prod([len(v) for v in HP_LEVELS.values()]))


class Design(dict):
    """{'w': {term: weight}, 'hp': {name: value}}"""

    @staticmethod
    def random(rng):
        return Design(w={n: float(rng.choice(W_LEVELS)) for n in TERM_NAMES},
                      hp={k: float(rng.choice(v)) for k, v in HP_LEVELS.items()})

    @staticmethod
    def zeros():
        return Design(w={n: 0.0 for n in TERM_NAMES},
                      hp={k: float(v[len(v) // 2]) for k, v in HP_LEVELS.items()})

    def mutate(self, rng, n_changes=2):
        d = Design(w=dict(self["w"]), hp=dict(self["hp"]))
        keys = TERM_NAMES + HP_NAMES
        for k in rng.choice(keys, size=n_changes, replace=False):
            if k in TERM_NAMES:
                cur = W_LEVELS.index(d["w"][k]) if d["w"][k] in W_LEVELS else 0
                step = int(rng.choice([-2, -1, 1, 2]))
                d["w"][k] = W_LEVELS[int(np.clip(cur + step, 0, len(W_LEVELS) - 1))]
            else:
                lv = HP_LEVELS[k]
                cur = lv.index(d["hp"][k]) if d["hp"][k] in lv else 0
                d["hp"][k] = lv[int(np.clip(cur + rng.choice([-1, 1]), 0, len(lv) - 1))]
        return d

    def sanitize(self):
        """Snap arbitrary (e.g. LLM-authored) values onto the legal grid."""
        w = {n: float(min(W_LEVELS, key=lambda L: abs(L - float(self.get("w", {}).get(n, 0.0)))))
             for n in TERM_NAMES}
        hp = {k: float(min(v, key=lambda L: abs(L - float(self.get("hp", {}).get(k, v[0])))))
              for k, v in HP_LEVELS.items()}
        return Design(w=w, hp=hp)

    @property
    def weight_vec(self):
        return np.array([self["w"][n] for n in TERM_NAMES])

    def decoy_mass(self):
        tot = sum(abs(v) for v in self["w"].values())
        if tot == 0: return float("nan")
        return sum(abs(self["w"][n]) for n in DECOYS) / tot

    def key(self):
        return hashlib.md5(json.dumps(self, sort_keys=True).encode()).hexdigest()[:10]

    def pretty(self):
        act = {k: v for k, v in self["w"].items() if v > 0}
        return " ".join(f"{k}={v:g}" for k, v in sorted(act.items(), key=lambda kv: -kv[1])) \
            + " | " + " ".join(f"{k}={v:g}" for k, v in self["hp"].items())


# ---- anonymisation --------------------------------------------------------
# Term names are replaced by t00..t16 under a PER-RUN RANDOM PERMUTATION, so that
# the proposer cannot carry knowledge of "t07 = fwd_vel" from one seed to the next.
# The permutation lives in the run's config.json and must not be read while
# answering an anonymous request.
def anon_maps(seed):
    perm = np.random.default_rng(1000 + seed).permutation(len(TERM_NAMES))
    to = {n: f"t{perm[i]:02d}" for i, n in enumerate(TERM_NAMES)}
    return to, {v: k for k, v in to.items()}

def anonymise(d, seed=0):
    to, _ = anon_maps(seed)
    return {"w": {to[k]: v for k, v in d["w"].items()}, "hp": dict(d["hp"])}

def deanonymise(d, seed=0):
    _, fr = anon_maps(seed)
    return Design(w={fr[k]: v for k, v in d["w"].items() if k in fr}, hp=dict(d["hp"]))
