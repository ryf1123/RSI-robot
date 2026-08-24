"""The two ZERO-FEEDBACK proposers.

These are the arms that get no results at all, so everything they produce is
prior. Writing that prior down as code (instead of asking the model 128 times)
makes it auditable and reproducible -- and the prior is the object of study, so
it should be inspectable.

  structural_prior : what a model proposes when the terms are anonymous.
                     No semantics available, but still a real prior: prefer
                     SPARSE designs with moderate weights and middling
                     hyper-parameters, rather than a uniform draw.
  semantic_prior   : what a model proposes when it can read the term names and
                     the task description. This is where robotics knowledge
                     enters, and it is the thing the bullseye experiment prices.

Note on honesty: `semantic_prior` deliberately keeps a 10% chance of putting a
small weight on a decoy. A model does not have perfect knowledge either -- e.g.
`foot_press` sounds like it might buy ground stability, and nothing in the term
list says `y_drift` is identically zero on a planar model.
"""
import numpy as np
from ..design import Design, W_LEVELS, HP_LEVELS
from ..rewards import TERM_NAMES

# my ranked belief over the named terms: (term, P(active), weight choices)
BELIEF = [
    ("fwd_vel",    0.98, [1.0, 2.0, 4.0]),
    ("alive",      0.90, [0.25, 0.5, 1.0]),
    ("height",     0.75, [0.5, 1.0, 2.0]),
    ("upright",    0.60, [0.25, 0.5, 1.0]),
    ("ctrl_cost",  0.55, [0.25, 0.5]),
    ("smooth",     0.40, [0.25, 0.5]),
    ("pitch_rate", 0.30, [0.25, 0.5]),
    ("hop",        0.30, [0.25, 0.5, 1.0]),
    ("foot_clear", 0.30, [0.25, 0.5]),
    ("energy",     0.25, [0.25, 0.5]),
    ("joint_lim",  0.25, [0.25, 0.5]),
    ("progress",   0.20, [0.5, 1.0]),
    ("back_vel",   0.03, [0.25]),
    ("stand_still", 0.05, [0.25]),
    ("foot_press", 0.10, [0.25, 0.5]),
    ("y_drift",    0.08, [0.25]),
    ("clock",      0.05, [0.25]),
]
# the dense-decoy variant's extra terms; a model reading these names should also
# rank them near-zero for "travel forward"
BELIEF += [("crouch", 0.04, [0.25]), ("freeze_joints", 0.06, [0.25]),
           ("backward_pos", 0.03, [0.25]), ("max_ctrl", 0.04, [0.25]),
           ("foot_glue", 0.10, [0.25, 0.5]), ("const_zero", 0.06, [0.25]),
           ("clock2", 0.05, [0.25])]

HP_BELIEF = {"step_size": [0.01, 0.02, 0.02, 0.05], "noise_std": [0.02, 0.02, 0.05, 0.05],
             "top_frac": [0.25, 0.5, 0.5], "term_height": [0.6, 0.7, 0.7]}


def semantic_prior(rng):
    d = Design.zeros()
    for name, p, levels in BELIEF:
        if name not in d["w"]: continue
        d["w"][name] = float(rng.choice(levels)) if rng.random() < p else 0.0
    if d["w"]["fwd_vel"] == 0 and d["w"]["progress"] == 0:
        d["w"]["fwd_vel"] = 2.0                       # never propose "go nowhere"
    for k, v in HP_BELIEF.items():
        d["hp"][k] = float(rng.choice(v))
    return d


def structural_prior(rng):
    """Sparse + moderate. No idea what any term means."""
    d = Design.zeros()
    k = int(rng.integers(3, 7))
    for name in rng.choice(TERM_NAMES, size=k, replace=False):
        d["w"][name] = float(rng.choice([0.25, 0.5, 1.0, 1.0, 2.0]))
    for key, lv in HP_LEVELS.items():
        d["hp"][key] = float(rng.choice(lv[max(0, len(lv) // 2 - 1):len(lv) // 2 + 1]))
    return d
