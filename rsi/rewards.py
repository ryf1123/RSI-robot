"""Reward-term library. Each term is written so that a POSITIVE weight means
"do more of the thing the term's name says". Five of the seventeen are decoys
planted on purpose:

  HARMFUL : back_vel, stand_still, foot_press  -- positive weight actively hurts
  NULL    : y_drift, clock                     -- identically ~0 contribution

Because we know which is which, "how much weight mass sits on decoys" is a
metric with a known floor (0.0) and a known chance level (5/17 = 0.294).
That is the instrument this project is built around.
"""
import numpy as np
from .env import JOINT_RANGE

def _jnorm(qpos):
    q = qpos[3:6]
    lo, hi = JOINT_RANGE[:, 0], JOINT_RANGE[:, 1]
    return 2 * (q - lo) / (hi - lo) - 1  # -> [-1, 1]

TERMS = {}
def term(name, kind):
    def deco(f):
        TERMS[name] = (f, kind); return f
    return deco

# ---------------- useful ----------------
@term("fwd_vel", "useful")
def _(s, t): return np.clip(s["dx"], -3, 3) / 2.0

@term("alive", "useful")
def _(s, t): return 1.0

@term("height", "useful")
def _(s, t): return np.clip(s["z"] - 0.7, 0, 0.6) / 0.6

@term("upright", "useful")
def _(s, t): return 1.0 - abs(s["pitch"]) / 0.2

@term("ctrl_cost", "useful")
def _(s, t): return -float(np.sum(s["u"] ** 2)) / 3.0

@term("smooth", "useful")
def _(s, t): return -float(np.sum((s["u"] - s["prev_u"]) ** 2)) / 3.0

@term("joint_lim", "useful")
def _(s, t): return -float(np.mean(np.clip(np.abs(_jnorm(s["qpos"])) - 0.9, 0, None))) / 0.1

@term("foot_clear", "useful")
def _(s, t): return np.clip(s["foot_z"], 0, 0.3) / 0.3

@term("hop", "useful")
def _(s, t): return 1.0 if s["airborne"] else 0.0

@term("energy", "useful")
def _(s, t): return -float(np.mean(np.abs(s["u"] * s["qvel"][3:6]))) / 5.0

@term("pitch_rate", "useful")
def _(s, t): return -abs(s["qvel"][2]) / 5.0

@term("progress", "useful")
def _(s, t): return s["x"] / 5.0

# ---------------- harmful decoys ----------------
@term("back_vel", "harmful")
def _(s, t): return -np.clip(s["dx"], -3, 3) / 2.0

@term("stand_still", "harmful")
def _(s, t): return float(np.exp(-s["dx"] ** 2))

@term("foot_press", "harmful")
def _(s, t): return -np.clip(s["foot_z"], 0, 0.3) / 0.3

# ---------------- null decoys ----------------
@term("y_drift", "null")
def _(s, t): return -abs(float(s["qpos"][0] * 0.0))   # planar model: always 0

@term("clock", "null")
def _(s, t): return float(np.sin(2 * np.pi * t / 50.0))

# ---------------- dense-decoy variant (ring 5) ----------------
# RSI_SPACE=dense adds 7 more decoys, taking the library to 24 terms of which 12
# are decoys. Uniform-random decoy mass moves from 0.294 to 0.500 -- across the
# damage knee measured in ring 2.
import os
if os.environ.get("RSI_SPACE") == "dense":
    @term("crouch", "harmful")
    def _(s, t): return -np.clip(s["z"] - 0.7, 0, 0.6) / 0.6

    @term("freeze_joints", "harmful")
    def _(s, t): return -float(np.mean(np.abs(s["qvel"][3:6]))) / 5.0

    @term("backward_pos", "harmful")
    def _(s, t): return -s["x"] / 5.0

    @term("max_ctrl", "harmful")
    def _(s, t): return float(np.sum(s["u"] ** 2)) / 3.0

    @term("foot_glue", "harmful")
    def _(s, t): return 1.0 if not s["airborne"] else 0.0

    @term("const_zero", "null")
    def _(s, t): return 0.0

    @term("clock2", "null")
    def _(s, t): return float(np.cos(2 * np.pi * t / 37.0))

TERM_NAMES = list(TERMS.keys())
KIND = {k: v[1] for k, v in TERMS.items()}
DECOYS = [k for k, v in KIND.items() if v != "useful"]
HARMFUL = [k for k, v in KIND.items() if v == "harmful"]
CHANCE_DECOY_MASS = len(DECOYS) / len(TERM_NAMES)

def reward_vector(s, t):
    return np.array([TERMS[n][0](s, t) for n in TERM_NAMES], dtype=np.float64)
