"""Inner loop: Augmented Random Search (ARS-V2) on a linear policy, plus the
evaluation that turns one design into (fitness, diagnostics).

The inner optimiser only ever sees the DESIGNED reward. The fitness reported to
the outer loop is the TASK metric (metres travelled before falling), which no
design can rewrite. That separation is what makes reward hacking visible."""
import os
import numpy as np
from .env import Hopper
from .rewards import reward_vector, TERM_NAMES
from .design import Design

N_ITERS, N_DIRS, TOP_DEFAULT = 120, 6, 3
# RSI_POLICY=mlp swaps the inner policy class for a 1-hidden-layer tanh net.
# Everything outside this module -- the design space, the decoys, the task metric,
# the outer loop -- is untouched, which is the point: it isolates the threat
# `external_validity_inner` to exactly one variable.
POLICY = os.environ.get("RSI_POLICY", "linear")
HIDDEN = int(os.environ.get("RSI_HIDDEN", "8"))
MLP_ITERS = int(os.environ.get("RSI_MLP_ITERS", "300"))
EVAL_EPISODES = 8
MAX_STEPS = 200  # 2.4 s at 83 Hz


class Normalizer:
    def __init__(self, n):
        self.n = np.zeros(n); self.mean = np.zeros(n); self.m2 = np.ones(n)
    def observe(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
    def __call__(self, x):
        return (x - self.mean) / np.sqrt(np.maximum(self.m2 / np.maximum(self.n, 1), 1e-8))


def act(M, x):
    """Linear policy, or a 1-hidden-layer tanh MLP when M is a (W1, W2) pair.
    Kept in one place so the outer-loop code never needs to know which is in use."""
    if isinstance(M, tuple):
        W1, W2 = M
        return W2 @ np.tanh(W1 @ x)
    return M @ x


def _perturb(M, d, s):
    if isinstance(M, tuple):
        return (M[0] + s * d[0], M[1] + s * d[1])
    return M + s * d


def rollout(env, M, norm, weights, train=True, max_steps=MAX_STEPS):
    o = env.reset()
    R = 0.0; rvec = np.zeros(len(TERM_NAMES)); steps = 0
    diag = dict(air=0, sat=0, z=0.0, pitch=0.0)
    while True:
        if train: norm.observe(o)
        u = act(M, norm(o))
        o, s, done, fell = env.step(u)
        rv = reward_vector(s, env.t)
        rvec += rv
        R += float(weights @ rv)
        steps += 1
        diag["air"] += s["airborne"]; diag["sat"] += float(np.mean(np.abs(np.clip(u, -1, 1)) > 0.95))
        diag["z"] += s["z"]; diag["pitch"] += abs(s["pitch"])
        if done: break
    return R, dict(x=float(env.d.qpos[0]), steps=steps, fell=bool(fell),
                   rvec=rvec, **{k: v / steps for k, v in diag.items()})


def train_and_eval(design, seed=0, n_iters=N_ITERS, n_dirs=N_DIRS):
    design = Design(design)
    w = design.weight_vec
    hp = design["hp"]
    rng = np.random.default_rng(seed)
    env = Hopper(term_height=hp["term_height"], max_steps=MAX_STEPS, seed=seed)
    if POLICY == "mlp":
        n_iters = MLP_ITERS if n_iters == N_ITERS else n_iters
        M = (rng.standard_normal((HIDDEN, Hopper.obs_dim)) / np.sqrt(Hopper.obs_dim),
             np.zeros((Hopper.act_dim, HIDDEN)))
    else:
        M = np.zeros((Hopper.act_dim, Hopper.obs_dim))
    norm = Normalizer(Hopper.obs_dim)
    top = max(1, int(round(hp["top_frac"] * n_dirs)))
    curve = []
    for it in range(n_iters):
        if isinstance(M, tuple):
            deltas = [(rng.standard_normal(M[0].shape), rng.standard_normal(M[1].shape))
                      for _ in range(n_dirs)]
        else:
            deltas = rng.standard_normal((n_dirs, *M.shape))
        rp, rm = np.zeros(n_dirs), np.zeros(n_dirs)
        for i, dlt in enumerate(deltas):
            rp[i] = rollout(env, _perturb(M, dlt, hp["noise_std"]), norm, w)[0]
            rm[i] = rollout(env, _perturb(M, dlt, -hp["noise_std"]), norm, w)[0]
        order = np.argsort(-np.maximum(rp, rm))[:top]
        sr = np.concatenate([rp[order], rm[order]]).std() + 1e-6
        step = hp["step_size"] / (top * sr)
        if isinstance(M, tuple):
            M = (M[0] + step * sum((rp[i] - rm[i]) * deltas[i][0] for i in order),
                 M[1] + step * sum((rp[i] - rm[i]) * deltas[i][1] for i in order))
        else:
            M += step * np.einsum("i,ijk->jk", rp[order] - rm[order], deltas[order])
        if (it + 1) % max(1, n_iters // 5) == 0:
            xs = [rollout(env, M, norm, w, train=False)[1]["x"] for _ in range(2)]
            curve.append(round(float(np.mean(xs)), 3))

    # ---- final evaluation on the TASK metric --------------------------------
    eenv = Hopper(term_height=0.7, max_steps=MAX_STEPS, seed=seed + 10_000)
    res = [rollout(eenv, M, norm, w, train=False)[1] for _ in range(EVAL_EPISODES)]
    xs = np.array([r["x"] for r in res])
    contrib = w * np.mean([r["rvec"] / r["steps"] for r in res], axis=0)
    fit = float(xs.mean())
    d = dict(
        fitness=round(fit, 3), fitness_std=round(float(xs.std()), 3),
        fall_rate=round(float(np.mean([r["fell"] for r in res])), 3),
        ep_seconds=round(float(np.mean([r["steps"] for r in res])) * eenv.dt, 2),
        speed_mps=round(fit / max(1e-6, float(np.mean([r["steps"] for r in res])) * eenv.dt), 3),
        airborne_frac=round(float(np.mean([r["air"] for r in res])), 3),
        action_sat=round(float(np.mean([r["sat"] for r in res])), 3),
        mean_height=round(float(np.mean([r["z"] for r in res])), 3),
        mean_abs_pitch=round(float(np.mean([r["pitch"] for r in res])), 3),
        task_curve=curve,
        term_contrib={n: round(float(c), 4) for n, c in zip(TERM_NAMES, contrib) if abs(c) > 1e-6},
        decoy_mass=round(float(design.decoy_mass()), 3),
    )
    d["failure_mode"] = failure_mode(d)
    return d


def failure_mode(d):
    if d["fitness"] < -0.15: return "在倒退：策略学会了往后走"
    if d["ep_seconds"] < 0.4: return f"立刻摔倒（{d['ep_seconds']}s），起步就失稳"
    if d["fall_rate"] > 0.5 and d["fitness"] < 0.5: return f"撑不住：{d['ep_seconds']}s 后摔倒，只走了 {d['fitness']}m"
    if abs(d["fitness"]) < 0.15 and d["fall_rate"] < 0.3: return "站着不动：不摔但也不前进"
    if d["fall_rate"] < 0.2 and d["fitness"] > 1.0: return "站得住且在前进"
    if d["fall_rate"] > 0.5: return f"能走 {d['fitness']}m 但会摔（{d['ep_seconds']}s）"
    return "中间状态：慢速前进"
