"""Honest reporting layer.

Three rules are enforced here rather than left to the writer, because this
project broke all three by hand at least once:
  1. report an EFFECT SIZE with an interval, never a bare p-value;
  2. correct for the size of the pre-registered family, not for what you looked at;
  3. if the comparison is underpowered for the effect you care about, say
     `underpowered` -- never `no difference`.
"""
import numpy as np
from scipy import stats as st


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        m = float(v.mean()) if len(v) else float("nan")
        return m, m, m
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def cliffs_delta(a, b):
    """Non-parametric effect size in [-1, 1]; 0 means the two are interleaved."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b): return float("nan")
    gt = sum((x > b).sum() for x in a); lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def min_detectable(a, b, power=0.80, alpha=0.05):
    """Given the observed spread and n, what is the smallest difference this
    comparison could have detected? If |observed| < this, the honest verdict is
    `underpowered`, not `no difference`."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2: return float("nan")
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    z = st.norm.ppf(1 - alpha) + st.norm.ppf(power)
    return float(z * sp * np.sqrt(1 / len(a) + 1 / len(b)))


def required_n(delta, sd, power=0.80, alpha=0.05):
    z = st.norm.ppf(1 - alpha) + st.norm.ppf(power)
    return int(np.ceil(2 * (z * sd / delta) ** 2)) if delta else -1


def compare(a, b, alt="greater", alpha=0.05):
    """One comparison, fully described."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return dict(n_a=len(a), n_b=len(b), verdict="insufficient_data")
    p = st.mannwhitneyu(a, b, alternative=alt).pvalue
    ma, la, ha = boot(a); mb, lb, hb = boot(b)
    mde = min_detectable(a, b, alpha=alpha)
    obs = ma - mb
    return dict(n_a=len(a), n_b=len(b),
                mean_a=round(ma, 3), ci_a=[round(la, 3), round(ha, 3)],
                mean_b=round(mb, 3), ci_b=[round(lb, 3), round(hb, 3)],
                diff=round(obs, 3), cliffs_delta=round(cliffs_delta(a, b), 3),
                p_raw=round(float(p), 5),
                min_detectable_diff=round(mde, 3),
                intervals_overlap=bool(la <= hb and lb <= ha))


def verdict(cmp, p_adj, alpha=0.05):
    """The only place allowed to turn numbers into a word."""
    if cmp.get("verdict") == "insufficient_data":
        return "insufficient_data"
    if p_adj <= alpha:
        return "supported"
    if abs(cmp["diff"]) < cmp["min_detectable_diff"]:
        return "underpowered"
    return "not_supported"
