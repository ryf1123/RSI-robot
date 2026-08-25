"""Budget planner.

Turns the measured noise floor into the two decisions an outer loop has to make
before it starts:

  how many SEEDS per arm  -- so a comparison can detect the effect you care about
  how to SPLIT the budget -- candidates x inner-seeds

The second one is not a constant. Ring 9 measured an interaction: with a weak
proposer, candidate count wins (few of your candidates are any good, so you need
volume); with a strong proposer, evaluation accuracy wins (most candidates are
decent, so what limits you is picking the right one)."""
import json, os
import numpy as np
from . import stats as S
from .spec import RESEARCH_DIR

PROBE = f"{RESEARCH_DIR}/probes/noise_floor.json"


def noise():
    nf = json.load(open(PROBE))
    good = max((k for k in nf if isinstance(nf[k], dict)), key=lambda k: nf[k]["mean"])
    return nf[good]["std"], nf[good]["mean"], good


def seeds_for(delta, sd=None, power=0.80, alpha=0.05):
    """How many runs per arm to detect a `delta`-metre difference."""
    sd = sd if sd is not None else noise()[0]
    return S.required_n(delta, sd, power, alpha)


def split(proposer_strength, budget=32):
    """candidates x inner-seeds, given how good the proposer is.

    `proposer_strength` in [0,1]: 0 = indistinguishable from uniform random,
    1 = most proposals already land in the good region. Calibrated on this
    project's two measured points (random -> k=1 fine; semantic prior -> k=4
    gives 2.3x)."""
    k = 1 if proposer_strength < 0.35 else (2 if proposer_strength < 0.6 else 4)
    return dict(inner_seeds=k, candidates=budget // k, budget=budget,
                rationale=("weak proposer: good designs are rare, buy volume" if k == 1
                           else "strong proposer: good designs are common, buy selection accuracy"))


def plan(delta_of_interest=0.5, budget=32, proposer_strength=0.7):
    sd, mean, which = noise()
    n = seeds_for(delta_of_interest, sd)
    sp = split(proposer_strength, budget)
    return dict(noise_std=sd, noise_ref_design=which, noise_mean=mean,
                delta_of_interest=delta_of_interest,
                seeds_per_arm=n,
                seeds_for_baseline=max(n, int(np.ceil(2.5 * n))),
                note_baseline=("the baseline is the denominator of every comparison; "
                               "give it 2-3x the seeds of any treatment arm and run it FIRST"),
                **sp)


def observed_sd(root, arm, metric="reeval"):
    from . import data
    v = data.series(root, arm, metric)
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def power_audit(comparisons, power=0.80, alpha=0.05):
    """For every comparison actually made: what could it have detected, and how
    many seeds would the effect of interest have required?

    This is the report the system should print BEFORE a project starts, not after."""
    from . import data
    rows = []
    for root, arm, base, metric, delta in comparisons:
        a, b = data.series(root, arm, metric), data.series(root, base, metric)
        if len(a) < 2 or len(b) < 2: continue
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append(dict(root=root, arm=arm, baseline=base, metric=metric,
                         n_a=len(a), n_b=len(b), sd=round(float(sd), 3),
                         observed_diff=round(float(a.mean() - b.mean()), 3),
                         min_detectable=round(S.min_detectable(a, b, power, alpha), 3),
                         n_needed_for_delta=S.required_n(delta, sd, power, alpha),
                         powered=bool(abs(a.mean() - b.mean()) >= S.min_detectable(a, b, power, alpha))))
    return rows
