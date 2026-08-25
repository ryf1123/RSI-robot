"""RSI-AutoResearch: an autonomous research loop with a built-in epistemic
discipline layer.

The object of study is a DESIGN SPACE (here: robot reward functions). The system
does not just search it -- it maintains a registry of what it believes, what
could still be wrong, and what the next most informative experiment is, and it
refuses to let the four failure modes that dominated this project happen again:

    unreadable curves      -> a noise-floor probe is a gate, not an option
    winner's curse         -> elites are re-evaluated on fresh seeds before any number is reported
    thin baselines         -> the denominator is scheduled first and 2-3x deeper
    forking paths          -> a comparison needs a pre-registration with a declared family

and it distinguishes `underpowered` from `no difference`, which is the single
most common way an honest researcher lies to themselves.
"""
from . import spec, data, stats, claims, probes, planner, threats, driver  # noqa: F401
