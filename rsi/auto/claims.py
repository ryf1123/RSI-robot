"""Claim registry + contradiction sweep.

Every number the project asserts is registered together with the *test* that
produced it. `sweep` re-runs every test against whatever data exists now and
reports which claims changed status. This is the mechanical version of the three
times this project had to hand-chase a number through the docs after n grew."""
import json, glob, os, datetime
import numpy as np
from scipy import stats as st
from . import data, stats as S
from .spec import Claim, RESEARCH_DIR, save


def run_test(t):
    """Execute a Test dict against current data. Returns (cmp_dict, p_raw)."""
    kind, root, metric = t["kind"], t.get("root", "runs"), t["metric"]
    if kind in ("mwu_greater", "mwu_less"):
        a = data.series(root, t["arm"], metric)
        b = data.series(root, t["baseline"], metric)
        alt = "greater" if kind == "mwu_greater" else "less"
        c = S.compare(a, b, alt=alt, alpha=t.get("alpha", 0.05))
        return c, c.get("p_raw", 1.0)
    if kind == "spearman_reported_vs_reeval":
        roots = t.get("roots", ["runs", "runs_dense", "runs_k4"])
        a, b = data.all_pairs(tuple(roots))
        if len(a) < 3: return dict(verdict="insufficient_data"), 1.0
        rho, p = st.spearmanr(a, b)
        return dict(n=len(a), rho=round(float(rho), 3), p_raw=round(float(p), 5),
                    mean_reported=round(float(a.mean()), 3),
                    mean_reeval=round(float(b.mean()), 3),
                    shrinkage=round(1 - float(b.mean()) / float(a.mean()), 3),
                    below_diagonal=int((b < a).sum())), float(p)
    if kind == "spearman_arm":
        a = data.series(t["root"], t["arm"], t["metric"])
        b = data.series(t["root"], t["arm"], t["metric2"])
        n = min(len(a), len(b))
        if n < 3: return dict(verdict="insufficient_data"), 1.0
        rho, p = st.spearmanr(a[:n], b[:n])
        return dict(n=n, rho=round(float(rho), 3), p_raw=round(float(p), 5)), float(p)
    raise ValueError(f"unknown test kind {kind}")


def all_claims():
    return [Claim(**json.load(open(p))) for p in sorted(glob.glob(f"{RESEARCH_DIR}/claims/*.json"))]


def families(cl):
    """Group by pre-registration: multiplicity correction is per declared family."""
    f = {}
    for c in cl: f.setdefault(c.prereg or "__unregistered__", []).append(c)
    return f


def sweep(write=True):
    """Re-run everything. Returns the list of claims whose status changed."""
    changed, out = [], []
    for fam, cl in families(all_claims()).items():
        results = [run_test(c.test) for c in cl]
        ps = [r[1] for r in results]
        adj = S.holm(ps) if fam != "__unregistered__" else ps
        for c, (cmp_, p), pa in zip(cl, results, adj):
            if c.superseded_by:
                c.status = "superseded"; c.evidence = dict(cmp_, p_adj=round(float(pa), 5),
                                                           superseded_by=c.superseded_by)
                out.append(c)
                if write: save(c)
                continue
            v = S.verdict(cmp_, pa) if c.test["kind"].startswith("mwu") else (
                "supported" if pa <= c.test.get("alpha", 0.05) else "not_supported")
            prev = c.status
            ev = dict(cmp_, p_adj=round(float(pa), 5), family=fam, family_size=len(cl),
                      checked=datetime.datetime.now().isoformat(timespec="seconds"))
            if prev not in ("untested",) and prev != v:
                c.history.append(dict(status=prev, evidence=c.evidence))
                changed.append((c.id, prev, v))
            c.status, c.evidence = v, ev
            out.append(c)
            if write: save(c)
    return changed, out


def report(cl=None):
    cl = cl or all_claims()
    order = {"supported": 0, "underpowered": 1, "not_supported": 2,
             "insufficient_data": 3, "superseded": 5}
    print(f"{'claim':<26}{'status':<18}{'effect':>26}{'p_adj':>9}  statement")
    print("-" * 132)
    for c in sorted(cl, key=lambda c: (order.get(c.status, 9), c.id)):
        e = c.evidence
        if "rho" in e:
            eff = f"rho={e['rho']:+.2f} (n={e['n']})"
        elif "diff" in e:
            eff = f"{e['mean_a']:.2f} vs {e['mean_b']:.2f} (d={e['cliffs_delta']:+.2f})"
        else:
            eff = "-"
        print(f"{c.id:<26}{c.status:<18}{eff:>26}{e.get('p_adj', float('nan')):>9.4f}  {c.statement[:56]}")
    n = len(cl)
    print("-" * 132)
    for k in ("supported", "underpowered", "not_supported", "insufficient_data", "superseded"):
        m = sum(1 for c in cl if c.status == k)
        if m: print(f"  {k:<18}{m:>3} / {n}")
