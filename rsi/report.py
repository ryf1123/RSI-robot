"""Aggregate runs into the two headline numbers, each with a bootstrap CI and
each with its floor and ceiling written next to it. Overlapping intervals are
not a difference."""
import json, os, glob, argparse
import numpy as np
from .design import Design
from .rewards import CHANCE_DECOY_MASS

ARMS = ["random", "evo", "llm_anon_nofb", "llm_anon_fb", "llm_named_nofb", "llm_named_fb"]


def load(run):
    cfg = json.load(open(f"{run}/config.json"))
    hist = [json.loads(l) for l in open(f"{run}/history.jsonl")]
    return cfg, hist


def boot(vals, n=10000, rng=None):
    rng = rng or np.random.default_rng(0)
    v = np.asarray(vals, float)
    if len(v) < 2: return (float(v.mean()), float(v.mean()), float(v.mean()))
    bs = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def arm_stats(arm, root="runs", budget=None):
    runs = sorted(glob.glob(f"{root}/{arm}_s*"))
    best_fit, best_decoy, curves, all_decoy, reeval = [], [], [], [], []
    for r in runs:
        cfg, hist = load(r)
        if budget: hist = hist[:budget]
        if not hist: continue
        b = max(hist, key=lambda h: h["fitness"])
        best_fit.append(b["fitness"])
        best_decoy.append(b["decoy_mass"])
        all_decoy += [h["decoy_mass"] for h in hist if h["decoy_mass"] == h["decoy_mass"]]
        c, m = [], -1e9
        for h in hist:
            m = max(m, h["fitness"]); c.append(m)
        curves.append(c)
        rp = f"{r}/reeval.json"
        if os.path.exists(rp): reeval.append(json.load(open(rp))["mean"])
    if not best_fit: return None
    L = min(len(c) for c in curves)
    return dict(arm=arm, n_runs=len(best_fit),
                fit=boot(best_fit), decoy_best=boot(best_decoy), decoy_all=boot(all_decoy),
                curve=np.mean([c[:L] for c in curves], axis=0).round(3).tolist(),
                reeval=boot(reeval) if reeval else None)


def table(root="runs", budget=None):
    rows = [s for a in ARMS if (s := arm_stats(a, root, budget))]
    w = "{:<16}{:>3}  {:>22}  {:>22}  {:>22}"
    print(w.format("arm", "n", "best fitness (m)", "elite decoy mass", "re-eval fitness (6 seeds)"))
    print("-" * 90)
    for s in rows:
        f = "{:.2f} [{:.2f},{:.2f}]".format(*s["fit"])
        d = "{:.3f} [{:.3f},{:.3f}]".format(*s["decoy_best"])
        r = "{:.2f} [{:.2f},{:.2f}]".format(*s["reeval"]) if s["reeval"] else "-"
        print(w.format(s["arm"], s["n_runs"], f, d, r))
    print("-" * 90)
    from .rewards import TERM_NAMES, DECOYS as _D
    print(f"decoy mass: floor 0.000 (perfect) / chance {CHANCE_DECOY_MASS:.3f} ({len(_D)} of {len(TERM_NAMES)} terms)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = table(a.root, a.budget)
    if a.json: json.dump(rows, open(a.json, "w"), indent=1)
