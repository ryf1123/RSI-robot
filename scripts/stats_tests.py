"""Every headline comparison in one place, with the multiple-comparison correction
written next to it. One-sided Mann-Whitney against `random`, plus Holm across the
family of comparisons in each block."""
import json, glob, os, numpy as np, sys
from scipy import stats
from rsi.report import boot

ARMS = ["random", "evo", "llm_anon_nofb", "llm_anon_fb", "llm_named_nofb", "llm_named_fb"]

def vals(root, arm, key):
    out = []
    for r in sorted(p for p in glob.glob(f"{root}/{arm}_s*") if os.path.isdir(p)):
        if key == "reeval":
            try: out.append(json.load(open(f"{r}/reeval.json"))["mean"])
            except FileNotFoundError: pass
        else:
            try: h = [json.loads(l) for l in open(f"{r}/history.jsonl")]
            except FileNotFoundError: continue
            if not h: continue
            out.append(max(x["fitness"] for x in h) if key == "best"
                       else max(h, key=lambda x: x["fitness"])["decoy_mass"])
    return np.array(out, float)

def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj

def block(root, key, alt="greater", label=""):
    print(f"\n### {label or key}  ({root})")
    base = vals(root, "random", key)
    names, ps = [], []
    for a in ARMS:
        v = vals(root, a, key)
        if len(v) == 0: continue
        m, lo, hi = boot(v)
        line = f"  {a:<16} n={len(v)}  mean={m:5.2f} [{lo:5.2f},{hi:5.2f}]  median={np.median(v):5.2f}"
        if a != "random":
            p = stats.mannwhitneyu(v, base, alternative=alt).pvalue
            names.append(a); ps.append(p); line += f"   MWU vs random p={p:.4f}"
        print(line)
    if ps:
        adj = holm(np.array(ps))
        print("  Holm-corrected: " + "  ".join(f"{n}={a:.3f}" for n, a in zip(names, adj)))

if __name__ == "__main__":
    block("runs", "best", "greater", "best-so-far（外层报出来的数）")
    block("runs", "reeval", "greater", "精英重评（6 个新种子）")
    block("runs", "decoy", "less", "精英 decoy mass（越小越好）")
    if os.path.isdir("runs_dense"):
        block("runs_dense", "best", "greater", "密集诱饵空间 · best-so-far")
        block("runs_dense", "reeval", "greater", "密集诱饵空间 · 精英重评")
