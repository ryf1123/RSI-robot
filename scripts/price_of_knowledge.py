"""Ring 8: what would it cost to BUY, by experiment, the knowledge the semantic
prior hands over for free?

Paired A/B per term against a fixed base design, same seeds on both sides.
The k-ladder is computed as nested subsets, so the whole thing costs
16 + 17*16 = 288 inner trainings."""
import json, numpy as np, multiprocessing as mp
from scipy import stats
from rsi.design import Design
from rsi.inner import train_and_eval
from rsi.rewards import TERM_NAMES, KIND

SEEDS = list(range(3000, 3016))          # 16 paired seeds
BASE = Design.zeros()
BASE["w"].update(fwd_vel=2.0, alive=0.5, height=1.0)
BASE["hp"].update(step_size=0.02, noise_std=0.05, top_frac=0.25, term_height=0.7)
ADD_W = 2.0

def work(a):
    tag, d, s = a
    return tag, s, train_and_eval(d, seed=s, n_iters=120)["fitness"]

if __name__ == "__main__":
    jobs = [("__base__", dict(BASE), s) for s in SEEDS]
    for n in TERM_NAMES:
        if BASE["w"][n] > 0: continue
        d = Design(w=dict(BASE["w"]), hp=dict(BASE["hp"])); d["w"][n] = ADD_W
        jobs += [(n, dict(d), s) for s in SEEDS]
    print(len(jobs), "trainings", flush=True)
    res = {}
    with mp.Pool(6) as p:
        for tag, s, f in p.map(work, jobs):
            res.setdefault(tag, {})[s] = f
    json.dump({k: {str(s): v for s, v in d.items()} for k, d in res.items()},
              open("runs/price_of_knowledge.json", "w"), indent=1)

    base = np.array([res["__base__"][s] for s in SEEDS])
    terms = [n for n in res if n != "__base__"]
    print(f"\nbase design: mean {base.mean():.2f}  std {base.std(ddof=1):.2f}")
    print(f"\n{'k':>3} {'判为有害的项':<44}{'TP':>4}{'FP':>4}{'召回':>7}{'精确率':>8}{'总训练次数':>10}")
    harmful = {n for n in TERM_NAMES if KIND[n] == "harmful"}
    for k in [1, 2, 4, 8, 16]:
        ss = SEEDS[:k]
        flagged = []
        for n in terms:
            d = np.array([res[n][s] - res["__base__"][s] for s in ss])
            if k == 1:
                if d[0] < -0.3: flagged.append(n)
            else:
                if stats.wilcoxon(d, alternative="less").pvalue < 0.05: flagged.append(n)
        tp = len([n for n in flagged if n in harmful]); fp = len(flagged) - tp
        print(f"{k:>3} {','.join(sorted(flagged))[:43]:<44}{tp:>4}{fp:>4}"
              f"{tp/len(harmful):>7.2f}{(tp/len(flagged) if flagged else float('nan')):>8.2f}"
              f"{(1+len(terms))*k:>10}")
    print(f"\n外层循环一整轮的预算 = 32 次训练")
    print("\n每一项的配对差值（16 个种子）：")
    for n in sorted(terms, key=lambda n: np.mean([res[n][s]-res["__base__"][s] for s in SEEDS])):
        d = np.array([res[n][s] - res["__base__"][s] for s in SEEDS])
        print(f"  {n:<14} {KIND[n]:<8} Δ={d.mean():+6.2f} ± {d.std(ddof=1):.2f}   "
              f"wilcoxon p={stats.wilcoxon(d, alternative='less').pvalue:.4f}")
