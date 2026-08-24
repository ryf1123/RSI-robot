"""Ring 5: fixed inner-training budget B=32, split as (n_designs x k_seeds).
Every arm draws designs from the SAME uniform distribution, so proposal quality
is identical by construction and only the budget split differs."""
import json, os, numpy as np, multiprocessing as mp, sys
from rsi.design import Design
from rsi.inner import train_and_eval

B = 32
ARMS = {"k1": 1, "k2": 2, "k4": 4, "k8": 8}
REEVAL = [100, 101, 102, 103, 104, 105]

def work(a):
    tag, d, s = a
    return tag, train_and_eval(d, seed=s, n_iters=120)["fitness"]

def run(rep):
    """one repetition of every arm; designs are re-drawn per arm from the same rng stream"""
    jobs, book = [], {}
    for arm, k in ARMS.items():
        n = B // k
        rng = np.random.default_rng(10_000 + rep * 97 + hash(arm) % 1000)
        ds = [Design.random(rng) for _ in range(n)]
        book[arm] = ds
        for i, d in enumerate(ds):
            for j in range(k):
                jobs.append((f"{arm}|{i}|{j}", dict(d), 5000 + rep * 31 + j))
    return jobs, book

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    out = []
    with mp.Pool(6) as pool:
        for rep in range(reps):
            jobs, book = run(rep)
            res = {}
            for tag, f in pool.map(work, jobs):
                arm, i, j = tag.split("|"); res.setdefault((arm, int(i)), []).append(f)
            # pick each arm's elite by its own (mean-of-k) criterion, then re-evaluate
            rj, elite = [], {}
            for arm in ARMS:
                sc = {i: np.mean(res[(arm, i)]) for (a, i) in res if a == arm}
                best = max(sc, key=sc.get)
                elite[arm] = (book[arm][best], float(sc[best]))
                rj += [(f"{arm}", dict(book[arm][best]), s) for s in REEVAL]
            rr = {}
            for tag, f in pool.map(work, rj): rr.setdefault(tag, []).append(f)
            row = {arm: dict(reported=elite[arm][1], reeval=float(np.mean(rr[arm])),
                             design=dict(elite[arm][0])) for arm in ARMS}
            out.append(row)
            print(f"rep {rep}: " + "  ".join(
                f"{a} rep={row[a]['reported']:5.2f} re={row[a]['reeval']:5.2f}" for a in ARMS), flush=True)
            json.dump(out, open("runs/budget_split.json", "w"), indent=1)
