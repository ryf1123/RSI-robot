"""Re-evaluate each run's elite over 6 fresh inner seeds. The gap between the
fitness the outer loop reported and this number is the winner's curse."""
import os, json, glob, sys, numpy as np, multiprocessing as mp
from rsi.inner import train_and_eval
SEEDS = [100, 101, 102, 103, 104, 105]
def work(a):
    tag, d, s = a; return tag, s, train_and_eval(d, seed=s, n_iters=120)["fitness"]
if __name__ == "__main__":
    jobs, elite = [], {}
    for r in sorted(glob.glob(os.environ.get("RSI_ROOT","runs")+"/*_s*")):
        try: hist = [json.loads(l) for l in open(f"{r}/history.jsonl")]
        except FileNotFoundError: continue
        if not hist: continue
        b = max(hist, key=lambda h: h["fitness"]); elite[r] = b
        jobs += [(r, b["design"], s) for s in SEEDS]
    print(len(jobs), "jobs", flush=True)
    out = {}
    with mp.Pool(6) as p:
        for tag, s, f in p.imap_unordered(work, jobs):
            out.setdefault(tag, []).append(f)
    for r, v in out.items():
        json.dump(dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)), vals=v,
                       reported=elite[r]["fitness"]), open(f"{r}/reeval.json", "w"), indent=1)
        print(f"{r:32s} reported={elite[r]['fitness']:.2f} re-eval={np.mean(v):.2f}±{np.std(v,ddof=1):.2f}")
