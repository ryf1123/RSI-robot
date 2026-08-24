"""Ring 7 follow-ups: does the structural prior decompose into
(a) few terms, (b) no decoys, (c) mid-range hyper-parameters?"""
import json, glob, os, numpy as np
from scipy import stats
from rsi.report import boot

def el(arm, key="reeval"):
    o = []
    for r in sorted(p for p in glob.glob(f"runs/{arm}_s*") if os.path.isdir(p)):
        if key == "reeval":
            try: o.append(json.load(open(f"{r}/reeval.json"))["mean"])
            except FileNotFoundError: pass
        else:
            h = [json.loads(l) for l in open(f"{r}/history.jsonl")]
            if not h: continue
            b = max(h, key=lambda x: x["fitness"])
            o.append(b["fitness"] if key == "best" else b["decoy_mass"])
    return np.array(o)

def line(a):
    b, r, d = boot(el(a, "best")), boot(el(a)), boot(el(a, "decoy"))
    return f"{a:<18}{len(el(a,'best')):>3}{b[0]:7.2f} [{b[1]:.2f},{b[2]:.2f}]{r[0]:9.2f} [{r[1]:.2f},{r[2]:.2f}]{d[0]:8.3f}"

hdr = f"{'arm':<18}{'n':>3}{'best-so-far':>22}{'re-eval':>22}{'elite decoy':>16}"

print("## A. 诱饵 on/off，在固定稀疏度下\n" + hdr)
for k in [3, 5, 8, 12]:
    print(line(f"sparse{k}")); print(line(f"sparseclean{k}")); print()
sc = np.concatenate([el(f"sparseclean{k}") for k in [3,5,8,12]])
sp = np.concatenate([el(f"sparse{k}") for k in [3,5,8,12]])
print(f"  pooled sparseclean n={len(sc)} {sc.mean():.2f} {[round(v,2) for v in boot(sc)[1:]]}")
print(f"  pooled sparse      n={len(sp)} {sp.mean():.2f} {[round(v,2) for v in boot(sp)[1:]]}")
if len(sc) and len(sp):
    print("  MWU sparseclean > sparse p=%.4f" % stats.mannwhitneyu(sc, sp, alternative="greater").pvalue)
    K = [3,5,8,12]
    x = np.concatenate([[k]*len(el(f"sparseclean{k}")) for k in K]); y = np.concatenate([el(f"sparseclean{k}") for k in K])
    print("  spearman(k, re-eval | 无诱饵) =", [round(v,4) for v in stats.spearmanr(x, y)])

print("\n## B. 2x2：{稀疏, 稠密} x {中档超参, 均匀超参}\n" + hdr)
for a in ["random", "midhp", "sparse5", "sparse5mid", "llm_anon_nofb"]:
    if len(el(a, "best")): print(line(a))
if len(el("midhp")) and len(el("random")):
    print("\n  中档超参主效应: (midhp+sparse5mid) vs (random+sparse5)")
    mid = np.concatenate([el("midhp"), el("sparse5mid")]); uni = np.concatenate([el("random"), el("sparse5")])
    print(f"    {mid.mean():.2f} vs {uni.mean():.2f}  MWU p={stats.mannwhitneyu(mid,uni,alternative='greater').pvalue:.4f}")
    print("  稀疏主效应: (sparse5+sparse5mid) vs (random+midhp)")
    s = np.concatenate([el("sparse5"), el("sparse5mid")]); d = np.concatenate([el("random"), el("midhp")])
    print(f"    {s.mean():.2f} vs {d.mean():.2f}  MWU p={stats.mannwhitneyu(s,d,alternative='greater').pvalue:.4f}")
