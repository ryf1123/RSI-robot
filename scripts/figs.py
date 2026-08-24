"""Two figures: (1) best-so-far fitness vs evaluations with bootstrap bands,
(2) decoy mass with its floor and chance level drawn in."""
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rsi.report import arm_stats, ARMS, boot
from rsi.rewards import CHANCE_DECOY_MASS

LBL = {"random":"random（均匀采样）","evo":"evo（变异精英）",
       "llm_anon_nofb":"匿名·无反馈（结构先验）","llm_anon_fb":"匿名·有反馈",
       "llm_named_nofb":"命名·无反馈（语义先验）","llm_named_fb":"命名·有反馈"}
C = {"random":"#888","evo":"#4C9F70","llm_anon_nofb":"#C7A76C","llm_anon_fb":"#D2691E",
     "llm_named_nofb":"#6A8CBF","llm_named_fb":"#B03A48"}

def curves(arm):
    out=[]
    for r in sorted(glob.glob(f"runs/{arm}_s*")):
        try: h=[json.loads(l) for l in open(f"{r}/history.jsonl")]
        except FileNotFoundError: continue
        if not h: continue
        c,m=[],-1e9
        for x in h: m=max(m,x["fitness"]); c.append(m)
        out.append(c)
    if not out: return None
    L=min(len(c) for c in out)
    return np.array([c[:L] for c in out])

fig,ax=plt.subplots(1,2,figsize=(13,4.6))
for arm in ARMS:
    C_=curves(arm)
    if C_ is None: continue
    m=C_.mean(0); n=len(C_)
    rng=np.random.default_rng(0)
    bs=C_[rng.integers(0,n,(2000,n))].mean(1)
    lo,hi=np.percentile(bs,[2.5,97.5],axis=0)
    x=np.arange(1,C_.shape[1]+1)
    ax[0].plot(x,m,color=C[arm],label=f"{LBL[arm]} (n={n})",lw=1.8)
    ax[0].fill_between(x,lo,hi,color=C[arm],alpha=0.13,lw=0)
ax[0].set_xlabel("评估次数（每次 = 一次完整内层 ARS 训练）"); ax[0].set_ylabel("best-so-far 任务适应度（米）")
ax[0].set_title("① 任务适应度：所有臂的区间都重叠"); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.25)

rows=[s for a in ARMS if (s:=arm_stats(a))]
y=np.arange(len(rows))
for i,s in enumerate(rows):
    m,lo,hi=s["decoy_best"]
    ax[1].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt="o",color=C[s["arm"]],capsize=4,ms=7)
ax[1].axvline(0,color="k",ls="--",lw=1); ax[1].axvline(CHANCE_DECOY_MASS,color="r",ls=":",lw=1.4)
ax[1].text(0.004,len(rows)-0.4,"下限 0（完美）",fontsize=8)
ax[1].text(CHANCE_DECOY_MASS+0.005,len(rows)-0.4,f"随机 {CHANCE_DECOY_MASS:.3f}",fontsize=8,color="r")
ax[1].set_yticks(y); ax[1].set_yticklabels([LBL[s["arm"]] for s in rows],fontsize=8.5)
ax[1].set_xlabel("精英设计的诱饵权重占比"); ax[1].set_title("② 诱饵占比：命名臂立刻贴到 0")
ax[1].grid(alpha=.25,axis="x"); ax[1].invert_yaxis()
plt.rcParams["font.sans-serif"]=["PingFang SC","Heiti SC","Arial Unicode MS"]
plt.tight_layout(); plt.savefig("docs/figs/main.png",dpi=150)
print("docs/figs/main.png")
