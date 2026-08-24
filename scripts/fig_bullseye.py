import json,glob,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["PingFang SC","Heiti SC","Arial Unicode MS"]; plt.rcParams["axes.unicode_minus"]=False
from rsi.report import boot, ARMS
from rsi.rewards import CHANCE_DECOY_MASS
LBL={"random":"random（均匀采样）","evo":"evo（变异精英）","llm_anon_nofb":"匿名·无反馈",
     "llm_anon_fb":"匿名·有反馈","llm_named_nofb":"命名·无反馈","llm_named_fb":"命名·有反馈"}
C={"random":"#888","evo":"#4C9F70","llm_anon_nofb":"#C7A76C","llm_anon_fb":"#D2691E",
   "llm_named_nofb":"#6A8CBF","llm_named_fb":"#B03A48"}
def pergen(arm, key):
    g={}
    for r in sorted(glob.glob(f"runs/{arm}_s*")):
        try: h=[json.loads(l) for l in open(f"{r}/history.jsonl")]
        except FileNotFoundError: continue
        for x in h:
            v=x[key] if key!="decoy_mass" else x["decoy_mass"]
            if v!=v: continue
            g.setdefault(x["i"]//8,[]).append(v)
    return g
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
for arm in ARMS:
    g=pergen(arm,"decoy_mass")
    if not g: continue
    ks=sorted(g); m=[boot(g[k]) for k in ks]
    ax[0].errorbar([k+1 for k in ks],[x[0] for x in m],
                   yerr=[[x[0]-x[1] for x in m],[x[2]-x[0] for x in m]],
                   color=C[arm],marker="o",ms=5,lw=1.8,capsize=3,label=LBL[arm])
ax[0].axhline(0,color="k",ls="--",lw=1); ax[0].axhline(CHANCE_DECOY_MASS,color="r",ls=":",lw=1.4)
ax[0].text(4.05,0.005,"下限 0",fontsize=8); ax[0].text(4.05,CHANCE_DECOY_MASS+0.005,"随机 0.294",color="r",fontsize=8)
ax[0].set_xlabel("第几代（每代 8 次评估）"); ax[0].set_ylabel("提议的 decoy mass（该代全部 8 个）")
ax[0].set_xticks([1,2,3,4]); ax[0].set_title("① 语义把 decoy mass 一步打到 0；反馈只走完一半")
ax[0].legend(fontsize=7.5,loc="center right"); ax[0].grid(alpha=.25)
for arm in ARMS:
    g=pergen(arm,"fitness")
    if not g: continue
    ks=sorted(g); m=[boot(g[k]) for k in ks]
    ax[1].errorbar([k+1 for k in ks],[x[0] for x in m],
                   yerr=[[x[0]-x[1] for x in m],[x[2]-x[0] for x in m]],
                   color=C[arm],marker="o",ms=5,lw=1.8,capsize=3,label=LBL[arm])
ax[1].set_xlabel("第几代"); ax[1].set_ylabel("该代 8 个提议的平均适应度（米）")
ax[1].set_xticks([1,2,3,4]); ax[1].set_title("② 提议质量：会爬坡的是 evo 和「匿名·有反馈」，命名臂反而更平")
ax[1].legend(fontsize=7.5); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig("docs/figs/bullseye.png",dpi=150); print("ok")
