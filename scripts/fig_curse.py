import json,glob,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["PingFang SC","Heiti SC","Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"]=False
from rsi.report import ARMS
LBL={"random":"random","evo":"evo","llm_anon_nofb":"匿名·无反馈","llm_anon_fb":"匿名·有反馈",
     "llm_named_nofb":"命名·无反馈","llm_named_fb":"命名·有反馈"}
C={"random":"#888","evo":"#4C9F70","llm_anon_nofb":"#C7A76C","llm_anon_fb":"#D2691E",
   "llm_named_nofb":"#6A8CBF","llm_named_fb":"#B03A48"}
fig,ax=plt.subplots(1,2,figsize=(12.5,4.6))
# left: noise floor
nf=json.load(open("runs/noise_floor.json"))
names=list(nf); 
for i,n in enumerate(names):
    v=np.array(nf[n]); ax[0].scatter([i]*len(v),v,s=45,color="#B03A48",alpha=.75,zorder=3)
    ax[0].plot([i-.22,i+.22],[v.mean()]*2,color="k",lw=2)
    ax[0].text(i+.28,v.mean(),f"mean {v.mean():.2f}\nstd {v.std(ddof=1):.2f}",fontsize=8,va="center")
ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(["中等设计","好设计","带诱饵"],fontsize=10)
ax[0].set_ylabel("任务适应度（米）"); ax[0].set_xlim(-.5,len(names)-.1)
ax[0].set_title("① 噪声底：同一个设计，只换内层种子（各 6 个）"); ax[0].grid(alpha=.25,axis="y")
# right: winner's curse
for a in ARMS:
    r_,e_=[],[]
    for r in sorted(glob.glob(f"runs/{a}_s*")):
        try: d=json.load(open(f"{r}/reeval.json"))
        except FileNotFoundError: continue
        r_.append(d["reported"]); e_.append(d["mean"])
    if r_: ax[1].scatter(r_,e_,s=55,color=C[a],label=f"{LBL[a]} (n={len(r_)})",alpha=.85,zorder=3)
lim=[-0.1,4.5]; ax[1].plot(lim,lim,"k--",lw=1,label="y = x（没有赢家诅咒）")
ax[1].set_xlim(lim); ax[1].set_ylim(-0.2,3.0)
ax[1].set_xlabel("外层报出来的 best-so-far（1 个内层种子）"); ax[1].set_ylabel("同一设计重评（6 个新种子）")
ax[1].set_title("② 赢家诅咒：48 个 run 里 43 个在对角线下方；ρ = 0.17（p = 0.24）"); ax[1].legend(fontsize=7.5); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig("docs/figs/noise_curse.png",dpi=150); print("ok")
