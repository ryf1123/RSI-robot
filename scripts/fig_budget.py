import json,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["PingFang SC","Heiti SC","Arial Unicode MS"]; plt.rcParams["axes.unicode_minus"]=False
from rsi.report import boot
d=json.load(open("runs/budget_split.json")); arms=["k1","k2","k4","k8"]
x=np.arange(4); lbl=["32 × 1","16 × 2","8 × 4","4 × 8"]
fig,ax=plt.subplots(1,2,figsize=(12.5,4.5))
for key,c,name in [("reported","#B03A48","外层报出来的精英分数"),("reeval","#4C7BBF","精英重评（6 个新种子）")]:
    m=[boot([r[a][key] for r in d]) for a in arms]
    ax[0].errorbar(x,[q[0] for q in m],yerr=[[q[0]-q[1] for q in m],[q[2]-q[0] for q in m]],
                   marker="o",ms=7,lw=2,capsize=4,color=c,label=name)
ax[0].set_xticks(x); ax[0].set_xticklabels(lbl); ax[0].set_xlabel("预算怎么切（设计数 × 每个设计的内层种子数），总训练次数恒为 32")
ax[0].set_ylabel("适应度（米）"); ax[0].set_title("① 报出来的数一路下滑，真实质量纹丝不动")
ax[0].legend(fontsize=9); ax[0].grid(alpha=.25)
bias=[boot([r[a]["reported"]-r[a]["reeval"] for r in d]) for a in arms]
ax[1].bar(x,[b[0] for b in bias],0.6,color="#C7773A",
          yerr=[[b[0]-b[1] for b in bias],[b[2]-b[0] for b in bias]],capsize=4)
ax[1].axhline(0,color="k",lw=1)
for i,b in enumerate(bias): ax[1].text(i,b[0]+0.06,f"{b[0]:+.2f}",ha="center",fontsize=10)
ax[1].set_xticks(x); ax[1].set_xticklabels(lbl); ax[1].set_ylabel("赢家诅咒（报出来的 − 重评）")
ax[1].set_xlabel("预算怎么切"); ax[1].set_title("② 多种子买到的不是更好的设计，是更诚实的数")
ax[1].grid(alpha=.25,axis="y")
plt.tight_layout(); plt.savefig("docs/figs/budget.png",dpi=150); print("ok")
