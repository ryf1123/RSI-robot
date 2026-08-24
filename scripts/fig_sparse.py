import json,glob,os,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["PingFang SC","Heiti SC","Arial Unicode MS"]; plt.rcParams["axes.unicode_minus"]=False
from rsi.report import boot
def el(arm,key):
    o=[]
    for r in sorted(p for p in glob.glob(f"runs/{arm}_s*") if os.path.isdir(p)):
        if key=="reeval":
            try: o.append(json.load(open(f"{r}/reeval.json"))["mean"])
            except FileNotFoundError: pass
        else:
            h=[json.loads(l) for l in open(f"{r}/history.jsonl")]
            if not h: continue
            b=max(h,key=lambda x:x["fitness"]); o.append(b["fitness"] if key=="best" else b["decoy_mass"])
    return np.array(o)
K=[2,3,5,8,12,14]
fig,ax=plt.subplots(1,2,figsize=(12.5,4.6))
for key,c,name in [("reeval","#4C7BBF","精英重评（6 个新种子）"),("best","#B03A48","best-so-far")]:
    m=[boot(el(f"sparse{k}",key)) for k in K]
    ax[0].errorbar(K,[q[0] for q in m],yerr=[[q[0]-q[1] for q in m],[q[2]-q[0] for q in m]],
                   marker="o",ms=6,lw=2,capsize=4,color=c,label=name)
for arm,c,lab,ls in [("random","#888","random（平均激活 14.2 项）","--"),
                     ("llm_anon_nofb","#C7A76C","匿名·无反馈（结构先验，3–6 项）",":"),
                     ("llm_named_nofb","#6A8CBF","命名·无反馈（语义先验）","-.")]:
    v=boot(el(arm,"reeval")); ax[0].axhline(v[0],color=c,ls=ls,lw=1.6,label=lab)
ax[0].set_xlabel("恰好激活几项（17 项里）"); ax[0].set_ylabel("适应度（米）")
ax[0].set_title("① 稀疏度阶梯：除了激活项数，和 random 没有任何区别")
ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.25); ax[0].set_xticks(K)
m=[boot(el(f"sparse{k}","decoy")) for k in K]
ax[1].errorbar(K,[q[0] for q in m],yerr=[[q[0]-q[1] for q in m],[q[2]-q[0] for q in m]],
               marker="s",ms=6,lw=2,capsize=4,color="#C7773A",label="sparse_k 精英的 decoy mass")
ax[1].axhline(0.294,color="r",ls=":",lw=1.4); ax[1].text(8,0.30,"随机水平 0.294",color="r",fontsize=8.5)
v=boot(el("random","decoy")); ax[1].axhline(v[0],color="#888",ls="--",lw=1.5,label="random 精英")
ax[1].set_xlabel("恰好激活几项"); ax[1].set_ylabel("精英的 decoy mass"); ax[1].set_xticks(K)
ax[1].set_title("② 精英的诱饵比例和 k 没有明显关系（ρ = 0.22, p = 0.13）")
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig("docs/figs/sparse.png",dpi=150); print("ok")
