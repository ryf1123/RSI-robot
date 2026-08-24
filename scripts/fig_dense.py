import json,glob,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["PingFang SC","Heiti SC","Arial Unicode MS"]; plt.rcParams["axes.unicode_minus"]=False
from rsi.report import boot
def stats(root, arm, key="fitness"):
    v=[]
    for r in sorted(glob.glob(f"{root}/{arm}_s*")):
        try: h=[json.loads(l) for l in open(f"{r}/history.jsonl")]
        except FileNotFoundError: continue
        if h: v.append(max(x["fitness"] for x in h))
    return boot(v)
DEC={"back_vel","stand_still","foot_press","y_drift","clock",
     "crouch","freeze_joints","backward_pos","max_ctrl","foot_glue","const_zero","clock2"}
def dmass(w):
    tot=sum(abs(v) for v in w.values())
    return sum(abs(v) for k,v in w.items() if k in DEC)/tot if tot else float("nan")
def dm_curve(root, hist_glob, bins):
    A=[]
    for r in sorted(glob.glob(hist_glob)):
        try: h=[json.loads(l) for l in open(f"{r}/history.jsonl")]
        except FileNotFoundError: continue
        for x in h:
            dm=dmass(x["design"]["w"])
            if dm==dm: A.append((dm,x["fitness"]))
    A=np.array(A); out=[]
    rng=np.random.default_rng(0)
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(A[:,0]>=lo)&(A[:,0]<hi)
        if m.sum()<5: out.append((np.nan,)*4); continue
        f=A[m,1]; bs=rng.choice(f,(3000,len(f))).mean(1)
        out.append(((lo+hi)/2, f.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)))
    return np.array(out)
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
B=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.75,1.01]
for root,lab,c,ch in [("runs","稀疏空间（17 项，5 个诱饵）","#4C7BBF",0.294),
                      ("runs_dense","密集空间（24 项，12 个诱饵）","#B03A48",0.500)]:
    C=dm_curve(root,f"{root}/*_s*",B)
    ok=~np.isnan(C[:,0])
    ax[0].plot(C[ok,0],C[ok,1],"-o",color=c,label=lab,ms=5)
    ax[0].fill_between(C[ok,0],C[ok,2],C[ok,3],color=c,alpha=.15,lw=0)
    ax[0].axvline(ch,color=c,ls=":",lw=1.4)
    ax[0].annotate(f"随机水平 {ch:.2f}",(ch,-0.13),color=c,fontsize=8.5,ha="center")
ax[0].set_ylim(-0.22,0.72); ax[0].set_xlabel("decoy mass"); ax[0].set_ylabel("任务适应度（米）")
ax[0].set_title("① 诱饵的伤害有拐点，而拐点就落在随机水平上"); ax[0].legend(fontsize=8.5); ax[0].grid(alpha=.25)
arms=["random","evo","llm_named_nofb"]; L=["random","evo","命名·无反馈\n（纯先验，零反馈）"]
x=np.arange(3); w=0.36
for i,(root,lab,c) in enumerate([("runs","稀疏（5/17 诱饵）","#4C7BBF"),("runs_dense","密集（12/24 诱饵）","#B03A48")]):
    m=[stats(root,a) for a in arms]
    ax[1].bar(x+(i-0.5)*w,[q[0] for q in m],w,color=c,label=lab,
              yerr=[[q[0]-q[1] for q in m],[q[2]-q[0] for q in m]],capsize=4,alpha=.9)
ax[1].set_xticks(x); ax[1].set_xticklabels(L,fontsize=9); ax[1].set_ylabel("32 次评估后的 best-so-far（米）")
ax[1].set_title("② 把诱饵密度加倍：随机腰斩，纯先验不受影响"); ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.25,axis="y")
plt.tight_layout(); plt.savefig("docs/figs/dense.png",dpi=150); print("ok")
