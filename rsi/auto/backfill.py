"""Register every assertion this project made, bound to the test that produced it.

The point is not bookkeeping. Once these exist, `claims.sweep()` re-derives all
of them from the current data -- so the three times this project had to hand-chase
a number through the docs after n grew become one command."""
import datetime, os
from .spec import Prereg, Claim, save, RESEARCH_DIR

NOW = "2026-08-25"

PREREGS = [
 Prereg(id="pr1_bullseye", created="2026-08-24T23:10",
        question="agent 迭代机器人时，起作用的是先验、反馈还是选择压力？",
        prediction="纯先验就能打过随机；匿名+反馈≈evo；命名的效应量>反馈；匿名·无反馈≈随机；decoy mass 比 fitness 更早分开",
        decision_rule="单侧 MWU，精英重评口径，Holm 校正跨 5 条臂，alpha=0.05",
        family=["c_named_nofb_k1","c_named_fb_k1","c_anon_nofb_k1","c_anon_fb_k1","c_evo_k1"]),
 Prereg(id="pr1_decoy", created="2026-08-24T23:10",
        question="同上，但用不含内层噪声的 decoy mass 口径",
        prediction="命名臂贴到 0.000；反馈只走一半；选择压力零贡献",
        decision_rule="单侧 MWU（越小越好），Holm 跨 5 条臂",
        family=["d_named_nofb","d_named_fb","d_anon_nofb","d_anon_fb","d_evo"]),
 Prereg(id="pr2_dense", created="2026-08-25T00:05",
        question="把诱饵密度加倍、让随机基线越过损伤拐点，先验的优势会不会在任务适应度上显现？",
        prediction="会；这是「先验买的那段区间里干净是免费的」这个机制的直接推论",
        decision_rule="单侧 MWU，精英重评口径，Holm 跨 2 条臂",
        family=["c_dense_named_nofb","c_dense_evo"]),
 Prereg(id="pr3_sparsity", created="2026-08-25T03:00",
        question="「结构先验 = 少写几项」能不能解释 LLM 臂的重评优势？",
        prediction="重评质量在 k=3-5 达峰，向 k=12+ 单调下降",
        decision_rule="单侧 MWU 跨 6 个 k，Holm 校正；另看 spearman(k, reeval)",
        family=["c_sparse2","c_sparse3","c_sparse5","c_sparse8","c_sparse12","c_sparse14"]),
 Prereg(id="pr4_sparse5_rep", created="2026-08-25T06:25",
        question="2x2 里那个 p=0.039 的稀疏主效应，是不是「先看阶梯再挑 k=5」的产物？",
        prediction="是；用 16 个全新种子重跑会消失",
        decision_rule="单侧 MWU，只做这一次比较，不校正（family=1）",
        family=["c_sparse5_replication"]),
 Prereg(id="pr5_k4", created="2026-08-25T07:10",
        question="「涨的是报出来的数、不是拿到的设计」会不会只是 k=1 评估的产物？",
        prediction="不是；k=4 下两条臂在真实质量上仍然测不出差别",
        decision_rule="单侧 MWU，精英重评口径，family=1",
        family=["c_k4_named_nofb"]),
 Prereg(id="pr_curse", created="2026-08-24T23:35",
        question="外层报出来的 best-so-far 和设计的真实质量有多大关系？",
        prediction="极弱；best-so-far 基本是 max-of-noise",
        decision_rule="spearman(reported, reeval) 跨全部 run",
        family=["c_winners_curse"]),
]

def T(kind, arm, baseline, metric, root="runs", **kw):
    return dict(kind=kind, arm=arm, baseline=baseline, metric=metric, root=root, alpha=0.05, **kw)

CLAIMS = [
 # --- pr1: sparse space, k=1, elite re-evaluation -------------------------
 Claim(id="c_named_nofb_k1", prereg="pr1_bullseye",
       statement="纯语义先验（零反馈）交付的设计，真实质量优于随机",
       test=T("mwu_greater","llm_named_nofb","random","reeval")),
 Claim(id="c_named_fb_k1", prereg="pr1_bullseye",
       statement="命名+反馈的完整 Eureka 式循环，真实质量优于随机",
       test=T("mwu_greater","llm_named_fb","random","reeval")),
 Claim(id="c_anon_nofb_k1", prereg="pr1_bullseye",
       statement="匿名·无反馈（结构先验）真实质量优于随机",
       test=T("mwu_greater","llm_anon_nofb","random","reeval")),
 Claim(id="c_anon_fb_k1", prereg="pr1_bullseye",
       statement="匿名+反馈真实质量优于随机",
       test=T("mwu_greater","llm_anon_fb","random","reeval")),
 Claim(id="c_evo_k1", prereg="pr1_bullseye",
       statement="变异精英（选择压力）真实质量优于随机",
       test=T("mwu_greater","evo","random","reeval")),
 # --- pr1_decoy: does the proposer understand what it edits? --------------
 Claim(id="d_named_nofb", prereg="pr1_decoy",
       statement="命名·无反馈的精英诱饵占比低于随机", test=T("mwu_less","llm_named_nofb","random","decoy")),
 Claim(id="d_named_fb", prereg="pr1_decoy",
       statement="命名·有反馈的精英诱饵占比低于随机", test=T("mwu_less","llm_named_fb","random","decoy")),
 Claim(id="d_anon_nofb", prereg="pr1_decoy",
       statement="匿名·无反馈的精英诱饵占比低于随机（协议对照：应当不显著）",
       test=T("mwu_less","llm_anon_nofb","random","decoy")),
 Claim(id="d_anon_fb", prereg="pr1_decoy",
       statement="匿名·有反馈的精英诱饵占比低于随机", test=T("mwu_less","llm_anon_fb","random","decoy")),
 Claim(id="d_evo", prereg="pr1_decoy",
       statement="选择压力能降低精英的诱饵占比", test=T("mwu_less","evo","random","decoy")),
 # --- pr2: dense-decoy space ---------------------------------------------
 Claim(id="c_dense_named_nofb", prereg="pr2_dense",
       statement="诱饵密集空间里，纯语义先验交付的设计真实质量优于随机",
       test=T("mwu_greater","llm_named_nofb","random","reeval",root="runs_dense")),
 Claim(id="c_dense_evo", prereg="pr2_dense",
       statement="诱饵密集空间里，变异精英真实质量优于随机",
       test=T("mwu_greater","evo","random","reeval",root="runs_dense")),
 # --- pr3: sparsity ladder ------------------------------------------------
 *[Claim(id=f"c_sparse{k}", prereg="pr3_sparsity",
         statement=f"恰好激活 {k} 项的稀疏提议，真实质量优于随机",
         test=T("mwu_greater",f"sparse{k}","random","reeval")) for k in (2,3,5,8,12,14)],
 # --- pr4: pre-registered replication -------------------------------------
 Claim(id="c_sparse5_replication", prereg="pr4_sparse5_rep",
       statement="sparse5 在全新种子上仍然优于随机（预注册重复）",
       test=T("mwu_greater","sparse5","random","reeval")),
 # --- pr5: the loop-configuration test ------------------------------------
 Claim(id="c_k4_named_nofb", prereg="pr5_k4",
       statement="同算力下改用 8 设计 x 4 内层种子，纯语义先验真实质量优于随机",
       test=T("mwu_greater","llm_named_nofb","random","reeval",root="runs_k4")),
 # --- winner's curse ------------------------------------------------------
 Claim(id="c_winners_curse", prereg="pr_curse",
       statement="外层报出来的 best-so-far 与设计真实质量相关",
       test=dict(kind="spearman_reported_vs_reeval", metric="reeval", arm="", alpha=0.05,
                 roots=["runs","runs_dense","runs_k4"])),
]

def run():
    os.makedirs(f"{RESEARCH_DIR}/prereg", exist_ok=True)
    os.makedirs(f"{RESEARCH_DIR}/claims", exist_ok=True)
    for p in PREREGS: save(p)
    for c in CLAIMS: save(c)
    return len(PREREGS), len(CLAIMS)
