"""Threats-to-validity register.

A claim registry says what you believe. This says what could still be wrong, and
names the control that would settle it. The driver treats an unaddressed threat
as an open task with a cost, so the system spends compute on falsifying itself
rather than on collecting more of the same evidence.

Every entry below is a real threat this project hit, and the `control` column is
the experiment that actually resolved it (or is still open)."""
import json, os, glob, datetime
from .spec import RESEARCH_DIR

DIR = f"{RESEARCH_DIR}/threats"

SEED = [
 dict(id="inner_noise", severity="fatal",
      threat="外层比较的两个数各自是一次随机 RL 训练，种子方差可能大到把设计差异完全盖住",
      control="probes.noise_floor：同一设计跑 k 个种子，报 std/mean",
      status="addressed", finding="好设计 std 1.17 > mean 0.85；噪声只在好的一侧爆炸"),
 dict(id="winners_curse", severity="fatal",
      threat="best-so-far 是对噪声取 max，可能与真实质量无关",
      control="精英用一批全新种子重评，报 spearman(reported, reeval)",
      status="addressed", finding="rho=0.29 (n=192)，缩水 53%；k=4 下升到 0.51"),
 dict(id="thin_baseline", severity="fatal",
      threat="所有比较共用一个分母；分母 n 太小会系统性地偏向处理组",
      control="baseline 种子数给到处理组的 2-3 倍并最先跑",
      status="addressed", finding="random n=8→24 后三处结论翻转，方向全部偏向「agent 有用」"),
 dict(id="forking_paths", severity="high",
      threat="在阶梯上挑最高的那个点再做比较，会必然产生假阳性",
      control="用全新种子对预注册的那个点重跑",
      status="addressed", finding="sparse5 的 p=0.039 在 16 个新种子上完全消失（0.78 vs 0.97, p=0.71）"),
 dict(id="loop_config", severity="fatal",
      threat="结论可能只在 k=1 评估下成立，也就是只是循环配置的产物",
      control="同算力换成 8 设计 x 4 种子重跑",
      status="addressed", finding="先验从 p=1.00 变成 2.3 倍 (p=0.008)：主结论确实是 k=1 的产物"),
 dict(id="semantic_leak", severity="fatal",
      threat="匿名臂里我可能记得 t07 是什么，语义从后门泄漏",
      control="每个种子一份随机置换 + 检查匿名·无反馈臂是否落在随机水平",
      status="addressed", finding="decoy mass 0.235 vs random 0.209，区间重叠，未泄漏"),
 dict(id="single_answerer", severity="high",
      threat="所有种子的 LLM 请求由同一个会话回答，种子之间隐性迁移",
      control="每个种子换独立会话回答",
      status="open", finding="方向为保守（污染只会让有反馈臂显得更好，而它没赢）"),
 dict(id="external_validity_inner", severity="high",
      threat="结论可能只在 ARS + 线性策略这一种内层学习器上成立",
      control="换 PPO + MLP 重跑靶心",
      status="open", finding=""),
 dict(id="external_validity_space", severity="high",
      threat="离散空间买到了随机下限，代价是外部效度；自由形式代码空间未必一样",
      control="在自由形式奖励代码空间上做交叉验证",
      status="open", finding=""),
 dict(id="k4_arms_incomplete", severity="high",
      threat="k=4 的结论只在 random 和 命名·无反馈 两条臂上测过",
      control="6 条主臂全部在 k=4 下重跑",
      status="in_progress", finding=""),
]


def init():
    os.makedirs(DIR, exist_ok=True)
    for t in SEED:
        p = f"{DIR}/{t['id']}.json"
        if not os.path.exists(p):
            json.dump(dict(t, updated=datetime.datetime.now().isoformat(timespec="seconds")),
                      open(p, "w"), ensure_ascii=False, indent=1)
    return load()


def load():
    return [json.load(open(p)) for p in sorted(glob.glob(f"{DIR}/*.json"))]


def open_threats():
    return [t for t in load() if t["status"] in ("open", "in_progress")]


def report():
    ts = load()
    rank = {"fatal": 0, "high": 1, "medium": 2}
    print(f"{'threat':<26}{'severity':<10}{'status':<13} finding / control")
    print("-" * 118)
    for t in sorted(ts, key=lambda t: (t["status"] != "open", rank.get(t["severity"], 9))):
        tail = t["finding"] or ("→ " + t["control"])
        print(f"{t['id']:<26}{t['severity']:<10}{t['status']:<13} {tail[:62]}")
    print("-" * 118)
    print(f"  addressed {sum(1 for t in ts if t['status']=='addressed')} / {len(ts)}"
          f"   open {sum(1 for t in ts if t['status']=='open')}"
          f"   in progress {sum(1 for t in ts if t['status']=='in_progress')}")
