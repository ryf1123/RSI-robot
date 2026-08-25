"""The autonomous driver.

Given the current state of the registry, it decides what to do next. The
priority order encodes what this project learned the hard way -- each rule
exists because skipping it cost real hours:

  0. a failed gate outranks everything (you cannot read a curve without a
     noise floor, and you cannot trust an instrument you have not calibrated)
  1. a FATAL open threat outranks any new measurement
  2. a thin BASELINE outranks a thin treatment arm (it is the denominator of
     every comparison, and its error is biased toward "the treatment works")
  3. an UNDERPOWERED claim outranks an untested one (you already spent the
     compute; finish the comparison rather than opening a new one)
  4. a claim with no pre-registration is a bug, not a result
  5. only then: new questions

Each task carries an estimated cost in inner trainings, so the plan is budgeted
rather than a wish list."""
import json, os
import numpy as np
from . import claims as C, threats as T, probes as P, planner as PL, data
from .spec import RESEARCH_DIR

TRAIN_SECONDS = 2.0          # amortised over 6 processes on an M4


def _cost(n_trainings):
    return dict(trainings=int(n_trainings), minutes=round(n_trainings * TRAIN_SECONDS / 60, 1))


def next_tasks(delta=0.5, budget_trainings=4000):
    tasks = []

    g = P.gate()
    for m in g["missing"]:
        tasks.append(dict(priority=0, kind="probe", what=m,
                          why="没有它，后面每一条曲线都读不懂", **_cost(18 if m == "noise_floor" else 0)))

    for t in T.open_threats():
        if t["severity"] == "fatal":
            tasks.append(dict(priority=1, kind="threat", what=t["id"], why=t["threat"],
                              action=t["control"], **_cost(1500)))

    cl = C.all_claims()
    # 2. thin baselines
    seen = set()
    for c in cl:
        b, root = c.test.get("baseline"), c.test.get("root", "runs")
        if not b or (root, b) in seen: continue
        seen.add((root, b))
        n_b = len(data.series(root, b, c.test["metric"]))
        need = PL.seeds_for(delta, PL.observed_sd(root, b, c.test["metric"]))
        if need > 0 and n_b < min(need, 24):
            tasks.append(dict(priority=2, kind="add_seeds", what=f"{root}/{b}",
                              why=f"基线只有 n={n_b}，要测 {delta}m 需要 n≈{need}；分母的误差偏向「处理组有效」",
                              **_cost((min(need, 24) - n_b) * 32)))

    # 3. underpowered claims (fitness metrics only; decoy mass has its own scale)
    for c in cl:
        if c.status != "underpowered" or c.test["metric"] not in ("reeval", "best"): continue
        root, arm = c.test.get("root", "runs"), c.test["arm"]
        sd = PL.observed_sd(root, arm, c.test["metric"])
        need = PL.seeds_for(delta, sd)
        have = len(data.series(root, arm, c.test["metric"]))
        tasks.append(dict(priority=3, kind="add_seeds", what=f"{root}/{arm}", claim=c.id,
                          why=f"能测到的最小差是 {c.evidence.get('min_detectable_diff','?')}m，"
                              f"观测差只有 {c.evidence.get('diff','?')}m；n={have}→{need}",
                          **_cost(max(0, need - have) * 32)))

    # 4. unregistered claims
    for c in cl:
        if not c.prereg:
            tasks.append(dict(priority=4, kind="prereg_missing", what=c.id,
                              why="这条结论没有预注册，多重比较的分母未知", **_cost(0)))

    for t in T.open_threats():
        if t["severity"] != "fatal":
            tasks.append(dict(priority=5, kind="threat", what=t["id"], why=t["threat"],
                              action=t["control"], **_cost(2000)))

    # one task per target, keep the most expensive (= the binding requirement)
    dedup = {}
    for t in tasks:
        if t["kind"] == "add_seeds" and t["trainings"] <= 0: continue
        k = (t["kind"], t["what"])
        if k not in dedup or t["trainings"] > dedup[k]["trainings"]: dedup[k] = t
    tasks = list(dedup.values())
    tasks.sort(key=lambda t: (t["priority"], -t["trainings"]))
    spent, plan = 0, []
    for t in tasks:
        if spent + t["trainings"] > budget_trainings and t["priority"] > 1:
            t["deferred"] = True
        else:
            spent += t["trainings"]
        plan.append(t)
    return plan, spent


def report(delta=0.5, budget=4000):
    plan, spent = next_tasks(delta, budget)
    print(f"预算 {budget} 次内层训练（≈{budget*TRAIN_SECONDS/3600:.1f} 小时），"
          f"排进计划 {spent} 次\n")
    print(f"{'P':<3}{'kind':<16}{'what':<28}{'训练次数':>9}{'分钟':>8}  why")
    print("-" * 128)
    for t in plan:
        mark = "  (deferred)" if t.get("deferred") else ""
        print(f"{t['priority']:<3}{t['kind']:<16}{t['what']:<28}{t['trainings']:>9}{t['minutes']:>8}  "
              f"{t['why'][:52]}{mark}")


# ---------------------------------------------------------------- execution --
AUTOMATIC = ("random", "evo", "contrast") + tuple(f"sparse{k}" for k in (2, 3, 5, 8, 12, 14)) \
    + tuple(f"sparseclean{k}" for k in (3, 5, 8, 12)) + ("midhp", "sparse5mid",
                                                          "llm_named_nofb", "llm_anon_nofb")

def executable(plan):
    """Which planned tasks the system can carry out with no human in the loop.

    The two zero-feedback LLM arms count as automatic because their prior is
    written down as auditable code (rsi/proposers/prior.py) rather than answered
    per-request -- that was a deliberate design choice so that the majority of
    the arm set stays machine-runnable."""
    out = []
    for t in plan:
        if t["kind"] != "add_seeds" or t.get("deferred"): continue
        root, arm = t["what"].split("/")
        if arm.split("_s")[0] in AUTOMATIC:
            out.append(dict(t, root=root, arm=arm))
    return out


def script(tasks, out="research/next_batch.sh"):
    """Emit a runnable batch for the tasks the system chose. Deliberately a
    script rather than an in-process call: the plan stays auditable and a human
    can veto a line before it burns two hours of compute."""
    import os, json, glob
    lines = ["#!/bin/bash", "set -e", 'cd "$(dirname "$0")/.."',
             "source .venv/bin/activate", "export PYTHONPATH=."]
    for t in tasks:
        root, arm = t["root"], t["arm"]
        have = len(glob.glob(f"{root}/{arm}_s*"))
        need = have + max(1, t["trainings"] // 32)
        env = "RSI_SPACE=dense " if root == "runs_dense" else ""
        extra = "--inner-seeds 4 --budget 8 --gen 4" if root == "runs_k4" else "--budget 32 --gen 8"
        for s in range(have, need):
            d = f"{root}/{arm}_s{s}"
            if arm.startswith("llm_"):
                anon = " --anon" if "anon" in arm else ""
                lines += [f'{env}python -m rsi.loop init --run {d} --arm llm --seed {s} {extra} --nofb{anon} >/dev/null',
                          f'{env}python -m rsi.loop request --run {d} >/dev/null']
            else:
                lines.append(f'{env}python -m rsi.loop init --run {d} --arm {arm} --seed {s} {extra} >/dev/null')
                lines.append(f'{env}python -m rsi.loop run --run {d} --procs 6')
    lines += ["python scripts/reeval.py > research/reeval.log 2>&1",
              "python -m rsi.auto sweep", "echo BATCH_DONE"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n"); os.chmod(out, 0o755)
    return out
