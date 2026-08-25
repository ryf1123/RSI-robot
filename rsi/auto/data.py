"""The one place that knows how to turn a run directory into a number.

Every test in the system goes through here, so a claim registered in August can
be re-evaluated in September against data that did not exist yet."""
import json, glob, os
import numpy as np

METRICS = ("reeval", "best", "decoy", "reported")


def runs(root, arm):
    return sorted(p for p in glob.glob(f"{root}/{arm}_s*") if os.path.isdir(p))


def series(root, arm, metric):
    """Per-run values. `reeval` is the honest one (elite re-trained on fresh
    seeds); `best`/`reported` is what the outer loop announced."""
    out = []
    for r in runs(root, arm):
        if metric in ("reeval", "reported"):
            try:
                d = json.load(open(f"{r}/reeval.json"))
            except FileNotFoundError:
                continue
            out.append(d["mean"] if metric == "reeval" else d["reported"])
        else:
            try:
                h = [json.loads(l) for l in open(f"{r}/history.jsonl")]
            except FileNotFoundError:
                continue
            if not h:
                continue
            b = max(h, key=lambda x: x["fitness"])
            out.append(b["fitness"] if metric == "best" else b["decoy_mass"])
    return np.array([v for v in out if v == v], dtype=float)


def paired(root, arm, metric="reeval"):
    """reported vs reeval, paired per run -- the winner's-curse pair."""
    a, b = [], []
    for r in runs(root, arm):
        try:
            d = json.load(open(f"{r}/reeval.json"))
        except FileNotFoundError:
            continue
        a.append(d["reported"]); b.append(d["mean"])
    return np.array(a), np.array(b)


def all_pairs(roots=("runs", "runs_dense", "runs_k4")):
    a, b = [], []
    for root in roots:
        for r in sorted(p for p in glob.glob(f"{root}/*_s*") if os.path.isdir(p)):
            try:
                d = json.load(open(f"{r}/reeval.json"))
            except FileNotFoundError:
                continue
            a.append(d["reported"]); b.append(d["mean"])
    return np.array(a), np.array(b)
