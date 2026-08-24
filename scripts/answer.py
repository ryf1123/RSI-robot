"""Helper for writing a generation's response by hand.
usage in python: A(run, gen, [ (base_i|None, {edits}), ... ])"""
import json
from rsi.design import Design, anonymise, TERM_NAMES, HP_NAMES

def A(run, gen, specs):
    req = json.load(open(f"runs/{run}/req_{gen:02d}.json"))
    hist = {h["i"]: h["design"] for h in req["history"]}
    anon = req["anon"]
    out = []
    for base, edits in specs:
        d = dict(w=dict(hist[base]["w"]), hp=dict(hist[base]["hp"])) if base is not None \
            else dict(w={k: 0.0 for k in (req["terms"])}, hp=dict(req["hp_levels"].items().__iter__().__next__() and
                                                                 {k: v[len(v)//2] for k, v in req["hp_levels"].items()}))
        for k, v in edits.items():
            (d["hp"] if k in d["hp"] else d["w"])[k] = float(v)
        out.append(d)
    assert len(out) == req["k"], f"{len(out)} != {req['k']}"
    json.dump({"designs": out}, open(f"runs/{run}/resp_{gen:02d}.json", "w"), indent=1)
    return f"runs/{run} gen{gen}: {len(out)} designs"

def show(run, gen, sort=True):
    req = json.load(open(f"runs/{run}/req_{gen:02d}.json"))
    h = sorted(req["history"], key=lambda x: -x["fitness"]) if sort else req["history"]
    print(f"--- {run} gen{gen} anon={req['anon']} k={req['k']}")
    for x in h:
        w = {k: v for k, v in x["design"]["w"].items() if v > 0}
        print(f" i={x['i']} fit={x['fitness']:6.2f} fall={x['fall_rate']:.2f} ep={x['ep_seconds']:.2f} "
              f"air={x['airborne_frac']:.2f} sat={x['action_sat']:.2f} hp={list(x['design']['hp'].values())}")
        print(f"    {w}")
