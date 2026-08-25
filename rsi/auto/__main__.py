import argparse, json
from . import claims as C, probes as P, planner as PL, threats as T, driver as D, backfill


def main():
    ap = argparse.ArgumentParser(prog="rsi.auto", description="RSI-AutoResearch")
    ap.add_argument("cmd", choices=["init", "gate", "probe", "sweep", "claims", "threats",
                                    "plan", "next", "audit", "verify-batch"])
    ap.add_argument("--delta", type=float, default=0.5, help="effect size you care about (m)")
    ap.add_argument("--budget", type=int, default=4000, help="inner trainings available")
    ap.add_argument("--which", default="all")
    a = ap.parse_args()

    if a.cmd == "init":
        print("preregs/claims:", backfill.run()); T.init(); print("threats:", len(T.load()))
    elif a.cmd == "gate":
        print(json.dumps(P.gate(), indent=1))
    elif a.cmd == "probe":
        if a.which in ("all", "instrument"): print("instrument:", P.instrument())
        if a.which in ("all", "protocol"): print("protocol  :", {k: P.protocol()[k] for k in ("mean_a", "mean_b", "ok")})
        if a.which in ("all", "noise_floor"): print("noise     :", P.noise_floor())
    elif a.cmd == "sweep":
        ch, cl = C.sweep()
        print("status changes:", ch or "none")
        C.report(cl)
    elif a.cmd == "claims":
        C.report()
    elif a.cmd == "threats":
        T.report()
    elif a.cmd == "plan":
        print(json.dumps(PL.plan(a.delta), indent=1, ensure_ascii=False))
    elif a.cmd == "next":
        D.report(a.delta, a.budget)
    elif a.cmd == "verify-batch":
        bad = D.verify_batch()
        if bad:
            print("⚠ 这些任务花了算力但样本量没涨（多半是 reeval 没覆盖到对应的 root）：")
            for b in bad: print(f"   {b['what']}  n {b['n_before']} → {b['n_now']}  花了 {b['spent']} 次训练")
        else:
            print("batch verified: every targeted claim gained sample size")
    elif a.cmd == "audit":
        print("== gate =="); print(json.dumps(P.gate(), indent=1)); print()
        print("== threats =="); T.report(); print()
        print("== claims =="); C.sweep(); C.report(); print()
        print("== next =="); D.report(a.delta, a.budget)


main()
