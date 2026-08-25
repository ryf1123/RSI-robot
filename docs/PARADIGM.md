# RSI-AutoResearch: an autonomous research system for design-space iteration

A search harness answers *what is the best design?*
A research system answers *what do I now believe, how sure am I, what could
still be wrong, and what should I run next?*

This is the second thing. It came out of nine hours of running the first thing
and getting four conclusions wrong in a row — each failure is now a component.

---

## The control loop

```
                    ┌──────────────────────────────────────────────┐
                    │                                              │
   ┌────────┐   ┌───▼────┐   ┌─────────┐   ┌────────┐   ┌────────┐ │
   │ PROBE  │──▶│ PREREG │──▶│ EXECUTE │──▶│ VERIFY │──▶│ SWEEP  │─┘
   │ gate   │   │ gate   │   │ arms    │   │ reeval │   │ claims
   └────────┘   └────────┘   └─────────┘   └────────┘   └───┬────┘
        ▲                                                   │
        │                    ┌──────────┐   ┌───────────┐   │
        └────────────────────│  DRIVER  │◀──│  THREATS  │◀──┘
                             │ next task│   │ register  │
                             └──────────┘   └───────────┘
```

Six stages, five of which are *gates* — they can refuse to proceed. That is the
whole design: in a noisy domain, the expensive mistakes are not bad experiments,
they are good experiments read too confidently.

---

## The five layers, and the failure each one exists to prevent

| Layer | Refuses to let you… | Because, measured here |
|-|-|-|
| **Probe gate** | compare anything before the noise floor is measured | the best design's seed-to-seed std (1.17 m) exceeds its own mean (0.85 m) |
| **Pre-registration gate** | run a comparison with an undeclared family | a `p = 0.039` picked off a 6-point ladder vanished on fresh seeds (0.78 vs 0.97, `p = 0.71`) |
| **Verification** | report `best-so-far` | ρ(reported, re-evaluated) = **0.34** over 256 runs; mean shrinkage **53%** |
| **Claim registry + sweep** | leave a number stale after `n` grows | the same ρ was published as 0.043 → 0.17 → 0.29 → 0.34 as data accumulated |
| **Threat register + driver** | spend compute on confirmation | the headline conclusion was *wrong* until the `loop_config` threat was tested |

### The rule that changes the most conclusions

`stats.verdict()` is the only function allowed to turn numbers into a word, and
it has three outcomes, not two:

```python
if p_adj <= alpha:                                  return "supported"
if abs(observed_diff) < min_detectable_diff:        return "underpowered"
                                                    return "not_supported"
```

Applied to this project's own 21 registered claims: **15 are `underpowered`**,
5 `supported`, 1 `not_supported`. Almost every place the write-up said
"no difference" it should have said "this comparison could not have seen a
difference of the size I care about."

### The rule that is easiest to skip and most expensive to skip

The baseline is the denominator of every comparison, so its error enters every
conclusion — and it enters *biased toward the treatment working*. The planner
schedules it first and 2–3× deeper. Going from `n = 8` to `n = 24` on the random
arm here flipped three published conclusions, all in the same direction.

---

## The planner: budget is a decision, not a default

Two numbers come out of the noise floor:

**How many seeds per arm.** At σ = 1.17 and a 0.5 m effect of interest:
68 per arm, 170 for the baseline. This project ran 8–24 — underpowered by
roughly 8×, which the audit table now says out loud.

**How to split the budget** — candidates × inner-seeds. This is *not* a
constant; it interacts with proposer quality, which is the system's sharpest
empirical finding:

| Proposer | Optimal split | Why | Evidence |
|-|-|-|-|
| weak (≈ uniform) | many candidates, `k = 1` | good designs are rare, buy volume | four splits tie, paired `p > 0.5` |
| strong (semantic prior) | few candidates, `k = 4` | good designs are common, buy selection accuracy | 1.18 vs 0.52, `p = 0.008` |

The same prior scores `p = 1.00` against random under `32 × 1` and **2.3×**
under `8 × 4` at *identical total compute*. A search harness cannot notice this,
because it holds the loop configuration fixed and searches inside it. A research
system treats the loop configuration as part of what is under study.

---

## Ground-truth instrumentation

The system requires the design space to carry a **known-answer probe** — here,
5 planted decoy terms out of 17, so "decoy mass" has a known floor (0.000) and a
known chance level (0.294), and `probes.instrument()` verifies both.

This is what lets the system measure *whether the proposer understands what it is
editing*, separately from the noisy objective. It is the only metric in the
project that survived every increase in `n`, and it is the one that made the
central result legible: the semantic prior sits at 0.000 from generation 1
(`p < 0.001`), while pure fitness selection never moves off chance (0.295 → 0.280).

Any design space this system is pointed at needs an equivalent. Without one you
can measure outcomes but never mechanism, and in a domain this noisy the outcome
signal alone is too weak to learn from.

---

## Using it

```bash
python -m rsi.auto init                 # register preregs, claims, threats
python -m rsi.auto gate                 # may I compare anything yet?
python -m rsi.auto probe                # noise floor / instrument / protocol control
python -m rsi.auto plan  --delta 0.5    # seeds per arm + budget split
python -m rsi.auto sweep                # re-derive every claim; print status changes
python -m rsi.auto threats              # what could still be wrong
python -m rsi.auto next --budget 3000   # a costed, prioritised plan
python -m rsi.auto audit                # all of the above
```

`driver.script(driver.executable(plan))` emits `research/next_batch.sh` — a
runnable batch for the tasks needing no human. It is a script rather than an
in-process call so the plan stays auditable and a person can veto a line before
it burns two hours of compute.

State lives in flat files under `research/` — `prereg/`, `claims/`, `threats/`,
`probes/`. Nothing is hidden in a database; a claim is a JSON file containing the
executable test that produced it, which is exactly what makes the sweep possible.

---

## Porting it to another domain

Four things are domain-specific; the rest is not.

1. `data.series()` — how a run directory becomes a number.
2. A **task metric the designer cannot rewrite** (here: metres travelled). The
   inner learner optimises the *designed* objective; the outer loop scores the
   fixed one. Reward hacking is invisible without this split.
3. A **known-answer probe** (here: planted decoys).
4. A **discrete, enumerable design space**, so that "uniform random" is a real
   lower bound. Free-form code cannot be uniformly sampled, so claims of the form
   "the agent beat search" have no denominator.

---

## What it does not do

- It does not choose research *questions*. The threat register is seeded by a
  human; the driver only prioritises within it.
- Its `proposer_strength` input is calibrated on two measured points in one
  space. Treat the `k` it recommends as a hypothesis to test, not a setting.
- It cannot detect a threat nobody wrote down. `loop_config` — the one that
  overturned the headline — was registered eight hours in, by hand.

That last limitation is the honest summary of the whole thing: the system
mechanises the discipline, not the imagination.

---

## Replaying this project through the system

The most useful test of the system is the record it came from.

| Gate / layer | Would have stopped | What actually happened |
|-|-|-|
| Probe gate | building arms before the noise floor is measured | measured it at hour 1 — *after* the harness was already written to a budget the noise floor invalidates |
| **Planner** | the whole arm programme | it prints **68 seeds/arm, 170 for the baseline** at hour 1. I ran 4. The k=1 programme was knowably underpowered before it started |
| Baseline-first | a thin denominator | `random` sat at n=8 for seven hours; deepening it to 24 flipped three published conclusions, all toward "the agent works" |
| Pre-registration | picking `k=5` off a ladder and then testing it | that `p = 0.039` got published; only a hand-initiated replication killed it (`p = 0.71` on fresh seeds) |
| `verdict()` | reading "no difference" off an underpowered test | 15 claims mislabelled — which sent effort into opening new rings instead of deepening old ones |
| Threat register | **nothing** | `loop_config`, the threat that overturned the headline, was written by hand at hour 8 |

### The uncomfortable number

```
one adequately-powered 2-arm comparison (68 + 170 seeds) =  7,616 inner trainings
the 6-arm bullseye at that power                         = 16,320
actually spent this session                              = 12,178
```

The planner's allocation is the opposite of mine: **two arms at 68 seeds, not six
arms at eight.** And that is checkable — ring 9 *is* a two-arm comparison
(prior vs random at k=4), and it is the only experiment in the project that was
both adequately powered and changed the headline. The other twenty-odd arms
produced 5 supported claims and 15 underpowered ones.

The project's real output came from about a sixth of its compute. The planner
could have said so in the first hour.

### What it cannot take over

Of the seven things that mattered this session, the system can mechanise four —
thin baseline, underpowered verdicts, stale numbers, forking paths. It cannot
mechanise the three that produced the actual insight: inventing the decoy probe,
inventing anonymisation as a semantic off-switch, and suspecting `loop_config`.

All four it takes over are discipline. None of the three it cannot are.

---

## Operating record

Two cycles the system chose, ran and judged on its own, plus one alarm it raised
against itself.

**Cycle 1 — 352 trainings spent to retract a published number.** The driver
ranks by *distance to decidable*, not cost. Top of the queue was
`runs_k4/llm_anon_nofb` at `promise = 0.85` — observed 0.48 m against a
0.56 m detection floor, 15% short, 352 trainings to close. It generated the
batch, answered its own code-written prior, ran 11 seeds, re-evaluated and swept:

```
n = 12   mean 1.00   Holm p = 0.061    (already published as "marginal")
n = 23   mean 0.86   Holm p = 0.21
```

The effect shrank with data. What the system bought was a retraction — and the
reason it ranked first is exactly the reason it was dangerous: *closest to
decidable* is also *most likely to be written up as a result*.

**Cycle 2 — same pattern again.** `runs/llm_anon_nofb`, n = 8 → 27:
mean 1.30 → 1.10, `p_adj = 1.00`. Two cycles, two shrinking effects. This is the
"don't report a mechanism below n = 8" rule, running without anyone remembering it.

**The alarm.** Cycle 2's sweep flipped the *protocol control* — the arm that must
stay indistinguishable from its baseline or the whole ablation is void:

```
d_anon_nofb   n =  8    0.235 vs 0.209   p_adj = 0.78   passes
              n = 27    0.156 vs 0.209   p_adj = 0.022  ALARM: it "beats" random
```

Resolving it needed a better control, not more data. Elite decoy mass falls with
the number of active terms on its own — fewer terms, higher chance of carrying no
decoy, and fitness selection then picks those. The control arm averages 4.4
active terms; uniform random averages 13.4. Against **sparsity-matched**
baselines the alarm vanishes: 0.156 vs 0.166 (`sparse5`, p = 0.51), 0.156 vs
0.093 (`sparse3`, p = 0.88).

No leak — a confound. The old claim is marked `superseded_by`, a matched-baseline
claim replaces it, `probes.protocol()` now defaults to a sparsity-matched
baseline, and the lesson is in its docstring.

That episode is the clearest statement of the division of labour: **the system
noticed the control had flipped and escalated it; it could not invent the
matched baseline.** Discipline is mechanisable. Judgement is not.
