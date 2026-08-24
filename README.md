# RSI-robot

Using an LLM agent to iterate on robot designs — and measuring **which part of the
agent's information actually does the work**.

Eureka-style loops (LLM writes a reward → RL trains → training statistics go back
to the LLM → repeat) report large wins over human experts. That number mixes at
least four things: the model's **prior**, the **feedback** it reads, the
**selection pressure** of keeping the best, and the **search space** a human wrote.
This repo pulls the first three apart on a Mac mini, with no CUDA.

## How it is made falsifiable

The search space is built on purpose. Of the 17 reward terms, **5 are planted
decoys** — three actively harmful (`back_vel`, `stand_still`, `foot_press`) and two
identically irrelevant (`y_drift`, `clock`). So **decoy mass** (decoy weight /
total weight) is a metric with a known floor (0.000) and a known chance level
(5/17 = 0.294). It measures whether the proposer understands what it is editing,
and — unlike task fitness — it carries no inner-loop training noise.

Six arms differ by exactly two switches: whether term **names** are visible, and
whether **feedback** is visible.

| arm | proposer | sees names | sees feedback |
|-|-|-|-|
| `random` | uniform | — | — |
| `evo` | mutate the elite (1+8) | — | fitness only |
| `llm_named_fb` | Claude | yes | yes |
| `llm_named_nofb` | semantic prior | yes | no |
| `llm_anon_fb` | Claude, terms shown as `t00..t16` | no | yes |
| `llm_anon_nofb` | structural prior | no | no |

`llm_anon_nofb` is the protocol control: it is information-equivalent to random,
and it measures a decoy mass of 0.31 against a chance level of 0.294. Good.

## Headline results (1408 evaluations)

- **The outer loop's own score is uninformative.** Spearman(best-so-far,
  re-evaluation of the same design on 6 fresh inner seeds) = **0.043, p = 0.82**.
  Mean shrinkage 60%.
- **Semantics is one-shot; feedback gets halfway; selection buys nothing.**
  Decoy mass: named arms **0.000** from generation 1 (chance 0.294); anonymous +
  feedback reaches 0.143 over three generations; `evo` does not move at all
  (0.295 → 0.283).
- **Better proposals do not reach the result.** `evo` proposes 6× better than
  random (0.59 vs 0.10 m mean) and its best-so-far is 5% higher.
- **Pre-registered intervention holds.** Doubling decoy density (12 of 24 terms,
  chance 0.500) halves random (2.05 → 0.99 m) and leaves the zero-feedback
  semantic prior untouched (2.59 → 3.45 m) — the project's first non-overlapping
  intervals, and the winner never looked at a single training result.

Full write-up (Chinese, with figures and video): see the Feishu doc linked from
`PLAN.md`. Pre-registration in `notes/00_preregistration.md`.

## Reproduce

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install mujoco torch numpy scipy imageio imageio-ffmpeg matplotlib pyyaml tqdm

python -m rsi.loop init --run runs/random_s0 --arm random --seed 0 --budget 32 --gen 8
python -m rsi.loop run  --run runs/random_s0 --procs 6

python scripts/noise_floor.py && python scripts/reeval.py
python -m rsi.report && python scripts/analyze_prop_vs_sel.py
RSI_SPACE=dense python -m rsi.report --root runs_dense
```

One evaluation = one full ARS training run from scratch (~10–12 s amortised over
6 processes on an M4).
