"""Typed records the autoresearch system passes around.

Everything here is JSON-serialisable on purpose: the whole system's state lives
in flat files under `research/`, so a run can be audited, diffed and replayed
without the code that produced it."""
from dataclasses import dataclass, field, asdict
from typing import Any
import json, hashlib, os

RESEARCH_DIR = "research"


def _stamp(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


@dataclass
class Prereg:
    """A prediction, written before the data exists. The runner refuses to
    execute a comparison that has no matching pre-registration."""
    id: str
    question: str
    prediction: str
    decision_rule: str            # e.g. "one-sided MWU on reeval, Holm across the family, alpha=0.05"
    family: list[str]             # every comparison in this family -> multiplicity is fixed in advance
    created: str
    outcome: str = "open"         # open | held | refuted | partial
    note: str = ""

    def path(self): return f"{RESEARCH_DIR}/prereg/{self.id}.json"


@dataclass
class Test:
    """A machine-checkable comparison. The point of making this data rather than
    code is that it can be RE-RUN against later data -- which is what turns a
    claim registry into a contradiction detector."""
    kind: str                     # mwu_greater | mwu_less | spearman | paired_wilcoxon
    metric: str                   # reeval | best | decoy
    arm: str
    baseline: str = ""
    root: str = "runs"
    alpha: float = 0.05

    def key(self): return _stamp(asdict(self))


@dataclass
class Claim:
    """One assertion the project makes, bound to the test that supports it."""
    id: str
    statement: str
    test: dict
    prereg: str = ""
    status: str = "untested"      # supported | not_supported | underpowered | refuted | superseded
    evidence: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    cited_in: list = field(default_factory=list)   # doc sections that repeat this number
    superseded_by: str = ""

    def path(self): return f"{RESEARCH_DIR}/claims/{self.id}.json"


def save(rec):
    os.makedirs(os.path.dirname(rec.path()), exist_ok=True)
    json.dump(asdict(rec), open(rec.path(), "w"), ensure_ascii=False, indent=1)
    return rec.path()


def load(cls, path):
    return cls(**json.load(open(path)))
