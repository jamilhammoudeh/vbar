"""Every metric in the catalogue has a complete card (template in docs/metrics.md)."""

import re
from pathlib import Path

import pytest

from vbar.metrics import METRICS

CARDS = Path(__file__).parents[1] / "src" / "vbar" / "metrics" / "cards"
FIELDS = (
    "Definition",
    "Why it matters",
    "Known-answer tests",
    "Convergence",
    "Invariance",
    "Mutation detection",
    "Anti-gaming",
    "Reproducibility",
    "Power",
    "Validity",
    "Changelog",
)


def cards():
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(CARDS.glob("*.md")))
    return dict(re.findall(r"^## (\S+ \S+ \(v\d+\))\n(.*?)(?=^## |\Z)", text, re.M | re.S))


@pytest.mark.parametrize("name", list(METRICS))
def test_every_metric_has_a_card_with_every_field(name):
    info = METRICS[name]
    body = cards().get(f"{info.id} {info.name} (v{info.version})")
    assert body is not None, f"no card for {info.id} {name}"
    for field in FIELDS:
        assert f"- {field}: " in body, f"{info.id} has no {field}"
    assert "{P:" not in body


def test_no_card_without_a_metric():
    known = {f"{m.id} {m.name} (v{m.version})" for m in METRICS.values()}
    assert set(cards()) <= known
