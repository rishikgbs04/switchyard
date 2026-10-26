"""Model routing.

Policies:
  - "cheapest": lowest-cost model, escalate on failure
  - "best":     highest tier, fall back downward on failure
  - "auto":     score prompt difficulty, pick the cheapest model in the
                matching tier, fall back to neighbouring tiers

The difficulty scorer is a transparent feature-based heuristic in v0. It is
deliberately structured as features -> score so it can be swapped for a
learned classifier (see README roadmap) without touching the gateway.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from switchyard.config import ModelConfig, Settings
from switchyard.schemas import Message

HARD_PATTERNS = re.compile(
    r"\b(prove|derive|step[- ]by[- ]step|analy[sz]e|architect|design a|trade-?offs?|"
    r"optimi[sz]e|refactor|debug|complexity|theorem|explain why|compare)\b",
    re.IGNORECASE,
)
CODE_PATTERN = re.compile(r"```|def |class |function |SELECT |#include|=>|\{\s*\n")
MATH_PATTERN = re.compile(r"[∑∫√≤≥]|\\frac|\\sum|\^\d|\d+\s*[*/^]\s*\d+")

POLICIES = {"auto", "cheapest", "best"}


@dataclass
class Difficulty:
    score: float  # 0..1
    features: dict[str, float]

    @property
    def tier(self) -> int:
        if self.score < 0.3:
            return 1
        if self.score < 0.55:
            return 2
        return 3


def score_difficulty(messages: list[Message]) -> Difficulty:
    user_text = "\n".join(m.content for m in messages if m.role == "user")
    last = messages[-1].content if messages else ""
    words = len(user_text.split())
    f = {
        "length": min(words / 400, 1.0),
        "hard_keywords": min(len(HARD_PATTERNS.findall(user_text)) / 2, 1.0),
        "code": 1.0 if CODE_PATTERN.search(user_text) else 0.0,
        "math": 1.0 if MATH_PATTERN.search(user_text) else 0.0,
        "multi_turn": min(sum(m.role == "user" for m in messages) / 6, 1.0),
        "multi_question": min(last.count("?") / 3, 1.0),
    }
    weights = {
        "length": 0.25,
        "hard_keywords": 0.35,
        "code": 0.3,
        "math": 0.2,
        "multi_turn": 0.1,
        "multi_question": 0.1,
    }
    score = min(sum(f[k] * weights[k] for k in f), 1.0)
    return Difficulty(score=round(score, 3), features=f)


def _cost(m: ModelConfig) -> float:
    # Blend assuming ~3:1 input:output tokens, a typical chat ratio.
    return 0.75 * m.input_cost_per_mtok + 0.25 * m.output_cost_per_mtok


@dataclass
class RoutePlan:
    candidates: list[ModelConfig]
    policy: str
    difficulty: Difficulty | None = None


def plan_route(settings: Settings, requested: str, messages: list[Message]) -> RoutePlan:
    models = settings.models
    if requested not in POLICIES:
        chosen = settings.model(requested)
        if chosen is None:
            raise KeyError(requested)
        # Explicit model: try it first, then same-tier peers, then others by cost.
        rest = sorted((m for m in models if m.name != chosen.name),
                      key=lambda m: (abs(m.tier - chosen.tier), _cost(m)))
        return RoutePlan([chosen, *rest], policy="explicit")

    if requested == "cheapest":
        return RoutePlan(sorted(models, key=_cost), policy="cheapest")
    if requested == "best":
        return RoutePlan(sorted(models, key=lambda m: (-m.tier, _cost(m))), policy="best")

    d = score_difficulty(messages)
    # Prefer the target tier; on failure escalate upward before falling back down,
    # because a too-weak answer is worse than a slightly pricier one.
    def rank(m: ModelConfig) -> tuple:
        gap = m.tier - d.tier
        order = 0 if gap == 0 else (1 if gap > 0 else 2)
        return (order, abs(gap), _cost(m))

    return RoutePlan(sorted(models, key=rank), policy="auto", difficulty=d)
