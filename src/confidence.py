"""Ambiguity / confidence resolution: floor threshold + top1/top2 margin (ADR-6)."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class ResolutionState(Enum):
    CONFIDENT = "confident"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


@dataclass
class ResolutionResult:
    state: ResolutionState
    primary: Optional[Any]  # Candidate; set for CONFIDENT and NOT_FOUND
    alternatives: List[Any] = field(default_factory=list)  # AMBIGUOUS: top-k including primary


def resolve(
    candidates: List[Any],
    floor_threshold: float = 0.75,
    margin_threshold: float = 0.10,
    top_k: int = 3,
) -> ResolutionResult:
    """Classify ranked match candidates into CONFIDENT, AMBIGUOUS, or NOT_FOUND.

    Floor is inclusive (top-1 combined_score >= floor_threshold). Margin is
    inclusive (top-1 minus top-2 >= margin_threshold), matching ADR-6.
    A missing top-2 (empty or single-candidate list) skips the margin check.
    """
    if not candidates:
        return ResolutionResult(
            state=ResolutionState.NOT_FOUND,
            primary=None,
            alternatives=[],
        )

    ranked = sorted(candidates, key=lambda c: c.combined_score, reverse=True)
    top1 = ranked[0]

    if top1.combined_score < floor_threshold:
        return ResolutionResult(
            state=ResolutionState.NOT_FOUND,
            primary=top1,
            alternatives=[],
        )

    if len(ranked) == 1:
        return ResolutionResult(
            state=ResolutionState.CONFIDENT,
            primary=top1,
            alternatives=[],
        )

    # Binary floats can make 0.85 - 0.75 slightly below 0.10; treat a
    # margin at the threshold as inclusive per ADR-6 / T4.5.
    margin = top1.combined_score - ranked[1].combined_score
    if margin >= margin_threshold or math.isclose(
        margin, margin_threshold, rel_tol=0.0, abs_tol=1e-9
    ):
        return ResolutionResult(
            state=ResolutionState.CONFIDENT,
            primary=top1,
            alternatives=[],
        )

    return ResolutionResult(
        state=ResolutionState.AMBIGUOUS,
        primary=None,
        alternatives=ranked[:top_k],
    )
