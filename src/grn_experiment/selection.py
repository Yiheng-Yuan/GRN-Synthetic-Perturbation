"""Budget-matched, leakage-free condition selection.

The active rule is a *design heuristic*, not Bayesian information gain: fitted
networks are not posterior samples.  A rival pair must already have passed the
independent-of-fitting replicate check before being supplied here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Mapping

import numpy as np

from .metrics import eliminated_fraction, sliced_wasserstein
from .protocol import Condition


Predictor = Callable[[Condition, float], np.ndarray]


@dataclass(frozen=True)
class Rival:
    identifier: str
    support: frozenset[tuple[int, int]]
    predict: Predictor


@dataclass(frozen=True)
class Choice:
    condition: Condition
    fallback: bool
    profiled_separation: float


def _sorted_candidates(candidates: Iterable[Condition]) -> list[Condition]:
    ordered = sorted(set(candidates), key=lambda c: (c.target, c.dose, c.time))
    if any(c.target is None or c.time <= 0 for c in ordered):
        raise ValueError("active candidates must be post-perturbation targets")
    return ordered


def uniform_order(candidates: Iterable[Condition]) -> list[Condition]:
    """Frozen cyclic target/dose/time coverage, then deterministic remainder."""

    ordered = _sorted_candidates(candidates)
    targets = sorted({c.target for c in ordered})
    doses = sorted({c.dose for c in ordered})
    times = sorted({c.time for c in ordered})
    available = set(ordered)
    first: list[Condition] = []
    for index, target in enumerate(targets):
        for offset in (0, 3):
            proposal = Condition(
                target=target,
                dose=doses[(index + offset) % len(doses)],
                time=times[(index + offset) % len(times)],
            )
            if proposal in available and proposal not in first:
                first.append(proposal)
    return first + [candidate for candidate in ordered if candidate not in first]


def uniform_sample(candidates: Iterable[Condition], budget: int = 12) -> list[Condition]:
    ordered = uniform_order(candidates)
    if not 0 <= budget <= len(ordered):
        raise ValueError("budget exceeds candidate pool")
    return ordered[:budget]


def random_sample(
    candidates: Iterable[Condition], *, budget: int = 12, seed: int
) -> list[Condition]:
    ordered = _sorted_candidates(candidates)
    if not 0 <= budget <= len(ordered):
        raise ValueError("budget exceeds candidate pool")
    indices = np.random.default_rng(seed).choice(len(ordered), budget, replace=False)
    return [ordered[int(index)] for index in indices]


def most_distinct_pair(rivals: Iterable[Rival]) -> tuple[Rival, Rival] | None:
    """Find a structurally discordant, qualified pair; never count restarts."""

    unique = {r.identifier: r for r in rivals}
    if len(unique) < 2:
        return None
    pairs = [
        (len(a.support ^ b.support), a.identifier, b.identifier, a, b)
        for a, b in combinations(sorted(unique.values(), key=lambda r: r.identifier), 2)
        if a.support != b.support
    ]
    if not pairs:
        return None
    _, _, _, left, right = sorted(pairs, key=lambda item: (-item[0], item[1], item[2]))[0]
    return left, right


def profiled_separation(
    condition: Condition,
    rivals: tuple[Rival, Rival],
    efficiency_values: Iterable[float],
    *,
    projection_seed: int = 0,
) -> float:
    """Worst-case prediction separation after *each* network varies rho.

    For an unobserved target this profiles a prespecified plausible range; it
    does not fit or reveal that target's actual efficiency before acquisition.
    """

    values = tuple(float(value) for value in efficiency_values)
    if not values or any(not 0 <= value <= 1 for value in values):
        raise ValueError("efficiency grid must be nonempty and within [0, 1]")
    left = [rivals[0].predict(condition, value) for value in values]
    right = [rivals[1].predict(condition, value) for value in values]
    return min(
        sliced_wasserstein(a, b, seed=projection_seed)
        for a in left
        for b in right
    )


def choose_active(
    candidates: Iterable[Condition],
    acquired: Iterable[Condition],
    qualified_rivals: Iterable[Rival],
    *,
    efficiency_grid: Mapping[int, Iterable[float]] | None = None,
    default_efficiency_grid: tuple[float, ...] = (0.5, 0.625, 0.75, 0.875, 1.0),
    projection_seed: int = 0,
) -> Choice:
    """Choose one not-yet-acquired condition using no candidate outcomes.

    If no distinct qualified rival pair exists, follow the frozen uniform
    order and mark the fallback so its frequency can be reported.
    """

    unavailable = set(acquired)
    remaining = [c for c in _sorted_candidates(candidates) if c not in unavailable]
    if not remaining:
        raise ValueError("no unacquired candidate conditions remain")
    pair = most_distinct_pair(qualified_rivals)
    if pair is None:
        chosen = next(c for c in uniform_order(candidates) if c not in unavailable)
        return Choice(chosen, True, float("nan"))
    grids = efficiency_grid or {}
    scored = [
        (
            profiled_separation(
                c,
                pair,
                grids.get(c.target, default_efficiency_grid),
                projection_seed=projection_seed,
            ),
            c,
        )
        for c in remaining
    ]
    best_score = max(score for score, _ in scored)
    # Explicit deterministic tie handling; no access to observed candidate data.
    best = min(
        (c for score, c in scored if np.isclose(score, best_score)),
        key=lambda c: (c.target, c.dose, c.time),
    )
    return Choice(best, False, best_score)


@dataclass
class FixedRegistry:
    """The denominator stays fixed even when strategies refit differently."""

    initial_supports: Mapping[str, frozenset[tuple[int, int]]]
    surviving: set[str]

    def __post_init__(self) -> None:
        supports = {str(identifier): frozenset(edges) for identifier, edges in self.initial_supports.items()}
        if len(set(supports.values())) != len(supports):
            raise ValueError("each registered identifier must denote a distinct edge structure")
        if not set(self.surviving) <= set(supports):
            raise ValueError("survivors must belong to the initial registry")
        self.initial_supports = supports
        self.surviving = set(self.surviving)

    @classmethod
    def from_rivals(cls, rivals: Iterable[Rival]) -> FixedRegistry:
        entries = tuple(rivals)
        supports = {r.identifier: r.support for r in entries}
        if len(supports) != len(entries):
            raise ValueError("rival identifiers must be unique")
        return cls(supports, set(supports))

    def update_survivors(self, qualified_ids: Iterable[str]) -> None:
        current = set(qualified_ids)
        if not current <= self.surviving:
            raise ValueError("eliminated initial structures cannot re-enter")
        self.surviving = current

    @property
    def eliminated_fraction(self) -> float:
        return eliminated_fraction(self.initial_supports, self.surviving)
