"""Auditable experiment sequencing without producing or inspecting truth.

The manifest precommits 4 development and 20 blind *network units*.  Each
unit has both generator families, using the same network and split seeds.
``CaseWorkflow`` then creates three independent, budget-matched selection
branches over one supplied observation pool.  It never accepts a truth matrix
or hidden dynamical parameters.  The caller is responsible for keeping the
source observation pool outside learner code and for constructing the initial
qualified rival registry from the two-fit/one-check replicate protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal, Mapping

import numpy as np

from .protocol import (
    ACTIVE_BUDGET,
    BlindStore,
    Condition,
    ReplicatedSnapshot,
    Split,
    candidate_grid,
)
from .selection import (
    Choice,
    FixedRegistry,
    Rival,
    choose_active,
    random_sample,
    uniform_order,
)


Family = Literal["sigmoid", "hill"]
Role = Literal["development", "blind"]
Strategy = Literal["active", "random", "uniform"]
FAMILIES: tuple[Family, Family] = ("sigmoid", "hill")
STRATEGIES: tuple[Strategy, Strategy, Strategy] = ("active", "random", "uniform")
MODEL_SELECTION_RULE = "minimum fixed-grid validation score, lexicographic tie"


@dataclass(frozen=True)
class NetworkPlan:
    identifier: str
    role: Role
    network_seed: int
    dynamics_seed: int
    observation_seed: int
    split_seed: int
    random_selection_seed: int
    families: tuple[Family, Family] = FAMILIES

    def __post_init__(self) -> None:
        if not self.identifier or self.role not in ("development", "blind"):
            raise ValueError("A network needs an identifier and a valid role")
        if self.families != FAMILIES:
            raise ValueError("Each network must be paired across both generator families")
        if min(
            self.network_seed,
            self.dynamics_seed,
            self.observation_seed,
            self.split_seed,
            self.random_selection_seed,
        ) < 0:
            raise ValueError("Manifest seeds must be nonnegative")


@dataclass(frozen=True)
class StudyManifest:
    networks: tuple[NetworkPlan, ...]
    master_seed: int

    def __post_init__(self) -> None:
        if len(self.networks) != 24:
            raise ValueError("The study requires exactly 24 network units")
        if sum(plan.role == "development" for plan in self.networks) != 4:
            raise ValueError("The study requires exactly 4 development networks")
        if sum(plan.role == "blind" for plan in self.networks) != 20:
            raise ValueError("The study requires exactly 20 blind networks")
        if len({plan.identifier for plan in self.networks}) != 24:
            raise ValueError("Network identifiers must be distinct")
        if len({plan.network_seed for plan in self.networks}) != 24:
            raise ValueError("Network construction seeds must be distinct")

    @property
    def development(self) -> tuple[NetworkPlan, ...]:
        return tuple(plan for plan in self.networks if plan.role == "development")

    @property
    def blind(self) -> tuple[NetworkPlan, ...]:
        return tuple(plan for plan in self.networks if plan.role == "blind")

    @property
    def n_statistical_units(self) -> int:
        """The main comparison has 20 independent networks, not 40 arms."""

        return len(self.blind)


def make_manifest(master_seed: int) -> StudyManifest:
    """Precommit network/split/selection seeds without generating any data."""

    rng = random.Random(master_seed)
    seeds = rng.sample(range(1, 2**31), 24 * 5)
    plans = []
    for index in range(24):
        role: Role = "development" if index < 4 else "blind"
        name = f"dev-{index:02d}" if index < 4 else f"blind-{index - 4:02d}"
        network_seed, dynamics_seed, observation_seed, split_seed, random_seed = seeds[
            5 * index : 5 * index + 5
        ]
        plans.append(
            NetworkPlan(
                name,
                role,
                network_seed,
                dynamics_seed,
                observation_seed,
                split_seed,
                random_seed,
            )
        )
    return StudyManifest(tuple(plans), master_seed)


@dataclass(frozen=True)
class AuditEvent:
    sequence: int
    network_id: str
    family: Family
    strategy: Strategy
    action: str
    condition: Condition | None = None
    detail: str = ""


@dataclass(frozen=True)
class Acquisition:
    choice: Choice
    observation: ReplicatedSnapshot
    ordinal: int


class StrategyRun:
    """One policy branch, with a sealed store and an append-only audit trail."""

    def __init__(
        self,
        plan: NetworkPlan,
        family: Family,
        strategy: Strategy,
        split: Split,
        observations: Mapping[Condition, ReplicatedSnapshot],
        initial_supports: Mapping[str, frozenset[tuple[int, int]]],
        *,
        n_genes: int = 24,
        cells_per_replicate: int = 128,
    ) -> None:
        if family not in plan.families or strategy not in STRATEGIES:
            raise ValueError("Invalid family or strategy")
        self.plan, self.family, self.strategy = plan, family, strategy
        self.split = split
        self._store = BlindStore(
            split,
            observations,
            n_genes=n_genes,
            cells_per_replicate=cells_per_replicate,
        )
        supports = {str(identifier): frozenset(edges) for identifier, edges in initial_supports.items()}
        if len(set(supports.values())) != len(supports):
            raise ValueError("Initial rival identifiers must represent different edge structures")
        self._registry = FixedRegistry(supports, set(supports))
        self._pool = candidate_grid(split)
        self._order = (
            random_sample(self._pool, budget=ACTIVE_BUDGET, seed=plan.random_selection_seed)
            if strategy == "random"
            else uniform_order(self._pool)[:ACTIVE_BUDGET]
            if strategy == "uniform"
            else []
        )
        self._audit: list[AuditEvent] = []
        self._selected_model: str | None = None
        self._validation_scores: tuple[tuple[str, float], ...] | None = None
        self._fallback_count = 0
        self._append("start", detail=f"initial_rivals={len(supports)}")

    def _append(self, action: str, *, condition: Condition | None = None, detail: str = "") -> None:
        self._audit.append(
            AuditEvent(
                len(self._audit) + 1,
                self.plan.identifier,
                self.family,
                self.strategy,
                action,
                condition,
                detail,
            )
        )

    @property
    def audit(self) -> tuple[AuditEvent, ...]:
        return tuple(self._audit)

    @property
    def acquired_conditions(self) -> tuple[Condition, ...]:
        return self._store.acquired_conditions

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

    @property
    def initial_registry_size(self) -> int:
        return len(self._registry.initial_supports)

    @property
    def eliminated_fraction(self) -> float:
        return self._registry.eliminated_fraction

    @property
    def validation_open(self) -> bool:
        return self._store.validation_open

    @property
    def scoring_open(self) -> bool:
        return self._store.predictions_locked

    @property
    def selected_model(self) -> str | None:
        return self._selected_model

    @property
    def active_cost(self):
        return self._store.active_cost

    def available_conditions(self) -> tuple[Condition, ...]:
        return self._store.available_conditions()

    def get(self, condition: Condition) -> ReplicatedSnapshot:
        return self._store.get(condition)

    def update_survivors(self, identifiers: set[str] | tuple[str, ...]) -> None:
        """Recheck only initially registered structures; denominator never grows."""

        if self.scoring_open:
            raise RuntimeError("Cannot update rivals after prediction lock")
        self._registry.update_survivors(identifiers)
        self._append("rivals_checked", detail=f"survivors={len(self._registry.surviving)}")

    def _check_current_rivals(self, rivals: tuple[Rival, ...]) -> None:
        identifiers = [r.identifier for r in rivals]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Current rival identifiers must be unique")
        if not set(identifiers) <= self._registry.surviving:
            raise ValueError("Current rivals must be surviving initial structures")
        if any(self._registry.initial_supports[r.identifier] != r.support for r in rivals):
            raise ValueError("A registered structure cannot change edge support")

    def acquire_next(self, *, qualified_rivals: tuple[Rival, ...] = ()) -> Acquisition:
        """Choose before reading the selected observation, then pay its cost."""

        ordinal = len(self.acquired_conditions)
        if ordinal >= ACTIVE_BUDGET:
            raise RuntimeError("The 12-condition candidate budget is exhausted")
        if self.strategy == "active":
            self._check_current_rivals(qualified_rivals)
            choice = choose_active(self._pool, self.acquired_conditions, qualified_rivals)
        else:
            if qualified_rivals:
                raise ValueError("Only the active policy takes fitted rival predictions")
            choice = Choice(self._order[ordinal], False, float("nan"))
        observation = self._store.acquire(choice.condition)
        self._fallback_count += int(choice.fallback)
        self._append(
            "acquired",
            condition=choice.condition,
            detail=f"ordinal={ordinal + 1};fallback={choice.fallback};cost_cells={self.active_cost.total_cells}",
        )
        return Acquisition(choice, observation, ordinal + 1)

    def select_model(self, validation_scores: Mapping[str, float]) -> str:
        """Freeze the lowest fixed-grid score, breaking ties by model ID."""

        if not self.validation_open:
            raise RuntimeError("Validation is sealed until all 12 acquisitions")
        if self._selected_model is not None:
            raise RuntimeError("Model selection is already frozen")
        if not validation_scores or any(
            not identifier or not math.isfinite(float(score))
            for identifier, score in validation_scores.items()
        ):
            raise ValueError("Validation scores must be nonempty and finite")
        scores = tuple(sorted((str(identifier), float(score)) for identifier, score in validation_scores.items()))
        selected = min(scores, key=lambda item: (item[1], item[0]))[0]
        self._validation_scores = scores
        self._selected_model = selected
        self._append("model_selected", detail=f"model={selected};rule={MODEL_SELECTION_RULE}")
        return selected

    def lock_predictions(self, predictions: Mapping[Condition, np.ndarray]) -> None:
        """Only a selected model with a full test commitment opens scoring."""

        if self._selected_model is None:
            raise RuntimeError("Select the final model on validation before locking predictions")
        self._store.lock_predictions(predictions)
        self._append("predictions_locked", detail=f"model={self._selected_model};count={len(predictions)}")

    def committed_prediction(self, condition: Condition) -> np.ndarray:
        return self._store.committed_prediction(condition)


class CaseWorkflow:
    """Three selection strategies sharing one case's starting observations."""

    def __init__(
        self,
        plan: NetworkPlan,
        family: Family,
        split: Split,
        observations: Mapping[Condition, ReplicatedSnapshot],
        initial_supports: Mapping[str, frozenset[tuple[int, int]]],
        *,
        n_genes: int = 24,
        cells_per_replicate: int = 128,
    ) -> None:
        self.plan, self.family, self.split = plan, family, split
        self._runs = {
            strategy: StrategyRun(
                plan,
                family,
                strategy,
                split,
                observations,
                initial_supports,
                n_genes=n_genes,
                cells_per_replicate=cells_per_replicate,
            )
            for strategy in STRATEGIES
        }

    def run(self, strategy: Strategy) -> StrategyRun:
        return self._runs[strategy]

    def audit_records(self) -> tuple[AuditEvent, ...]:
        return tuple(event for strategy in STRATEGIES for event in self._runs[strategy].audit)


def paired_blind_differences(
    manifest: StudyManifest,
    scores: Mapping[tuple[str, Family, Strategy], float],
    strategy: Strategy,
    comparator: Strategy,
) -> np.ndarray:
    """Return 20x2 differences, preserving each network's family pairing.

    Pass this array to ``paired_network_bootstrap_ci``.  Cell/replicate rows
    and the two generator families are not independent network units.
    """

    if strategy not in STRATEGIES or comparator not in STRATEGIES:
        raise ValueError("Unknown comparison strategy")
    rows = []
    for plan in manifest.blind:
        row = []
        for family in FAMILIES:
            try:
                difference = float(scores[(plan.identifier, family, strategy)]) - float(
                    scores[(plan.identifier, family, comparator)]
                )
            except KeyError as exc:
                raise ValueError(f"Missing blind result: {exc}") from None
            if not math.isfinite(difference):
                raise ValueError("Blind results must be finite")
            row.append(difference)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)
