"""Register and re-test competing *structures*, not optimizer restarts.

The thresholds and candidate-generation rule must be chosen on development
networks before a blind network is opened.  A check replicate is independent
of parameter fitting, but it may be used during sequential acquisition and
is therefore not an untouched final test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from .learning import SparseRNAODE, TrainConfig, fit_unpaired_snapshots, predict_population
from .metrics import control_adjusted_rmse, sliced_wasserstein
from .protocol import Condition, ReplicatedSnapshot
from .selection import Rival


Edge = tuple[int, int]  # (target, regulator)


@dataclass(frozen=True)
class RivalSpecification:
    identifier: str
    support: frozenset[Edge]

    def __post_init__(self) -> None:
        if not self.identifier or not self.support:
            raise ValueError("a rival needs an identifier and nonempty edge support")


@dataclass(frozen=True)
class QualificationThresholds:
    """Frozen from development networks, before blind network inspection."""

    max_distribution_error: float
    max_effect_error: float
    min_local_edge_effect: float

    def __post_init__(self) -> None:
        values = (
            self.max_distribution_error,
            self.max_effect_error,
            self.min_local_edge_effect,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("qualification thresholds must be finite and positive")


@dataclass(frozen=True)
class QualificationReport:
    identifier: str
    max_distribution_error: float
    max_effect_error: float
    minimum_edge_effect: float
    qualified: bool


def support_mask(n_genes: int, support: frozenset[Edge]) -> np.ndarray:
    """Constrain a refit to the exact directed edge support being tested."""

    mask = np.zeros((n_genes, n_genes), dtype=bool)
    for target, regulator in support:
        if not (0 <= target < n_genes and 0 <= regulator < n_genes) or target == regulator:
            raise ValueError("support contains an invalid directed edge")
        mask[target, regulator] = True
    return mask


def top_edge_support(
    model: SparseRNAODE, regulators: Sequence[int], *, n_edges: int = 40,
    max_indegree: int = 3,
    max_outdegree: int = 5,
) -> frozenset[Edge]:
    """Deterministic proposal rule; this is not a claim that edges are true."""

    allowed = set(regulators)
    if not allowed or len(allowed) != len(regulators):
        raise ValueError("regulators must be distinct and nonempty")
    weights = model.edge_weights.detach().cpu().numpy()
    choices = sorted(
        (
            (-abs(float(weights[target, regulator])), target, regulator)
            for target in range(model.n_genes)
            for regulator in allowed
            if target != regulator
        )
    )
    if (
        not 0 < n_edges <= len(choices)
        or max_indegree < 1
        or max_outdegree < 1
        or n_edges < model.n_genes - len(allowed)
    ):
        raise ValueError("invalid edge count")
    chosen: set[Edge] = set()
    indegrees = np.zeros(model.n_genes, dtype=int)
    outdegrees = np.zeros(model.n_genes, dtype=int)
    for target in range(model.n_genes):
        if target in allowed:
            continue
        options = (
            (score, regulator) for score, row, regulator in choices
            if row == target and outdegrees[regulator] < max_outdegree
        )
        try:
            _, regulator = min(options)
        except ValueError:
            raise ValueError("cannot cover every non-regulator under degree caps") from None
        chosen.add((target, regulator))
        indegrees[target] += 1
        outdegrees[regulator] += 1
    for _, target, regulator in choices:
        if (
            (target, regulator) not in chosen
            and indegrees[target] < max_indegree
            and outdegrees[regulator] < max_outdegree
        ):
            chosen.add((target, regulator))
            indegrees[target] += 1
            outdegrees[regulator] += 1
        if len(chosen) == n_edges:
            break
    if len(chosen) != n_edges:
        raise ValueError("edge budget exceeds the indegree-constrained search space")
    return frozenset(chosen)


def single_edge_swaps(
    support: frozenset[Edge],
    *,
    n_genes: int,
    regulators: Sequence[int],
    count: int,
    seed: int,
    max_indegree: int = 3,
    max_outdegree: int = 5,
) -> tuple[RivalSpecification, ...]:
    """Predeclared search of distinct supports at the same edge budget."""

    if count < 0 or max_indegree < 1 or max_outdegree < 1:
        raise ValueError("count and degree limits are invalid")
    support_mask(n_genes, support)
    possible = frozenset(
        (target, regulator)
        for target in range(n_genes)
        for regulator in regulators
        if target != regulator
    )
    if not support <= possible:
        raise ValueError("reference support includes a nonregulator edge")
    absent = tuple(sorted(possible - support))
    present = tuple(sorted(support))
    indegrees = np.zeros(n_genes, dtype=int)
    outdegrees = np.zeros(n_genes, dtype=int)
    for target, regulator in support:
        indegrees[target] += 1
        outdegrees[regulator] += 1
    nonregulators = set(range(n_genes)) - set(regulators)
    if (
        np.any(indegrees > max_indegree)
        or np.any(outdegrees > max_outdegree)
        or any(indegrees[target] < 1 for target in nonregulators)
    ):
        raise ValueError("reference support violates the graph constraints")
    all_swaps = [
        (removed, added)
        for removed in present
        for added in absent
        if indegrees[added[0]] - int(removed[0] == added[0]) < max_indegree
        and outdegrees[added[1]] - int(removed[1] == added[1]) < max_outdegree
        and (removed[0] not in nonregulators or indegrees[removed[0]] > 1 or added[0] == removed[0])
    ]
    if count > len(all_swaps):
        raise ValueError("not enough valid single-edge swaps")
    chosen = np.random.default_rng(seed).choice(len(all_swaps), size=count, replace=False)
    return tuple(
        RivalSpecification(
            f"swap-{index:03d}",
            frozenset((support - {all_swaps[int(pair)][0]}) | {all_swaps[int(pair)][1]}),
        )
        for index, pair in enumerate(chosen)
    )


def minimum_local_effect(
    model: SparseRNAODE, support: frozenset[Edge], states: np.ndarray
) -> float:
    """A forced-present edge must have a non-negligible actual local action."""

    points = np.asarray(states, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != model.n_genes or not len(points):
        raise ValueError("states must be nonempty cells-by-genes")
    if not support:
        return float("inf")
    device = next(model.parameters()).device
    effects: dict[Edge, list[float]] = {edge: [] for edge in support}
    for state in points:
        jacobian = model.local_jacobian(torch.as_tensor(state, device=device)).detach().cpu().numpy()
        for target, regulator in support:
            effects[(target, regulator)].append(abs(float(jacobian[target, regulator])))
    # A single saturated state should not make a real edge count as absent.
    return min(float(np.median(values)) for values in effects.values())


def qualify_rival(
    specification: RivalSpecification,
    model: SparseRNAODE,
    baseline: ReplicatedSnapshot,
    observations: Mapping[Condition, ReplicatedSnapshot],
    thresholds: QualificationThresholds,
    *,
    effect_states: np.ndarray,
    step: float = 0.1,
    projection_seed: int = 0,
) -> QualificationReport:
    """Apply frozen thresholds to the third replicate only, plus edge action."""

    if not observations:
        raise ValueError("at least one acquired condition is required")
    initial = baseline.check_treated
    distribution_errors = []
    effect_errors = []
    for condition, snapshot in observations.items():
        if condition.target is None or snapshot.check_control is None:
            raise ValueError("a treated condition and its check control are required")
        predicted_treated = predict_population(model, initial, condition, step=step)
        predicted_control = predict_population(
            model, initial, Condition(None, 0, condition.time), step=step
        )
        distribution_errors.extend(
            (
                sliced_wasserstein(predicted_treated, snapshot.check_treated, seed=projection_seed),
                sliced_wasserstein(predicted_control, snapshot.check_control, seed=projection_seed),
            )
        )
        effect_errors.append(
            control_adjusted_rmse(
                predicted_treated, predicted_control,
                snapshot.check_treated, snapshot.check_control,
            )
        )
    minimum_effect = minimum_local_effect(model, specification.support, effect_states)
    distribution = max(distribution_errors)
    effect = max(effect_errors)
    qualifies = (
        distribution <= thresholds.max_distribution_error
        and effect <= thresholds.max_effect_error
        and minimum_effect >= thresholds.min_local_edge_effect
    )
    return QualificationReport(
        specification.identifier, distribution, effect, minimum_effect, qualifies
    )


def fit_structural_rivals(
    specifications: Sequence[RivalSpecification],
    baseline: ReplicatedSnapshot,
    observations: Mapping[Condition, ReplicatedSnapshot],
    thresholds: QualificationThresholds,
    *,
    config: TrainConfig,
    effect_states: np.ndarray,
    min_efficiency: float = 0.5,
) -> tuple[tuple[Rival, ...], tuple[QualificationReport, ...]]:
    """Refit each structure to acquired fit data, then qualify on check data.

    The returned rivals predict unobserved candidate outcomes by profiling
    efficiency externally; no candidate snapshot enters this function.
    """

    if len({spec.identifier for spec in specifications}) != len(specifications):
        raise ValueError("rival identifiers must be unique")
    if len({spec.support for spec in specifications}) != len(specifications):
        raise ValueError("rival supports must be structurally distinct")
    if not observations:
        raise ValueError("initial observations are required")
    n_genes = baseline.treated.shape[-1]
    rivals: list[Rival] = []
    reports: list[QualificationReport] = []
    initial_cells = baseline.fit_treated.reshape(-1, n_genes)
    for index, specification in enumerate(specifications):
        torch.manual_seed(config.seed + index)
        model = SparseRNAODE(
            n_genes,
            min_efficiency=min_efficiency,
            allowed_edges=support_mask(n_genes, specification.support),
        )
        fit_unpaired_snapshots(model, baseline, observations, config=config)
        report = qualify_rival(
            specification, model, baseline, observations, thresholds,
            effect_states=effect_states, step=config.step,
        )
        reports.append(report)
        if report.qualified:
            rivals.append(
                Rival(
                    specification.identifier,
                    specification.support,
                    lambda condition, efficiency, fitted=model, cells=initial_cells: predict_population(
                        fitted, cells, condition, rho_override=efficiency, step=config.step
                    ),
                )
            )
    return tuple(rivals), tuple(reports)
