"""Condition grids and information barriers for the perturbation experiment.

This module deliberately contains no simulator and no ground-truth network.  A
``BlindStore`` holds *observations only*.  Ground truth belongs to a separate
scorer so that the training and acquisition interfaces cannot accidentally
hand it to a learner.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from numbers import Integral
import random
from typing import Mapping

import numpy as np


N_GENES = 24
N_REGULATORS = 16
N_REPLICATES = 3
CELLS_PER_REPLICATE = 128
ACTIVE_BUDGET = 12

INITIAL_DOSES = (0.3, 0.7)
INITIAL_TIMES = (0.25, 1.0, 4.0)
ACTIVE_DOSES = (0.3, 0.5, 0.7)
ACTIVE_TIMES = (0.125, 0.25, 0.5, 1.0, 4.0)
VALIDATION_DOSES = INITIAL_DOSES
VALIDATION_TIMES = INITIAL_TIMES
SEALED_TIMES = (2.0, 8.0)


@dataclass(frozen=True)
class Condition:
    """A sampled condition; ``target=None`` denotes an untreated control.

    The sole condition at time zero is ``Condition(None, 0, 0)``: the common
    pre-intervention baseline.  A targeted condition therefore always has
    ``time > 0``.  Time is sampling time, never intervention start time; every
    intervention begins at ``t=0+`` and persists.
    """

    target: int | None
    dose: float
    time: float

    def __post_init__(self) -> None:
        dose = float(self.dose)
        time = float(self.time)
        if not math.isfinite(dose) or not math.isfinite(time):
            raise ValueError("Dose and time must be finite")
        if time < 0:
            raise ValueError("Sampling time must be nonnegative")
        if self.target is None:
            if dose != 0:
                raise ValueError("An untreated control must have dose zero")
        else:
            if isinstance(self.target, bool) or not isinstance(self.target, Integral) or self.target < 0:
                raise ValueError("A target must be a nonnegative integer")
            if not 0 < dose <= 1:
                raise ValueError("A targeted dose must be in (0, 1]")
            if time == 0:
                raise ValueError("Time zero is the shared pre-intervention baseline")
        object.__setattr__(self, "dose", dose)
        object.__setattr__(self, "time", time)
        if self.target is not None:
            object.__setattr__(self, "target", int(self.target))


COMMON_BASELINE = Condition(None, 0, 0)


@dataclass(frozen=True)
class Split:
    initial: tuple[int, ...]
    active: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = (self.initial, self.active, self.validation, self.test)
        if tuple(map(len, groups)) != (6, 6, 2, 2):
            raise ValueError("Regulators must be split 6/6/2/2")
        values = tuple(regulator for group in groups for regulator in group)
        if len(set(values)) != N_REGULATORS:
            raise ValueError("Split groups must be disjoint")
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in values):
            raise ValueError("Regulators must be nonnegative integers")
        for name, group in zip(("initial", "active", "validation", "test"), groups):
            object.__setattr__(self, name, tuple(map(int, group)))


def make_split(regulators: tuple[int, ...] | list[int], seed: int) -> Split:
    """Reproducibly split exactly 16 distinct regulators into 6/6/2/2."""

    values = list(regulators)
    if len(values) != N_REGULATORS or len(set(values)) != N_REGULATORS:
        raise ValueError("Exactly 16 distinct regulators are required")
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in values):
        raise ValueError("Regulators must be nonnegative integers")
    random.Random(seed).shuffle(values)
    return Split(tuple(values[:6]), tuple(values[6:12]), tuple(values[12:14]), tuple(values[14:16]))


def _grid(targets: tuple[int, ...], doses: tuple[float, ...], times: tuple[float, ...]) -> tuple[Condition, ...]:
    return tuple(Condition(target, dose, time) for target in targets for dose in doses for time in times)


def initial_grid(split: Split) -> tuple[Condition, ...]:
    """36 post-intervention conditions, excluding the one shared baseline."""

    return _grid(split.initial, INITIAL_DOSES, INITIAL_TIMES)


def candidate_grid(split: Split) -> tuple[Condition, ...]:
    """The fixed pool of 90 candidate conditions for each acquisition strategy."""

    return _grid(split.active, ACTIVE_DOSES, ACTIVE_TIMES)


def validation_grid(split: Split) -> tuple[Condition, ...]:
    """The fixed 12-condition validation grid, independent of acquisitions."""

    return _grid(split.validation, VALIDATION_DOSES, VALIDATION_TIMES)


def test_grids(split: Split) -> dict[str, tuple[Condition, ...]]:
    """Five named grids comprising the four non-independent test categories."""

    return {
        "new_target": _grid(split.test, (0.7,), (1.0,)),
        "new_dose": _grid(split.initial, (0.9,), (1.0,)),
        "new_time_interp": _grid(split.initial, (0.7,), (2.0,)),
        "new_time_extrap": _grid(split.initial, (0.7,), (8.0,)),
        "combination": _grid(split.test, (0.9,), (8.0,)),
    }


def required_test_conditions(split: Split) -> tuple[Condition, ...]:
    """Every prediction to commit before final observations become available.

    The two held-out control times are explicit so their measured values
    cannot silently be substituted into a prediction before scoring.
    """

    conditions = [condition for grid in test_grids(split).values() for condition in grid]
    conditions.extend(Condition(None, 0, time) for time in SEALED_TIMES)
    return tuple(dict.fromkeys(conditions))


@dataclass(frozen=True, eq=False)
class ReplicatedSnapshot:
    """Independent cell snapshots with two fit and one check replicate.

    Arrays have shape ``(3, cells_per_replicate, n_genes)``.  A targeted
    condition has its own matched untreated controls.  Only the common
    baseline or an explicit standalone control may omit ``matched_control``.
    ``BlindStore`` enforces the latter rule based on the condition key.
    """

    treated: np.ndarray
    matched_control: np.ndarray | None = None

    def __post_init__(self) -> None:
        treated = np.asarray(self.treated, dtype=np.float64).copy()
        if treated.ndim != 3 or treated.shape[0] != N_REPLICATES or min(treated.shape[1:]) < 1:
            raise ValueError("Treated RNA must have shape (3, cells, genes)")
        if not np.isfinite(treated).all() or (treated < 0).any():
            raise ValueError("RNA observations must be finite and nonnegative")
        treated.setflags(write=False)
        object.__setattr__(self, "treated", treated)
        if self.matched_control is not None:
            control = np.asarray(self.matched_control, dtype=np.float64).copy()
            if control.shape != treated.shape:
                raise ValueError("Matched controls must have the treated shape")
            if not np.isfinite(control).all() or (control < 0).any():
                raise ValueError("RNA controls must be finite and nonnegative")
            control.setflags(write=False)
            object.__setattr__(self, "matched_control", control)

    @property
    def fit_treated(self) -> np.ndarray:
        return self.treated[:2].copy()

    @property
    def check_treated(self) -> np.ndarray:
        return self.treated[2].copy()

    @property
    def fit_control(self) -> np.ndarray | None:
        return None if self.matched_control is None else self.matched_control[:2].copy()

    @property
    def check_control(self) -> np.ndarray | None:
        return None if self.matched_control is None else self.matched_control[2].copy()


@dataclass(frozen=True)
class SamplingCost:
    """Marginal cost of acquired candidate conditions, excluding shared data."""

    conditions: int
    treated_cells: int
    matched_control_cells: int

    @property
    def total_cells(self) -> int:
        return self.treated_cells + self.matched_control_cells


@dataclass(slots=True, eq=False)
class BlindStore:
    """Gate observations by protocol phase, without storing any truth.

    Construct one store per acquisition strategy with the *same* initial
    observations and candidate pool.  ``acquire`` exposes only the chosen
    condition (including its matched control) and charges its full cell cost.
    Three replicates remain available for the predeclared fit/check division;
    the check replicate is not a final independent test set.
    """

    split: Split
    _observations: dict[Condition, ReplicatedSnapshot] = field(repr=False)
    n_genes: int = N_GENES
    cells_per_replicate: int = CELLS_PER_REPLICATE
    _acquired: list[Condition] = field(init=False, default_factory=list, repr=False)
    _committed_predictions: dict[Condition, np.ndarray] | None = field(init=False, default=None, repr=False)

    def __init__(
        self,
        split: Split,
        observations: Mapping[Condition, ReplicatedSnapshot],
        *,
        n_genes: int = N_GENES,
        cells_per_replicate: int = CELLS_PER_REPLICATE,
    ) -> None:
        if n_genes < 1 or cells_per_replicate < 1:
            raise ValueError("n_genes and cells_per_replicate must be positive")
        self.split = split
        self.n_genes = n_genes
        self.cells_per_replicate = cells_per_replicate
        self._acquired = []
        self._committed_predictions = None
        allowed = self._all_protocol_conditions(split)
        copied: dict[Condition, ReplicatedSnapshot] = {}
        for condition, snapshot in observations.items():
            if condition not in allowed:
                raise ValueError(f"Condition outside protocol: {condition}")
            if not isinstance(snapshot, ReplicatedSnapshot):
                raise TypeError("Observations must be ReplicatedSnapshot instances")
            if snapshot.treated.shape[1:] != (cells_per_replicate, n_genes):
                raise ValueError("Observation shape does not match store dimensions")
            if condition.target is not None and snapshot.matched_control is None:
                raise ValueError("Every targeted condition requires a matched control")
            if condition.target is None and snapshot.matched_control is not None:
                raise ValueError("Standalone controls must not contain a second control")
            copied[condition] = deepcopy(snapshot)
        if COMMON_BASELINE not in copied:
            raise ValueError("The shared t=0 baseline is required")
        absent_initial = set(initial_grid(split)) - copied.keys()
        if absent_initial:
            raise ValueError(f"Missing {len(absent_initial)} initial conditions")
        self._observations = copied

    @staticmethod
    def _all_protocol_conditions(split: Split) -> set[Condition]:
        all_conditions = {COMMON_BASELINE}
        all_conditions.update(initial_grid(split))
        all_conditions.update(candidate_grid(split))
        all_conditions.update(validation_grid(split))
        all_conditions.update(required_test_conditions(split))
        return all_conditions

    @property
    def acquired_conditions(self) -> tuple[Condition, ...]:
        return tuple(self._acquired)

    @property
    def active_cost(self) -> SamplingCost:
        n = len(self._acquired)
        cells = n * N_REPLICATES * self.cells_per_replicate
        return SamplingCost(n, cells, cells)

    @property
    def common_baseline_cells(self) -> int:
        """Cost of the shared baseline, counted exactly once."""

        return N_REPLICATES * self.cells_per_replicate

    @property
    def validation_open(self) -> bool:
        return len(self._acquired) == ACTIVE_BUDGET

    @property
    def predictions_locked(self) -> bool:
        return self._committed_predictions is not None

    def _may_read(self, condition: Condition) -> bool:
        if condition == COMMON_BASELINE or condition in initial_grid(self.split):
            return True
        if condition in candidate_grid(self.split):
            return condition in self._acquired
        if condition in validation_grid(self.split):
            return self.validation_open
        if condition in required_test_conditions(self.split):
            return self.predictions_locked
        return False

    def available_conditions(self) -> tuple[Condition, ...]:
        """List *only* visible observations; hidden keys never appear here."""

        return tuple(condition for condition in self._observations if self._may_read(condition))

    def get(self, condition: Condition) -> ReplicatedSnapshot:
        """Return an isolated copy of a currently visible observation."""

        if not self._may_read(condition):
            raise PermissionError("This observation is not yet available")
        try:
            return deepcopy(self._observations[condition])
        except KeyError:
            raise KeyError("Visible observation was not supplied") from None

    def acquire(self, condition: Condition) -> ReplicatedSnapshot:
        """Spend one candidate unit and expose its treated and control cells."""

        if self.predictions_locked:
            raise RuntimeError("The experiment is already locked for scoring")
        if condition not in candidate_grid(self.split):
            raise ValueError("Only the 90 active-pool conditions may be acquired")
        if condition in self._acquired:
            raise ValueError("A candidate condition may be acquired only once")
        if len(self._acquired) >= ACTIVE_BUDGET:
            raise RuntimeError("The 12-condition acquisition budget is exhausted")
        if condition not in self._observations:
            raise KeyError("Candidate observation was not supplied")
        self._acquired.append(condition)
        return self.get(condition)

    def lock_predictions(self, predictions: Mapping[Condition, np.ndarray]) -> None:
        """Commit nonnegative cell-population predictions before final reveal."""

        if not self.validation_open:
            raise RuntimeError("Complete all 12 acquisitions before final scoring")
        if self.predictions_locked:
            raise RuntimeError("Predictions have already been locked")
        required = set(required_test_conditions(self.split))
        if set(predictions) != required:
            raise ValueError("Prediction keys must exactly match the final test grid and sealed controls")
        committed: dict[Condition, np.ndarray] = {}
        for condition, prediction in predictions.items():
            try:
                cells = np.asarray(prediction, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid prediction for {condition}") from exc
            if (
                cells.ndim != 2
                or cells.shape[0] < 1
                or cells.shape[1] != self.n_genes
                or not np.isfinite(cells).all()
                or (cells < 0).any()
            ):
                raise ValueError("Each final prediction must be finite nonnegative cells-by-genes")
            committed[condition] = cells.copy()
        self._committed_predictions = committed

    def committed_prediction(self, condition: Condition) -> np.ndarray:
        if self._committed_predictions is None:
            raise PermissionError("Predictions are not locked")
        return deepcopy(self._committed_predictions[condition])
