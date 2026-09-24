"""Independent, unpaired RNA snapshots from a hidden simulator.

This module belongs on the *simulator/scorer side* of the information
barrier.  Give learners only ``BlindStore`` views of the returned
``ReplicatedSnapshot`` objects, never the ``SnapshotSampler`` or its
``Dynamics`` object (which contains A, kinetic parameters, and true rho).

Cells are destroyed by sampling: a condition, time, replicate, and treatment
arm each draws new initial cells.  Treated and matched-control arms share a
replicate-level shift but never an initial cell or observation-noise draw.
No cell IDs or paired trajectories are returned.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral

import numpy as np

from .protocol import (
    COMMON_BASELINE,
    CELLS_PER_REPLICATE,
    N_REPLICATES,
    Condition,
    ReplicatedSnapshot,
    Split,
    candidate_grid,
    initial_grid,
    required_test_conditions,
    validation_grid,
)
from .simulation import Dynamics


@dataclass(frozen=True)
class ObservationConfig:
    """Continuous-RNA observation settings; technical noise is log-normal.

    These defaults are development assumptions, not estimates from a real
    sequencing platform.  A count-measurement layer is a separate stress
    test and must not be silently substituted into the primary experiment.
    """

    cells_per_replicate: int = CELLS_PER_REPLICATE
    cell_log_sd: float = 0.15
    replicate_log_sd: float = 0.04
    observation_log_sd: float = 0.03
    integration_step: float = 0.05

    def __post_init__(self) -> None:
        if (
            isinstance(self.cells_per_replicate, bool)
            or not isinstance(self.cells_per_replicate, Integral)
            or self.cells_per_replicate < 1
        ):
            raise ValueError("cells_per_replicate must be a positive integer")
        for name in ("cell_log_sd", "replicate_log_sd", "observation_log_sd"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.integration_step) or not 0 < self.integration_step <= 0.25:
            raise ValueError("integration_step must lie in (0, 0.25]")


def protocol_conditions(split: Split) -> tuple[Condition, ...]:
    """Unique conditions needed to populate one fully gated ``BlindStore``.

    This is a *key list only*: it does not generate data or expose any hidden
    response.  Time-zero baseline occurs exactly once.  Matched controls for
    targeted conditions live inside each ``ReplicatedSnapshot``. Standalone
    controls exist only for the sealed t=2 and t=8 scoring times.
    """

    keys = [COMMON_BASELINE]
    keys.extend(initial_grid(split))
    keys.extend(candidate_grid(split))
    keys.extend(validation_grid(split))
    keys.extend(required_test_conditions(split))
    return tuple(dict.fromkeys(keys))


class SnapshotSampler:
    """Deterministic-by-condition snapshot sampler with private generator truth.

    Calling conditions in different orders yields the same data.  This lets
    active, random, and uniform policies share one initial dataset and one
    sealed outcome pool, while each policy only *sees* acquired conditions.
    """

    def __init__(
        self,
        dynamics: Dynamics,
        *,
        seed: int,
        config: ObservationConfig = ObservationConfig(),
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self._dynamics = dynamics
        self._seed = seed
        self.config = config
        self._baseline_equilibrium = dynamics.steady_state()

    @property
    def n_genes(self) -> int:
        return self._dynamics.network.n_genes

    def _rng(self, condition: Condition) -> np.random.Generator:
        # Python's salted hash() would change on every process.  Float.hex()
        # retains the exact specified dose/time and keeps seeds order-free.
        signature = repr(
            (
                self._seed,
                condition.target,
                condition.dose.hex(),
                condition.time.hex(),
            )
        ).encode("ascii")
        entropy = int.from_bytes(hashlib.blake2b(signature, digest_size=16).digest(), "big")
        return np.random.default_rng(entropy)

    def _draw_initial(
        self, rng: np.random.Generator, replicate_shift: np.ndarray
    ) -> np.ndarray:
        config = self.config
        n = self.n_genes
        deviations = rng.normal(
            loc=-0.5 * config.cell_log_sd**2,
            scale=config.cell_log_sd,
            size=(N_REPLICATES, config.cells_per_replicate, n),
        )
        return self._baseline_equilibrium[None, None, :] * np.exp(
            replicate_shift + deviations
        )

    def _observe(self, rng: np.random.Generator, latent: np.ndarray) -> np.ndarray:
        sd = self.config.observation_log_sd
        noise = rng.normal(loc=-0.5 * sd**2, scale=sd, size=latent.shape)
        return latent * np.exp(noise)

    def _advance(
        self,
        cells: np.ndarray,
        condition: Condition,
        *,
        control: bool,
    ) -> np.ndarray:
        if condition.time == 0:
            return cells
        flat = cells.reshape(-1, self.n_genes)
        if control:
            evolved = self._dynamics.integrate(
                flat, condition.time, step=self.config.integration_step
            )
        else:
            evolved = self._dynamics.integrate(
                flat,
                condition.time,
                condition.target,
                condition.dose,
                step=self.config.integration_step,
            )
        return evolved.reshape(cells.shape)

    def sample_condition(self, condition: Condition) -> ReplicatedSnapshot:
        """Draw one condition; the same key always yields the same snapshot.

        ``Condition(None, 0, 0)`` is the sole pre-intervention baseline.
        For a targeted condition, the matched control is independently drawn
        from the same replicate-level population and evolved without an
        intervention.  Array row positions do *not* identify paired cells.
        """

        if condition.target is not None and condition.target not in self._dynamics.network.regulators:
            raise ValueError("condition target is not a designated regulator")
        rng = self._rng(condition)
        sd = self.config.replicate_log_sd
        replicate_shift = rng.normal(
            loc=-0.5 * sd**2,
            scale=sd,
            size=(N_REPLICATES, 1, self.n_genes),
        )
        treated_start = self._draw_initial(rng, replicate_shift)
        treated = self._observe(rng, self._advance(treated_start, condition, control=False))
        if condition.target is None:
            return ReplicatedSnapshot(treated=treated)
        control_start = self._draw_initial(rng, replicate_shift)
        control = self._observe(rng, self._advance(control_start, condition, control=True))
        return ReplicatedSnapshot(treated=treated, matched_control=control)

    def sample_many(
        self,
        conditions: tuple[Condition, ...] | list[Condition],
        *,
        include_baseline: bool = True,
    ) -> dict[Condition, ReplicatedSnapshot]:
        """Generate requested observations, de-duplicating the common baseline."""

        keys = ([COMMON_BASELINE] if include_baseline else []) + list(conditions)
        return {condition: self.sample_condition(condition) for condition in dict.fromkeys(keys)}

    def build_protocol_observations(self, split: Split) -> dict[Condition, ReplicatedSnapshot]:
        """Build a sealed pool for ``BlindStore``; keep this call scorer-side.

        Only observation arrays are returned.  The ``Dynamics`` object held
        by this sampler is the separate ground-truth record for scoring and
        must never be passed to training or acquisition code.
        """

        if set(split.initial + split.active + split.validation + split.test) != set(
            self._dynamics.network.regulators
        ):
            raise ValueError("split must contain exactly this network's regulators")
        return self.sample_many(protocol_conditions(split), include_baseline=False)
