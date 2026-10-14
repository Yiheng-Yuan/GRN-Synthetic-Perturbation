"""Scorer-side snapshots for the scMultiSim-derived continuous-time extension.

The public object remains ``ReplicatedSnapshot``. Initial/final hidden states,
parameters, and random-stream seeds must remain with the generator/scorer.
Every condition, replicate, and arm has fresh independent initial lineages.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

import numpy as np

from .protocol import Condition, SteadyKey, ReplicatedSnapshot, N_REPLICATES
from .scmultisim import KineticParameters, Simulator, observation_counts, parameter_digest
from .scmultisim_stationary import StationarityResult, sample_initial_states


def stream_seed(seed: int, *parts: object) -> int:
    """Stable, order-independent streams; never use Python's salted hash()."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    payload = json.dumps([int(seed), *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "big")


def key_record(key: Condition | SteadyKey) -> dict:
    record = {"target": key.target, "dose": key.dose}
    if isinstance(key, Condition):
        record.update(kind="timecourse", time=key.time)
    else:
        record["kind"] = "stationary"
    return record


def key_id(key: Condition | SteadyKey) -> str:
    payload = json.dumps(key_record(key), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class GeneratedSnapshot:
    """Generator-side bundle, intentionally NOT a learner-facing API."""

    snapshot: ReplicatedSnapshot
    truth: dict[str, np.ndarray]
    provenance: dict


class SnapshotSampler:
    def __init__(
        self,
        params: KineticParameters,
        *,
        seed: int,
        tau: float,
        baseline_stationarity: StationarityResult,
        cells_per_replicate: int = 128,
    ) -> None:
        if isinstance(cells_per_replicate, bool) or not isinstance(cells_per_replicate, Integral) or cells_per_replicate < 1:
            raise ValueError("cells_per_replicate must be a positive integer")
        stream_seed(seed, "validate")
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive")
        if not baseline_stationarity.passed or baseline_stationarity.burnin is None:
            raise ValueError("Baseline stationarity diagnostics must pass before sampling")
        self._check_diagnostic(params, tau, baseline_stationarity, None, 0.)
        self._params = params
        self._seed = seed
        self.tau = float(tau)
        self.cells_per_replicate = int(cells_per_replicate)
        self.burnin = float(baseline_stationarity.burnin)

    @staticmethod
    def _check_diagnostic(params, tau, result, target, dose):
        report = result.report
        if (report.get("parameter_sha256") != parameter_digest(params) or
                report.get("tau") != tau or report.get("target") != target or
                report.get("dose") != dose):
            raise ValueError("Stationarity diagnostic does not match parameters, tau or intervention")
        if not np.isfinite(result.burnin) or result.burnin <= 0:
            raise ValueError("Invalid accepted burnin")

    @property
    def n_genes(self) -> int:
        return self._params.n_genes

    def sample_with_truth(
        self,
        key: Condition | SteadyKey,
        *,
        stationary_result: StationarityResult | None = None,
    ) -> GeneratedSnapshot:
        if key.target is not None and key.target not in self._params.regulators:
            raise ValueError("target must be a designated regulator")
        if isinstance(key, SteadyKey):
            if stationary_result is None or not stationary_result.passed or stationary_result.burnin is None:
                raise ValueError("A passed condition-specific stationary diagnostic is required")
            self._check_diagnostic(self._params, self.tau, stationary_result, key.target, key.dose)
            duration = float(stationary_result.burnin)
        elif isinstance(key, Condition):
            duration = key.time
        else:
            raise TypeError("Expected Condition or SteadyKey")
        record = key_record(key)
        truth: dict[str, np.ndarray] = {}
        streams = []
        observations = {}
        for arm in (("treated",) if key.target is None else ("treated", "control")):
            counts = []
            states = {f"{stage}_{species}": [] for stage in ("initial", "final") for species in ("G", "U", "M")}
            production_jacobians = []
            for replicate in range(N_REPLICATES):
                initial_seed = stream_seed(self._seed, record, arm, replicate, "initial")
                trajectory_seed = stream_seed(self._seed, record, arm, replicate, "trajectory")
                initial = sample_initial_states(
                    self._params, self.cells_per_replicate, seed=initial_seed,
                    tau=self.tau, burnin=self.burnin,
                )
                target = key.target if arm == "treated" else None
                dose = key.dose if arm == "treated" else 0.
                simulator = Simulator(
                    self._params, initial, seed=trajectory_seed, tau=self.tau,
                    target=target, dose=dose,
                )
                final = simulator.advance(duration)
                counts.append(observation_counts(final))
                # Protocol J^s is the UNPERTURBED production sensitivity at
                # this reference M, distinct from the KD-scaled instantaneous
                # derivative of a treated cell's production rate.
                production_jacobians.append(self._params.production_jacobian(final.M))
                for stage, state in (("initial", initial), ("final", final)):
                    for species in ("G", "U", "M"):
                        states[f"{stage}_{species}"].append(getattr(state, species))
                streams.append({"arm": arm, "replicate": replicate,
                                "initial_seed": str(initial_seed), "trajectory_seed": str(trajectory_seed)})
            observations[arm] = np.stack(counts)
            truth.update({f"{arm}_{name}": np.stack(values) for name, values in states.items()})
            truth[f"{arm}_production_jacobian"] = np.stack(production_jacobians)
        return GeneratedSnapshot(
            ReplicatedSnapshot(observations["treated"], observations.get("control")),
            truth,
            {"condition": record, "burnin": self.burnin, "duration": duration,
             "tau": self.tau, "streams": streams, "paired_cells_visible": False,
             "production_jacobian_definition": "unperturbed_ds0_dM_at_final_continuous_M"},
        )

    def sample_condition(self, condition: Condition) -> ReplicatedSnapshot:
        return self.sample_with_truth(condition).snapshot
