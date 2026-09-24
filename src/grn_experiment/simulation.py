"""Ground-truth networks and positive, contractive RNA dynamics.

``weights[i, j]`` means regulator ``j`` acts on target gene ``i``.  The
simulator owns these weights and the fixed, target-specific intervention
efficiencies; neither belongs in the learner's training input.

All interventions start at ``t=0+`` and persist.  In particular,
``integrate(x0, 0, target=q, dose=d)`` returns the *pre-intervention* ``x0``.
Times are dimensionless and are sampling times, never intervention times.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np


Family = Literal["sigmoid", "hill"]
_MAX_INDEGREE = 3
_MAX_OUTDEGREE = 5
_CONTRACTION_LIMIT = 0.65


@dataclass(frozen=True)
class NetworkSpec:
    """A directed, signed ground-truth GRN, with rows as target genes."""

    weights: np.ndarray
    regulators: tuple[int, ...]
    seed: int

    def __post_init__(self) -> None:
        weights = np.array(self.weights, dtype=float, copy=True)
        if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
            raise ValueError("weights must be a square gene-by-gene matrix")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite")
        if np.any(np.diag(weights) != 0):
            raise ValueError("self-loops are excluded")
        regulators = tuple(int(gene) for gene in self.regulators)
        if len(set(regulators)) != len(regulators) or any(
            gene < 0 or gene >= weights.shape[0] for gene in regulators
        ):
            raise ValueError("regulators must be distinct valid gene indices")
        nonregulators = set(range(weights.shape[0])) - set(regulators)
        if nonregulators and np.any(weights[:, sorted(nonregulators)] != 0):
            raise ValueError("non-regulator genes cannot have outgoing edges")
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "regulators", regulators)

    @property
    def n_genes(self) -> int:
        return self.weights.shape[0]


def _capacity_matching(
    regulators: tuple[int, ...],
    n_genes: int,
    occupied: set[tuple[int, int]],
    out_capacity: np.ndarray,
    in_capacity: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Maximum feasible regulator->target edges under both degree capacities.

    A small integral max-flow calculation avoids a random greedy placement
    consuming the last slot of a target or regulator.  The RNG only changes
    which equally feasible maximum matching is returned.
    """

    source = 0
    first_regulator = 1
    first_target = first_regulator + len(regulators)
    sink = first_target + n_genes
    residual = np.zeros((sink + 1, sink + 1), dtype=np.int16)
    neighbors: list[list[int]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int) -> None:
        if capacity <= 0:
            return
        residual[start, end] = capacity
        neighbors[start].append(end)
        neighbors[end].append(start)

    regulator_nodes = {regulator: first_regulator + index for index, regulator in enumerate(regulators)}
    regulator_order = list(regulators)
    rng.shuffle(regulator_order)
    for regulator in regulator_order:
        add_edge(source, regulator_nodes[regulator], int(out_capacity[regulator]))

    candidates = [
        (target, regulator)
        for regulator in regulators
        if out_capacity[regulator] > 0
        for target in range(n_genes)
        if in_capacity[target] > 0
        and target != regulator
        and (target, regulator) not in occupied
    ]
    rng.shuffle(candidates)
    for target, regulator in candidates:
        add_edge(regulator_nodes[regulator], first_target + target, 1)
    target_order = list(range(n_genes))
    rng.shuffle(target_order)
    for target in target_order:
        add_edge(first_target + target, sink, int(in_capacity[target]))

    while True:
        previous = [-1] * (sink + 1)
        previous[source] = source
        queue = deque([source])
        while queue and previous[sink] == -1:
            current = queue.popleft()
            for neighbor in neighbors[current]:
                if previous[neighbor] == -1 and residual[current, neighbor] > 0:
                    previous[neighbor] = current
                    queue.append(neighbor)
                    if neighbor == sink:
                        break
        if previous[sink] == -1:
            break
        amount = sink
        flow = np.iinfo(np.int16).max
        while amount != source:
            before = previous[amount]
            flow = min(flow, int(residual[before, amount]))
            amount = before
        amount = sink
        while amount != source:
            before = previous[amount]
            residual[before, amount] -= flow
            residual[amount, before] += flow
            amount = before

    return [
        (target, regulator)
        for target, regulator in candidates
        if residual[regulator_nodes[regulator], first_target + target] == 0
    ]


def make_network(
    seed: int,
    n_genes: int = 24,
    n_regulators: int = 16,
    n_edges: int = 40,
) -> NetworkSpec:
    """Construct a sparse signed GRN with required feedforward/feedback motifs.

    Five regulators and seven edges are the minimum needed for the fixed
    motif scaffold.  Every non-regulator receives at least one incoming
    edge.  Remaining edges are capacity-matched without replacement while
    keeping indegree <= 3 and regulator outdegree <= 5.  The sign quota gives
    40--60% positive and 40--60% negative edges for every valid edge count.
    Magnitudes are sampled in [0.4, 1.2] and are never altered here.
    """

    if n_genes < 5 or not 5 <= n_regulators <= n_genes:
        raise ValueError("require n_genes >= n_regulators >= 5")
    minimum_edges = 7 + n_genes - n_regulators
    maximum_edges = min(_MAX_INDEGREE * n_genes, _MAX_OUTDEGREE * n_regulators)
    if not minimum_edges <= n_edges <= maximum_edges:
        raise ValueError("n_edges must fit motifs, non-regulator coverage, and degree caps")

    rng = np.random.default_rng(seed)
    regulators = tuple(int(gene) for gene in rng.permutation(n_genes)[:n_regulators])
    a, b, c, d, e = regulators[:5]
    # a->b->c and a->c; c<->d is positive; d<->e is negative.
    scaffold = [
        (b, a, -1),
        (c, b, +1),
        (c, a, -1),
        (d, c, +1),
        (c, d, +1),
        (e, d, +1),
        (d, e, -1),
    ]
    scaffold_edges = {(target, regulator) for target, regulator, _ in scaffold}
    nonregulators = tuple(gene for gene in range(n_genes) if gene not in regulators)
    topology: set[tuple[int, int]] | None = None
    # The mandatory-coverage matching is randomized.  If a particular one
    # leaves too few opportunities for the remaining edges, retry without
    # changing the public seed or weakening any requirement.
    for _ in range(64):
        occupied = set(scaffold_edges)
        indegree = np.bincount([target for target, _ in occupied], minlength=n_genes)
        outdegree = np.bincount([regulator for _, regulator in occupied], minlength=n_genes)
        mandatory_capacity = np.zeros(n_genes, dtype=int)
        mandatory_capacity[list(nonregulators)] = 1
        mandatory = _capacity_matching(
            regulators,
            n_genes,
            occupied,
            _MAX_OUTDEGREE - outdegree,
            mandatory_capacity,
            rng,
        )
        if len(mandatory) != len(nonregulators):
            raise ValueError("cannot cover every non-regulator under the degree caps")
        occupied.update(mandatory)
        indegree = np.bincount([target for target, _ in occupied], minlength=n_genes)
        outdegree = np.bincount([regulator for _, regulator in occupied], minlength=n_genes)
        remaining = n_edges - len(occupied)
        optional = _capacity_matching(
            regulators,
            n_genes,
            occupied,
            _MAX_OUTDEGREE - outdegree,
            _MAX_INDEGREE - indegree,
            rng,
        )
        if len(optional) >= remaining:
            topology = occupied | set(optional[:remaining])
            break
    if topology is None:
        raise ValueError("cannot place the requested edge count under all degree constraints")

    weights = np.zeros((n_genes, n_genes), dtype=float)
    for target, regulator, sign in scaffold:
        weights[target, regulator] = sign * rng.uniform(0.4, 1.2)

    positive_quota = (n_edges + 1) // 2
    remaining_signs = np.array(
        [+1] * (positive_quota - 4) + [-1] * (n_edges - positive_quota - 3),
        dtype=int,
    )
    remaining_edges = sorted(topology - scaffold_edges)
    rng.shuffle(remaining_edges)
    rng.shuffle(remaining_signs)
    for (target, regulator), sign in zip(remaining_edges, remaining_signs, strict=True):
        weights[target, regulator] = int(sign) * rng.uniform(0.4, 1.2)

    network = NetworkSpec(weights=weights, regulators=regulators, seed=seed)
    validate_network(network, n_edges=n_edges, require_motifs=True)
    return network


def network_diagnostics(network: NetworkSpec) -> dict[str, object]:
    """Report edge counts, degree caps, coverage, signs, and motifs."""

    weights = network.weights
    present = weights != 0
    has_chain = False
    has_feedforward = False
    for a in network.regulators:
        for b in network.regulators:
            if a == b or not present[b, a]:
                continue
            for c in range(network.n_genes):
                if c in (a, b) or not present[c, b]:
                    continue
                has_chain = True
                if present[c, a]:
                    has_feedforward = True
            if has_feedforward:
                break
        if has_feedforward:
            break
    has_positive_feedback = False
    has_negative_feedback = False
    for a in network.regulators:
        for b in network.regulators:
            if a >= b or not (present[a, b] and present[b, a]):
                continue
            product = weights[a, b] * weights[b, a]
            has_positive_feedback |= product > 0
            has_negative_feedback |= product < 0
    return {
        "n_edges": int(np.count_nonzero(present)),
        "n_positive": int(np.count_nonzero(weights > 0)),
        "n_negative": int(np.count_nonzero(weights < 0)),
        "indegree": np.count_nonzero(present, axis=1),
        "outdegree": np.count_nonzero(present, axis=0),
        "nonregulators": tuple(gene for gene in range(network.n_genes) if gene not in network.regulators),
        "has_chain": has_chain,
        "has_feedforward": has_feedforward,
        "has_positive_feedback": has_positive_feedback,
        "has_negative_feedback": has_negative_feedback,
    }


def validate_network(
    network: NetworkSpec, *, n_edges: int | None = None, require_motifs: bool = True
) -> dict[str, object]:
    """Raise for a malformed predeclared network; otherwise return diagnostics."""

    report = network_diagnostics(network)
    if n_edges is not None and report["n_edges"] != n_edges:
        raise ValueError("incorrect number of edges")
    if np.any(report["indegree"] > _MAX_INDEGREE):
        raise ValueError("indegree exceeds three")
    if np.any(report["outdegree"] > _MAX_OUTDEGREE):
        raise ValueError("regulator outdegree exceeds five")
    if any(report["indegree"][gene] == 0 for gene in report["nonregulators"]):
        raise ValueError("each non-regulator must have an incoming edge")
    count = report["n_edges"]
    if count == 0 or not (0.4 <= report["n_positive"] / count <= 0.6):
        raise ValueError("positive/negative edge balance must be within 40--60%")
    if require_motifs and not all(
        report[key]
        for key in (
            "has_chain",
            "has_feedforward",
            "has_positive_feedback",
            "has_negative_feedback",
        )
    ):
        raise ValueError("network is missing a required motif")
    return report


@dataclass(frozen=True)
class Dynamics:
    """Positive ODE with a provable unique attracting equilibrium.

    For either family, production is globally nonnegative on ``x >= 0``.
    The induced steady-state map ``x -> production(x)/gamma`` has an
    infinity-norm Lipschitz constant at most 0.65, including under any
    admissible intervention.  This makes the fixed point unique and also
    bounds the ODE's matrix measure by a negative number.
    """

    network: NetworkSpec
    family: Family
    gamma: np.ndarray
    basal: np.ndarray
    amplitude: np.ndarray
    bias: np.ndarray
    hill_k: np.ndarray
    coupling: np.ndarray
    rho: np.ndarray

    def __post_init__(self) -> None:
        if self.family not in ("sigmoid", "hill"):
            raise ValueError("family must be 'sigmoid' or 'hill'")
        n = self.network.n_genes
        for field in ("gamma", "basal", "amplitude", "bias", "hill_k", "coupling", "rho"):
            value = np.array(getattr(self, field), dtype=float, copy=True)
            if value.shape != (n,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{field} must be a finite length-{n} vector")
            value.setflags(write=False)
            object.__setattr__(self, field, value)
        if np.any(self.gamma <= 0) or np.any(self.basal <= 0):
            raise ValueError("gamma and basal must be positive")
        if np.any(self.amplitude <= 0) or np.any(self.hill_k <= 0):
            raise ValueError("amplitude and hill_k must be positive")
        if np.any(self.coupling <= 0) or np.any(self.coupling > 1):
            raise ValueError("coupling must lie in (0, 1]")
        if np.any(self.rho < 0.7) or np.any(self.rho > 1):
            raise ValueError("rho must lie in [0.7, 1]")
        if self.max_contraction_ratio > _CONTRACTION_LIMIT + 1e-12:
            raise ValueError("production is not contractive enough")

    @property
    def max_contraction_ratio(self) -> float:
        weights = np.abs(self.network.weights)
        if self.family == "sigmoid":
            row_bound = 0.25 * weights.sum(axis=1)
        else:
            max_hill_derivative = 9.0 / (8.0 * np.sqrt(3.0) * self.hill_k)
            row_bound = weights @ max_hill_derivative
        ratios = self.amplitude * self.coupling * row_bound / self.gamma
        return float(np.max(ratios))

    def _condition(self, target: int | None, dose: float) -> tuple[int | None, float]:
        if not np.isfinite(dose) or not 0 <= dose <= 1:
            raise ValueError("nominal dose must lie in [0, 1]")
        if target is None:
            if dose != 0:
                raise ValueError("a nonzero dose requires a target")
            return None, 0.0
        target = int(target)
        if target not in self.network.regulators:
            raise ValueError("target must be a designated regulator")
        return target, float(dose)

    def _state(self, x: np.ndarray, *, batch: bool = True) -> np.ndarray:
        state = np.asarray(x, dtype=float)
        valid_ndim = (1, 2) if batch else (1,)
        if state.ndim not in valid_ndim or state.shape[-1] != self.network.n_genes:
            raise ValueError("state must have shape (genes,) or (cells, genes)")
        if not np.all(np.isfinite(state)) or np.any(state < 0):
            raise ValueError("RNA states must be finite and nonnegative")
        return state

    def production(
        self, x: np.ndarray, target: int | None = None, dose: float = 0.0
    ) -> np.ndarray:
        """Nonnegative synthesis rates; dose attenuates only the target row."""

        state = self._state(x)
        target, dose = self._condition(target, dose)
        weights = self.network.weights
        if self.family == "sigmoid":
            drive = (state - 1.5) @ weights.T
            z = self.bias + self.coupling * drive
            # Tanh form avoids overflow for large but finite signed drives.
            signal = 0.5 * (1.0 + np.tanh(0.5 * z))
            rate = self.basal + self.amplitude * signal
        else:
            square = state * state
            k_square = self.hill_k * self.hill_k
            signal = square / (k_square + square)
            positive = np.maximum(weights, 0.0)
            negative = np.maximum(-weights, 0.0)
            drive = signal @ positive.T + (1.0 - signal) @ negative.T
            rate = self.basal + self.amplitude * (0.5 + self.coupling * drive)
        if target is not None and dose != 0:
            rate = np.array(rate, copy=True)
            rate[..., target] *= 1.0 - self.rho[target] * dose
        return rate

    def vector_field(
        self, x: np.ndarray, target: int | None = None, dose: float = 0.0
    ) -> np.ndarray:
        """Derivative ``dx/dt`` after a sustained intervention at ``t=0+``."""

        state = self._state(x)
        return self.production(state, target=target, dose=dose) - self.gamma * state

    def jacobian(
        self, x: np.ndarray, target: int | None = None, dose: float = 0.0
    ) -> np.ndarray:
        """Analytic local derivative of the full vector field, not raw ``A``."""

        state = self._state(x, batch=False)
        target, dose = self._condition(target, dose)
        weights = self.network.weights
        if self.family == "sigmoid":
            z = self.bias + self.coupling * (weights @ (state - 1.5))
            signal = 0.5 * (1.0 + np.tanh(0.5 * z))
            row_factor = self.amplitude * self.coupling * signal * (1.0 - signal)
            jac = row_factor[:, None] * weights
        else:
            k_square = self.hill_k * self.hill_k
            derivative = 2.0 * k_square * state / (k_square + state * state) ** 2
            jac = (self.amplitude * self.coupling)[:, None] * weights * derivative[None, :]
        if target is not None and dose != 0:
            jac[target] *= 1.0 - self.rho[target] * dose
        jac[np.diag_indices(self.network.n_genes)] -= self.gamma
        return jac

    def integrate(
        self,
        x0: np.ndarray,
        time: float,
        target: int | None = None,
        dose: float = 0.0,
        *,
        step: float = 0.05,
    ) -> np.ndarray:
        """Fixed-step RK4 trajectory for a scalar sampling time.

        ``x0`` may be one cell or a batch of cells.  The intervention is
        applied over the open interval ``(0, time]``; at exactly zero the
        original pre-intervention state is returned without modification.
        """

        state = np.array(self._state(x0), copy=True)
        self._condition(target, dose)
        if not np.isfinite(time) or time < 0:
            raise ValueError("sampling time must be finite and nonnegative")
        if not np.isfinite(step) or not 0 < step <= 0.25:
            raise ValueError("step must lie in (0, 0.25]")
        if time == 0:
            return state
        n_steps = max(1, int(np.ceil(time / step)))
        h = time / n_steps
        for _ in range(n_steps):
            k1 = self.vector_field(state, target, dose)
            k2 = self.vector_field(np.maximum(state + 0.5 * h * k1, 0.0), target, dose)
            k3 = self.vector_field(np.maximum(state + 0.5 * h * k2, 0.0), target, dose)
            k4 = self.vector_field(np.maximum(state + h * k3, 0.0), target, dose)
            next_state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if np.any(next_state < -1e-10) or not np.all(np.isfinite(next_state)):
                raise FloatingPointError("integration produced an invalid RNA state")
            state = np.maximum(next_state, 0.0)
        return state

    def steady_state(
        self,
        target: int | None = None,
        dose: float = 0.0,
        *,
        initial: np.ndarray | None = None,
        tolerance: float = 1e-11,
        max_iterations: int = 10_000,
    ) -> np.ndarray:
        """Solve the fixed point separately; never treat a late sample as steady."""

        self._condition(target, dose)
        if tolerance <= 0 or max_iterations <= 0:
            raise ValueError("tolerance and max_iterations must be positive")
        state = (
            np.zeros(self.network.n_genes, dtype=float)
            if initial is None
            else np.array(self._state(initial, batch=False), copy=True)
        )
        for _ in range(max_iterations):
            updated = self.production(state, target, dose) / self.gamma
            if np.max(np.abs(updated - state)) <= tolerance:
                return updated
            state = updated
        raise RuntimeError("fixed-point iteration did not converge")


def make_dynamics(network: NetworkSpec, family: Family, seed: int) -> Dynamics:
    """Sample family-specific kinetics and fixed target efficiencies."""

    if family not in ("sigmoid", "hill"):
        raise ValueError("family must be 'sigmoid' or 'hill'")
    rng = np.random.default_rng(seed)
    n = network.n_genes
    gamma = rng.uniform(0.9, 1.1, size=n)
    basal = rng.uniform(0.2, 0.35, size=n)
    amplitude = rng.uniform(2.0, 2.6, size=n) if family == "sigmoid" else rng.uniform(1.5, 1.9, size=n)
    bias = rng.uniform(-0.2, 0.2, size=n)
    hill_k = rng.uniform(1.5, 2.0, size=n)
    rho = rng.uniform(0.7, 1.0, size=n)
    weights = np.abs(network.weights)
    if family == "sigmoid":
        row_bound = 0.25 * weights.sum(axis=1)
    else:
        max_hill_derivative = 9.0 / (8.0 * np.sqrt(3.0) * hill_k)
        row_bound = weights @ max_hill_derivative
    coupling = np.minimum(
        1.0,
        np.divide(
            _CONTRACTION_LIMIT * gamma,
            amplitude * row_bound,
            out=np.full(n, np.inf),
            where=row_bound > 0,
        ),
    )
    return Dynamics(
        network=network,
        family=family,
        gamma=gamma,
        basal=basal,
        amplitude=amplitude,
        bias=bias,
        hill_k=hill_k,
        coupling=coupling,
        rho=rho,
    )


def dynamics_diagnostics(dynamics: Dynamics, *, weak_edge_threshold: float = 0.02) -> dict[str, object]:
    """Development-only numerical checks, reporting weak edges without deletion."""

    if weak_edge_threshold < 0:
        raise ValueError("weak_edge_threshold must be nonnegative")
    baseline = dynamics.steady_state()
    alternate = dynamics.steady_state(initial=np.full(dynamics.network.n_genes, 10.0))
    if np.max(np.abs(baseline - alternate)) > 1e-7:
        raise RuntimeError("steady-state estimates disagree across initial states")
    if np.max(np.abs(dynamics.vector_field(baseline))) > 1e-8:
        raise RuntimeError("steady state has a large residual")
    full_jac = dynamics.jacobian(baseline)
    regulatory_jac = full_jac + np.diag(dynamics.gamma)
    edges = dynamics.network.weights != 0
    strength = np.abs(regulatory_jac[edges])
    if not np.all(np.sign(regulatory_jac[edges]) == np.sign(dynamics.network.weights[edges])):
        raise RuntimeError("local regulation has the wrong sign")
    # Test integration on a perturbed trajectory, not only the fixed point.
    target = dynamics.network.regulators[0]
    coarse = dynamics.integrate(baseline, 1.0, target, 0.7, step=0.1)
    fine = dynamics.integrate(baseline, 1.0, target, 0.7, step=0.05)
    step_error = float(np.max(np.abs(coarse - fine)))
    if step_error > 1e-4:
        raise RuntimeError("integration step-halving discrepancy is too large")
    return {
        "max_contraction_ratio": dynamics.max_contraction_ratio,
        "min_state": float(min(np.min(baseline), np.min(fine))),
        "steady_residual": float(np.max(np.abs(dynamics.vector_field(baseline)))),
        "step_halving_error": step_error,
        "min_edge_effect_at_baseline": float(np.min(strength)),
        "n_weak_edges_at_baseline": int(np.count_nonzero(strength < weak_edge_threshold)),
        "weak_edge_threshold": weak_edge_threshold,
    }
