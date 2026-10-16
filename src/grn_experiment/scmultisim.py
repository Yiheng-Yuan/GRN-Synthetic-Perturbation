"""Continuous-intervention extension of scMultiSim's promoter/U/M mechanism.

This is an independently implemented Python kinetic kernel, not a wrapper of
the R package and not a bitwise reproduction of its developmental simulation.
See docs/protocol/scmultisim-v1.md for the approved changes.  Only promoter
events and constant-production RNA segments are exact; feedback is held over
the numerical interval ``tau`` and requires convergence calibration.

All objects here are GENERATOR/SCORER PRIVATE.  Learners receive only the
ReplicatedSnapshot objects made by scmultisim_observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


SOURCE_COMMIT = "563c3b542ada289fb4d9789bb1bc944229e78699"
KERNEL_VERSION = "scmultisim-continuous-v1"


def _positive(value: float, name: str, *, zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def stable_seed(seed: int, *parts: Any) -> int:
    """Private, request-order-independent entropy; never use salted hash()."""
    seed = _integer(seed, "seed")
    payload = json.dumps([seed, *parts], sort_keys=True, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


@dataclass(frozen=True)
class PositiveMapping:
    """Frozen log-linear positive map; parameters are development assumptions.

    g(z) = scale * 10**(log10_offset + log10_slope*z).  No density rematching,
    historical ranks, clipping, or random draws occur during evaluation.
    Choosing this map does NOT certify biological realism or calibration.
    """

    log10_offset: float = 1.0
    log10_slope: float = 0.25
    scale: float = 1.0

    def __post_init__(self) -> None:
        if (isinstance(self.log10_offset, (bool, np.bool_)) or
                not isinstance(self.log10_offset, Real) or not math.isfinite(self.log10_offset)):
            raise ValueError("log10_offset must be finite")
        object.__setattr__(self, "log10_offset", float(self.log10_offset))
        object.__setattr__(self, "log10_slope", _positive(self.log10_slope, "log10_slope"))
        object.__setattr__(self, "scale", _positive(self.scale, "scale"))

    def __call__(self, z: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            out = np.exp(math.log(self.scale) + math.log(10) *
                         (self.log10_offset + self.log10_slope * np.asarray(z)))
        if not np.isfinite(out).all() or np.any(out <= 0):
            raise ValueError("positive mapping overflow/underflow: revise parameter domain")
        return out

    def to_dict(self) -> dict[str, float]:
        """Legacy parameter-file representation, preserved for existing data."""
        return {"log10_offset": self.log10_offset, "log10_slope": self.log10_slope,
                "scale": self.scale}

    def to_config(self) -> dict[str, str | float]:
        """Canonical typed calibration definition of the implemented function."""
        return {"kind": "positive_log_linear", **self.to_dict()}

    @classmethod
    def from_config(cls, value: Any) -> PositiveMapping:
        """Reconstruct exactly the calibrated map, with no family substitution.

        Parameter files continue to use ``to_dict``/``KineticParameters.from_dict``.
        Formal calibration uses this typed schema so an interpolated lookup
        table cannot silently stand in for the exponential map used at runtime.
        """
        expected = {"kind", "log10_offset", "log10_slope", "scale"}
        if not isinstance(value, dict):
            raise ValueError("mapping must be an object")
        if set(value) != expected:
            raise ValueError("mapping requires exactly kind, log10_offset, log10_slope, scale")
        if value["kind"] != "positive_log_linear":
            raise ValueError("mapping.kind must be positive_log_linear")
        return cls(log10_offset=value["log10_offset"],
                   log10_slope=value["log10_slope"], scale=value["scale"])


@dataclass(frozen=True, eq=False)
class KineticParameters:
    """One fixed network shared by every cell, dose, time and replicate.

    A[i,j] is j -> i. rho has one entry per gene; only designated regulators
    may be targeted. b is a PRE-GRN background, not an already-mapped R rate.
    """

    A: np.ndarray
    b: np.ndarray
    kon: np.ndarray
    koff: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    rho: np.ndarray
    regulators: tuple[int, ...]
    K: float
    mapping: PositiveMapping = field(default_factory=PositiveMapping)

    def __post_init__(self) -> None:
        A = np.array(self.A, dtype=np.float64, copy=True)
        if A.ndim != 2 or not A.shape[0] or A.shape[0] != A.shape[1] or not np.isfinite(A).all():
            raise ValueError("A must be a nonempty finite square target-by-regulator matrix")
        n = A.shape[0]
        regulators = tuple(_integer(q, "regulator") for q in self.regulators)
        if len(set(regulators)) != len(regulators) or any(q >= n for q in regulators):
            raise ValueError("regulators must be unique valid gene indices")
        nonregulators = sorted(set(range(n)) - set(regulators))
        if nonregulators and np.any(A[:, nonregulators]):
            raise ValueError("nonregulators cannot have outgoing edges")
        if np.any(np.diag(A)):
            raise ValueError("self-loops are excluded by this protocol")
        A.setflags(write=False)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "regulators", regulators)
        object.__setattr__(self, "K", _positive(self.K, "K"))
        if not isinstance(self.mapping, PositiveMapping):
            raise TypeError("mapping must be a frozen PositiveMapping")
        for name in ("b", "kon", "koff", "beta", "gamma", "rho"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite vector of length {n}")
            if name in ("kon", "koff", "beta", "gamma") and np.any(value <= 0):
                raise ValueError(f"{name} must be strictly positive")
            if name == "rho" and np.any((value < 0.6) | (value > 1)):
                raise ValueError("rho must lie within the registered [0.6, 1] domain")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        # r lies in [0,1], so check the entire reachable input box up front.
        self.mapping(self.b + np.minimum(self.A, 0).sum(axis=1))
        self.mapping(self.b + np.maximum(self.A, 0).sum(axis=1))

    @property
    def n_genes(self) -> int:
        return len(self.b)

    def _multiplier(self, target: int | None, dose: float) -> np.ndarray:
        dose = _positive(dose, "dose", zero=True)
        if dose > 1:
            raise ValueError("dose must lie in [0, 1]")
        factor = np.ones(self.n_genes)
        if target is None:
            if dose:
                raise ValueError("untreated control requires dose=0")
        else:
            target = _integer(target, "target")
            if target not in self.regulators:
                raise ValueError("target is not a designated regulator")
            factor[target] = 1 - self.rho[target] * dose
        return factor

    def _check_M(self, M: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=np.float64)
        if M.ndim < 1 or M.shape[-1] != self.n_genes or not np.isfinite(M).all() or np.any(M < 0):
            raise ValueError("M must be finite, nonnegative, with genes on the last axis")
        return M

    def production(self, M: np.ndarray, target: int | None = None, dose: float = 0) -> np.ndarray:
        M = self._check_M(M)
        with np.errstate(over="ignore"):
            denominator = M + self.K
        r = np.divide(M, denominator)
        # Overflow of M+K need not mean the dimensionless ratio is invalid.
        large = np.isinf(denominator)
        if np.any(large):
            r[large] = 1.0 / (1.0 + self.K / M[large])
        return self.mapping(self.b + r @ self.A.T) * self._multiplier(target, dose)

    def production_jacobian(self, M: np.ndarray, target: int | None = None,
                            dose: float = 0) -> np.ndarray:
        """J^s_ij = ds_i/dM_j, NOT the Jacobian of observed dM/dt."""
        M = self._check_M(M)
        rates = self.production(M, target, dose)
        with np.errstate(over="ignore"):
            derivative_r = (self.K / (M + self.K)) / (M + self.K)
        return (rates[..., :, None] * math.log(10) * self.mapping.log10_slope *
                self.A * derivative_r[..., None, :])

    def derivative(self, state: CellState, target: int | None = None,
                   dose: float = 0) -> tuple[np.ndarray, np.ndarray]:
        return (state.G * self.production(state.M, target, dose) - self.beta * state.U,
                self.beta * state.U - self.gamma * state.M)

    def to_dict(self) -> dict[str, Any]:
        return {**{name: getattr(self, name).tolist() for name in
                   ("A", "b", "kon", "koff", "beta", "gamma", "rho")},
                "regulators": list(self.regulators), "K": self.K, "mapping": self.mapping.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> KineticParameters:
        data = dict(value)
        data["mapping"] = PositiveMapping(**data["mapping"])
        return cls(**data)


@dataclass(eq=False)
class CellState:
    """Full joint state, never constructed by separately shuffling G/U/M."""

    G: np.ndarray
    U: np.ndarray
    M: np.ndarray

    def __post_init__(self) -> None:
        G = np.asarray(self.G)
        if G.ndim != 2 or min(G.shape) < 1 or not np.isin(G, (0, 1)).all():
            raise ValueError("G must be a nonempty cells-by-genes binary array (0=off, 1=on)")
        self.G = G.astype(np.uint8, copy=True)
        for name in ("U", "M"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != G.shape or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError(f"{name} must match G and be finite nonnegative continuous RNA")
            setattr(self, name, value)

    def copy(self) -> CellState:
        return CellState(self.G, self.U, self.M)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name).tolist() for name in ("G", "U", "M")}


def observation_counts(state: CellState) -> np.ndarray:
    """Single final floor; no intermediate rounding or measurement noise."""
    M = state.M
    if not np.isfinite(M).all() or np.any(M < 0) or np.any(M >= 2.0**63):
        raise ValueError("M outside finite nonnegative int64 observation range")
    return np.floor(M).astype(np.int64)


def exact_rna_segment(U: np.ndarray, M: np.ndarray, production: np.ndarray,
                      beta: np.ndarray, gamma: np.ndarray,
                      dt: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytic U/M solution for constant G*s, including beta=gamma.

    Written as nonnegative linear coefficients. A small-time series avoids
    subtracting almost equal exponentials in the newly transcribed M term.
    No negative-value clipping conceals integration failures.
    """
    U, M, p, b, g, h = np.broadcast_arrays(*[np.asarray(x, dtype=np.float64)
                                          for x in (U, M, production, beta, gamma, dt)])
    if any(not np.isfinite(x).all() for x in (U, M, p, b, g, h)):
        raise ValueError("RNA segment inputs must be finite")
    if any(np.any(x < 0) for x in (U, M, p, h)) or np.any(b <= 0) or np.any(g <= 0):
        raise ValueError("RNA/state/time must be nonnegative and kinetic rates positive")
    bh, gh = b * h, g * h
    eb, eg = np.exp(-bh), np.exp(-gh)
    fb, fg = -np.expm1(-bh) / b, -np.expm1(-gh) / g
    diff = np.abs(g - b)
    cross = np.empty_like(h)
    equal = diff == 0
    cross[equal] = h[equal] * eb[equal]
    np.divide(-np.expm1(-diff * h), diff, out=cross, where=~equal)
    cross[~equal] *= np.exp(-np.minimum(bh, gh)[~equal])
    # integral of (1-exp(-beta*t))*exp(-gamma*(h-t)).
    newly_spliced = fg - cross
    separated = diff > 0.1 * np.maximum(b, g)
    np.divide(b * (fb - fg), g - b, out=newly_spliced, where=separated)
    small = np.maximum(bh, gh) < 0.05
    if np.any(small):
        B, C = bh[small], gh[small]
        symmetric = np.ones_like(B)
        series = np.full_like(B, 0.5)
        # S_k(B,C) = sum_{j=0}^k B^j*C^(k-j).
        power_b = np.ones_like(B)
        for k in range(1, 10):
            power_b *= B
            symmetric = C * symmetric + power_b
            series += (-1)**k * symmetric / math.factorial(k + 2)
        newly_spliced[small] = h[small] * B * series
    out_u = U * eb + p * fb
    out_m = M * eg + b * U * cross + p * newly_spliced
    if not np.isfinite(out_u).all() or not np.isfinite(out_m).all() or np.any(out_u < 0) or np.any(out_m < 0):
        raise FloatingPointError("analytic RNA segment left its valid numerical domain")
    return out_u, out_m


def _uniform(keys: np.ndarray, counters: np.ndarray) -> np.ndarray:
    """Counter-based SplitMix64 streams, one stream per cell/promoter.

    Event scheduling and feedback resolution cannot change which variate
    belongs to a promoter. uint64 overflow is intentional modulo arithmetic.
    This is a simulation PRNG, not cryptography. Keys/counters are checkpointed.
    """
    with np.errstate(over="ignore"):
        z = keys + counters * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    # 52-bit cell midpoints guarantee a representable open (0,1) interval.
    return ((z >> np.uint64(12)).astype(np.float64) + 0.5) * 2.0**-52


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def parameter_digest(parameters: KineticParameters) -> str:
    """Bind diagnostics to the exact hidden network, mapping and kinetic rates."""
    return _digest(parameters.to_dict())


class Simulator:
    """Joint-cell simulator with observation-invariant feedback and restart.

    Committed state is anchored on the feedback grid. A snapshot inside an
    interval is evaluated on a COPY, without resampling or splitting the
    committed path. This makes extra observation requests and arbitrary
    checkpoint times bitwise invariant within one NumPy/platform runtime.
    """

    def __init__(self, parameters: KineticParameters, initial: CellState, *, seed: int,
                 tau: float, target: int | None = None, dose: float = 0) -> None:
        self.parameters = parameters
        self.tau = _positive(tau, "tau")
        self.target = None if target is None else _integer(target, "target")
        self.dose = _positive(dose, "dose", zero=True)
        parameters._multiplier(self.target, self.dose)
        if initial.M.shape[1] != parameters.n_genes:
            raise ValueError("initial state gene dimension does not match parameters")
        self._state = initial.copy()
        self._tick = 0
        self.time = 0.0
        rng = np.random.Generator(np.random.PCG64(_integer(seed, "seed")))
        self._keys = rng.integers(0, np.iinfo(np.uint64).max, size=initial.G.shape,
                                  dtype=np.uint64)
        self._counters = np.ones(initial.G.shape, dtype=np.uint64)
        rates = np.where(initial.G, parameters.koff, parameters.kon)
        self._next_event = -np.log(_uniform(self._keys, self._counters)) / rates
        if not np.isfinite(self._next_event).all() or np.any(self._next_event <= 0):
            raise ValueError("initial promoter waiting times outside finite clock domain")
        self._held_rates = parameters.production(initial.M, target, dose)

    def _interval(self, end: float) -> tuple[CellState, np.ndarray, np.ndarray]:
        """Advance a copy from the last grid boundary to end <= next grid."""
        p = self.parameters
        state = self._state.copy()
        counters, events = self._counters.copy(), self._next_event.copy()
        local_time = np.full(state.M.shape, self._tick * self.tau)
        # Each promoter has its own exact waiting-time stream. Vectorized
        # batches need not order independent promoters relative to each other.
        while np.any(active := (events <= end)):
            rows, genes = np.nonzero(active)
            u, m = exact_rna_segment(state.U[active], state.M[active],
                                     state.G[active] * self._held_rates[active],
                                     p.beta[genes], p.gamma[genes],
                                     events[active] - local_time[active])
            state.U[active], state.M[active] = u, m
            local_time[active] = events[active]
            state.G[active] ^= np.uint8(1)
            counters[active] += np.uint64(1)
            rates = np.where(state.G[active], p.koff[genes], p.kon[genes])
            waiting = -np.log(_uniform(self._keys[active], counters[active])) / rates
            updated = events[active] + waiting
            if np.any(updated <= events[active]) or not np.isfinite(updated).all():
                raise FloatingPointError("promoter waiting time outside clock precision")
            events[active] = updated
        state.U, state.M = exact_rna_segment(state.U, state.M, state.G * self._held_rates,
                                              p.beta, p.gamma, end - local_time)
        return state, counters, events

    def advance(self, time: float) -> CellState:
        time = _positive(time, "time", zero=True)
        if time < self.time:
            raise ValueError("cannot move simulation time backwards")
        while (boundary := (self._tick + 1) * self.tau) <= time:
            self._state, self._counters, self._next_event = self._interval(boundary)
            self._tick += 1
            self._held_rates = self.parameters.production(self._state.M, self.target, self.dose)
        self.time = time
        if time == self._tick * self.tau:
            return self._state.copy()
        return self._interval(time)[0]

    def checkpoint(self) -> dict[str, Any]:
        value = {"schema": KERNEL_VERSION, "parameters": self.parameters.to_dict(),
                 "tau": self.tau, "target": self.target, "dose": self.dose,
                 "time": self.time, "feedback_tick": self._tick,
                 "anchor_state": self._state.to_dict(),
                 "held_rates": self._held_rates.tolist(),
                 "next_event": self._next_event.tolist(),
                 "rng_state": {"algorithm": "splitmix64-per-promoter-v1",
                               "keys": self._keys.tolist(), "counters": self._counters.tolist()}}
        return {**value, "sha256": _digest(value)}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> Simulator:
        value = dict(checkpoint)
        digest = value.pop("sha256", None)
        if digest != _digest(value) or value.get("schema") != KERNEL_VERSION:
            raise ValueError("checkpoint integrity or schema mismatch")
        obj = cls.__new__(cls)
        obj.parameters = KineticParameters.from_dict(value["parameters"])
        obj.tau = _positive(value["tau"], "tau")
        obj.target = None if value["target"] is None else _integer(value["target"], "target")
        obj.dose = _positive(value["dose"], "dose", zero=True)
        obj.parameters._multiplier(obj.target, obj.dose)
        obj.time = _positive(value["time"], "time", zero=True)
        obj._tick = _integer(value["feedback_tick"], "feedback_tick")
        if not obj._tick * obj.tau <= obj.time < (obj._tick + 1) * obj.tau:
            raise ValueError("invalid checkpoint feedback phase")
        obj._state = CellState(**value["anchor_state"])
        if obj._state.M.shape[1] != obj.parameters.n_genes:
            raise ValueError("checkpoint state dimension mismatch")
        rng = value["rng_state"]
        if rng["algorithm"] != "splitmix64-per-promoter-v1":
            raise ValueError("unsupported checkpoint random stream")
        obj._keys = np.asarray(rng["keys"], dtype=np.uint64)
        obj._counters = np.asarray(rng["counters"], dtype=np.uint64)
        obj._next_event = np.asarray(value["next_event"], dtype=np.float64)
        obj._held_rates = np.asarray(value["held_rates"], dtype=np.float64)
        shape = obj._state.M.shape
        if any(x.shape != shape for x in (obj._keys, obj._counters, obj._next_event, obj._held_rates)):
            raise ValueError("checkpoint array shape mismatch")
        if (not np.isfinite(obj._next_event).all() or
                np.any(obj._next_event <= obj._tick * obj.tau) or
                not np.isfinite(obj._held_rates).all() or np.any(obj._held_rates < 0) or
                np.any(obj._counters < 1)):
            raise ValueError("invalid pending events, counters or held rates")
        expected = obj.parameters.production(obj._state.M, obj.target, obj.dose)
        if not np.array_equal(expected, obj._held_rates):
            raise ValueError("checkpoint held rates inconsistent with anchor state")
        return obj


def development_parameters(seed: int, n_genes: int = 24) -> KineticParameters:
    """Reproducible DEVELOPMENT defaults, never frozen blind parameters.

    The 24-gene graph reuses only the existing topology constructor. No old
    sigmoid/Hill Dynamics or their stability rescaling is used. The four-gene
    fixture has a signed feed-forward chain and one uncoupled negative control.
    K=20 and the explicit rate ranges below still need development calibration.
    """
    seed = _integer(seed, "seed")
    n_genes = _integer(n_genes, "n_genes", minimum=1)
    if n_genes == 24:
        from .simulation import make_network
        network = make_network(seed)
        A, regulators = network.weights.copy(), network.regulators
    elif n_genes == 4:
        A = np.zeros((4, 4))
        A[1, 0], A[2, 1], A[2, 0] = 0.8, -0.8, 0.5
        regulators = (0, 1, 2, 3)
    else:
        raise ValueError("development fixtures support 4 or 24 genes; provide explicit parameters otherwise")
    rng = np.random.default_rng(stable_seed(seed, "development-kinetics-v1"))
    return KineticParameters(A=A, b=rng.uniform(-0.2, 0.2, n_genes),
                             kon=rng.uniform(0.7, 1.3, n_genes),
                             koff=rng.uniform(0.7, 1.3, n_genes),
                             beta=rng.uniform(0.7, 1.3, n_genes),
                             gamma=rng.uniform(0.7, 1.3, n_genes),
                             rho=rng.uniform(0.6, 1, n_genes),
                             regulators=regulators, K=20.0)
