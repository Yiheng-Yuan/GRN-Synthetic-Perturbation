"""Scorer-private, finite parameter equivalence for the M-only experiment.

These helpers use hidden parameters and must never be imported by a learner or
an acquisition policy.  They register one exact symmetry of the implemented
architecture; they do not assert that the remaining parameters are identifiable.
See docs/protocol/parameter-equivalence.md for the proof and domain conditions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .scmultisim import CellState, KineticParameters
from .metrics import jacobian_rmse


ParameterBounds = Mapping[str, Sequence[float]]
_PARAMETER_FIELDS = ("A", "b", "kon", "koff", "beta", "gamma", "rho")


@dataclass(frozen=True)
class RateSwap:
    """One gene-wise transformation, shared across every condition and state."""

    parameters: KineticParameters
    initial: CellState | None
    swapped_genes: tuple[int, ...]


def _mask(value: Sequence[bool] | np.ndarray, n_genes: int) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (n_genes,) or result.dtype.kind != "b":
        raise ValueError("swap_mask must be a Boolean vector with one entry per gene")
    return result


def _bounds_failures(parameters: KineticParameters,
                     bounds: ParameterBounds | None) -> list[str]:
    if bounds is None:
        return []
    if "A" in bounds and "effective_A" in bounds:
        raise ValueError("bounds must use only one of A or its formal-config alias effective_A")
    failures = []
    for name, interval in bounds.items():
        attribute = "A" if name == "effective_A" else name
        if attribute not in (*_PARAMETER_FIELDS, "K"):
            raise ValueError(f"unsupported bounded parameter: {name}")
        values = np.asarray(interval, dtype=float)
        if (values.shape != (2,) or not np.isfinite(values).all() or
                values[0] > values[1]):
            raise ValueError(f"bounds for {name} must be a finite [lower, upper] pair")
        value = np.asarray(getattr(parameters, attribute))
        if np.any(value < values[0]) or np.any(value > values[1]):
            failures.append(name)
    return failures


def swap_splicing_degradation(
    parameters: KineticParameters,
    swap_mask: Sequence[bool] | np.ndarray,
    initial: CellState | None = None,
    *,
    bounds: ParameterBounds | None = None,
) -> RateSwap:
    """Swap beta/gamma with the compensating production and hidden-U changes.

    For selected genes, r=beta/gamma, b'=b+log10(r)/mapping.log10_slope,
    U'=r*U+(r-1)*M; G and M are unchanged.  A, promoter rates, rho, K and
    the positive map itself stay fixed.  Supplied bounds must admit BOTH
    parameter sets.  A supplied initial state must have nonnegative U'.
    Formal configuration calls A `effective_A`; either alias is accepted in
    bounds, but passing both is rejected to avoid ambiguous constraints.

    Without an initial state this function establishes only the parameter
    transformation, not admissibility of an unprovided initial distribution.
    No clipping is used to conceal an inadmissible transformed state.
    """
    selected = _mask(swap_mask, parameters.n_genes)
    failures = _bounds_failures(parameters, bounds)
    if failures:
        raise ValueError(f"original parameters outside supplied bounds: {failures}")
    values = parameters.to_dict()
    beta, gamma, b = parameters.beta.copy(), parameters.gamma.copy(), parameters.b.copy()
    ratio = parameters.beta / parameters.gamma
    beta[selected], gamma[selected] = parameters.gamma[selected], parameters.beta[selected]
    b[selected] += np.log10(ratio[selected]) / parameters.mapping.log10_slope
    values.update(beta=beta, gamma=gamma, b=b)
    transformed = KineticParameters.from_dict(values)
    failures = _bounds_failures(transformed, bounds)
    if failures:
        raise ValueError(f"transformed parameters outside supplied bounds: {failures}")
    transformed_state = None
    if initial is not None:
        if initial.M.shape[1] != parameters.n_genes:
            raise ValueError("initial state gene dimension does not match parameters")
        U = initial.U.copy()
        U[:, selected] = (ratio[selected] * initial.U[:, selected] +
                          (ratio[selected] - 1) * initial.M[:, selected])
        if not np.isfinite(U).all() or np.any(U < 0):
            raise ValueError("rate swap gives inadmissible negative/nonfinite hidden U")
        transformed_state = CellState(initial.G, U, initial.M)
    return RateSwap(transformed, transformed_state, tuple(np.flatnonzero(selected).tolist()))


def canonicalize_rate_order(parameters: KineticParameters) -> RateSwap:
    """Algebraic representative beta>=gamma, independently for each gene.

    This is a scoring convention, not a biological assumption or a claim that
    this representative lies in frozen bounds or admits an arbitrary U state.
    Canonicalization is performed once per parameter set, never per condition.
    """
    return swap_splicing_degradation(parameters, parameters.beta < parameters.gamma)


def equivalence_invariants(parameters: KineticParameters) -> dict[str, np.ndarray]:
    """Unordered rates and production normalized by gamma are swap-invariant.

    `b_minus_log10_gamma_over_slope` encodes s(M)/gamma under the SAME frozen
    map and K.  It is not the derivative of a closed observed-M dynamical law.
    """
    return {
        "rate_min": np.minimum(parameters.beta, parameters.gamma),
        "rate_max": np.maximum(parameters.beta, parameters.gamma),
        "b_minus_log10_gamma_over_slope": (
            parameters.b - np.log10(parameters.gamma) / parameters.mapping.log10_slope
        ),
    }


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    # hypot avoids the avoidable overflow of squaring a large finite error.
    return float(np.hypot.reduce(np.ravel(difference)) / math.sqrt(difference.size))


def _parameter_errors(first: KineticParameters,
                      second: KineticParameters) -> dict[str, float]:
    return {name: _rmse(getattr(first, name), getattr(second, name))
            for name in _PARAMETER_FIELDS}


def parameter_equivalence_report(
    estimate: KineticParameters,
    reference: KineticParameters,
    reference_M: np.ndarray | None = None,
    *,
    bounds: ParameterBounds | None = None,
    estimate_initial: CellState | None = None,
    reference_initial: CellState | None = None,
) -> dict[str, Any]:
    """Report raw and finite-orbit errors on aligned genes, without enumeration.

    Both models must use exactly the same registered g, K and regulator list.
    Parameter errors are RMSE per field (not a dimensionally mixed total).
    Jacobian RMSE uses off-diagonal directed pairs, consistent with metrics.py;
    a one-gene model has no such pairs and reports None for Jacobian errors.
    `canonical_rmse` compares beta>=gamma representatives and implements the
    independent gene-wise 2**n symmetry in O(n**2), dominated by copying A.
    One mask per model applies jointly to all doses, times and reference states.

    Physical admissibility is separately reported.  When bounds/states are
    supplied, the estimate is aligned to the reference rate ordering and the
    SAME transformation is checked against them.  Canonical coordinates are
    only a scoring chart and need not themselves lie in frozen bounds.
    Absence of initial states cannot certify the hidden initial distribution.
    This report uses truth and belongs only to the isolated scoring side.
    """
    if (estimate.n_genes != reference.n_genes or estimate.regulators != reference.regulators or
            estimate.K != reference.K or estimate.mapping.to_dict() != reference.mapping.to_dict()):
        raise ValueError("equivalence scoring requires aligned genes and identical frozen map, K, regulators")
    if (estimate_initial is None) != (reference_initial is None):
        raise ValueError("provide both initial states or neither")
    canonical_estimate = canonicalize_rate_order(estimate)
    canonical_reference = canonicalize_rate_order(reference)
    aligned_mask = ((estimate.beta >= estimate.gamma) !=
                    (reference.beta >= reference.gamma))
    admissibility: dict[str, Any] = {
        "bounds_supplied": bounds is not None,
        "initial_states_supplied": estimate_initial is not None,
        "reference_aligned_estimate_swaps": np.flatnonzero(aligned_mask).tolist(),
        "parameter_bounds_accepted": None,
        "initial_pair_accepted": None,
    }
    failures = []
    aligned_parameters = swap_splicing_degradation(estimate, aligned_mask).parameters
    if bounds is not None:
        bound_failures = {
            name: _bounds_failures(parameters, bounds)
            for name, parameters in (("reference", reference), ("estimate", estimate),
                                     ("reference_aligned_estimate", aligned_parameters))
        }
        admissibility["parameter_bounds_accepted"] = not any(bound_failures.values())
        admissibility["parameter_bound_failures"] = bound_failures
        for name, failed_fields in bound_failures.items():
            if failed_fields:
                failures.append(f"{name} parameters outside supplied bounds: {failed_fields}")
    if estimate_initial is not None:
        try:
            aligned = swap_splicing_degradation(estimate, aligned_mask, estimate_initial)
        except ValueError as exc:
            failures.append(str(exc))
            admissibility["initial_pair_accepted"] = False
        else:
            # Initial-state matching is pathwise evidence for the supplied pair,
            # not proof that separately sampled stationary laws are equal.
            assert aligned.initial is not None and reference_initial is not None
            if reference_initial.M.shape != aligned.initial.M.shape:
                raise ValueError("initial state shapes must match for pathwise admissibility")
            admissibility["initial_pair_accepted"] = bool(
                np.array_equal(aligned.initial.G, reference_initial.G) and
                np.allclose(aligned.initial.U, reference_initial.U, rtol=1e-12, atol=1e-12) and
                np.allclose(aligned.initial.M, reference_initial.M, rtol=1e-12, atol=1e-12)
            )
    if failures:
        admissibility["failure"] = "; ".join(failures)
    inv_estimate, inv_reference = equivalence_invariants(estimate), equivalence_invariants(reference)
    result: dict[str, Any] = {
        "equivalence_rule": "independent_gene_beta_gamma_swap_v1",
        "canonical_rule": "beta >= gamma; one fixed gene mask over all conditions",
        "scope": "registered finite symmetry only; other nonidentifiabilities are not excluded",
        "estimate_canonical_swaps": list(canonical_estimate.swapped_genes),
        "reference_canonical_swaps": list(canonical_reference.swapped_genes),
        "raw_rmse": _parameter_errors(estimate, reference),
        "canonical_rmse": _parameter_errors(canonical_estimate.parameters, canonical_reference.parameters),
        "invariant_rmse": {key: _rmse(inv_estimate[key], inv_reference[key]) for key in inv_estimate},
        "admissibility": admissibility,
        "jacobian_errors": None,
    }
    if reference_M is not None:
        M = reference._check_M(reference_M)
        if M.size == 0:
            raise ValueError("reference_M must contain at least one reference state")
        raw_estimate, raw_reference = estimate.production_jacobian(M), reference.production_jacobian(M)
        can_estimate = canonical_estimate.parameters.production_jacobian(M)
        can_reference = canonical_reference.parameters.production_jacobian(M)
        no_pairs = reference.n_genes == 1
        result["jacobian_errors"] = {
            "raw_Js_rmse": None if no_pairs else jacobian_rmse(raw_estimate, raw_reference),
            "canonical_Js_rmse": None if no_pairs else jacobian_rmse(can_estimate, can_reference),
            "invariant_Js_over_gamma_rmse": None if no_pairs else jacobian_rmse(
                raw_estimate / estimate.gamma[:, None], raw_reference / reference.gamma[:, None],
            ),
            "entries": "off-diagonal directed pairs, including absent edges",
            "directed_pairs_per_state": reference.n_genes * (reference.n_genes - 1),
            "definition": "J^s=ds/dM; J^s/gamma is not the Jacobian of observed dM/dt",
        }
    return result
