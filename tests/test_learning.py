import numpy as np
import torch

from grn_experiment.learning import (
    SparseRNAODE,
    TrainConfig,
    audit_function_family,
    fit_unpaired_snapshots,
    predict_efficiency_interval,
    predict_population,
)
from grn_experiment.protocol import COMMON_BASELINE, Condition, ReplicatedSnapshot


def _tiny_snapshots(check_value: float = 1.0):
    baseline_values = np.ones((3, 4, 3))
    baseline = ReplicatedSnapshot(baseline_values)
    treated = np.ones((3, 4, 3)) * 0.9
    treated[2] = check_value
    control = np.ones((3, 4, 3))
    control[2] = check_value
    condition = Condition(0, 0.3, 0.25)
    return baseline, {condition: ReplicatedSnapshot(treated, control)}


def test_time_zero_is_shared_baseline_and_positive():
    model = SparseRNAODE(3)
    state = torch.ones(4, 3)
    original = model.integrate(state, 0, target=0, dose=0.7)
    assert torch.equal(original, state)
    assert torch.all(model.synthesis(state) > 0)
    efficiencies = model.efficiencies.detach()
    assert 0.5 <= float(efficiencies.min()) <= float(efficiencies.max()) <= 1


def test_support_mask_excludes_edges_and_gradients():
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 0] = True
    model = SparseRNAODE(3, allowed_edges=mask)
    assert torch.count_nonzero(model.edge_weights) == 1
    result = model.vector_field(torch.ones(3)).sum()
    result.backward()
    assert torch.count_nonzero(model.weights.grad) == 1


def test_check_replicate_never_enters_snapshot_fit():
    baseline, first = _tiny_snapshots(1.0)
    _, changed = _tiny_snapshots(7.0)
    config = TrainConfig(epochs=2, conditions_per_epoch=1, cells_per_condition=4, projections=3, step=0.125)
    torch.manual_seed(9)
    one = SparseRNAODE(3)
    torch.manual_seed(9)
    two = SparseRNAODE(3)
    first_report = fit_unpaired_snapshots(one, baseline, first, config=config)
    second_report = fit_unpaired_snapshots(two, baseline, changed, config=config)
    assert first_report.losses == second_report.losses
    assert torch.equal(one.edge_weights, two.edge_weights)
    assert one.observed_targets == {0}
    outcome = predict_population(one, baseline.check_treated, next(iter(first)))
    assert outcome.shape == (4, 3)
    assert np.all(outcome >= 0)


def test_development_oracle_audit_is_explicit_and_isolated():
    model = SparseRNAODE(3)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    states = np.ones((2, 3), dtype=np.float32)
    report = audit_function_family(
        model,
        states,
        lambda points: np.zeros_like(points),
        lambda point: np.zeros((3, 3)),
        epochs=1,
    )
    assert np.isfinite(report.relative_vector_field_rmse)
    assert not report.passed
    assert all(torch.equal(before[name], parameter) for name, parameter in model.named_parameters())


def test_unseen_target_efficiency_is_profiled_not_claimed_estimated():
    model = SparseRNAODE(3)
    cells = np.ones((4, 3))
    condition = Condition(0, 0.7, 1.0)
    point = predict_population(model, cells, condition)
    midpoint = predict_population(model, cells, condition, rho_override=0.75)
    np.testing.assert_allclose(point, midpoint)
    envelope = predict_efficiency_interval(model, cells, condition)
    assert np.all(envelope.lower <= point)
    assert np.all(point <= envelope.upper)
