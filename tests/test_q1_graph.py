"""Explicit graph, gradient and solver contracts, not biological validation."""

import math

import pytest
import torch

from grn_experiment.q1_graph import GraphRNAODE


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def chain_model():
    model = GraphRNAODE(4, hidden=4, baseline_mean=torch.full((4,), 5.)).double().eval()
    with torch.no_grad():
        model.gate_logits.fill_(-20.)
        model.gate_logits[1, 0] = model.gate_logits[2, 1] = 20.
        model.raw_H.zero_()
        model.raw_H[1, 0] = math.atanh(.6 / model.weight_bound)
        model.raw_H[2, 1] = math.atanh(-.4 / model.weight_bound)
    return model


def test_state_is_mature_rna_only_and_export_has_binary_gated_signed_graph():
    model = chain_model()
    state = torch.ones(2, 4, dtype=torch.float64)
    assert model.initial_state(state) is state
    assert model.decode(state) is state
    assert model.integrate(state, torch.zeros_like(state), 0) is state
    assert not any(hasattr(model, name) for name in ("encoder", "history", "memory", "core"))
    assert not any(isinstance(layer, (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM))
                   for layer in model.modules())
    matrices = model.graph_matrices()
    assert set(matrices) == {"W", "H", "Z", "probability"}
    assert set(matrices["W"].unique().tolist()) <= {0., 1.}
    assert matrices["W"].sum() == 2
    torch.testing.assert_close(matrices["Z"], matrices["W"] * matrices["H"], atol=0, rtol=0)
    assert matrices["H"][1, 0] > 0 and matrices["H"][2, 1] < 0
    assert torch.count_nonzero(matrices["W"].diag()) == 0
    torch.testing.assert_close(model.sample_gates(), matrices["W"], atol=0, rtol=0)
    with pytest.raises(ValueError, match="latent noise"):
        model.initial_state(state, state)


def test_hard_concrete_expected_active_count_matches_independent_edge_sample():
    # Many identically initialized independent edges give a Monte Carlo check
    # of analytic P(gate>0), rather than restating the implementation formula.
    torch.manual_seed(17)
    model = GraphRNAODE(64)
    gates = model.sample_gates()
    actual = (gates > 0).sum().item() / (64 * 63)
    probability = model.expected_edges().item() / (64 * 63)
    assert abs(actual - probability) < .025
    assert ((gates > 0) & (gates < 1)).any()
    model.expected_edges().backward()
    assert (model.gate_logits.grad[~torch.eye(64, dtype=torch.bool)] > 0).all()
    assert torch.count_nonzero(model.gate_logits.grad.diag()) == 0


def test_source_functions_are_anchored_and_monotone():
    model = GraphRNAODE(3, state_scale=torch.tensor([1., 2., .5])).double()
    zero = torch.zeros(1, 3, dtype=torch.float64)
    unit = torch.expm1(model.state_scale).unsqueeze(0)
    torch.testing.assert_close(model.source_features(zero), zero, atol=0, rtol=0)
    torch.testing.assert_close(model.source_features(unit), torch.ones_like(unit))
    state = torch.tensor([[.1, 2., 9.]], dtype=torch.float64, requires_grad=True)
    derivative, = torch.autograd.grad(model.source_features(state).sum(), state)
    assert (derivative > 0).all()


def test_zero_edges_block_direct_path_and_sign_matches_local_derivative():
    model = chain_model()
    state = torch.full((2, 4), 5., dtype=torch.float64)
    jacobian = model.local_jacobian(state)
    assert jacobian.shape == (2, 4, 4)
    assert (jacobian[:, 1, 0] > 0).all()
    assert (jacobian[:, 2, 1] < 0).all()
    off_diagonal = ~torch.eye(4, dtype=torch.bool)
    blocked = (model.W == 0) & off_diagonal
    assert torch.count_nonzero(jacobian[:, blocked]) == 0
    torch.testing.assert_close(jacobian[:, range(4), range(4)],
                               -model.decay.expand(2, 4))
    with torch.no_grad():
        shifted = state.clone()
        shifted[:, 0] += 3.
        difference = model.vector_field(shifted, torch.zeros_like(state)) - model.vector_field(
            state, torch.zeros_like(state))
    assert difference[:, 1].abs().min() > 0
    assert torch.count_nonzero(difference[:, [2, 3]]) == 0


def test_multihop_response_and_cut_edge_control_use_same_graph():
    model = chain_model()
    state = torch.full((2, 4), 5., dtype=torch.float64)
    dose = torch.zeros_like(state)
    dose[:, 0] = .7
    control = model.integrate(state, torch.zeros_like(state), 8.)
    treated = model.integrate(state, dose, 8.)
    assert ((treated - control)[:, [0, 1]] < 0).all()
    assert ((treated - control)[:, 2] > 0).all()
    torch.testing.assert_close(treated[:, 3], control[:, 3], atol=0, rtol=0)
    cut = model.W.clone()
    cut[1, 0] = 0
    treated_cut = model.integrate(state, dose, 8., gates=cut)
    control_cut = model.integrate(state, torch.zeros_like(state), 8., gates=cut)
    torch.testing.assert_close(treated_cut[:, 1:], control_cut[:, 1:], atol=0, rtol=0)


def test_knockdown_changes_only_target_production_at_identical_state():
    model = chain_model()
    state = torch.full((2, 4), 5., dtype=torch.float64)
    dose = torch.zeros_like(state)
    dose[:, 1] = .7
    no_dose = model.vector_field(state, torch.zeros_like(state))
    treated = model.vector_field(state, dose)
    expected = -dose * model.rho * model._production(state, model.W)
    torch.testing.assert_close(treated - no_dose, expected)
    assert torch.count_nonzero((treated - no_dose)[:, [0, 2, 3]]) == 0
    torch.testing.assert_close(model.rho, torch.full((4,), .75, dtype=torch.float64))


def test_training_has_gate_weight_nonlinearity_and_efficiency_gradients():
    torch.manual_seed(3)
    model = GraphRNAODE(4, hidden=4).double().train()
    state = torch.tensor([[1., 2., 4., 8.], [2., 3., 5., 9.]], dtype=torch.float64)
    dose = torch.zeros_like(state)
    dose[:, 0] = .7
    # A differentiable relaxed draw is shared by treatment and control.
    gates = model.sample_gates()
    prediction = model.integrate(state, dose, 1., gates=gates)
    control = model.integrate(state, torch.zeros_like(state), 1., gates=gates)
    loss = prediction.square().mean() + .1 * control.square().mean() + 1e-4 * model.expected_edges()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert torch.count_nonzero(model.raw_H.grad.diag()) == 0
    assert torch.count_nonzero(model.gate_logits.grad.diag()) == 0
    # Targets with no dose do not silently acquire measured efficiency evidence.
    assert torch.count_nonzero(model.raw_rho.grad[1:]) == 0


def test_explicit_gates_are_reused_for_entire_rollout_and_eval_is_repeatable(monkeypatch):
    model = GraphRNAODE(3).double()
    state = torch.ones(2, 3, dtype=torch.float64)
    original = model.sample_gates
    calls = []

    def record():
        calls.append(1)
        return original()

    monkeypatch.setattr(model, "sample_gates", record)
    model.integrate(state, torch.zeros_like(state), 1.)
    assert len(calls) == 1
    model.eval()
    first = model.integrate(state, torch.zeros_like(state), 1.)
    model.integrate(state * 3, torch.zeros_like(state), 2.)
    second = model.integrate(state, torch.zeros_like(state), 1.)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_nonnegative_drift_at_boundary_and_long_rollout_finite():
    model = chain_model()
    state = torch.zeros(2, 4, dtype=torch.float64)
    dose = torch.full_like(state, 1.)
    assert (model.vector_field(state, dose) >= 0).all()
    result = model.integrate(state, dose, 64.)
    assert result.shape == state.shape and torch.isfinite(result).all()
    assert (result >= 0).all()


def test_solver_matches_affine_analytic_dynamics_and_step_halving():
    model = GraphRNAODE(2, baseline_mean=torch.tensor([2., 3.])).double().eval()
    with torch.no_grad():
        model.raw_decay.fill_(math.log(math.expm1(1.7)))
    state = torch.tensor([[7., 11.]], dtype=torch.float64)
    no_dose = torch.zeros_like(state)
    no_graph = torch.zeros(2, 2, dtype=torch.float64)
    production = model._production(state, no_graph)
    equilibrium = production / model.decay
    expected = equilibrium + (state - equilibrium) * math.exp(-1.7 * 1.3)
    coarse = model.integrate(state, no_dose, 1.3, step=.25, gates=no_graph)
    fine = model.integrate(state, no_dose, 1.3, step=.125, gates=no_graph)
    assert (fine - expected).abs().max() < (coarse - expected).abs().max() / 8
    fine.sum().backward()
    assert torch.isfinite(model.raw_decay.grad).all()


@pytest.mark.parametrize("kwargs", [
    {"n_genes": True}, {"hidden": 1}, {"state_scale": torch.zeros(3)},
    {"state_scale": torch.ones(4)}, {"baseline_mean": torch.full((3,), -1.)},
])
def test_invalid_initialization(kwargs):
    with pytest.raises(ValueError):
        GraphRNAODE(**({"n_genes": 3} | kwargs))


def test_invalid_state_intervention_and_gates_do_not_silently_broadcast():
    model = GraphRNAODE(3)
    state = torch.ones(2, 3)
    for invalid in (torch.ones(3), -state, state.double(), state.to(torch.int64), state * float("nan")):
        with pytest.raises(ValueError):
            model.initial_state(invalid)
    for invalid in (torch.ones(1, 3), state * -1, state * 2, state.double()):
        with pytest.raises(ValueError):
            model.vector_field(state, invalid)
    for invalid in (torch.ones(3), torch.ones(3, 3) * 2, torch.ones(3, 3).double()):
        with pytest.raises(ValueError):
            model.vector_field(state, torch.zeros_like(state), gates=invalid)
    for duration, step in ((-1., .25), (1., 0), (True, .25), (1., .5)):
        with pytest.raises(ValueError):
            model.integrate(state, torch.zeros_like(state), duration, step=step)
