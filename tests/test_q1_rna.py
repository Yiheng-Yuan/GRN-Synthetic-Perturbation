"""Numerical and state-contract checks, not biological validation."""

import math

import pytest
import torch

from grn_experiment.q1_rna import RNAPerturbationODE


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ScalarField(RNAPerturbationODE):
    """Independent analytic system for checking the numerical solver."""

    def __init__(self, *, lag=None, rate=1.0, mode="decay"):
        super().__init__(2, hidden=4, history_lag=lag)
        self.rate = torch.nn.Parameter(torch.tensor(rate, dtype=torch.float64))
        self.mode = mode
        self.seen_stages = []

    def vector_field(self, state, intervention, delayed=None):
        self.seen_stages.append(state.detach())
        if self.mode == "constant":
            return torch.ones_like(state) * self.rate
        if self.mode == "delay":
            return delayed * self.rate
        return -state * self.rate


@pytest.mark.parametrize("lag", [None, 1.0])
def test_state_is_only_rna_with_identity_decode_and_no_latent_noise(lag):
    model = RNAPerturbationODE(4, hidden=8, history_lag=lag)
    counts = torch.tensor([[0., 1., 2., 9.], [3., 4., 0., 2.]])
    initial = model.initial_state(counts)
    assert initial is counts
    assert model.decode(initial) is initial
    assert initial.shape == (2, 4)
    assert not hasattr(model, "encoder")
    assert not hasattr(model, "memory")
    assert not list(model.buffers())
    assert not any(isinstance(layer, (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM))
                   for layer in model.modules())
    with pytest.raises(ValueError, match="latent noise"):
        model.initial_state(counts, torch.zeros_like(counts))
    assert model.integrate(initial, torch.zeros_like(initial), 0) is initial


@pytest.mark.parametrize("lag", [None, 1.0])
def test_generic_cross_gene_field_and_shared_local_intervention(lag):
    model = RNAPerturbationODE(4, hidden=8, history_lag=lag)
    state = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    delayed = state + 2 if lag else None
    features = torch.log1p(state)
    if delayed is not None:
        features = torch.cat((features, torch.log1p(delayed)), dim=-1)
    torch.testing.assert_close(model.vector_field(state, torch.zeros_like(state), delayed),
                               model.core(features))
    mask = torch.zeros_like(state)
    mask[:, 1] = .7
    difference = model.vector_field(state, mask, delayed) - model.vector_field(
        state, torch.zeros_like(state), delayed)
    torch.testing.assert_close(difference[:, [0, 2, 3]], torch.zeros(2, 3))
    assert not any(isinstance(module, torch.nn.Embedding) for module in model.modules())
    assert not any(hasattr(model, name) for name in ("A", "rho", "gamma"))


def test_ordinary_ode_matches_analytic_decay_and_step_halving_improves_error():
    model = ScalarField(rate=1.7)
    initial = torch.tensor([[1., 4.]], dtype=torch.float64)
    mask = torch.zeros_like(initial)
    expected = initial * math.exp(-1.7 * 1.3)
    coarse = model.integrate(initial, mask, 1.3, step=.25)
    fine = model.integrate(initial, mask, 1.3, step=.125)
    assert torch.max(torch.abs(fine - expected)) < torch.max(torch.abs(coarse - expected)) / 8
    # Independent RK4 stability polynomial checks implementation separately
    # from the nonzero discretization error against the exponential solution.
    steps = math.ceil(1.3 / .125)
    z = -1.7 * 1.3 / steps
    rk4 = initial * (1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)**steps
    torch.testing.assert_close(fine, rk4, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(fine, expected, atol=2e-5, rtol=0)


@pytest.mark.parametrize("duration", [.7, 1., 1.3, 2.])
def test_delay_method_of_steps_agrees_with_analytic_constant_prehistory(duration):
    # x'=a*x(t-1), x(t<=0)=x0. Until t=2 this has an exact quadratic form.
    model = ScalarField(lag=1., rate=.4, mode="delay")
    initial = torch.tensor([[1., 4.]], dtype=torch.float64)
    actual = model.integrate(initial, torch.zeros_like(initial), duration, step=.25)
    factor = 1 + .4 * duration + .4**2 / 2 * max(duration - 1, 0)**2
    # An irregular grid can straddle the t=1 derivative breakpoint.
    torch.testing.assert_close(actual, initial * factor, atol=2e-4, rtol=2e-4)


def test_delayed_gradient_keeps_past_parameter_and_initial_state_paths():
    model = ScalarField(lag=1., rate=.4, mode="delay")
    initial = torch.tensor([[1., 4.]], dtype=torch.float64, requires_grad=True)
    actual = model.integrate(initial, torch.zeros_like(initial), 2.)
    actual.sum().backward()
    # At t=2, x=x0*(1+2a+a^2/2). Dropping the historical graph loses
    # part of the a^2 gradient, although the forward prediction stays right.
    torch.testing.assert_close(model.rate.grad, initial.detach().sum() * 2.4)
    torch.testing.assert_close(initial.grad, torch.full_like(initial, 1.88))


def test_delayed_interpolation_half_step_keeps_physical_lag_fixed():
    model = ScalarField(lag=1., rate=.4, mode="delay")
    initial = torch.tensor([[1., 4.]], dtype=torch.float64)
    duration = 2.7
    factor = (1 + .4 * duration + .4**2 * (duration - 1)**2 / 2
              + .4**3 * (duration - 2)**3 / 6)
    coarse = model.integrate(initial, torch.zeros_like(initial), duration, step=.25)
    fine = model.integrate(initial, torch.zeros_like(initial), duration, step=.125)
    assert model.history_lag == 1.
    assert torch.max(torch.abs(fine - initial * factor)) < torch.max(torch.abs(coarse - initial * factor))
    torch.testing.assert_close(fine, initial * factor, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("lag", [None, 1.0])
def test_projection_keeps_all_stages_nonnegative_and_boundary_finite(lag):
    model = ScalarField(lag=lag, rate=-20., mode="constant")
    initial = torch.tensor([[0., 1.]], dtype=torch.float64, requires_grad=True)
    actual = model.integrate(initial, torch.zeros_like(initial), 1.)
    torch.testing.assert_close(actual, torch.zeros_like(initial))
    assert all((stage >= 0).all() for stage in model.seen_stages)
    actual.sum().backward()
    assert torch.isfinite(initial.grad).all()
    assert torch.isfinite(model.rate.grad).all()


@pytest.mark.parametrize("lag", [None, 1.0])
def test_batched_rollout_is_repeatable_and_does_not_retain_cross_call_history(lag):
    torch.manual_seed(5)
    model = RNAPerturbationODE(4, hidden=8, history_lag=lag)
    counts = torch.rand(2, 3, 4)
    original = counts.clone()
    mask = torch.zeros_like(counts)
    mask[..., 1] = .7
    first = model.integrate(counts, mask, 1.35)
    model.integrate(counts * 4, mask, 3.)
    second = model.integrate(counts, mask, 1.35)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(counts, original, atol=0, rtol=0)
    assert first.shape == counts.shape and torch.isfinite(first).all()
    assert (first >= 0).all()
    first.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize("kwargs", [
    {"n_genes": 1}, {"n_genes": True}, {"n_genes": 2.5}, {"hidden": 1},
    {"hidden": True}, {"history_lag": 0}, {"history_lag": -.1},
    {"history_lag": float("nan")}, {"history_lag": float("inf")}, {"history_lag": True},
])
def test_invalid_model_arguments(kwargs):
    with pytest.raises(ValueError):
        RNAPerturbationODE(**({"n_genes": 4} | kwargs))


@pytest.mark.parametrize("values", [
    torch.ones(4), torch.ones(2, 3), torch.ones(2, 4, dtype=torch.int64),
    torch.full((2, 4), -1.), torch.full((2, 4), float("nan")),
    torch.full((2, 4), float("inf")),
])
def test_invalid_initial_rna(values):
    with pytest.raises(ValueError):
        RNAPerturbationODE(4).initial_state(values)


@pytest.mark.parametrize("duration,step", [
    (-1., .25), (float("nan"), .25), (float("inf"), .25), (True, .25),
    (1., 0.), (1., .5), (1., float("nan")), (1., True),
])
def test_invalid_integration_times(duration, step):
    model = RNAPerturbationODE(4)
    state = torch.ones(2, 4)
    with pytest.raises(ValueError):
        model.integrate(state, torch.zeros_like(state), duration, step=step)


@pytest.mark.parametrize("intervention", [
    torch.ones(2, 3), torch.ones(2, 4, dtype=torch.float64),
    torch.full((2, 4), float("nan")), torch.full((2, 4), -.1),
    torch.full((2, 4), 1.1),
])
def test_invalid_interventions(intervention):
    with pytest.raises(ValueError):
        RNAPerturbationODE(4).integrate(torch.ones(2, 4), intervention, 1.)


def test_delay_input_contract_and_causality():
    state = torch.ones(2, 4)
    mask = torch.zeros_like(state)
    with pytest.raises(ValueError, match="current-state"):
        RNAPerturbationODE(4).vector_field(state, mask, state)
    delayed = RNAPerturbationODE(4, history_lag=.125)
    with pytest.raises(ValueError, match="delayed RNA"):
        delayed.vector_field(state, mask)
    with pytest.raises(ValueError, match="at least the integration step"):
        delayed.integrate(state, mask, 1., step=.25)
    with pytest.raises(ValueError, match="beyond generated history"):
        RNAPerturbationODE._past_state([state], .5, .25)


def test_current_model_restarts_from_rna_without_any_hidden_carry():
    torch.manual_seed(7)
    model = RNAPerturbationODE(4, hidden=8)
    x = torch.rand(2, 4)
    u = torch.zeros_like(x)
    u[:, 1] = .7
    whole = model.integrate(x, u, 1.5)
    halfway = model.integrate(x, u, 1.)
    restarted = model.integrate(halfway, u, .5)
    torch.testing.assert_close(restarted, whole, rtol=0, atol=0)


def test_same_current_rna_can_have_different_drift_with_explicit_history():
    torch.manual_seed(8)
    model = RNAPerturbationODE(4, hidden=8, history_lag=1.)
    now = torch.ones(2, 4)
    u = torch.zeros_like(now)
    assert not torch.allclose(model.vector_field(now, u, now),
                              model.vector_field(now, u, 4 * now))
