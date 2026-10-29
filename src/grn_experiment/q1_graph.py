"""An explicit signed graph and an RNA-only effective neural ODE.

These are declared modelling priors, not recovered generator equations:
monotone neural source/receiver functions, additive signed graph inputs,
nonnegative production and linear degradation. Mature RNA alone is an
effective Markov closure of the simulator's hidden U/G/M process. No hidden
memory, simulator rates, graph or true perturbation efficiency is used here.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GraphRNAODE(nn.Module):
    """Learn a graph-gated vector field with ``Z[i,j]`` meaning j -> i.

    ``H = 2*tanh(raw_H)`` supplies bounded signed edge parameters; binary W
    supplies edge existence. Training uses reparameterized hard-concrete gates
    (temperature 2/3, stretch [-0.1, 1.1]) and their analytic expected count.
    Evaluation and export both use ``W = 1[P(gate > 0) >= 0.5]``; the threshold
    is fixed, not selected using the true network. Continuous training gates
    are a relaxation, not binary graph samples. Sample once per rollout and
    pass the same ``gates`` to both treatment and control integrations.

    Efficiency is a gene-specific parameter in (0.5,1), shared across that
    gene's doses, cells and times. A target with no training data retains its
    initial value 0.75; that value is an assumption, not an estimated efficiency.
    Source normalization and H's bound reduce scale freedom without proving
    unique identifiability or equality to simulator edge weights.
    """

    temperature = 2.0 / 3.0
    stretch_low = -0.1
    stretch_high = 1.1
    graph_threshold = 0.5
    weight_bound = 2.0

    def __init__(self, n_genes: int, hidden: int = 8,
                 state_scale: torch.Tensor | None = None,
                 baseline_mean: torch.Tensor | None = None):
        super().__init__()
        for name, value in (("n_genes", n_genes), ("hidden", hidden)):
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        self.n_genes, self.hidden = n_genes, hidden
        scale = self._initial_vector(state_scale, n_genes, 1., "state_scale", positive=True)
        mean = self._initial_vector(baseline_mean, n_genes, 1., "baseline_mean", positive=False)
        self.register_buffer("state_scale", scale)
        # Initialize near a baseline with a 0.1 decay rate. This is not fitting
        # per-cell equilibrium, nor does it assert that every control is fixed.
        self.register_buffer("production_scale", .1 * mean.clamp_min(1.) / math.log(2.))
        self.register_buffer("candidate_mask", 1. - torch.eye(n_genes))
        self.raw_H = nn.Parameter(torch.empty(n_genes, n_genes))
        nn.init.normal_(self.raw_H, std=.02)
        self.gate_logits = nn.Parameter(torch.zeros(n_genes, n_genes))
        self.raw_rho = nn.Parameter(torch.zeros(n_genes))
        self.raw_decay = nn.Parameter(torch.full((n_genes,), math.log(math.expm1(.1))))
        self.rate_bias = nn.Parameter(torch.zeros(n_genes))
        # Separate monotone scalar neural functions for each source and receiver.
        # Their hidden activations are temporary, not propagated state variables.
        self.source_slope = nn.Parameter(torch.zeros(n_genes, hidden))
        self.source_weight = nn.Parameter(torch.zeros(n_genes, hidden))
        self.source_bias = nn.Parameter(torch.linspace(-1., 1., hidden).repeat(n_genes, 1))
        self.receiver_slope = nn.Parameter(torch.zeros(n_genes, hidden))
        self.receiver_weight = nn.Parameter(torch.zeros(n_genes, hidden))
        self.receiver_bias = nn.Parameter(torch.linspace(-1., 1., hidden).repeat(n_genes, 1))

    @staticmethod
    def _initial_vector(value, n_genes, default, name, *, positive):
        vector = torch.full((n_genes,), default) if value is None else torch.as_tensor(value)
        if (vector.shape != (n_genes,) or not vector.is_floating_point()
                or not torch.isfinite(vector).all().item()
                or (vector <= 0 if positive else vector < 0).any().item()):
            domain = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must be a finite {domain} floating gene vector")
        return vector.detach().to(dtype=torch.get_default_dtype(), device="cpu").clone()

    @property
    def H(self) -> torch.Tensor:
        return self.weight_bound * torch.tanh(self.raw_H) * self.candidate_mask

    @property
    def W(self) -> torch.Tensor:
        return (self.graph_probabilities() >= self.graph_threshold).to(self.raw_H.dtype)

    @property
    def Z(self) -> torch.Tensor:
        return self.W * self.H

    @property
    def rho(self) -> torch.Tensor:
        return .5 + .5 * torch.sigmoid(self.raw_rho)

    @property
    def decay(self) -> torch.Tensor:
        return F.softplus(self.raw_decay)

    def graph_probabilities(self) -> torch.Tensor:
        """Analytic P(hard-concrete gate > 0), not its mean gate magnitude."""
        shift = self.temperature * math.log(-self.stretch_low / self.stretch_high)
        return torch.sigmoid(self.gate_logits - shift) * self.candidate_mask

    def expected_edges(self) -> torch.Tensor:
        return self.graph_probabilities().sum()

    def sample_gates(self) -> torch.Tensor:
        """A single train-time relaxed graph, or the fixed binary eval graph."""
        if not self.training:
            return self.W
        eps = torch.finfo(self.gate_logits.dtype).eps
        uniform = torch.rand_like(self.gate_logits).clamp(eps, 1. - eps)
        concrete = torch.sigmoid((torch.logit(uniform) + self.gate_logits) / self.temperature)
        stretched = concrete * (self.stretch_high - self.stretch_low) + self.stretch_low
        return stretched.clamp(0., 1.) * self.candidate_mask

    def graph_matrices(self) -> dict[str, torch.Tensor]:
        """Deterministic graph matching evaluation, with no train-time sampling."""
        W, H = self.W, self.H
        return {"W": W, "H": H, "Z": W * H,
                "probability": self.graph_probabilities()}

    def _check_state(self, state: torch.Tensor) -> None:
        if (not isinstance(state, torch.Tensor) or not state.is_floating_point()
                or state.ndim < 2 or state.shape[-1] != self.n_genes
                or state.numel() == 0 or state.dtype != self.raw_H.dtype
                or state.device != self.raw_H.device):
            raise ValueError("RNA must be a nonempty floating cells-by-genes tensor matching the model")
        if not torch.isfinite(state).all().item() or (state < 0).any().item():
            raise ValueError("RNA must be finite and nonnegative")

    def initial_state(self, counts: torch.Tensor,
                      noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is not None:
            raise ValueError("graph RNA models do not accept latent noise")
        self._check_state(counts)
        return counts

    @staticmethod
    def decode(state: torch.Tensor) -> torch.Tensor:
        return state

    def _check_intervention(self, state, intervention):
        if (not isinstance(intervention, torch.Tensor)
                or intervention.shape != state.shape or intervention.dtype != state.dtype
                or intervention.device != state.device
                or not torch.isfinite(intervention).all().item()
                or ((intervention < 0) | (intervention > 1)).any().item()):
            raise ValueError("intervention must match RNA and contain nominal doses in [0,1]")

    def _resolve_gates(self, gates):
        if gates is None:
            return self.sample_gates()
        if (not isinstance(gates, torch.Tensor) or gates.shape != self.raw_H.shape
                or gates.dtype != self.raw_H.dtype or gates.device != self.raw_H.device
                or not torch.isfinite(gates).all().item()
                or ((gates < 0) | (gates > 1)).any().item()):
            raise ValueError("gates must be a finite matching genes-by-genes tensor in [0,1]")
        return gates * self.candidate_mask

    def source_features(self, state: torch.Tensor) -> torch.Tensor:
        """Monotone phi_j(log1p(M_j)/scale_j), anchored at phi(0)=0, phi(1)=1."""
        coordinates = torch.log1p(state) / self.state_scale
        slope, weight = F.softplus(self.source_slope), F.softplus(self.source_weight)
        origin = F.softplus(self.source_bias)
        value = ((F.softplus(coordinates.unsqueeze(-1) * slope + self.source_bias)
                  - origin) * weight).sum(-1)
        unit = ((F.softplus(slope + self.source_bias) - origin) * weight).sum(-1)
        return value / unit.clamp_min(torch.finfo(state.dtype).eps)

    def _production(self, state, gates):
        inputs = self.source_features(state) @ (gates * self.H).T
        slope, weight = F.softplus(self.receiver_slope), F.softplus(self.receiver_weight)
        response = ((torch.tanh(inputs.unsqueeze(-1) * slope + self.receiver_bias)
                     - torch.tanh(self.receiver_bias)) * weight).mean(-1)
        return self.production_scale * F.softplus(self.rate_bias + response)

    def _drift(self, state, intervention, gates):
        return ((1. - intervention * self.rho) * self._production(state, gates)
                - self.decay * state)

    def vector_field(self, state: torch.Tensor, intervention: torch.Tensor,
                     gates: torch.Tensor | None = None) -> torch.Tensor:
        self._check_state(state)
        self._check_intervention(state, intervention)
        return self._drift(state, intervention, self._resolve_gates(gates))

    def integrate(self, initial: torch.Tensor, intervention: torch.Tensor,
                  duration: float, *, step: float = .25,
                  gates: torch.Tensor | None = None) -> torch.Tensor:
        """Projected RK4; one graph per rollout and no persistent hidden state.

        Stage projection guarantees nonnegative numerical states, but changes
        RK4's formal order at the boundary. Halving-step acceptance remains a
        separate numerical check. ``step`` is a maximum, not a biological lag.
        """
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration < 0
                or isinstance(step, bool) or not isinstance(step, (int, float))
                or not math.isfinite(step) or not 0 < step <= .25):
            raise ValueError("invalid integration time or step")
        state = self.initial_state(initial)
        self._check_intervention(state, intervention)
        graph = self._resolve_gates(gates)
        if duration == 0:
            return state
        steps = max(1, math.ceil(duration / step))
        dt = duration / steps
        for _ in range(steps):
            k1 = self._drift(state, intervention, graph)
            k2 = self._drift((state + dt * k1 / 2).clamp_min(0), intervention, graph)
            k3 = self._drift((state + dt * k2 / 2).clamp_min(0), intervention, graph)
            k4 = self._drift((state + dt * k3).clamp_min(0), intervention, graph)
            state = (state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6).clamp_min(0)
        return state

    def local_jacobian(self, state: torch.Tensor,
                       intervention: torch.Tensor | None = None) -> torch.Tensor:
        """Return dF_i/dM_j per cell using the deterministic exported graph.

        This is the learned mature-RNA effective drift Jacobian, not the
        simulator's production-rate Jacobian; they must not share a truth score.
        """
        self._check_state(state)
        if state.ndim != 2:
            raise ValueError("local_jacobian requires a cells-by-genes matrix")
        dose = torch.zeros_like(state) if intervention is None else intervention
        self._check_intervention(state, dose)
        gates = self.W.detach()
        with torch.enable_grad():
            return torch.stack([
                torch.autograd.functional.jacobian(
                    lambda x: self._drift(x.unsqueeze(0), u.unsqueeze(0), gates)[0],
                    cell, create_graph=False, vectorize=True)
                for cell, u in zip(state, dose)
            ])
