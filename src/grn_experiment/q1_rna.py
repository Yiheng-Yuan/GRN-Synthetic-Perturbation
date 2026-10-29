"""RNA-state learners with optional explicit, self-generated RNA history.

The persistent state is the nonnegative RNA abundance itself. MLP activations
are temporary computations, never extra dynamical coordinates. The delayed
model uses a fixed lag in simulation-time units, not an earlier observed cell:
the dataset contains unpaired population snapshots.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RNAPerturbationODE(nn.Module):
    """Generic RNA vector field, optionally conditioned on its own past RNA.

    ``history_lag=None`` gives dx/dt = f(x, u). A positive lag gives
    dx/dt = f(x, x(t-lag), u), with x(t<=0)=x(0) as an explicit initialization
    assumption. Only the nominal, gene-local perturbation is supplied; neither
    the true GRN nor the generator's hidden efficiency enters the model.

    Both variants use the same projected RK4 rule. Intermediate and final RNA
    states are projected onto the nonnegative orthant. Away from the boundary
    the current-state rule is ordinary RK4; projection changes its formal order
    near the boundary. Linear delay interpolation also limits delayed accuracy.
    This is a numerical model choice, not a claim of globally fourth-order
    accuracy or a known biological production/degradation equation.
    """

    def __init__(self, n_genes: int, hidden: int = 64,
                 history_lag: float | None = None):
        super().__init__()
        for name, value in (("n_genes", n_genes), ("hidden", hidden)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        if history_lag is not None and (
            isinstance(history_lag, bool) or not math.isfinite(history_lag)
            or history_lag <= 0
        ):
            raise ValueError("history_lag must be finite and positive or None")
        self.n_genes = n_genes
        self.hidden = hidden
        self.history_lag = None if history_lag is None else float(history_lag)
        coordinates = 1 if self.history_lag is None else 2
        self.core = nn.Sequential(
            nn.Linear(coordinates * n_genes, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_genes),
        )
        self.local_input = nn.Sequential(
            nn.Linear(coordinates + 1, 32), nn.Tanh(), nn.Linear(32, 1),
        )
        # Start close to the measured baseline; the sign of drift is learned.
        nn.init.normal_(self.core[-1].weight, std=0.003)
        nn.init.zeros_(self.core[-1].bias)
        nn.init.normal_(self.local_input[-1].weight, std=0.02)
        nn.init.zeros_(self.local_input[-1].bias)

    def _check_state(self, state: torch.Tensor) -> None:
        if (not isinstance(state, torch.Tensor) or not state.is_floating_point()
                or state.ndim < 2 or state.shape[-1] != self.n_genes):
            raise ValueError("RNA state must be a floating cells-by-genes tensor")

    @staticmethod
    def _check_matching(state: torch.Tensor, other: torch.Tensor, name: str) -> None:
        if (not isinstance(other, torch.Tensor) or other.shape != state.shape
                or other.dtype != state.dtype or other.device != state.device):
            raise ValueError(f"{name} must match the RNA shape, dtype and device")

    def initial_state(self, counts: torch.Tensor,
                      noise: torch.Tensor | None = None) -> torch.Tensor:
        """Use observed RNA directly; a latent-noise argument is an error."""
        if noise is not None:
            raise ValueError("RNA-state models do not accept latent noise")
        self._check_state(counts)
        if not torch.isfinite(counts).all().item() or (counts < 0).any().item():
            raise ValueError("initial RNA must be finite and nonnegative")
        return counts

    @staticmethod
    def decode(state: torch.Tensor) -> torch.Tensor:
        """Identity observation: no hidden coordinates or many-to-one decoder."""
        return state

    def vector_field(self, state: torch.Tensor, intervention: torch.Tensor,
                     delayed: torch.Tensor | None = None) -> torch.Tensor:
        """Compute RNA drift; the integrator supplies nonnegative stage states."""
        self._check_state(state)
        self._check_matching(state, intervention, "intervention")
        features = [torch.log1p(state)]
        if self.history_lag is None:
            if delayed is not None:
                raise ValueError("current-state model does not accept delayed RNA")
        else:
            self._check_matching(state, delayed, "delayed RNA")
            features.append(torch.log1p(delayed))
        global_update = self.core(torch.cat(features, dim=-1))
        local_features = torch.stack((*features, intervention), dim=-1)
        local_update = self.local_input(local_features).squeeze(-1) * intervention
        return global_update + local_update

    @staticmethod
    def _past_state(states: list[torch.Tensor], query: float,
                    step_size: float) -> torch.Tensor:
        """Interpolate only already generated RNA, retaining its gradient graph."""
        if query <= 0:
            return states[0]
        position = query / step_size
        lower = math.floor(position)
        if lower >= len(states) - 1:
            # A lag of at least one step makes every RK stage query causal.
            if position > len(states) - 1 + 1e-9:
                raise ValueError("delayed RNA query extends beyond generated history")
            return states[-1]
        fraction = position - lower
        return states[lower] * (1.0 - fraction) + states[lower + 1] * fraction

    def integrate(self, initial: torch.Tensor, intervention: torch.Tensor,
                  duration: float, *, step: float = 0.25) -> torch.Tensor:
        """Roll out from the baseline; no observed post-perturbation history is accepted.

        The requested step is a maximum: equal steps divide the requested duration.
        The physical history lag is unchanged when the integration step is halved.
        All history is local to this call, so predictions cannot inherit another
        condition's or a previous call's RNA trajectory.
        """
        if (isinstance(duration, bool) or not math.isfinite(duration) or duration < 0
                or isinstance(step, bool) or not math.isfinite(step) or not 0 < step <= 0.25):
            raise ValueError("invalid integration time or step")
        if self.history_lag is not None and self.history_lag < step:
            raise ValueError("history_lag must be at least the integration step")
        state = self.initial_state(initial)
        self._check_matching(state, intervention, "intervention")
        if (not torch.isfinite(intervention).all().item()
                or ((intervention < 0) | (intervention > 1)).any().item()):
            raise ValueError("intervention must contain finite nominal doses in [0, 1]")
        if duration == 0:
            return state
        n_steps = max(1, math.ceil(duration / step))
        step_size = duration / n_steps
        history = [state] if self.history_lag is not None else None

        def drift(stage: torch.Tensor, time: float) -> torch.Tensor:
            delayed = (None if history is None else
                       self._past_state(history, time - self.history_lag, step_size))
            return self.vector_field(stage, intervention, delayed)

        for index in range(n_steps):
            time = index * step_size
            k1 = drift(state, time)
            k2 = drift(torch.clamp_min(state + step_size * k1 / 2, 0), time + step_size / 2)
            k3 = drift(torch.clamp_min(state + step_size * k2 / 2, 0), time + step_size / 2)
            k4 = drift(torch.clamp_min(state + step_size * k3, 0), time + step_size)
            state = torch.clamp_min(state + step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0)
            if history is not None:
                history.append(state)
        return state
