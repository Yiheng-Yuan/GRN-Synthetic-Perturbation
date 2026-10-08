"""Centralized configuration management for freezing experimental parameters.

Before blind networks are opened, all thresholds, seeds, and model selection
rules must be frozen on development networks. This module provides:
- Type-safe configuration dataclasses
- YAML serialization for audit trails
- Validation of frozen parameters

Usage:
    # Development phase
    config = ExperimentConfig.from_defaults()
    config.validate_on_development_networks(...)
    config.save("frozen_config.yaml")

    # Blind phase
    config = ExperimentConfig.load("frozen_config.yaml")
    config.ensure_frozen()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import json


@dataclass(frozen=True)
class NetworkConfig:
    """Fixed network topology parameters."""

    n_genes: int = 24
    n_regulators: int = 16
    n_edges: int = 40
    max_indegree: int = 3
    max_outdegree: int = 5
    min_positive_fraction: float = 0.4
    max_positive_fraction: float = 0.6

    def __post_init__(self) -> None:
        if not (self.n_regulators <= self.n_genes):
            raise ValueError("n_regulators cannot exceed n_genes")
        if not (0 <= self.min_positive_fraction <= self.max_positive_fraction <= 1):
            raise ValueError("invalid positive fraction bounds")


@dataclass(frozen=True)
class DynamicsConfig:
    """Dynamics family parameters and contractivity bounds."""

    families: tuple[Literal["sigmoid", "hill"], ...] = ("sigmoid", "hill")
    contraction_limit: float = 0.65
    gamma_min: float = 0.9
    gamma_max: float = 1.1
    basal_min: float = 0.2
    basal_max: float = 0.35
    amplitude_sigmoid_min: float = 2.0
    amplitude_sigmoid_max: float = 2.6
    amplitude_hill_min: float = 1.5
    amplitude_hill_max: float = 1.9
    hill_k_min: float = 1.5
    hill_k_max: float = 2.0
    efficiency_min: float = 0.7
    efficiency_max: float = 1.0
    coupling_max: float = 1.0

    def __post_init__(self) -> None:
        if not (0 < self.contraction_limit < 1):
            raise ValueError("contraction limit must be in (0, 1)")
        if not all(family in ("sigmoid", "hill") for family in self.families):
            raise ValueError("families must be sigmoid or hill")


@dataclass(frozen=True)
class ObservationConfig:
    """RNA observation noise parameters."""

    cells_per_replicate: int = 128
    n_replicates: int = 3
    cell_log_sd: float = 0.15
    replicate_log_sd: float = 0.04
    observation_log_sd: float = 0.03
    integration_step: float = 0.05

    def __post_init__(self) -> None:
        if self.cells_per_replicate < 1 or self.n_replicates < 2:
            raise ValueError("need positive cells and at least 2 replicates")
        if any(
            sd < 0
            for sd in (self.cell_log_sd, self.replicate_log_sd, self.observation_log_sd)
        ):
            raise ValueError("noise standard deviations must be nonnegative")
        if not (0 < self.integration_step <= 0.25):
            raise ValueError("integration step must be in (0, 0.25]")


@dataclass(frozen=True)
class ProtocolConfig:
    """Experimental protocol timing and sampling design."""

    initial_targets: int = 6
    active_targets: int = 6
    validation_targets: int = 2
    test_targets: int = 2

    initial_doses: tuple[float, ...] = (0.3, 0.7)
    active_doses: tuple[float, ...] = (0.3, 0.5, 0.7)
    validation_doses: tuple[float, ...] = (0.3, 0.7)

    initial_times: tuple[float, ...] = (0.25, 1.0, 4.0)
    active_times: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 4.0)
    validation_times: tuple[float, ...] = (0.25, 1.0, 4.0)
    sealed_times: tuple[float, ...] = (2.0, 8.0)

    active_budget: int = 12

    def __post_init__(self) -> None:
        if (
            self.initial_targets
            + self.active_targets
            + self.validation_targets
            + self.test_targets
        ) != 16:
            raise ValueError("target split must sum to 16 regulators")
        if self.active_budget < 1:
            raise ValueError("active budget must be positive")


@dataclass(frozen=True)
class LearningConfig:
    """Problem 1 learning hyperparameters."""

    epochs: int = 250
    conditions_per_epoch: int = 4
    cells_per_condition: int = 128
    projections: int = 32
    learning_rate: float = 0.001
    sparsity_weight: float = 0.001
    effect_weight: float = 0.2
    step: float = 0.1
    min_efficiency: float = 0.5

    # Capacity audit thresholds (frozen on development networks)
    max_relative_field_error: float = 0.05
    min_sign_accuracy: float = 0.95

    def __post_init__(self) -> None:
        if any(
            x < 1
            for x in (
                self.epochs,
                self.conditions_per_epoch,
                self.cells_per_condition,
                self.projections,
            )
        ):
            raise ValueError("training counts must be positive")
        if not (0 < self.learning_rate and 0 <= self.sparsity_weight):
            raise ValueError("invalid learning rate or sparsity weight")


@dataclass(frozen=True)
class AmbiguityConfig:
    """Problem 1 structural ambiguity thresholds."""

    max_distribution_error: float = 0.15
    max_effect_error: float = 0.10
    min_local_edge_effect: float = 0.02
    n_rival_swaps: int = 50

    def __post_init__(self) -> None:
        if any(
            x <= 0
            for x in (
                self.max_distribution_error,
                self.max_effect_error,
                self.min_local_edge_effect,
            )
        ):
            raise ValueError("thresholds must be positive")


@dataclass(frozen=True)
class ParameterSetConfig:
    """Problem 2 parameter set approximation settings."""

    n_initial_samples: int = 1000
    weights_min: float = -1.2
    weights_max: float = 1.2
    gamma_min: float = 0.5
    gamma_max: float = 1.5
    basal_min: float = 0.1
    basal_max: float = 0.5
    amplitude_min: float = 1.0
    amplitude_max: float = 3.0
    coupling_min: float = 0.0
    coupling_max: float = 1.0
    efficiency_min: float = 0.5
    efficiency_max: float = 1.0

    # Per-condition tolerance (set during development)
    distribution_tolerance: float = 0.12
    effect_tolerance: float = 0.08

    def __post_init__(self) -> None:
        if self.n_initial_samples < 10:
            raise ValueError("need at least 10 initial samples")
        bounds = [
            (self.weights_min, self.weights_max),
            (self.gamma_min, self.gamma_max),
            (self.basal_min, self.basal_max),
            (self.amplitude_min, self.amplitude_max),
            (self.coupling_min, self.coupling_max),
            (self.efficiency_min, self.efficiency_max),
        ]
        if any(low >= high for low, high in bounds):
            raise ValueError("all bounds must have min < max")


@dataclass(frozen=True)
class StudyManifestConfig:
    """Study-level design: 4 development + 20 blind networks."""

    n_development: int = 4
    n_blind: int = 20
    master_seed: int = 42

    def __post_init__(self) -> None:
        if self.n_development < 1 or self.n_blind < 1:
            raise ValueError("need at least one development and one blind network")


@dataclass
class ExperimentConfig:
    """Complete frozen experiment configuration.

    This is the single source of truth for all thresholds and parameters that
    must not change between development and blind phases.
    """

    network: NetworkConfig = field(default_factory=NetworkConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    ambiguity: AmbiguityConfig = field(default_factory=AmbiguityConfig)
    parameter_set: ParameterSetConfig = field(default_factory=ParameterSetConfig)
    manifest: StudyManifestConfig = field(default_factory=StudyManifestConfig)

    _frozen: bool = field(default=False, init=False, repr=False)
    _frozen_path: Path | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_defaults(cls) -> ExperimentConfig:
        """Create configuration with documented defaults."""
        return cls()

    def save(self, path: str | Path) -> None:
        """Save frozen configuration to JSON for audit trail."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "network": asdict(self.network),
            "dynamics": asdict(self.dynamics),
            "observation": asdict(self.observation),
            "protocol": asdict(self.protocol),
            "learning": asdict(self.learning),
            "ambiguity": asdict(self.ambiguity),
            "parameter_set": asdict(self.parameter_set),
            "manifest": asdict(self.manifest),
            "_frozen": True,
            "_frozen_path": str(path.absolute()),
        }

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

        object.__setattr__(self, "_frozen", True)
        object.__setattr__(self, "_frozen_path", path)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        """Load frozen configuration from JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        config = cls(
            network=NetworkConfig(**data["network"]),
            dynamics=DynamicsConfig(**data["dynamics"]),
            observation=ObservationConfig(**data["observation"]),
            protocol=ProtocolConfig(**data["protocol"]),
            learning=LearningConfig(**data["learning"]),
            ambiguity=AmbiguityConfig(**data["ambiguity"]),
            parameter_set=ParameterSetConfig(**data["parameter_set"]),
            manifest=StudyManifestConfig(**data["manifest"]),
        )

        object.__setattr__(config, "_frozen", data.get("_frozen", False))
        object.__setattr__(config, "_frozen_path", path)

        return config

    def ensure_frozen(self) -> None:
        """Raise if configuration has not been frozen via save()."""
        if not self._frozen:
            raise RuntimeError(
                "configuration must be frozen before blind networks. "
                "Call config.save() after development validation."
            )

    def is_frozen(self) -> bool:
        """Check if configuration has been saved and frozen."""
        return self._frozen

    def validate_on_development_networks(self, diagnostics: dict[str, Any]) -> None:
        """Run development-phase checks before freezing.

        Args:
            diagnostics: Results from running all development networks,
                        including capacity audits and numerical stability checks.
        """
        required_keys = [
            "all_capacity_passed",
            "all_weak_edges_acceptable",
            "all_steady_states_converged",
        ]

        for key in required_keys:
            if key not in diagnostics:
                raise ValueError(f"missing required diagnostic: {key}")

        if not diagnostics["all_capacity_passed"]:
            raise ValueError(
                "some development networks failed capacity audit. "
                "Adjust learning config or model architecture."
            )

        if not diagnostics["all_steady_states_converged"]:
            raise ValueError(
                "some development networks have unstable steady states. "
                "Check dynamics parameters and contraction limit."
            )

        print("✓ All development network checks passed")
        print(f"  Capacity audits: {diagnostics.get('n_networks_checked', 0)} networks")
        print(f"  Weak edges: {diagnostics.get('n_weak_edges', 0)} below threshold")
        print("Configuration is ready to be frozen.")
