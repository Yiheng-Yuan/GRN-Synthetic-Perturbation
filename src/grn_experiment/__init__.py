"""GRN Synthetic Perturbation Experiment Framework.

This package implements a research protocol for studying gene regulatory network
(GRN) learning and parameter identification from time-series perturbation data.

Main components:
- simulation: Ground-truth network generation and dynamics
- observations: Deterministic snapshot sampling with controlled noise
- protocol: Information barriers and experimental gates
- learning: Problem 1 - Unknown dynamics learning
- parameter_sets: Problem 2 - Known-form parameter identification
- baselines: Endpoint, blackbox, and oracle-form comparators
- ambiguity: Structural rival registration and qualification
- selection: Active experimental design strategies
- metrics: Scoring functions for edges, Jacobians, and predictions
- workflow: Multi-strategy experiment orchestration
- config: Centralized configuration management
- efficiency: Target efficiency estimation and tracking
"""

__version__ = "0.2.0"

from .ambiguity import (
    QualificationReport,
    QualificationThresholds,
    RivalSpecification,
    fit_structural_rivals,
    minimum_local_effect,
    single_edge_swaps,
    top_edge_support,
)
from .baselines import (
    BlackBoxRNAODE,
    EndpointResponsePredictor,
    KnownFormRNAODE,
    fit_blackbox_snapshots,
    fit_known_form_snapshots,
)
from .config import (
    AmbiguityConfig,
    DynamicsConfig,
    ExperimentConfig,
    LearningConfig,
    NetworkConfig,
    ObservationConfig,
    ParameterSetConfig,
    ProtocolConfig,
    StudyManifestConfig,
)
from .efficiency import (
    EfficiencyEstimate,
    EfficiencyTracker,
    calibrate_efficiencies_from_observations,
)
from .learning import (
    CapacityReport,
    EfficiencyInterval,
    SparseRNAODE,
    TrainConfig,
    TrainingReport,
    audit_function_family,
    fit_unpaired_snapshots,
    predict_efficiency_interval,
    predict_population,
)
from .metrics import (
    PairedCI,
    control_adjusted_rmse,
    edge_average_precision,
    eliminated_fraction,
    interval_coverage,
    jacobian_rmse,
    paired_network_bootstrap_ci,
    signed_edge_f1,
    sliced_wasserstein,
    steady_state_rmse,
)
from .observations import (
    ObservationConfig,
    SnapshotSampler,
    protocol_conditions,
)
from .parameter_sets import (
    ConditionTolerance,
    ParameterBounds,
    ParameterSample,
    ParameterSetApproximation,
    choose_parameter_informed,
)
from .protocol import (
    ACTIVE_BUDGET,
    ACTIVE_DOSES,
    ACTIVE_TIMES,
    CELLS_PER_REPLICATE,
    COMMON_BASELINE,
    INITIAL_DOSES,
    INITIAL_TIMES,
    N_GENES,
    N_REGULATORS,
    N_REPLICATES,
    SEALED_TIMES,
    VALIDATION_DOSES,
    VALIDATION_TIMES,
    BlindStore,
    Condition,
    ReplicatedSnapshot,
    SamplingCost,
    Split,
    candidate_grid,
    initial_grid,
    make_split,
    required_test_conditions,
    test_grids,
    validation_grid,
)
from .selection import (
    Choice,
    FixedRegistry,
    Rival,
    choose_active,
    random_sample,
    uniform_order,
)
from .simulation import (
    Dynamics,
    Family,
    NetworkSpec,
    dynamics_diagnostics,
    make_dynamics,
    make_network,
    network_diagnostics,
    validate_network,
)
from .workflow import (
    FAMILIES,
    MODEL_SELECTION_RULE,
    STRATEGIES,
    AuditEvent,
    CaseWorkflow,
    NetworkPlan,
    Role,
    Strategy,
    StrategyRun,
    StudyManifest,
    make_manifest,
    paired_blind_differences,
)

__all__ = [
    # Core protocol
    "Condition",
    "ReplicatedSnapshot",
    "Split",
    "BlindStore",
    "SamplingCost",
    "make_split",
    "initial_grid",
    "candidate_grid",
    "validation_grid",
    "test_grids",
    "required_test_conditions",
    "protocol_conditions",
    "COMMON_BASELINE",
    # Simulation
    "NetworkSpec",
    "Dynamics",
    "Family",
    "make_network",
    "make_dynamics",
    "validate_network",
    "network_diagnostics",
    "dynamics_diagnostics",
    # Observations
    "ObservationConfig",
    "SnapshotSampler",
    # Learning (Problem 1)
    "SparseRNAODE",
    "TrainConfig",
    "TrainingReport",
    "CapacityReport",
    "EfficiencyInterval",
    "fit_unpaired_snapshots",
    "predict_population",
    "predict_efficiency_interval",
    "audit_function_family",
    # Parameter sets (Problem 2)
    "ParameterBounds",
    "ConditionTolerance",
    "ParameterSample",
    "ParameterSetApproximation",
    "choose_parameter_informed",
    # Efficiency tracking
    "EfficiencyEstimate",
    "EfficiencyTracker",
    "calibrate_efficiencies_from_observations",
    # Baselines
    "EndpointResponsePredictor",
    "BlackBoxRNAODE",
    "KnownFormRNAODE",
    "fit_blackbox_snapshots",
    "fit_known_form_snapshots",
    # Ambiguity (structural rivals)
    "RivalSpecification",
    "QualificationThresholds",
    "QualificationReport",
    "top_edge_support",
    "single_edge_swaps",
    "minimum_local_effect",
    "fit_structural_rivals",
    # Selection
    "Rival",
    "Choice",
    "FixedRegistry",
    "choose_active",
    "random_sample",
    "uniform_order",
    # Metrics
    "sliced_wasserstein",
    "control_adjusted_rmse",
    "edge_average_precision",
    "signed_edge_f1",
    "jacobian_rmse",
    "steady_state_rmse",
    "interval_coverage",
    "eliminated_fraction",
    "PairedCI",
    "paired_network_bootstrap_ci",
    # Workflow
    "NetworkPlan",
    "StudyManifest",
    "AuditEvent",
    "StrategyRun",
    "CaseWorkflow",
    "Role",
    "Strategy",
    "make_manifest",
    "paired_blind_differences",
    "STRATEGIES",
    "FAMILIES",
    "MODEL_SELECTION_RULE",
    # Configuration
    "ExperimentConfig",
    "NetworkConfig",
    "DynamicsConfig",
    "ObservationConfig",
    "ProtocolConfig",
    "LearningConfig",
    "AmbiguityConfig",
    "ParameterSetConfig",
    "StudyManifestConfig",
    # Constants
    "N_GENES",
    "N_REGULATORS",
    "N_REPLICATES",
    "CELLS_PER_REPLICATE",
    "ACTIVE_BUDGET",
    "INITIAL_DOSES",
    "INITIAL_TIMES",
    "ACTIVE_DOSES",
    "ACTIVE_TIMES",
    "VALIDATION_DOSES",
    "VALIDATION_TIMES",
    "SEALED_TIMES",
]
