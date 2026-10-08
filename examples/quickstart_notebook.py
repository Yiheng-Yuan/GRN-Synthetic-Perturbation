"""Quick start guide: Run your first GRN perturbation experiment.

This notebook demonstrates:
1. Creating and validating a network
2. Generating synthetic observations
3. Training a Problem 1 learner
4. Building a Problem 2 parameter set
5. Running active acquisition

Run with: jupyter notebook examples/quickstart.ipynb
"""

# %% [markdown]
# # GRN Perturbation Experiment - Quick Start
#
# This notebook walks through a minimal end-to-end experiment.

# %% Setup
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from grn_experiment import (
    # Network and dynamics
    make_network,
    make_dynamics,
    validate_network,
    dynamics_diagnostics,

    # Protocol
    make_split,
    COMMON_BASELINE,
    initial_grid,
    candidate_grid,

    # Observations
    SnapshotSampler,
    ObservationConfig,

    # Problem 1
    SparseRNAODE,
    TrainConfig,
    fit_unpaired_snapshots,
    audit_function_family,

    # Problem 2
    ParameterSetApproximation,
    ParameterBounds,
    ConditionTolerance,
    choose_parameter_informed,

    # Metrics
    edge_average_precision,
    signed_edge_f1,
    sliced_wasserstein,
)

# %% [markdown]
# ## Step 1: Generate a Test Network

# %%
# Create network with fixed seed for reproducibility
network = make_network(
    seed=42,
    n_genes=24,
    n_regulators=16,
    n_edges=40,
)

# Validate network structure
diagnostics = validate_network(network, n_edges=40, require_motifs=True)

print(f"Network created successfully!")
print(f"  Genes: {network.n_genes}")
print(f"  Regulators: {len(network.regulators)}")
print(f"  Edges: {diagnostics['n_edges']}")
print(f"  Positive edges: {diagnostics['n_positive']}")
print(f"  Has feedforward: {diagnostics['has_feedforward']}")
print(f"  Has feedback: {diagnostics['has_positive_feedback']} (pos), "
      f"{diagnostics['has_negative_feedback']} (neg)")

# Visualize edge matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(network.weights, cmap='RdBu_r', vmin=-1.2, vmax=1.2)
ax.set_xlabel('Regulator')
ax.set_ylabel('Target')
ax.set_title('Ground Truth GRN (weights)')
plt.colorbar(im, ax=ax, label='Edge weight')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2: Create Dynamics and Sample Observations

# %%
# Create sigmoid dynamics
dynamics = make_dynamics(network, family="sigmoid", seed=123)

# Check dynamics properties
dyn_diag = dynamics_diagnostics(dynamics)
print(f"\nDynamics diagnostics:")
print(f"  Contraction ratio: {dyn_diag['max_contraction_ratio']:.4f}")
print(f"  Min edge effect: {dyn_diag['min_edge_effect_at_baseline']:.4f}")
print(f"  Weak edges: {dyn_diag['n_weak_edges_at_baseline']}")
print(f"  Integration error: {dyn_diag['step_halving_error']:.2e}")

# Create observation sampler
obs_config = ObservationConfig(
    cells_per_replicate=128,
    cell_log_sd=0.15,
    replicate_log_sd=0.04,
)

sampler = SnapshotSampler(dynamics, seed=456, config=obs_config)

# Sample baseline
baseline = sampler.sample_condition(COMMON_BASELINE)
print(f"\nBaseline snapshot shape: {baseline.treated.shape}")
print(f"  Mean expression: {baseline.treated.mean():.3f}")
print(f"  Std expression: {baseline.treated.std():.3f}")

# %% [markdown]
# ## Step 3: Generate Initial Training Data

# %%
# Create 6/6/2/2 split
split = make_split(network.regulators, seed=789)

print(f"Regulator split:")
print(f"  Initial: {split.initial}")
print(f"  Active: {split.active}")
print(f"  Validation: {split.validation}")
print(f"  Test: {split.test}")

# Sample initial grid
initial_conditions = initial_grid(split)
print(f"\nInitial grid: {len(initial_conditions)} conditions")

initial_obs = {
    cond: sampler.sample_condition(cond)
    for cond in initial_conditions
}

print(f"Sampled {len(initial_obs)} initial observations")

# %% [markdown]
# ## Step 4: Problem 1 - Train Unknown Dynamics Learner

# %%
# Create model
model = SparseRNAODE(
    n_genes=network.n_genes,
    min_efficiency=0.5,
)

# Training configuration
train_config = TrainConfig(
    epochs=100,  # Reduced for demo
    conditions_per_epoch=4,
    learning_rate=0.001,
    sparsity_weight=0.001,
    seed=42,
)

# Train
print("Training Problem 1 model...")
report = fit_unpaired_snapshots(model, baseline, initial_obs, config=train_config)

print(f"\nTraining complete:")
print(f"  Conditions seen: {report.conditions_seen}")
print(f"  Final loss: {report.losses[-1]:.4f}")

# Plot training curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(report.losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)

# Compare learned vs true edges
pred_weights = model.edge_weights.detach().cpu().numpy()

plt.subplot(1, 2, 2)
plt.scatter(network.weights.flatten(), pred_weights.flatten(), alpha=0.5, s=20)
plt.xlabel('True weight')
plt.ylabel('Predicted weight')
plt.title('Edge Weight Recovery')
plt.grid(True, alpha=0.3)
plt.axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Compute metrics
edge_ap = edge_average_precision(pred_weights, network.weights)
edge_f1 = signed_edge_f1(pred_weights, network.weights, threshold=0.1)

print(f"\nEdge recovery metrics:")
print(f"  Average Precision: {edge_ap:.4f}")
print(f"  Signed F1: {edge_f1:.4f}")

# %% [markdown]
# ## Step 5: Capacity Audit (Development Only)

# %%
# Generate test states
audit_states = np.random.RandomState(999).uniform(0.5, 2.5, size=(30, network.n_genes))

def true_field(x):
    return dynamics.vector_field(x)

def true_jacobian(x):
    return dynamics.jacobian(x)

# Run audit
print("Running capacity audit...")
capacity_report = audit_function_family(
    model,
    audit_states,
    true_field,
    true_jacobian,
    epochs=200,
    learning_rate=0.003,
)

print(f"\nCapacity audit results:")
print(f"  Relative field RMSE: {capacity_report.relative_vector_field_rmse:.4f}")
print(f"  Strong edge sign accuracy: {capacity_report.strong_edge_sign_accuracy:.4f}")
print(f"  Passed: {'✅' if capacity_report.passed else '❌'}")

# %% [markdown]
# ## Step 6: Problem 2 - Parameter Set Construction

# %%
# Initialize parameter set
bounds = ParameterBounds(
    weights_min=-1.2,
    weights_max=1.2,
    efficiency_min=0.5,
    efficiency_max=1.0,
)

# Create tolerances for initial conditions
tolerances = [
    ConditionTolerance(cond, max_distribution_error=0.15, max_effect_error=0.10)
    for cond in initial_obs.keys()
]

param_set = ParameterSetApproximation(
    family="sigmoid",
    n_genes=network.n_genes,
    bounds=bounds,
    tolerances=tolerances,
    n_samples=500,  # Reduced for demo
    seed=42,
)

print(f"Parameter set initialized:")
print(f"  Initial samples: {param_set.n_initial}")
print(f"  Survival fraction: {param_set.survival_fraction:.4f}")

# Refine with initial observations
print("\nRefining with initial observations...")
eliminated, remaining = param_set.refine(baseline, initial_obs, step=0.1)

print(f"  Eliminated: {eliminated}")
print(f"  Remaining: {remaining}")
print(f"  Volume shrinkage: {param_set.volume_shrinkage:.4f}")

# Check parameter ranges
ranges = param_set.parameter_ranges()
print(f"\nParameter ranges:")
for name, (low, high) in ranges.items():
    print(f"  {name}: [{low:.4f}, {high:.4f}]")

# %% [markdown]
# ## Step 7: Active Selection Demonstration

# %%
# Get candidate pool
candidate_pool = candidate_grid(split)

print(f"Candidate pool: {len(candidate_pool)} conditions")

# Select next condition
if param_set.n_survivors > 0:
    next_condition = choose_parameter_informed(
        candidate_pool,
        tuple(initial_obs.keys()),
        param_set,
        baseline,
    )

    print(f"\nNext condition selected:")
    print(f"  Target: {next_condition.target}")
    print(f"  Dose: {next_condition.dose}")
    print(f"  Time: {next_condition.time}")

    # Compute prediction disagreement
    lower, upper = param_set.prediction_disagreement(baseline, next_condition)

    disagreement = upper - lower
    mean_disagreement = disagreement.mean()

    print(f"\nPrediction disagreement:")
    print(f"  Mean: {mean_disagreement:.4f}")
    print(f"  Max: {disagreement.max():.4f}")
    print(f"  Min: {disagreement.min():.4f}")

    # Visualize disagreement
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(range(network.n_genes), disagreement)
    plt.xlabel('Gene')
    plt.ylabel('Disagreement (upper - lower)')
    plt.title('Per-Gene Prediction Disagreement')
    plt.grid(True, alpha=0.3, axis='y')

    plt.subplot(1, 2, 2)
    gene_means = (lower + upper) / 2
    plt.errorbar(
        range(network.n_genes),
        gene_means,
        yerr=disagreement/2,
        fmt='o',
        capsize=3,
        alpha=0.6,
    )
    plt.xlabel('Gene')
    plt.ylabel('Predicted expression')
    plt.title('Ensemble Prediction (mean ± disagreement/2)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No compatible parameters remaining!")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
# 1. ✅ Network generation and validation
# 2. ✅ Dynamics creation and diagnostics
# 3. ✅ Observation sampling
# 4. ✅ Problem 1: Unknown dynamics learning
# 5. ✅ Capacity audit
# 6. ✅ Problem 2: Parameter set construction
# 7. ✅ Parameter-informed active selection
#
# Next steps:
# - Run the full 12-round acquisition loop
# - Compare active vs random vs uniform strategies
# - Scale up to multiple networks
# - See `examples/run_complete_experiment.py` for production usage

# %%
print("\n" + "="*60)
print("Quick start complete! 🎉")
print("="*60)
