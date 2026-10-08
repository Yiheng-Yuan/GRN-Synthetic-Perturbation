"""Complete example: running the full experiment from development to blind testing.

This script demonstrates:
1. Freezing configuration on development networks
2. Running capacity audits
3. Executing 12-round active sampling
4. Model selection on validation
5. Final test predictions

Usage:
    python examples/run_complete_experiment.py --config config/frozen_config.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from grn_experiment.ambiguity import (
    QualificationThresholds,
    RivalSpecification,
    fit_structural_rivals,
    single_edge_swaps,
    top_edge_support,
)
from grn_experiment.baselines import EndpointResponsePredictor, KnownFormRNAODE
from grn_experiment.config import ExperimentConfig
from grn_experiment.learning import SparseRNAODE, TrainConfig, audit_function_family, fit_unpaired_snapshots
from grn_experiment.metrics import (
    control_adjusted_rmse,
    edge_average_precision,
    signed_edge_f1,
    sliced_wasserstein,
)
from grn_experiment.observations import ObservationConfig, SnapshotSampler
from grn_experiment.parameter_sets import (
    ConditionTolerance,
    ParameterBounds,
    ParameterSetApproximation,
    choose_parameter_informed,
)
from grn_experiment.protocol import (
    COMMON_BASELINE,
    BlindStore,
    Condition,
    candidate_grid,
    initial_grid,
    make_split,
    validation_grid,
)
from grn_experiment.selection import choose_active, random_sample, uniform_order
from grn_experiment.simulation import (
    Dynamics,
    dynamics_diagnostics,
    make_dynamics,
    make_network,
    validate_network,
)
from grn_experiment.workflow import CaseWorkflow, NetworkPlan, make_manifest


def run_development_phase(config: ExperimentConfig, output_dir: Path):
    """Phase 1: Validate on 4 development networks and freeze configuration."""

    print("\n" + "=" * 80)
    print("DEVELOPMENT PHASE: Capacity audits and threshold calibration")
    print("=" * 80)

    manifest = make_manifest(config.manifest.master_seed)
    dev_plans = manifest.development

    all_capacity_passed = True
    all_steady_states_converged = True
    all_weak_edges_acceptable = True
    n_weak_edges_total = 0

    for idx, plan in enumerate(dev_plans):
        print(f"\nDevelopment Network {idx + 1}/4: {plan.identifier}")
        print("-" * 80)

        # Generate network and dynamics
        network = make_network(
            seed=plan.network_seed,
            n_genes=config.network.n_genes,
            n_regulators=config.network.n_regulators,
            n_edges=config.network.n_edges,
        )

        validate_network(network, n_edges=config.network.n_edges, require_motifs=True)

        for family in plan.families:
            print(f"  Family: {family}")

            dynamics = make_dynamics(network, family, seed=plan.dynamics_seed)
            diagnostics = dynamics_diagnostics(dynamics, weak_edge_threshold=0.02)

            print(f"    Contraction ratio: {diagnostics['max_contraction_ratio']:.4f}")
            print(f"    Min edge effect: {diagnostics['min_edge_effect_at_baseline']:.4f}")
            print(f"    Weak edges: {diagnostics['n_weak_edges_at_baseline']}")
            print(f"    Step error: {diagnostics['step_halving_error']:.2e}")

            if diagnostics["step_halving_error"] > 1e-4:
                all_steady_states_converged = False

            n_weak_edges_total += diagnostics["n_weak_edges_at_baseline"]

            # Capacity audit for Problem 1
            split = make_split(network.regulators, seed=plan.split_seed)
            sampler = SnapshotSampler(
                dynamics,
                seed=plan.observation_seed,
                config=ObservationConfig(**config.observation.__dict__),
            )

            baseline = sampler.sample_condition(COMMON_BASELINE)
            initial_obs = {
                cond: sampler.sample_condition(cond) for cond in initial_grid(split)[:6]
            }

            # Train a model
            model = SparseRNAODE(
                config.network.n_genes, min_efficiency=config.learning.min_efficiency
            )
            train_config = TrainConfig(
                epochs=config.learning.epochs // 5,  # Reduced for development
                learning_rate=config.learning.learning_rate,
                sparsity_weight=config.learning.sparsity_weight,
                effect_weight=config.learning.effect_weight,
                seed=42,
            )

            fit_unpaired_snapshots(model, baseline, initial_obs, config=train_config)

            # Audit function family capacity
            audit_states = np.random.RandomState(123).uniform(
                0.5, 2.5, size=(20, config.network.n_genes)
            )

            def true_field(x):
                return dynamics.vector_field(x)

            def true_jacobian(x):
                return dynamics.jacobian(x)

            capacity_report = audit_function_family(
                model,
                audit_states,
                true_field,
                true_jacobian,
                epochs=500,
                learning_rate=0.003,
            )

            print(f"    Capacity audit:")
            print(f"      Field RMSE: {capacity_report.relative_vector_field_rmse:.4f}")
            print(f"      Sign accuracy: {capacity_report.strong_edge_sign_accuracy:.4f}")
            print(f"      Passed: {capacity_report.passed}")

            if not capacity_report.passed:
                all_capacity_passed = False

    # Summary diagnostics
    diagnostics_summary = {
        "all_capacity_passed": all_capacity_passed,
        "all_weak_edges_acceptable": n_weak_edges_total < config.network.n_edges * 4 * 0.1,
        "all_steady_states_converged": all_steady_states_converged,
        "n_networks_checked": len(dev_plans) * len(plan.families),
        "n_weak_edges": n_weak_edges_total,
    }

    # Validate and freeze
    config.validate_on_development_networks(diagnostics_summary)

    frozen_path = output_dir / "frozen_config.json"
    config.save(frozen_path)

    print(f"\n✓ Configuration frozen at: {frozen_path}")

    return config


def run_problem1_experiment(
    config: ExperimentConfig, network_plan: NetworkPlan, output_dir: Path
):
    """Run Problem 1 (unknown dynamics) for one network."""

    print(f"\n{'=' * 80}")
    print(f"Problem 1: {network_plan.identifier}")
    print(f"{'=' * 80}")

    # Setup
    network = make_network(
        seed=network_plan.network_seed,
        n_genes=config.network.n_genes,
        n_regulators=config.network.n_regulators,
        n_edges=config.network.n_edges,
    )

    split = make_split(network.regulators, seed=network_plan.split_seed)

    results = {}

    for family in network_plan.families:
        print(f"\n  Family: {family}")

        dynamics = make_dynamics(network, family, seed=network_plan.dynamics_seed)
        sampler = SnapshotSampler(
            dynamics,
            seed=network_plan.observation_seed,
            config=ObservationConfig(**config.observation.__dict__),
        )

        # Generate all protocol observations
        from grn_experiment.observations import protocol_conditions

        all_conditions = protocol_conditions(split)
        observations = sampler.sample_many(all_conditions, include_baseline=True)

        # Create blind store
        store = BlindStore(split, observations)

        # Initial training
        baseline = store.get(COMMON_BASELINE)
        initial_obs = {cond: store.get(cond) for cond in initial_grid(split)}

        model = SparseRNAODE(config.network.n_genes, min_efficiency=config.learning.min_efficiency)

        train_config = TrainConfig(
            epochs=config.learning.epochs,
            learning_rate=config.learning.learning_rate,
            sparsity_weight=config.learning.sparsity_weight,
            effect_weight=config.learning.effect_weight,
            seed=42,
        )

        fit_unpaired_snapshots(model, baseline, initial_obs, config=train_config)

        # 12-round active acquisition
        acquired = []
        candidate_pool = candidate_grid(split)

        for round_idx in range(config.protocol.active_budget):
            # Propose rivals
            support = top_edge_support(
                model, split.initial + split.active, n_edges=config.network.n_edges
            )

            rivals_specs = single_edge_swaps(
                support,
                n_genes=config.network.n_genes,
                regulators=list(network.regulators),
                count=config.ambiguity.n_rival_swaps,
                seed=1000 + round_idx,
            )

            thresholds = QualificationThresholds(
                max_distribution_error=config.ambiguity.max_distribution_error,
                max_effect_error=config.ambiguity.max_effect_error,
                min_local_edge_effect=config.ambiguity.min_local_edge_effect,
            )

            effect_states = np.random.RandomState(2000 + round_idx).uniform(
                0.5, 2.5, size=(20, config.network.n_genes)
            )

            rivals, reports = fit_structural_rivals(
                rivals_specs,
                baseline,
                {cond: store.get(cond) for cond in initial_grid(split) + tuple(acquired)},
                thresholds,
                config=train_config,
                effect_states=effect_states,
            )

            # Select next condition
            if len(rivals) >= 2:
                choice = choose_active(candidate_pool, tuple(acquired), rivals)
                next_cond = choice.condition
                using_active = True
            else:
                available = [c for c in candidate_pool if c not in acquired]
                next_cond = available[0] if available else None
                using_active = False

            if next_cond is None:
                break

            # Acquire
            acquired.append(next_cond)
            store.acquire(next_cond)

            # Retrain
            all_acquired_obs = {
                cond: store.get(cond) for cond in initial_grid(split) + tuple(acquired)
            }
            fit_unpaired_snapshots(model, baseline, all_acquired_obs, config=train_config)

            print(
                f"    Round {round_idx + 1}: target={next_cond.target}, "
                f"dose={next_cond.dose:.1f}, time={next_cond.time:.2f}, "
                f"rivals={len(rivals)}, active={using_active}"
            )

        # Validation and scoring
        validation_obs = {cond: store.get(cond) for cond in validation_grid(split)}

        val_errors = []
        for cond, snapshot in validation_obs.items():
            pred = model.integrate(
                torch.from_numpy(baseline.fit_treated.reshape(-1, config.network.n_genes).astype(np.float32)),
                cond.time,
                cond.target,
                cond.dose,
            ).detach().cpu().numpy()

            error = sliced_wasserstein(pred, snapshot.fit_treated.reshape(-1, config.network.n_genes))
            val_errors.append(error)

        val_score = np.mean(val_errors)

        # Edge recovery
        pred_weights = model.edge_weights.detach().cpu().numpy()
        true_weights = network.weights

        edge_ap = edge_average_precision(pred_weights, true_weights)
        edge_f1 = signed_edge_f1(pred_weights, true_weights, threshold=0.1)

        print(f"\n    Results:")
        print(f"      Validation score: {val_score:.4f}")
        print(f"      Edge AP: {edge_ap:.4f}")
        print(f"      Signed F1: {edge_f1:.4f}")

        results[family] = {
            "validation_score": val_score,
            "edge_ap": edge_ap,
            "edge_f1": edge_f1,
            "n_acquired": len(acquired),
        }

    # Save results
    result_path = output_dir / f"problem1_{network_plan.identifier}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✓ Saved to {result_path}")

    return results


def run_problem2_experiment(
    config: ExperimentConfig, network_plan: NetworkPlan, output_dir: Path
):
    """Run Problem 2 (known form, parameter identification) for one network."""

    print(f"\n{'=' * 80}")
    print(f"Problem 2: {network_plan.identifier}")
    print(f"{'=' * 80}")

    # Setup
    network = make_network(
        seed=network_plan.network_seed,
        n_genes=config.network.n_genes,
        n_regulators=config.network.n_regulators,
        n_edges=config.network.n_edges,
    )

    split = make_split(network.regulators, seed=network_plan.split_seed)

    results = {}

    for family in network_plan.families:
        print(f"\n  Family: {family}")

        dynamics = make_dynamics(network, family, seed=network_plan.dynamics_seed)
        sampler = SnapshotSampler(
            dynamics,
            seed=network_plan.observation_seed,
            config=ObservationConfig(**config.observation.__dict__),
        )

        from grn_experiment.observations import protocol_conditions

        all_conditions = protocol_conditions(split)
        observations = sampler.sample_many(all_conditions, include_baseline=True)

        store = BlindStore(split, observations)

        # Initial observations
        baseline = store.get(COMMON_BASELINE)
        initial_obs = {cond: store.get(cond) for cond in initial_grid(split)}

        # Initialize parameter set
        bounds = ParameterBounds(
            weights_min=config.parameter_set.weights_min,
            weights_max=config.parameter_set.weights_max,
            gamma_min=config.parameter_set.gamma_min,
            gamma_max=config.parameter_set.gamma_max,
            efficiency_min=config.parameter_set.efficiency_min,
            efficiency_max=config.parameter_set.efficiency_max,
        )

        tolerances = [
            ConditionTolerance(
                cond,
                config.parameter_set.distribution_tolerance,
                config.parameter_set.effect_tolerance,
            )
            for cond in initial_obs.keys()
        ]

        param_set = ParameterSetApproximation(
            family=family,
            n_genes=config.network.n_genes,
            bounds=bounds,
            tolerances=tolerances,
            n_samples=config.parameter_set.n_initial_samples,
            seed=42,
        )

        # Refine with initial observations
        param_set.refine(baseline, initial_obs)

        print(f"    Initial survivors: {param_set.n_survivors}/{param_set.n_initial}")

        # 12-round parameter-informed acquisition
        acquired = []
        candidate_pool = candidate_grid(split)

        for round_idx in range(config.protocol.active_budget):
            if param_set.n_survivors == 0:
                print(f"    Warning: No compatible parameters at round {round_idx + 1}")
                break

            # Select condition maximizing disagreement
            next_cond = choose_parameter_informed(
                candidate_pool, tuple(acquired), param_set, baseline
            )

            acquired.append(next_cond)
            store.acquire(next_cond)

            # Refine parameter set
            new_obs = {next_cond: store.get(next_cond)}
            eliminated, remaining = param_set.refine(baseline, new_obs)

            disagreement_before = param_set.prediction_disagreement(baseline, next_cond)
            mean_disagreement = np.mean(disagreement_before[1] - disagreement_before[0])

            print(
                f"    Round {round_idx + 1}: target={next_cond.target}, "
                f"eliminated={eliminated}, remaining={remaining}, "
                f"disagreement={mean_disagreement:.4f}"
            )

        # Final metrics
        shrinkage = param_set.volume_shrinkage
        ranges = param_set.parameter_ranges()

        print(f"\n    Results:")
        print(f"      Volume shrinkage: {shrinkage:.4f}")
        print(f"      Final survivors: {param_set.n_survivors}")
        print(f"      Parameter ranges:")
        for name, (low, high) in ranges.items():
            print(f"        {name}: [{low:.4f}, {high:.4f}]")

        results[family] = {
            "volume_shrinkage": shrinkage,
            "final_survivors": param_set.n_survivors,
            "parameter_ranges": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
            "n_acquired": len(acquired),
        }

    # Save results
    result_path = output_dir / f"problem2_{network_plan.identifier}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✓ Saved to {result_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run complete GRN perturbation experiment")
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to frozen config (if already exists)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results"), help="Output directory"
    )
    parser.add_argument(
        "--skip-development", action="store_true", help="Skip development phase if config exists"
    )
    parser.add_argument(
        "--problem", choices=["1", "2", "both"], default="both", help="Which problem to run"
    )
    parser.add_argument(
        "--n-networks", type=int, default=2, help="Number of blind networks to run (max 20)"
    )

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Phase 1: Development or load config
    if args.config and args.config.exists() and args.skip_development:
        print(f"Loading frozen config from {args.config}")
        config = ExperimentConfig.load(args.config)
        config.ensure_frozen()
    else:
        config = ExperimentConfig.from_defaults()
        config = run_development_phase(config, args.output)

    # Phase 2: Generate manifest
    manifest = make_manifest(config.manifest.master_seed)
    blind_plans = manifest.blind[: args.n_networks]

    print(f"\n{'=' * 80}")
    print(f"Running {len(blind_plans)} blind networks")
    print(f"{'=' * 80}")

    # Phase 3: Run experiments
    for plan in blind_plans:
        if args.problem in ["1", "both"]:
            run_problem1_experiment(config, plan, args.output)

        if args.problem in ["2", "both"]:
            run_problem2_experiment(config, plan, args.output)

    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {args.output}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
