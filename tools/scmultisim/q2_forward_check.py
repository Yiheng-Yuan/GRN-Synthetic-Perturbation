"""Score independent mature-RNA predictions from a known synthetic mechanism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from grn_experiment.protocol import COMMON_BASELINE, Condition
from grn_experiment.scmultisim import development_parameters, parameter_digest
from grn_experiment.scmultisim_observations import SnapshotSampler
from grn_experiment.scmultisim_stationary import assess_stationarity


NETWORK_SEED = 101
STATIONARITY_SEED = 501
OBSERVATION_SEED = 505
ORACLE_SEED = 506
TAU = 1 / 32
CELLS = 128
TARGETS = (0, 1)
DOSE = 0.7
DEVELOPMENT_TIMES = (1.0, 4.0)
HELD_TIME = 2.0


def response(snapshot, repeats):
    """Calculate one response vector from the specified independent repeats."""
    treated = snapshot.treated[repeats].mean(axis=(0, 1))
    control = snapshot.matched_control[repeats].mean(axis=(0, 1))
    return treated - control


def squared_errors(conditions, receivers):
    values = {"oracle": [], "observed_first_two_repeats": [], "zero_response": []}
    for condition in conditions:
        genes = receivers[condition["target"]]
        check = np.asarray(condition["check_response"])[genes]
        oracle = np.asarray(condition["oracle_response"])[genes]
        replicate = np.asarray(condition["observed_fit_response"])[genes]
        values["oracle"].extend((oracle - check) ** 2)
        values["observed_first_two_repeats"].extend((replicate - check) ** 2)
        values["zero_response"].extend(check ** 2)
    return {"pairs": len(values["oracle"]),
            **{name + "_rmse": float(np.sqrt(np.mean(errors)))
               for name, errors in values.items()}}


def run():
    start = perf_counter()
    params = development_parameters(NETWORK_SEED, n_genes=4)
    stationarity = assess_stationarity(params, seed=STATIONARITY_SEED, tau=TAU)
    if not stationarity.passed:
        raise RuntimeError(f"Baseline stationarity was not established: {stationarity.report['reason']}")
    sampler_kwargs = dict(tau=TAU, baseline_stationarity=stationarity,
                          cells_per_replicate=CELLS, save_production_jacobians=False)
    observed = SnapshotSampler(params, seed=OBSERVATION_SEED, **sampler_kwargs)
    oracle = SnapshotSampler(params, seed=ORACLE_SEED, **sampler_kwargs)
    observed_baseline = observed.sample_condition(COMMON_BASELINE).check_treated.mean(axis=0)
    oracle_baseline = oracle.sample_condition(COMMON_BASELINE).fit_treated.mean(axis=(0, 1))
    isolated = np.flatnonzero(~(params.A.any(axis=0) | params.A.any(axis=1))).tolist()
    receivers = {
        "direct_target": {target: [target] for target in TARGETS},
        "one_hop": {target: np.flatnonzero(params.A[:, target]).tolist() for target in TARGETS},
        "isolated": {target: isolated for target in TARGETS},
    }
    rows = []
    scores = {}
    for split, times in (("development_times", DEVELOPMENT_TIMES), ("held_time", (HELD_TIME,))):
        current_rows = []
        for time in times:
            for target in TARGETS:
                condition = Condition(target, DOSE, time)
                snapshots = observed.sample_condition(condition)
                independent_prediction = oracle.sample_condition(condition)
                current_rows.append({
                    "split": split, "target": target, "dose": DOSE, "time": time,
                    "check_response": response(snapshots, slice(2, 3)).tolist(),
                    "observed_fit_response": response(snapshots, slice(0, 2)).tolist(),
                    "oracle_response": response(independent_prediction, slice(0, 2)).tolist(),
                })
        rows.extend(current_rows)
        scores[split] = {group: squared_errors(current_rows, by_target)
                         for group, by_target in receivers.items()}
    return {
        "schema": "q2-four-gene-forward-check-v1",
        "status": "completed",
        "scope": "scorer-side development oracle with true parameters; no fitted parameters",
        "runtime": {"platform": platform.system(), "machine": platform.machine(),
                    "python": platform.python_version(), "numpy": np.__version__, "device": "cpu"},
        "seeds": {"network": NETWORK_SEED, "stationarity": STATIONARITY_SEED,
                  "observed": OBSERVATION_SEED, "oracle": ORACLE_SEED},
        "parameter_sha256": parameter_digest(params),
        "tau": TAU,
        "cells_per_replicate_per_arm": CELLS,
        "stationarity": {"status": stationarity.report["status"], "burnin": stationarity.burnin,
                         "config": stationarity.report["config"]},
        "used_mature_rna_counts_only": True,
        "replicates_zero_based": {"oracle_prediction": [0, 1],
                                  "observed_reference": [0, 1], "observed_scoring": [2]},
        "scoring_groups_by_target": receivers,
        "baseline": {"observed_third_repeat_mean": observed_baseline.tolist(),
                     "oracle_first_two_repeat_mean": oracle_baseline.tolist(),
                     "rmse": float(np.sqrt(np.mean((oracle_baseline - observed_baseline) ** 2)))},
        "conditions": rows, "scores": scores, "elapsed_seconds": perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
