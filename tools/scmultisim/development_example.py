"""Small real-data example for local container acceptance, not the batch CMD."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from grn_experiment.scmultisim import development_parameters
from grn_experiment.scmultisim_stationary import assess_stationarity
from grn_experiment.scmultisim_observations import SnapshotSampler
from grn_experiment.protocol import COMMON_BASELINE, Condition, SteadyKey


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    params = development_parameters(101, n_genes=4)
    baseline = assess_stationarity(params, seed=501, tau=1/32)
    steady = assess_stationarity(params, seed=502, tau=1/32, target=0, dose=.7)
    if not baseline.passed or not steady.passed:
        raise RuntimeError("Example distribution was not established")
    sampler = SnapshotSampler(params, seed=505, tau=1/32,
                              baseline_stationarity=baseline, cells_per_replicate=4)
    entries = []
    for name, key, result in (("baseline", COMMON_BASELINE, None),
                              ("perturbation", Condition(0, .7, 1), None),
                              ("stationary", SteadyKey(0, .7), steady)):
        data = sampler.sample_with_truth(key, stationary_result=result)
        arrays = {"treated": data.snapshot.treated}
        if data.snapshot.matched_control is not None:
            arrays["matched_control"] = data.snapshot.matched_control
        np.savez_compressed(args.output / f"{name}.npz", **arrays)
        entries.append({"name": name, "shape": list(data.snapshot.treated.shape),
                        "minimum": float(data.snapshot.treated.min()),
                        "maximum": float(data.snapshot.treated.max())})
        print(json.dumps(entries[-1]), flush=True)
    report = {"status": "complete", "development_only": True,
              "entries": entries, "baseline_burnin": baseline.burnin,
              "perturbation_burnin": steady.burnin, "elapsed_seconds": time.monotonic()-start}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
