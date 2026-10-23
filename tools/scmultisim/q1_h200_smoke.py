"""One CPU training update from two public RNA files; no held/late scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from grn_experiment.protocol import COMMON_BASELINE, Condition
from grn_experiment.q1_pilot import PilotConfig, fit_pilot, predict_population
from grn_experiment.q1_pilot_data import (
    _MAX_MANIFEST_BYTES, _MAX_OBSERVATION_BYTES, _decode, _directory_payload,
    _json, _modular_layout, _record, _select_entry,
)


def smoke(dataset: Path) -> dict:
    """Use the registered first fit target at t=.25, never any held response."""
    dataset = dataset.resolve()
    manifest = _json(_directory_payload(dataset, "manifest.json", _MAX_MANIFEST_BYTES))
    _, shape, conditions, registration = _modular_layout(manifest)
    key = Condition(registration["train"][0], .7, .25)
    snapshots, verified = {}, {}
    for condition in (COMMON_BASELINE, key):
        relative, checksum, _ = _select_entry(conditions, _record(condition), "observations/q1")
        payload = _directory_payload(dataset, relative, _MAX_OBSERVATION_BYTES)
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError(f"Observation checksum mismatch: {relative}")
        snapshots[condition] = _decode(payload, condition, shape)
        verified[relative] = checksum
    torch.set_num_threads(1)
    baseline = snapshots[COMMON_BASELINE].treated[:2].reshape(-1, shape[-1])
    config = PilotConfig(updates=1, conditions_per_update=1, cells=2, projections=4,
                         hidden=64, step=.25, fit_times=(.25,), target_weight=0,
                         learning_rate=.001, seed=20261022)
    model, history = fit_pilot(baseline, {key: snapshots[key]}, config=config, device="cpu")
    predicted, control = predict_population(model, baseline, key, cells=2, seed=config.seed)
    if not np.isfinite(predicted).all() or not np.isfinite(control).all():
        raise RuntimeError("Nonfinite smoke prediction")
    return {"status": "passed", "scope": "one CPU forward/backward update; not GPU or scientific validation",
            "n_genes": shape[-1], "updates": 1, "target": key.target, "time": key.time,
            "loss": history[-1]["loss"], "verified_observation_sha256": verified,
            "held_or_longterm_responses_read": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = smoke(args.dataset)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
