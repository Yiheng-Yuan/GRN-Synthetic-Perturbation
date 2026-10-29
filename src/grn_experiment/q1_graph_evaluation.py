"""Independent directed-graph scoring; simulator truth never enters fitting.

All matrices use row=receiver, column=source. A learned mature-RNA graph is an
effective model: correspondence with the simulator's transcription graph A is
a benchmark, not proof of mechanistic equivalence or parameter identifiability.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np

from .metrics import edge_average_precision


ORIENTATION = "row_receiver_col_source"
GRAPH_KEYS = ("W", "H", "Z", "probability")


def _square(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[0] < 2
            or array.shape[0] != array.shape[1]
            or not (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_)
            or np.iscomplexobj(array) or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite real square matrix of at least two genes")
    return array


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate_graph(graph: Mapping[str, np.ndarray], true_A: np.ndarray) -> dict:
    """Score exported binary structure and signs against an explicit truth A.

    Ranking uses gate probability, not H magnitude. W is scored as exported;
    this scorer never chooses a threshold using truth. Undefined scores are
    JSON null, not an implicit perfect score. Amplitude error is deliberately
    absent: H and simulator A need not have matching scale or interpretation.
    """
    if not isinstance(graph, Mapping) or not set(GRAPH_KEYS) <= graph.keys():
        raise ValueError("graph requires W, H, Z and probability")
    if "orientation" in graph:
        orientation = np.asarray(graph["orientation"])
        if orientation.shape != () or str(orientation.item()) != ORIENTATION:
            raise ValueError("graph orientation must be row_receiver_col_source")
    W, H, Z, probability = (_square(graph[key], key) for key in GRAPH_KEYS)
    truth = _square(true_A, "A")
    if len({array.shape for array in (W, H, Z, probability, truth)}) != 1:
        raise ValueError("graph and truth must have matching gene dimensions")
    if not np.isin(W, (0, 1)).all():
        raise ValueError("exported W must be binary")
    if not np.array_equal(Z, W * H):
        raise ValueError("exported Z must equal W * H elementwise")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("gate probabilities must lie in [0, 1]")
    if any(np.diag(array).any() for array in (W, Z, probability, truth)):
        raise ValueError("self edges are excluded from this registered benchmark")
    mask = ~np.eye(len(W), dtype=bool)
    predicted = W[mask].astype(bool)
    present = truth[mask] != 0
    pred_sign, true_sign = np.sign(Z[mask]), np.sign(truth[mask])
    tp = int(np.count_nonzero(predicted & present))
    fp = int(np.count_nonzero(predicted & ~present))
    fn = int(np.count_nonzero(~predicted & present))
    signed = {}
    for sign, label in ((1, "activating"), (-1, "repressing")):
        called = predicted & (pred_sign == sign)
        actual = true_sign == sign
        signed_tp = int(np.count_nonzero(called & actual))
        signed_fp = int(np.count_nonzero(called & ~actual))
        signed_fn = int(np.count_nonzero(~called & actual))
        signed[label] = {
            "true_positive": signed_tp, "false_positive": signed_fp,
            "false_negative": signed_fn,
            "f1": _ratio(2 * signed_tp, 2 * signed_tp + signed_fp + signed_fn),
        }
    defined_f1 = [entry["f1"] for entry in signed.values() if entry["f1"] is not None]
    correct_sign = int(np.count_nonzero(predicted & present & (pred_sign == true_sign)))
    return {
        "schema": "q1-graph-score-v1", "orientation": ORIENTATION,
        "n_genes": len(W), "candidate_edges": int(mask.sum()),
        "predicted_edges": int(predicted.sum()), "true_edges": int(present.sum()),
        "expected_edges": float(probability[mask].sum()),
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "true_negative": int(np.count_nonzero(~predicted & ~present)),
        "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "edge_average_precision": edge_average_precision(probability, truth) if present.any() else None,
        "correct_sign_recovered_edges": correct_sign,
        "sign_accuracy_on_recovered_true_edges": _ratio(correct_sign, tp),
        "active_zero_weight_edges": int(np.count_nonzero(predicted & (pred_sign == 0))),
        "signed_f1_macro": float(np.mean(defined_f1)) if defined_f1 else None,
        "signed_classes": signed,
        "scope": "One development network; self edges excluded; no independent-network significance claim",
        "interpretation": (
            "Mature-RNA effective graph compared with simulator production A. "
            "Matching edges is not proof of exact dynamics or unique edge amplitudes. "
            "No mature-RNA Jacobian is compared to a production-rate Jacobian."
        ),
    }


def _load_graph(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not set(GRAPH_KEYS) <= set(archive.files):
            raise ValueError("graph NPZ requires W, H, Z and probability")
        return {key: archive[key] for key in (*GRAPH_KEYS, "orientation") if key in archive.files}


def _load_truth(path: Path) -> np.ndarray:
    """Read only an explicitly named file; no dataset/private-directory search."""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "A" not in archive.files:
                raise ValueError("truth NPZ requires A")
            return archive["A"]
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, dict) and isinstance(value.get("parameters"), dict):
            value = value["parameters"]
        if not isinstance(value, dict) or "A" not in value:
            raise ValueError("truth JSON requires A or parameters.A")
        return np.asarray(value["A"])
    raise ValueError("truth must be an explicit .json parameter file or .npz with A")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = evaluate_graph(_load_graph(args.graph), _load_truth(args.truth))
    report["input_sha256"] = {
        "graph": hashlib.sha256(args.graph.read_bytes()).hexdigest(),
        "truth": hashlib.sha256(args.truth.read_bytes()).hexdigest(),
    }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Preserve previous scores; changing any inputs requires a new file.
        with args.output.open("x") as stream:
            stream.write(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
