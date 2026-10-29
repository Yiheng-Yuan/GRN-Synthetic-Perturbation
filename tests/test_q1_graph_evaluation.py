import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grn_experiment.q1_graph_evaluation import ORIENTATION, evaluate_graph, main


def exported(weights, probabilities=None):
    H = np.asarray(weights, dtype=float)
    W = (H != 0).astype(int)
    return {"W": W, "H": H, "Z": W * H,
            "probability": W.astype(float) if probabilities is None else probabilities,
            "orientation": np.asarray(ORIENTATION)}


class GraphEvaluationTests(unittest.TestCase):
    def test_perfect_directed_signed_graph(self):
        # j -> i; the deliberately asymmetric graph detects transposition.
        truth = np.array([[0, 0, 0], [1, 0, 0], [0, -2, 0]])
        graph = exported(truth)
        score = evaluate_graph(graph, truth)
        for key in ("precision", "recall", "f1", "edge_average_precision",
                    "sign_accuracy_on_recovered_true_edges", "signed_f1_macro"):
            self.assertEqual(score[key], 1.0)
        self.assertEqual(score["true_positive"], 2)
        transposed = evaluate_graph(exported(truth.T), truth)
        self.assertEqual(transposed["true_positive"], 0)
        self.assertEqual(transposed["f1"], 0)

    def test_false_missing_and_wrong_sign_are_separate(self):
        truth = np.array([[0, 0, 1], [1, 0, 0], [0, -2, 0]])
        graph = exported([[0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        result = evaluate_graph(graph, truth)
        self.assertEqual((result["true_positive"], result["false_positive"], result["false_negative"]), (2, 1, 1))
        self.assertEqual(result["correct_sign_recovered_edges"], 1)
        self.assertEqual(result["sign_accuracy_on_recovered_true_edges"], .5)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["signed_classes"]["repressing"]["f1"], 2 / 3)
        self.assertEqual(result["signed_classes"]["activating"]["f1"], 0)

    def test_ap_uses_gate_probabilities_not_amplitude(self):
        truth = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        graph = exported([[0, 999, 999], [.0001, 0, 999], [999, 999, 0]],
                         np.array([[0, .1, .1], [.9, 0, .1], [.1, .1, 0]]))
        self.assertEqual(evaluate_graph(graph, truth)["edge_average_precision"], 1)
        graph["probability"] = np.ones((3, 3)) - np.eye(3)
        self.assertAlmostEqual(evaluate_graph(graph, truth)["edge_average_precision"], 1 / 6)

    def test_zero_denominators_are_json_null(self):
        empty = np.zeros((3, 3))
        result = evaluate_graph(exported(empty), empty)
        for key in ("precision", "recall", "f1", "edge_average_precision",
                    "sign_accuracy_on_recovered_true_edges", "signed_f1_macro"):
            self.assertIsNone(result[key])
        json.dumps(result, allow_nan=False)

    def test_active_zero_weight_edge_counts_as_a_structural_edge(self):
        graph = exported(np.zeros((2, 2)))
        graph["W"][1, 0] = 1
        result = evaluate_graph(graph, np.array([[0, 0], [1, 0]]))
        self.assertEqual(result["predicted_edges"], 1)
        self.assertEqual(result["active_zero_weight_edges"], 1)
        self.assertEqual(result["sign_accuracy_on_recovered_true_edges"], 0)

    def test_rejects_corrupt_artifact_and_truth(self):
        good = exported([[0, 0], [1, 0]])
        cases = []
        for key, index, value in (("W", (1, 0), .5), ("Z", (1, 0), 2),
                                  ("probability", (0, 1), 1.1), ("H", (0, 1), np.nan),
                                  ("probability", (0, 0), .1)):
            bad = {k: v.copy() for k, v in good.items()}
            bad[key] = bad[key].astype(float)
            bad[key][index] = value
            cases.append(bad)
        cases += [{k: v for k, v in good.items() if k != "W"},
                  {**good, "orientation": np.asarray("row_source_col_receiver")},
                  {**good, "Z": np.zeros((3, 3))}]
        for graph in cases:
            with self.subTest(graph=graph):
                with self.assertRaises(ValueError):
                    evaluate_graph(graph, good["Z"])
        with self.assertRaises(ValueError):
            evaluate_graph(good, np.eye(2))
        with self.assertRaises(ValueError):
            evaluate_graph(good, np.array([[0, np.nan], [0, 0]]))

    def test_cli_reads_explicit_generator_json_and_preserves_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = exported([[0, 0], [-2, 0]])
            np.savez(root / "learned.npz", **graph)
            (root / "generator.json").write_text(json.dumps({
                "parameters": {"A": [[0, 0], [-1, 0]], "unrelated": "not a training input"}}))
            argv = ["--graph", str(root / "learned.npz"), "--truth", str(root / "generator.json"),
                    "--output", str(root / "score.json")]
            self.assertEqual(main(argv), 0)
            score = json.loads((root / "score.json").read_text())
            self.assertEqual(score["edge_average_precision"], 1)
            self.assertEqual(len(score["input_sha256"]["truth"]), 64)
            with self.assertRaises(FileExistsError):
                main(argv)

    def test_cli_npz_truth_needs_no_simulator_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "learned.npz", **exported([[0, 1], [0, 0]]))
            np.savez(root / "truth.npz", A=np.array([[0, 1], [0, 0]]))
            self.assertEqual(main(["--graph", str(root / "learned.npz"),
                                   "--truth", str(root / "truth.npz")]), 0)


if __name__ == "__main__":
    unittest.main()
