import unittest

import numpy as np

from grn_experiment.protocol import Condition
from grn_experiment.selection import (
    FixedRegistry,
    Rival,
    choose_active,
    random_sample,
    uniform_sample,
)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            Condition(q, d, t)
            for q in range(6)
            for d in (0.3, 0.5, 0.7)
            for t in (0.125, 0.25, 0.5, 1.0, 4.0)
        ]

    def test_budget_matched_distinct_choices(self):
        self.assertEqual(len(self.candidates), 90)
        for sampled in (
            uniform_sample(self.candidates),
            random_sample(self.candidates, seed=7),
        ):
            self.assertEqual(len(sampled), 12)
            self.assertEqual(len(set(sampled)), 12)

    def test_fallback_and_fixed_denominator(self):
        chosen = choose_active(self.candidates, [], [])
        self.assertTrue(chosen.fallback)
        self.assertIn(chosen.condition, self.candidates)
        registry = FixedRegistry.from_rivals(
            [
                Rival("a", frozenset({(1, 0)}), lambda c, r: np.zeros((2, 2))),
                Rival("b", frozenset({(0, 1)}), lambda c, r: np.ones((2, 2))),
            ]
        )
        registry.update_survivors(["a"])
        self.assertAlmostEqual(registry.eliminated_fraction, 0.5)
        with self.assertRaises(ValueError):
            FixedRegistry.from_rivals(
                [
                    Rival("restart-1", frozenset({(1, 0)}), lambda c, r: np.zeros((2, 2))),
                    Rival("restart-2", frozenset({(1, 0)}), lambda c, r: np.zeros((2, 2))),
                ]
            )

    def test_active_uses_profiled_rival_disagreement(self):
        contenders = [
            Rival("a", frozenset({(1, 0)}), lambda c, r: np.zeros((4, 2))),
            Rival("b", frozenset({(0, 1)}), lambda c, r: np.full((4, 2), c.time)),
        ]
        choice = choose_active(
            self.candidates[:5], [], contenders, default_efficiency_grid=(0.5, 1.0)
        )
        self.assertFalse(choice.fallback)
        self.assertEqual(choice.condition.time, 4.0)


if __name__ == "__main__":
    unittest.main()
