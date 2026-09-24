import numpy as np
import pytest
import torch

from grn_experiment.ambiguity import (
    QualificationThresholds,
    RivalSpecification,
    minimum_local_effect,
    qualify_rival,
    single_edge_swaps,
    support_mask,
    top_edge_support,
)
from grn_experiment.learning import SparseRNAODE
from grn_experiment.protocol import Condition, ReplicatedSnapshot


def test_distinct_structures_with_fixed_budget_and_indegree():
    torch.manual_seed(2)
    model = SparseRNAODE(6)
    support = top_edge_support(model, (0, 1, 2, 3), n_edges=9, max_indegree=2, max_outdegree=3)
    assert len(support) == 9
    assert all(sum(target == gene for target, _ in support) >= 1 for gene in (4, 5))
    rivals = single_edge_swaps(
        support, n_genes=6, regulators=(0, 1, 2, 3), count=8,
        seed=4, max_indegree=2, max_outdegree=3,
    )
    assert len({r.support for r in rivals}) == 8
    for rival in rivals:
        assert len(rival.support) == len(support)
        assert sum(target == 0 for target, _ in rival.support) <= 2
        assert all(sum(regulator == gene for _, regulator in rival.support) <= 3 for gene in (0, 1, 2, 3))
        assert all(sum(target == gene for target, _ in rival.support) >= 1 for gene in (4, 5))
    assert support_mask(6, support).sum() == 9


def test_actual_edge_effect_prevents_near_zero_edge_from_qualifying():
    support = frozenset({(1, 0)})
    model = SparseRNAODE(3, allowed_edges=support_mask(3, support))
    with torch.no_grad():
        model.weights.zero_()
        model.weights[1, 0] = 1e-8
    effect = minimum_local_effect(model, support, np.ones((2, 3)))
    assert effect < 1e-5
    baseline = ReplicatedSnapshot(np.ones((3, 4, 3)))
    condition = Condition(0, 0.3, 0.25)
    snapshot = ReplicatedSnapshot(np.ones((3, 4, 3)), np.ones((3, 4, 3)))
    report = qualify_rival(
        RivalSpecification("near-zero", support),
        model,
        baseline,
        {condition: snapshot},
        QualificationThresholds(100, 100, 1e-5),
        effect_states=np.ones((2, 3)),
    )
    assert not report.qualified


def test_malformed_support_or_duplicate_budget_rejected():
    with pytest.raises(ValueError):
        support_mask(3, frozenset({(1, 1)}))
    with pytest.raises(ValueError):
        QualificationThresholds(0, 1, 1)
