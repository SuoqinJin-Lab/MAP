from __future__ import annotations

import numpy as np

from map.eval.metrics import (
    condition_metrics,
    covariance_structure_score,
    maximum_mean_discrepancy,
)


def test_mmd_is_distinct_from_wasserstein_and_zero_for_identical_values():
    observed = np.linspace(0.0, 2.0, 64)
    shifted = observed + 0.75
    assert maximum_mean_discrepancy(observed, observed) == 0.0
    metrics = condition_metrics(shifted, observed, np.zeros_like(observed))
    assert metrics["mmd"] > 0.0
    assert metrics["wasserstein"] > 0.0
    assert not np.isclose(metrics["mmd"], metrics["wasserstein"])


def test_css_measures_hvg_response_covariance_difference():
    observed = np.asarray([
        [1.0, 0.0, 2.0],
        [2.0, 1.0, 1.0],
        [3.0, 1.5, 0.0],
        [4.0, 3.0, -1.0],
    ])
    assert covariance_structure_score(observed, observed) == 0.0
    distorted = observed.copy()
    distorted[:, 1] *= -2.0
    assert covariance_structure_score(distorted, observed) > 0.0
