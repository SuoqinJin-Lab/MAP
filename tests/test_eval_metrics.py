from __future__ import annotations

import numpy as np
import pandas as pd

from map.eval.metrics import (
    condition_metrics,
    covariance_structure_score,
    maximum_mean_discrepancy,
)
from map.eval.program import _condition_degs


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


def test_condition_degs_respects_row_level_evaluation_split():
    class Dataset:
        conditions = pd.DataFrame(
            [{"condition_id": 11, "population": "P0"}]
        ).set_index("condition_id")
        rows_by_condition = {11: np.asarray([2], dtype=np.int64)}

        def _open_population(self, population):
            assert population == "P0"
            return {
                "row_group": np.asarray([7, 7, 7, 7]),
                "hvg": np.asarray(
                    [[100.0, 100.0], [50.0, 50.0], [3.0, 5.0], [1.0, 2.0]]
                ),
            }

        @staticmethod
        def _group_rows(arrays, prefix, group_id):
            if prefix == "condition":
                return np.asarray([0, 1, 2], dtype=np.int64)
            assert prefix == "control_group" and group_id == 7
            return np.asarray([3], dtype=np.int64)

    seen = {}

    def mask(control, condition, fdr):
        seen["condition"] = condition.copy()
        return np.zeros(condition.shape[1], dtype=bool)

    _, true_delta = _condition_degs(
        Dataset(), 11, 0.05, None, 42, mask
    )
    np.testing.assert_array_equal(seen["condition"], [[3.0, 5.0]])
    np.testing.assert_array_equal(true_delta, [2.0, 3.0])
