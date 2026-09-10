"""Verifier reproductions for the 2026-09-10 step-rule diff (uncommitted).

Each test here FAILS on the tree it was written against; it documents a
confirmed defect for the orchestrator.  Spec:
``memory/plans/2026-09-10-step-rule-spec.md``.
"""

from __future__ import annotations

import glob
import sys
import unittest
from pathlib import Path

import numpy as np

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

from zap.planning.problem_abstract import _plateau
from zap.tests.test_ra_planning_methods import PLANNING_AVAILABLE, TASK_A, PlanningFixtureMixin, deep_merge
from zap.tests.test_ra_step_rules import GRADIENT

try:
    from experiments.ra.planning import history as history_mod
except Exception:  # noqa: BLE001  # pragma: no cover
    history_mod = None  # type: ignore[assignment]

C4_RUNS = "/Users/kamran/Documents/CH3_SGD_Planning/experiments/runs/plan_z4_2020_c4_grad_det-*"


class TestPlateauWindow(unittest.TestCase):
    """Finding 1: the plateau test is live from the 2nd value, not over ``tol_window``.

    Spec section 3: the relative decrease is measured **over a window W** as
    the difference of two half-window means.  ``_plateau`` shrinks the window
    to whatever history exists, so for the first ``W`` iterations it is a
    single-iteration test -- roughly ``W/2`` times stricter than the spec's --
    and signed, so one non-improving iteration ends the run.
    """

    def test_shrinking_window_stops_c4_like_progress_at_iteration_1(self):
        # c4's measured per-iteration relative decrease is 4.7e-5 (its recorded
        # sampled_objective_raw); over a W = 20 window that is ~4.7e-4 > 1e-4,
        # so the spec's test must NOT fire.
        f0 = 6.336e9
        series = [f0 * (1.0 - 4.7e-5) ** t for t in range(21)]
        self.assertFalse(_plateau(series, 20, 1e-4), "20-value window must not stop")
        # ... but the shipped test, handed the first two values, stops the run.
        self.assertFalse(
            _plateau(series[:2], 20, 1e-4),
            "plateau fired on a 2-value history although tol_window = 20",
        )

    def test_c4_recorded_series_would_have_stopped_after_one_iteration(self):
        files = glob.glob(f"{C4_RUNS}/iterations/*.iterations.parquet")
        if not files:
            self.skipTest("c4 run directory not present")
        import pandas as pd

        L = pd.read_parquet(files[0]).sort_values("iteration")["sampled_objective_raw"]
        L = L.astype(float).tolist()
        first = next((n for n in range(2, len(L) + 1) if _plateau(L[:n], 20, 1e-4)), None)
        self.assertTrue(first is None or first >= 20, f"fired at history index {first}")


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestTrustRegionRadiusIsRealised(PlanningFixtureMixin, unittest.TestCase):
    """Finding 2: the trust-region step is normalised over ALL rows.

    ``TrustRegionDescent.step`` divides by the 2-norm over every parameter row,
    frozen ones included -- the exact defect the spec diagnosed in the
    ``gradient`` rule (section 1: 98.7 % of the norm sits on rows that cannot
    move, so the realised step was 1.3 % of the reported one).  The radius is
    therefore not the step in MW the design takes; on ca2040_z4 A3 saturates at
    ``max_radius_mw`` and moves ~1 % of it.
    """

    def test_realised_free_step_matches_the_radius(self):
        radius = 50.0
        result = self.plan(
            planning=deep_merge(
                GRADIENT,
                {
                    "optimizer": {
                        "rule": "trust_region",
                        "num_iterations": 1,
                        "trust_region": {"initial_radius_mw": radius},
                    }
                },
            )
        )
        frame = history_mod.build_iteration_tables(result)["iterations"]
        row = frame[frame["iteration"] == 1].iloc[0]
        self.assertGreater(int(row["n_free"]), 0)
        # 4 of 14 rows are free on this fixture; the realised step is 8 MW of 50.
        self.assertGreaterEqual(
            float(row["step_norm_free_mw"]),
            0.5 * radius,
            f"radius {radius} MW, realised free step {row['step_norm_free_mw']:.2f} MW",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
