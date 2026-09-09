"""The single-level LPs must price capital cost exactly as the stochastic forward pass.

WP5-A verification defect 1 (WP5 spec sections 4 and 6).  Every subproblem of a
:class:`~zap.planning.problem_abstract.StochasticPlanningProblem` is built from
``dev.sample_time(block, total_hours)``, which pro-rates capital cost by
``block_hours / total_hours``.  ``StochasticPlanningProblem.forward`` therefore
evaluates ``sum_i w_i * (snapshot_weight_i * op_i + inv_i)`` and recovers
``coverage * CAPEX``.  ``MonolithicPlanningProblem`` and
``RelaxedPlanningProblem`` used to build their investment term from
``subproblems[0]`` alone, charging ``(block_0_hours / total_hours) * CAPEX``:
capital cost under-weighted by the number of blocks.

Runs on the 48-hour WP1 tiny dataset fixture with HiGHS.
"""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

from zap.planning.monolithic import MonolithicPlanningProblem
from zap.planning.relaxation import RelaxedPlanningProblem
from zap.tests.test_ra_planning_core import TinyPlanningMixin, _BuildOnlyMethod


class CapexWeightingMixin(TinyPlanningMixin):
    """Contexts over the tiny fixture, with a knob to force expansion."""

    def scaled_system(self, factor: float = 1.0):
        """The loaded fixture with demand multiplied by ``factor``."""
        from zap.importers.wy_store import LoadedSystem

        loaded = self.loaded()
        if factor == 1.0:
            return loaded
        devices = [copy.deepcopy(d) for d in loaded.devices]
        idx = loaded.index.device_index["Load"]
        devices[idx].load = np.asarray(devices[idx].load, dtype=float) * float(factor)
        return LoadedSystem(
            network=loaded.network,
            devices=devices,
            index=loaded.index,
            meta=dict(loaded.meta),
        )

    def build(self, block_size: int, factor: float = 1.0):
        cfg = self.cfg(selection={"strategy": "all", "block_size": block_size})
        return _BuildOnlyMethod(cfg).build(self.scaled_system(factor))

    @staticmethod
    def off_nominal(ctx) -> dict:
        """Parameters away from the as-built point, so the capex term is non-zero."""
        return {
            p: np.asarray(v, dtype=float) * 1.1 + 1.0 for p, v in ctx.initial_parameters.items()
        }

    @staticmethod
    def investment_value(lp, theta) -> tuple[float, float]:
        """``(weighted investment term, legacy subproblems[0]-only term)`` at ``theta``."""
        import cvxpy as cp

        net_params, _lower, _upper, investment = lp.model_outer_problem()
        for name, var in net_params.items():
            var.value = theta[name].reshape(var.shape)
        legacy = lp.problem.subproblems[0].investment_objective(la=cp, **net_params)
        return float(investment.value), float(legacy.value)


class TestMonolithicCapexWeighting(CapexWeightingMixin):
    def _lp(self, ctx, **kwargs):
        import cvxpy as cp

        return MonolithicPlanningProblem(ctx.problem, solver=cp.HIGHS, **kwargs)

    def test_investment_term_is_invariant_to_blocking(self):
        """(a) 2 x 24 h at coverage 1.0 costs the same capital as 1 x 48 h."""
        whole = self.build(48)
        split = self.build(24)
        self.assertEqual(whole.blocks, [(0, 48)])
        self.assertEqual(split.blocks, [(0, 24), (24, 48)])
        self.assertEqual(whole.coverage, 1.0)
        self.assertEqual(split.coverage, 1.0)

        theta = self.off_nominal(whole)
        inv_whole, legacy_whole = self.investment_value(self._lp(whole), theta)
        inv_split, legacy_split = self.investment_value(self._lp(split), theta)

        self.assertGreater(abs(inv_whole), 1.0)
        self.assertAlmostEqual(inv_split / inv_whole, 1.0, delta=1e-6)

        # And the shape of the old defect: one subproblem alone carries only its
        # own block's share of the capital cost.
        self.assertAlmostEqual(legacy_whole / inv_whole, 1.0, delta=1e-9)
        self.assertAlmostEqual(legacy_split / inv_split, 0.5, delta=1e-9)

    def test_investment_term_matches_the_stochastic_forward_pass(self):
        """The cvxpy capex expression equals ``StochasticPlanningProblem.inv_cost``."""
        ctx = self.build(24)
        theta = self.off_nominal(ctx)
        inv_lp, _legacy = self.investment_value(self._lp(ctx), theta)

        ctx.problem.forward(**theta)
        self.assertAlmostEqual(inv_lp / float(ctx.problem.inv_cost), 1.0, delta=1e-9)

    def test_lp_optimum_matches_the_stochastic_forward_pass(self):
        """(b) the LP objective at its optimum == forward() at the same parameters."""
        # Demand is scaled up so the design actually expands; at as-built demand
        # the tiny fixture is over-built and the investment term is exactly 0.
        ctx = self.build(24, factor=3.0)
        lp = self._lp(ctx)
        params, data = lp.solve()

        self.assertEqual(data["problem"].status, "optimal")
        self.assertGreater(float(data["investment_objective"].value), 0.0)

        forward = float(ctx.problem.forward(**params))
        self.assertAlmostEqual(forward / float(data["problem"].value), 1.0, delta=1e-5)


class TestRelaxedCapexWeighting(CapexWeightingMixin):
    """(c) the same investment term in ``RelaxedPlanningProblem``.

    Only ``model_outer_problem`` is exercised: ``zap.dual.dualize`` has no dual
    class for ``DirectedLine``, so the relaxation cannot be solved on this
    fixture (the links are what make it a two-zone system).
    """

    def _lp(self, ctx):
        import cvxpy as cp

        return RelaxedPlanningProblem(ctx.problem, solver=cp.HIGHS, solver_kwargs={})

    def test_investment_term_is_invariant_to_blocking(self):
        whole = self.build(48)
        split = self.build(24)
        theta = self.off_nominal(whole)

        inv_whole, legacy_whole = self.investment_value(self._lp(whole), theta)
        inv_split, legacy_split = self.investment_value(self._lp(split), theta)

        self.assertGreater(abs(inv_whole), 1.0)
        self.assertAlmostEqual(inv_split / inv_whole, 1.0, delta=1e-6)
        self.assertAlmostEqual(legacy_whole / inv_whole, 1.0, delta=1e-9)
        self.assertAlmostEqual(legacy_split / inv_split, 0.5, delta=1e-9)

    def test_investment_term_matches_the_stochastic_forward_pass(self):
        ctx = self.build(24)
        theta = self.off_nominal(ctx)
        inv_lp, _legacy = self.investment_value(self._lp(ctx), theta)

        ctx.problem.forward(**theta)
        self.assertAlmostEqual(inv_lp / float(ctx.problem.inv_cost), 1.0, delta=1e-9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
