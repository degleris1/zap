"""Per-method tests for the WP5 planning methods (spec section 7.1, tests 3-19).

Everything runs on the WP1 tiny fixture: 48 hours of one weather year, one or
two blocks, HiGHS for the LP presets and CLARABEL for the gradient path (the
KKT system the implicit-differentiation gradient factors is built from the
dispatch duals, and CLARABEL's interior-point duals are better conditioned than
HiGHS's vertex duals on a degenerate LP).

Two deliberate departures from the fixture the spec assumes, both so the tests
are not vacuous:

* the capital costs in ``static/*.csv`` are pro-rated to the 48-hour horizon
  (``capital_cost * 48/8760``).  The tiny fixture carries *annual* capital costs
  and ``sample_time`` pro-rates only by ``block_hours / total_hours``, so over a
  48-hour horizon a full year of capex is charged against two days of
  operations and every extendable row collapses to its floor -- no design
  question is left to test;
* several rows are made ``p_nom_extendable`` with a finite ``p_nom_max``.  The
  checked-in tiny fixture freezes every row, which makes ``expansion.mode:
  pypsa`` a no-op.

Tests 1, 2 and 20 of spec section 7.1 belong to Task A
(``test_ra_planning_core.py``); the equivalence tests are in
``test_ra_planning_equivalence.py``.
"""

from __future__ import annotations

import itertools
import json
import sys
import tempfile
import types
import typing
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

from zap.importers.wy_store import (
    HourWindow,
    LoadOptions,
    convert_dataset,
    load_system,
)
from zap.tests.fixtures.tiny_dataset import write_tiny_dataset

try:
    from experiments.ra import planning
    from experiments.ra.config import ConfigError

    PLANNING_AVAILABLE = True
    PLANNING_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - Task A not merged
    planning = None  # type: ignore[assignment]
    ConfigError = ValueError  # type: ignore[misc,assignment]
    PLANNING_AVAILABLE = False
    PLANNING_IMPORT_ERROR = exc

TASK_A = f"experiments.ra.planning is not importable: {PLANNING_IMPORT_ERROR!r}"

N_HOURS = 48
VOLL = 10_000.0

#: Generator rows made extendable, with their ``p_nom_max`` in MW.
EXTENDABLE_GENERATORS = {
    "z1 CCGT": 300.0,
    "z1 solar": 400.0,
    "z2 CCGT": 300.0,
    "z2 onwind": 400.0,
}
EXTENDABLE_STORAGE = {"z1 battery": 200.0}

#: Index of ``z1 solar`` and of the two CCGT rows in ``generator_capacity``.
SOLAR_ROW = 1
CCGT_ROWS = (0, 2)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def write_planning_dataset(root: Path, *, n_hours: int = N_HOURS) -> Path:
    """The tiny dataset with real expansion bounds and horizon-scaled capex."""
    root = write_tiny_dataset(root, n_hours=n_hours, years=(2020,))
    capital_scale = n_hours / 8760.0
    static = root / "static"

    for fname, extendable, scale_capex in (
        ("generators.csv", EXTENDABLE_GENERATORS, True),
        ("storage_units.csv", EXTENDABLE_STORAGE, True),
        ("links.csv", {}, False),
    ):
        df = pd.read_csv(static / fname, index_col=0)
        df["p_nom_min"] = df["p_nom"]
        df["p_nom_max"] = df["p_nom"]
        for name, p_nom_max in extendable.items():
            df.loc[name, "p_nom_extendable"] = True
            df.loc[name, "p_nom_min"] = 0.0
            df.loc[name, "p_nom_max"] = p_nom_max
        if scale_capex:
            df["capital_cost"] = df["capital_cost"] * capital_scale
        df.to_csv(static / fname)

    convert_dataset(root)
    return root


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def plan_config(dataset: Path, **overrides) -> dict:
    """A ``mode: plan`` config; only the keys the planning core reads."""
    cfg = {
        "mode": "plan",
        "dataset": {
            "dir": str(dataset),
            "years": [2020],
            "window": {"start": 0, "stop": N_HOURS},
        },
        "heuristics": {"name": "none", "ucap_derate": False, "outage_draws": []},
        "selection": {
            "strategy": "all",
            "block_size": 24,
            "num_blocks": None,
            "seed": 42,
        },
        "planning": {
            "method": "monolithic",
            "dispatch_solver": "CLARABEL",
            "dispatch_solver_kwargs": {"verbose": False},
            "expansion": {"mode": "pypsa"},
            "single_level": {"kind": "primal", "solver": "HIGHS"},
        },
    }
    for key, value in overrides.items():
        cfg[key] = deep_merge(cfg[key], value) if isinstance(cfg.get(key), dict) else value
    return cfg


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class PlanningFixtureMixin:
    """One dataset and one loaded system for the whole module."""

    _tmp: tempfile.TemporaryDirectory
    dataset: Path
    loaded: object

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dataset = write_planning_dataset(Path(cls._tmp.name) / "tiny")
        cls.loaded = load_system(
            cls.dataset,
            LoadOptions(years=(2020,), window=HourWindow(0, N_HOURS), voll=VOLL),
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # -- helpers ---------------------------------------------------------

    def plan(self, **overrides):
        return planning.plan(self.loaded, plan_config(self.dataset, **overrides))

    def assert_within_bounds(self, result):
        for param, value in result.parameters.items():
            value = np.asarray(value, dtype=float).reshape(-1)
            lower = np.asarray(result.lower_bounds[param], dtype=float).reshape(-1)
            upper = np.asarray(result.upper_bounds[param], dtype=float).reshape(-1)
            self.assertTrue(np.all(value >= lower - 1e-9), f"{param} below its lower bound")
            self.assertTrue(np.all(value <= upper + 1e-9), f"{param} above its upper bound")

    @staticmethod
    def flat(result, param="generator_capacity"):
        return np.asarray(result.parameters[param], dtype=float).reshape(-1)


# ===========================================================================
# Single-level LP presets (spec 7.1 tests 3-7)
# ===========================================================================


class TestSingleLevel(PlanningFixtureMixin, unittest.TestCase):
    def test_single_level_primal_produces_design(self):
        """Spec 7.1 test 3: one block of 48 h solves, and the parts add up."""
        result = self.plan(selection={"block_size": None})
        self.assertEqual(result.solver["status"], "optimal")
        self.assertEqual(result.solver["n_subproblems"], 1)

        devices = {type(d).__name__: d for d in self.loaded.devices}
        for param, (device_idx, _attr) in result.parameter_names.items():
            cls_name = type(self.loaded.devices[device_idx]).__name__
            self.assertEqual(
                self.flat(result, param).size,
                devices[cls_name].num_devices,
                f"{param}: one capacity per source row",
            )
        self.assert_within_bounds(result)

        objective = result.objective
        self.assertAlmostEqual(
            (objective["capex_raw"] + objective["opex_raw"]) / objective["raw"],
            1.0,
            delta=1e-6,
        )

    def test_stochastic_preset_two_blocks(self):
        """Spec 7.1 test 4: two 24 h blocks cover the horizon exactly."""
        result = self.plan(
            selection={"strategy": "all", "block_size": 24},
            planning={"method": "stochastic"},
        )
        self.assertEqual(result.solver["n_subproblems"], 2)
        self.assertEqual(result.annualization["coverage"], 1.0)
        self.assertEqual(result.annualization["annualization_factor"], 1.0)
        self.assertEqual(result.objective_annual, result.objective_raw)
        self.assert_within_bounds(result)

    def test_partial_coverage_annualizes(self):
        """Spec 7.1 test 5: half the horizon doubles the annualization factor."""
        result = self.plan(
            selection={"strategy": "random", "block_size": 24, "num_blocks": 1, "seed": 0}
        )
        self.assertEqual(result.annualization["sampled_hours"], 24)
        self.assertEqual(result.annualization["coverage"], 0.5)
        self.assertEqual(result.annualization["annualization_factor"], 2.0)
        self.assertEqual(result.objective_annual, 2.0 * result.objective_raw)
        self.assertEqual(
            result.objective["emissions_tonnes_annual"],
            2.0 * result.objective["emissions_tonnes_raw"],
        )

    def test_selection_seed_is_reproducible(self):
        """Spec 7.1 test 6: the seed pins the block list and the design."""
        selection = {"strategy": "random", "block_size": 24, "num_blocks": 1, "seed": 0}
        first = self.plan(selection=dict(selection))
        again = self.plan(selection=dict(selection))
        other = self.plan(selection=dict(selection, seed=7))

        self.assertEqual(first.selection["blocks"], again.selection["blocks"])
        np.testing.assert_allclose(self.flat(first), self.flat(again), rtol=1e-9)
        self.assertNotEqual(first.selection["blocks"], other.selection["blocks"])

    @unittest.expectedFailure
    def test_strong_duality_matches_primal(self):
        """Spec 7.1 test 7.

        Expected failure, reported rather than deleted: ``RelaxedPlanningProblem``
        dualizes the device list through ``zap.dual.dualize``, whose
        ``DUAL_CLASS`` table has no entry for WP3's ``DirectedLine`` (it knows
        ``DCLine``/``ACLine`` only), so ``kind: strong_duality`` raises
        ``KeyError: DirectedLine`` on every system this harness builds.  A dual
        ``DirectedLine`` is a phase-2 zap change.
        """
        primal = self.plan(selection={"block_size": None})
        relaxed = self.plan(
            selection={"block_size": None},
            planning={
                "method": "relaxed",
                "single_level": {"kind": "strong_duality", "price_bound": 2.0 * VOLL},
            },
        )
        self.assertAlmostEqual(relaxed.objective_raw / primal.objective_raw, 1.0, delta=1e-4)


# ===========================================================================
# Emissions (spec 7.1 tests 8-11)
# ===========================================================================


class TestEmissions(PlanningFixtureMixin, unittest.TestCase):
    def test_emissions_price_shifts_design(self):
        """Spec 7.1 test 8: a carbon price moves capacity onto the clean carriers.

        Restated for the no-retirement decision (WP5-A verification defect 2):
        ``apply_expansion`` now floors every existing row at its as-built
        capacity, so no design can shrink the dirtiest carrier -- the untaxed
        design already leaves both CCGTs at their floor.  The price shows up as
        *more* zero-emission capacity, and as lower emissions.
        """
        base = self.plan()
        priced = self.plan(planning={"emissions": {"mode": "price", "price": 500.0}})

        rates = np.asarray(self.loaded.index.emission_rates, dtype=float).reshape(-1)
        carriers = np.asarray(self.loaded.index.carrier["Generator"])
        dirtiest = int(np.argmax(rates))
        self.assertGreater(rates[dirtiest], 0.0)
        dirty_rows = np.flatnonzero(carriers == carriers[dirtiest])
        clean_rows = np.flatnonzero(rates <= 0.0)

        # the dirtiest carrier never grows under a price ...
        dirty_moved = self.flat(base)[dirty_rows].sum() - self.flat(priced)[dirty_rows].sum()
        self.assertGreaterEqual(dirty_moved, -1e-6)
        # ... and clean capacity does
        clean_added = self.flat(priced)[clean_rows].sum() - self.flat(base)[clean_rows].sum()
        self.assertGreaterEqual(
            clean_added, 1.0, f"clean capacity barely moved ({clean_added:.3f} MW)"
        )
        self.assertLess(
            priced.objective["emissions_tonnes_annual"],
            base.objective["emissions_tonnes_annual"],
        )

    def test_carbon_payment_is_not_part_of_system_cost(self):
        """WP5 verification: the objective is system cost, not cost + carbon payment.

        Under ``mode: price`` the LP minimizes ``capex + dispatch cost + price *
        emissions``.  The payment is a transfer, not a resource cost
        (PROJECT.md 2.4), so it is recorded on its own and netted out of both
        ``opex_raw`` and ``raw``; what is left is capex plus the *unpriced*
        dispatch cost of the dispatch this design runs.

        The reference numbers are taken by re-solving the same LP here, so the
        test does not just restate ``SingleLevelMethod``'s own arithmetic.
        """
        import cvxpy as cp

        from zap.planning import MonolithicPlanningProblem

        price = 500.0
        emissions_cfg = {"emissions": {"mode": "price", "price": price}}
        priced = self.plan(planning=dict(emissions_cfg))
        obj = priced.objective

        ctx = planning.make_method(plan_config(self.dataset, planning=dict(emissions_cfg))).build(
            self.loaded
        )
        _params, data = MonolithicPlanningProblem(
            ctx.problem, inf_value=100.0, solver=cp.HIGHS, solver_kwargs={}
        ).solve()
        priced_value = float(data["problem"].value)
        capex = float(data["investment_objective"].value)
        opex_priced = float(sum(float(t.value) for t in data["operation_objective"]))
        emissions = float(sum(float(t.value) for t in data["emissions_terms"]))
        payment = price * emissions
        self.assertGreater(payment, 0.0)

        # emissions are read off the solved LP, i.e. from the dispatch the LP
        # actually chose under the price
        self.assertAlmostEqual(obj["emissions_tonnes_raw"] / emissions, 1.0, delta=1e-6)
        self.assertAlmostEqual(obj["carbon_payment_raw"] / payment, 1.0, delta=1e-6)
        # opex is the unpriced dispatch cost; the objective is capex + that
        self.assertAlmostEqual(obj["opex_raw"] / (opex_priced - payment), 1.0, delta=1e-6)
        self.assertAlmostEqual(obj["raw"] / (capex + opex_priced - payment), 1.0, delta=1e-6)
        self.assertAlmostEqual(obj["raw"], obj["capex_raw"] + obj["opex_raw"], places=6)
        # ... and it is *not* the LP's own (priced) optimum
        self.assertAlmostEqual(
            (obj["raw"] + obj["carbon_payment_raw"]) / priced_value, 1.0, delta=1e-6
        )
        self.assertAlmostEqual(
            obj["carbon_payment_annual"],
            obj["carbon_payment_raw"] * priced.annualization["annualization_factor"],
            places=6,
        )

    def test_emissions_cap_binds(self):
        """Spec 7.1 test 9: a horizon cap at half the free emissions binds."""
        free = self.plan()
        target = 0.5 * free.objective["emissions_tonnes_raw"]
        capped = self.plan(
            planning={"emissions": {"mode": "cap", "cap_basis": "horizon", "cap": target}}
        )
        self.assertLessEqual(capped.objective["emissions_tonnes_raw"], target * (1.0 + 1e-6) + 1e-9)
        self.assertGreater(capped.objective_raw, free.objective_raw)
        self.assertEqual(capped.emissions["cap_applied"], target)

    def test_emissions_cap_annual_basis_is_scaled(self):
        """Spec 7.1 test 10: the same physical cap, two bases, one answer."""
        selection = {"strategy": "random", "block_size": 24, "num_blocks": 1, "seed": 0}
        free = self.plan(selection=dict(selection))
        af = free.annualization["annualization_factor"]
        self.assertEqual(af, 2.0)

        horizon_cap = 0.5 * free.objective["emissions_tonnes_raw"]
        by_horizon = self.plan(
            selection=dict(selection),
            planning={"emissions": {"mode": "cap", "cap_basis": "horizon", "cap": horizon_cap}},
        )
        by_annual = self.plan(
            selection=dict(selection),
            planning={
                "emissions": {
                    "mode": "cap",
                    "cap_basis": "annual",
                    "cap": horizon_cap * af,
                }
            },
        )
        self.assertAlmostEqual(by_annual.emissions["cap_applied"], horizon_cap, places=9)
        self.assertAlmostEqual(
            by_annual.objective["emissions_tonnes_raw"],
            by_horizon.objective["emissions_tonnes_raw"],
            delta=1e-6 * max(1.0, horizon_cap),
        )

    def test_emissions_cap_rejected_for_gradient_and_relaxed(self):
        """Spec 7.1 test 11: D-W5 is a config error, not a runtime surprise."""
        for method, kind in (
            ("gradient", "primal"),
            ("admm", "primal"),
            ("relaxed", "strong_duality"),
        ):
            with self.subTest(method=method):
                cfg = plan_config(
                    self.dataset,
                    planning={
                        "method": method,
                        "single_level": {"kind": kind},
                        "emissions": {"mode": "cap", "cap": 1.0},
                    },
                )
                with self.assertRaises(ConfigError):
                    planning.make_method(cfg)

    def test_dual_ascent_rejected_for_single_level(self):
        """The other half of D-W5: dual ascent needs an outer loop to run in."""
        cfg = plan_config(
            self.dataset,
            planning={
                "method": "monolithic",
                "emissions": {"mode": "dual_ascent", "dual_ascent": {"target": 1.0}},
            },
        )
        with self.assertRaises(ConfigError):
            planning.make_method(cfg)


# ===========================================================================
# Gradient method (spec 7.1 tests 12-16)
# ===========================================================================


#: ``step_size`` / ``clip`` are deliberately *not* set here: the tests below
#: exercise the shipped defaults in ``PLANNING_DEFAULTS`` (0.2 / 5e3), because a
#: step size that moves nothing is exactly the defect these tests must catch.
GRADIENT_PLAN = {
    "method": "gradient",
    "dispatch_solver": "CLARABEL",
    "optimizer": {"num_iterations": 3, "batch_size": 0},
}


def fleet_mw(mapping) -> float:
    """Total capacity, in MW, of a parameter dict."""
    return float(sum(np.asarray(v, dtype=float).sum() for v in (mapping or {}).values()))


class TestGradient(PlanningFixtureMixin, unittest.TestCase):
    def test_gradient_method_runs_and_descends(self):
        """Spec 7.1 test 13, plus the two assertions that make it non-vacuous.

        Deviation: the spec asks for ``batch_size: 1``.  With one of two blocks
        per step, ``history["loss"]`` alternates between two *different* batch
        losses, so ``loss[-1] <= loss[0]`` is not a descent statement at all.
        The descent assertion therefore runs full batch; ``batch_size: 1`` is
        exercised separately below.

        ``loss[-1] <= loss[0]`` on its own passes for ``step_size = 0``, which
        is what the shipped configs effectively were: ``GradientDescent.step``
        caps one iteration's movement at ``step_size * clip`` MW, and the old
        (1e-3, 1e+3) pair capped it at 1 MW on a 702 MW fleet.  So this also
        requires a *material* decrease in the loss and a *material* change in
        the design, both measured against the shipped defaults.
        """
        result = self.plan(planning=dict(GRADIENT_PLAN))
        loss = [float(x) for x in result.history["loss"]]
        self.assertTrue(np.all(np.isfinite(loss)), f"non-finite loss: {loss}")
        self.assertLessEqual(loss[-1], loss[0] + 1e-6)
        self.assert_within_bounds(result)

        # (a) the loss falls by at least 0.1 % of its initial magnitude
        self.assertGreaterEqual(
            loss[0] - loss[-1],
            1.0e-3 * abs(loss[0]),
            f"loss barely moved over {len(loss) - 1} iterations: {loss}",
        )

        # (b) the design moves by at least 1 % of the as-built fleet
        initial = result.meta["initial_parameters"]
        fleet = fleet_mw(initial)
        moved = sum(
            float(
                np.abs(
                    np.asarray(result.parameters[p], dtype=float).reshape(-1) - np.asarray(v)
                ).sum()
            )
            for p, v in initial.items()
        )
        self.assertGreater(fleet, 0.0)
        self.assertGreaterEqual(
            moved,
            0.01 * fleet,
            f"the design moved {moved:.3f} MW of a {fleet:.1f} MW fleet: "
            "step_size * clip is too small for the method to do anything",
        )

    def test_gradient_history_serializes(self):
        """The other half of spec 7.1 test 13: the history must round-trip as JSON.

        ``trackers.track_loss`` returns ``J.cpu().detach().numpy()`` -- a 0-d
        ``np.ndarray``, which the ported ``serialize_history`` left untouched, so
        ``json.dumps`` raised for every gradient run.  Fixed in
        ``planning/design.py``; this used to be an expected failure.
        """
        from experiments.ra.planning.design import serialize_history

        result = self.plan(planning=dict(GRADIENT_PLAN))
        json.dumps(serialize_history(result.history))

    def test_gradient_minibatch_runs(self):
        """``batch_size: 1`` over two blocks: every batch loss is finite."""
        result = self.plan(planning=deep_merge(GRADIENT_PLAN, {"optimizer": {"batch_size": 1}}))
        loss = [float(x) for x in result.history["loss"]]
        self.assertEqual(len(loss), 4)  # init_full_loss + 3 iterations
        self.assertTrue(np.all(np.isfinite(loss)))
        self.assert_within_bounds(result)

    def test_gradient_warm_start_uses_lp_solution(self):
        """Spec 7.1 test 14, over **two** blocks.

        This used to be pinned to a single block because
        ``MonolithicPlanningProblem`` built its investment term from
        ``subproblems[0]`` alone and so disagreed with
        ``StochasticPlanningProblem.forward`` whenever there was more than one
        block.  That is fixed (``test_single_level_capex_weighting.py``), so the
        real statement -- the gradient loop starts at the LP optimum, and the LP
        optimum is a lower bound on it -- is tested where it can fail: two 24 h
        blocks, warm-started on both of them.
        """
        warm = self.plan(
            selection={"strategy": "all", "block_size": 24},
            planning=deep_merge(
                GRADIENT_PLAN,
                {
                    "warm_start": {"enabled": True, "solver": "HIGHS"},
                    "optimizer": {"num_iterations": 2},
                },
            ),
        )
        self.assertEqual(warm.solver["n_subproblems"], 2)
        lower_bound = warm.objective["lower_bound_raw"]
        self.assertIsNotNone(lower_bound)
        self.assertAlmostEqual(float(warm.history["loss"][0]) / lower_bound, 1.0, delta=1e-6)
        # The warm start saw the whole block set, so its objective *is* the bound.
        self.assertAlmostEqual(warm.objective["warm_start_objective_raw"], lower_bound, places=9)
        self.assertEqual(warm.objective["warm_start_sampled_hours"], 48)
        self.assertIsNotNone(warm.objective["optimality_gap"])
        self.assertGreaterEqual(warm.objective["optimality_gap"], -1e-9)

        cold = self.plan(
            selection={"strategy": "all", "block_size": 24},
            planning=deep_merge(GRADIENT_PLAN, {"warm_start": {"enabled": False}}),
        )
        self.assertIsNone(cold.objective["lower_bound_raw"])
        self.assertIsNone(cold.objective["optimality_gap"])
        self.assertIsNone(cold.objective["warm_start_objective_raw"])

    def test_partial_warm_start_reports_no_bound(self):
        """A warm start on a *subset* of the blocks is not a bound (WP5 verification).

        ``sample_time`` pro-rates capital cost by ``block_hours / total_hours``,
        so an LP over 1 of 2 blocks charges half the capex the gradient loop
        charges: its optimum is a smaller number in different units, and
        reporting it as ``lower_bound_raw`` invented an optimality gap of ~100 %.
        It is now recorded as ``warm_start_objective_raw``, with the hours it was
        measured over, and no bound is claimed.
        """
        partial = self.plan(
            selection={"strategy": "all", "block_size": 24},
            planning=deep_merge(
                GRADIENT_PLAN,
                {
                    "warm_start": {"enabled": True, "num_blocks": 1, "solver": "HIGHS"},
                    "optimizer": {"num_iterations": 2},
                },
            ),
        )
        self.assertIsNone(partial.objective["lower_bound_raw"])
        self.assertIsNone(partial.objective["optimality_gap"])
        self.assertIsNotNone(partial.objective["warm_start_objective_raw"])
        self.assertEqual(partial.objective["warm_start_sampled_hours"], 24)
        # The suboptimality tracker is fed NaN, not 1.0, so nothing downstream
        # can mistake `J - 1` for a gap.
        self.assertTrue(np.all(np.isnan([float(x) for x in partial.history["suboptimality"]])))

    def test_gradient_objective_is_the_full_forward_not_the_last_minibatch(self):
        """WP5 verification: ``objective.raw`` must describe the whole design.

        ``history["loss"][-1]`` is the loss of the last *minibatch* (rescaled by
        the number of subproblems), which is a different number from the design's
        objective whenever ``batch_size`` is set.  Reporting it made the card's
        objective a sample of one block.
        """
        result = self.plan(
            planning=deep_merge(GRADIENT_PLAN, {"optimizer": {"batch_size": 1}}),
        )
        raw = result.objective["raw"]
        self.assertAlmostEqual(
            raw, result.objective["capex_raw"] + result.objective["opex_raw"], places=6
        )

        # Rebuild the same problem and evaluate it at the returned design.
        cfg = plan_config(self.dataset, planning=deep_merge(GRADIENT_PLAN, {"batch_size": 0}))
        ctx = planning.make_method(cfg).build(self.loaded)
        full = float(
            ctx.problem.forward(**{k: np.asarray(v) for k, v in result.parameters.items()})
        )
        self.assertAlmostEqual(raw / full, 1.0, delta=1e-6)

        # ... and it is not the last minibatch loss, which is what used to be
        # reported (one of two blocks, so the two differ by ~a factor of two).
        self.assertGreater(abs(raw - float(result.history["loss"][-1])), 1e-3 * abs(raw))

    def test_gradient_workers_ordering(self):
        """Spec 7.1 test 15: workers are initialized after the warm-start deepcopy."""
        result = self.plan(
            planning=deep_merge(
                GRADIENT_PLAN,
                {
                    "num_workers": 2,
                    "warm_start": {"enabled": True, "solver": "HIGHS"},
                    "optimizer": {"num_iterations": 2},
                },
            )
        )
        self.assertEqual(result.compute["num_workers"], 2)
        self.assertTrue(np.all(np.isfinite([float(x) for x in result.history["loss"]])))
        self.assert_within_bounds(result)

    def test_budget_constraint_binds(self):
        """Spec 7.1 test 12, through the ``monolithic`` preset *and* the gradient.

        The docstring here used to say ``MonolithicPlanningProblem`` builds no
        budget rows, so only the gradient path's projection QP could enforce one.
        That is stale: the LP now states its ``BudgetConstraintSet``
        (``MonolithicPlanningProblem.budget_constraints``), so the constraint is
        checked on both paths -- the LP is the one a reader would expect to hold.
        """
        free_lp = self.plan()
        solar_free_lp = float(self.flat(free_lp)[SOLAR_ROW])
        self.assertGreater(solar_free_lp, 1.0)

        free = self.plan(
            planning=deep_merge(GRADIENT_PLAN, {"warm_start": {"enabled": True, "solver": "HIGHS"}})
        )
        solar_free = float(self.flat(free)[SOLAR_ROW])
        self.assertGreater(solar_free, 1.0)
        cap = 0.5 * min(solar_free, solar_free_lp)

        csv = Path(self._tmp.name) / "budget.csv"
        pd.DataFrame(
            [
                {
                    "constraint_name": "max_solar",
                    "attribute": "nominal_capacity",
                    "device_name": "z1 solar",
                    "multiplier": 1.0,
                    "rhs_value": np.nan,
                    "sense": "",
                },
                {
                    "constraint_name": "max_solar",
                    "attribute": "rhs",
                    "device_name": "",
                    "multiplier": np.nan,
                    "rhs_value": cap,
                    "sense": "le",
                },
            ]
        ).to_csv(csv, index=False)

        bounded_lp = self.plan(planning={"budget_constraints": str(csv)})
        self.assertLess(float(self.flat(bounded_lp)[SOLAR_ROW]), solar_free_lp - 1e-6)
        self.assertLessEqual(float(self.flat(bounded_lp)[SOLAR_ROW]), cap + 1e-6)

        bounded = self.plan(
            planning=deep_merge(
                GRADIENT_PLAN,
                {
                    "budget_constraints": str(csv),
                    "warm_start": {"enabled": True, "solver": "HIGHS"},
                },
            )
        )
        self.assertLessEqual(float(self.flat(bounded)[SOLAR_ROW]), cap + 1e-6)

    def test_dual_ascent_outer_loop(self):
        """Spec 7.1 test 16: the ported loop runs (it used to raise NameError)."""
        free = self.plan(planning=dict(GRADIENT_PLAN))
        target = 0.5 * free.objective["emissions_tonnes_raw"]

        result = self.plan(
            planning=deep_merge(
                GRADIENT_PLAN,
                {
                    "optimizer": {"num_iterations": 2},
                    "emissions": {
                        "mode": "dual_ascent",
                        "cap_basis": "horizon",
                        "dual_ascent": {
                            "target": target,
                            "initial_weight": 0.0,
                            "dual_step_size": 1.0,
                            "max_weight": 1000.0,
                            "num_outer_iterations": 2,
                            "tolerance": 0.05,
                        },
                    },
                },
            )
        )
        dual = result.emissions["dual_ascent"]
        self.assertEqual(dual["num_outer_iterations_completed"], 2)
        lam = dual["lambda_history"]
        self.assertTrue(all(b >= a - 1e-12 for a, b in itertools.pairwise(lam)), lam)
        emissions = dual["emissions_history"]
        self.assertLessEqual(emissions[-1], emissions[0] + 1e-6)
        self.assertEqual(dual["target_horizon"], target)

        # The multiplier must stay inside its range: a lambda pinned at
        # `max_weight` after one update means the outer loop learned nothing,
        # which is what happened while the inner loop could not move capacity.
        self.assertLess(
            dual["final_lambda"],
            1000.0,
            f"the dual saturated at max_weight on iteration 1: {lam}",
        )
        # The carbon payment reported for the design is the applied multiplier
        # times its emissions, and it is netted out of the objective.
        self.assertAlmostEqual(
            result.objective["carbon_payment_raw"],
            dual["applied_lambda"] * result.objective["emissions_tonnes_raw"],
            places=6,
        )
        self.assertAlmostEqual(
            result.objective["raw"],
            result.objective["capex_raw"] + result.objective["opex_raw"],
            places=6,
        )


# ===========================================================================
# ADMM (spec 7.1 tests 17-18)
# ===========================================================================


class TestAdmm(PlanningFixtureMixin, unittest.TestCase):
    def test_admm_method_smoke(self):
        """Spec 7.1 test 17: cpu / float32, one 24 h block, one gradient step.

        Used to be skipped (``ADMM_BLOCKER``): the sampler passed
        ``budget_constraints=`` to ``PlanningProblemADMM``, which takes no such
        argument, and handed a torch problem numpy bounds.  Both are fixed in
        ``experiments/ra/planning/sampler.py``.
        """
        result = self.plan(
            selection={"strategy": "all", "block_size": 24},
            planning={
                "method": "admm",
                "admm": {
                    "machine": "cpu",
                    "dtype": "float32",
                    "solver_kwargs": {"num_iterations": 200, "rho_power": 1.0, "verbose": 0},
                },
                "optimizer": {"num_iterations": 1, "batch_size": 0},
            },
        )
        self.assert_within_bounds(result)

    def test_admm_layer_factory_and_torchify(self):
        """The two hooks ``AdmmGradientMethod`` overrides, in isolation."""
        import torch

        from experiments.ra.planning import parameters as parameters_mod
        from zap.admm import ADMMLayer

        cfg = plan_config(
            self.dataset,
            planning={
                "method": "admm",
                "admm": {"machine": "cpu", "dtype": "float32"},
            },
        )
        method = planning.make_method(cfg)
        method.network = self.loaded.network
        method.parameter_names = parameters_mod.setup_parameter_names(self.loaded.devices)

        devices = method.prepare_devices(list(self.loaded.devices))
        self.assertTrue(
            any(torch.is_tensor(v) for v in devices[0].__dict__.values()),
            "prepare_devices did not torchify the device arrays",
        )

        factory = method.layer_kwargs()["layer_factory"]
        layer = factory(devices, N_HOURS)
        self.assertIsInstance(layer, ADMMLayer)
        self.assertEqual(layer.time_horizon, N_HOURS)
        self.assertEqual(layer.solver.machine, "cpu")
        self.assertIs(layer.solver.dtype, torch.float32)

    def test_admm_mps_float64_rejected(self):
        """Spec 7.1 test 18: torch MPS has no float64."""
        cfg = plan_config(
            self.dataset,
            planning={"method": "admm", "admm": {"machine": "mps", "dtype": "float64"}},
        )
        with self.assertRaises(ConfigError) as ctx:
            planning.make_method(cfg)
        self.assertIn("float64", str(ctx.exception))

    def test_admm_cuda_unavailable_rejected(self):
        """Spec 7.1 test 18: a cuda config on a machine without cuda is an error."""
        import torch

        if torch.cuda.is_available():  # pragma: no cover - not this laptop
            self.skipTest("torch.cuda.is_available() is True on this machine")
        cfg = plan_config(
            self.dataset,
            planning={"method": "admm", "admm": {"machine": "cuda", "dtype": "float32"}},
        )
        with self.assertRaises(ConfigError) as ctx:
            planning.make_method(cfg)
        self.assertIn("cuda", str(ctx.exception))

    def test_admm_unknown_machine_rejected(self):
        cfg = plan_config(
            self.dataset,
            planning={"method": "admm", "admm": {"machine": "tpu", "dtype": "float32"}},
        )
        with self.assertRaises(ConfigError):
            planning.make_method(cfg)


# ===========================================================================
# ADMM warm starts (warm-start spec 2026-09-09, tests 7h)
# ===========================================================================


WARM_START_SPEC = "warm-start spec 2026-09-09"


def _admm_layer_feature(name: str) -> bool:
    """Whether the installed ``ADMMLayer`` carries a warm-start feature (task W1)."""
    import inspect

    from zap.admm import ADMMLayer

    if name in inspect.signature(ADMMLayer.__init__).parameters:
        return True
    # `warm_start_stats` is an instance attribute, so look for it in the source.
    return name in (inspect.getsource(ADMMLayer) or "")


class TestAdmmWarmStart(PlanningFixtureMixin, unittest.TestCase):
    """The `planning.admm.warm_start` keys, end to end (spec sections 2.6, 6)."""

    #: 2 blocks x 24 h over the 48-hour fixture, float64 so the residuals mean
    #: something, and a `minimum_iterations` floor high enough that a warm
    #: restart cannot converge on a stale dual residual (spec section 3.4).
    SOLVER_KWARGS: typing.ClassVar[dict] = {
        "num_iterations": 4000,
        "rho_power": 1.0,
        "minimum_iterations": 100,
        "verbose": 0,
    }

    #: A deliberately tiny step: the second forward pass then sees almost the
    #: parameters the carried state was produced at, which is what isolates
    #: "the state was reused" from "the state was stale".  A full-size step on
    #: this fixture moves capacities by more than the fleet (clip 5e3 MW on a
    #: ~700 MW system) and a warm start can then cost *more* than a cold one --
    #: risk R3 of the spec, and the reason the run card reports both means.
    STEP_SIZE = 1.0e-9

    def admm_plan(self, *, warm_start=True, reset_every=None, num_iterations=2):
        return self.plan(
            selection={"strategy": "all", "block_size": 24},
            planning={
                "method": "admm",
                "admm": {
                    "machine": "cpu",
                    "dtype": "float64",
                    "warm_start": warm_start,
                    "warm_start_reset_every": reset_every,
                    "solver_kwargs": dict(self.SOLVER_KWARGS),
                },
                "optimizer": {
                    "num_iterations": num_iterations,
                    "batch_size": 0,
                    "step_size": self.STEP_SIZE,
                },
            },
        )

    # -- 7h1 -------------------------------------------------------------

    def test_admm_warm_start_config_reaches_the_layer(self):
        """The two config keys arrive on every block's ``ADMMLayer``."""
        from experiments.ra.planning import parameters as parameters_mod

        if not _admm_layer_feature("warm_start_reset_every"):
            self.skipTest(f"ADMMLayer has no `warm_start_reset_every` yet ({WARM_START_SPEC} 2.3)")

        for warm_start, reset_every in ((False, 3), (True, None)):
            with self.subTest(warm_start=warm_start, reset_every=reset_every):
                cfg = plan_config(
                    self.dataset,
                    planning={
                        "method": "admm",
                        "admm": {
                            "machine": "cpu",
                            "dtype": "float32",
                            "warm_start": warm_start,
                            "warm_start_reset_every": reset_every,
                        },
                    },
                )
                method = planning.make_method(cfg)
                method.network = self.loaded.network
                method.parameter_names = parameters_mod.setup_parameter_names(self.loaded.devices)
                devices = method.prepare_devices(list(self.loaded.devices))
                layer = method.layer_kwargs()["layer_factory"](devices, 24)
                self.assertIs(layer.warm_start, warm_start)
                self.assertEqual(layer.warm_start_reset_every, reset_every)

    def test_admm_warm_start_defaults_to_true(self):
        """The default is on: today's behaviour, now explicit (spec section 0)."""
        from experiments.ra.planning import base as planning_base

        admm = planning_base.PLANNING_DEFAULTS["admm"]
        self.assertIs(admm["warm_start"], True)
        self.assertIsNone(admm["warm_start_reset_every"])

    # -- 7h2 -------------------------------------------------------------

    def test_admm_warm_start_reset_every_zero_rejected(self):
        cfg = plan_config(
            self.dataset,
            planning={
                "method": "admm",
                "admm": {"machine": "cpu", "dtype": "float32", "warm_start_reset_every": 0},
            },
        )
        with self.assertRaises(ConfigError) as ctx:
            planning.make_method(cfg)
        self.assertIn("warm_start_reset_every", str(ctx.exception))

    def test_low_minimum_iterations_warns(self):
        """Spec section 3.4: the only defence against a stale dual residual."""
        from experiments.ra.planning.methods import admm as admm_method

        with self.assertWarns(UserWarning) as ctx:
            admm_method.validate_warm_start(
                {"warm_start": True, "solver_kwargs": {"minimum_iterations": 10}}
            )
        self.assertIn("minimum_iterations", str(ctx.warning))

        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            admm_method.validate_warm_start(
                {"warm_start": True, "solver_kwargs": {"minimum_iterations": 100}}
            )
            admm_method.validate_warm_start(
                {"warm_start": False, "solver_kwargs": {"minimum_iterations": 1}}
            )

    # -- 7h3 -------------------------------------------------------------

    def test_admm_method_reports_warm_start(self):
        """Two planner passes over two blocks reuse each block's ADMM state.

        `optimizer.num_iterations: 2` with `init_full_loss` gives at least two
        forward passes per block, so every block contributes one cold pass (its
        first, there is no state yet) and at least one warm pass, and -- at the
        near-zero step size of `STEP_SIZE` -- the warm passes stop at the
        `minimum_iterations` floor because they start from the previous pass's
        converged iterate.
        """
        if not _admm_layer_feature("warm_start_stats"):
            self.skipTest(f"ADMMLayer records no `warm_start_stats` yet ({WARM_START_SPEC} 2.3)")

        result = self.admm_plan(warm_start=True)
        summary = result.compute["admm_warm_start"]

        self.assertIs(summary["enabled"], True)
        self.assertIsNone(summary["reset_every"])
        self.assertGreaterEqual(summary["forward_passes"], 4)  # 2 blocks x >= 2 passes
        self.assertGreaterEqual(summary["cold_passes"], 2)  # one per block
        self.assertGreaterEqual(summary["warm_passes"], 1)
        self.assertEqual(summary["refusal_reasons"], {})
        self.assertLess(
            summary["mean_iterations_warm"],
            summary["mean_iterations_cold"],
            "a warm-started pass should need fewer ADMM iterations than a cold one",
        )
        self.assert_within_bounds(result)

    def test_admm_warm_start_disabled_reports_all_cold(self):
        """`warm_start: false` restores cold behaviour, and says so on the card."""
        if not _admm_layer_feature("warm_start_stats"):
            self.skipTest(f"ADMMLayer records no `warm_start_stats` yet ({WARM_START_SPEC} 2.3)")

        result = self.admm_plan(warm_start=False)
        summary = result.compute["admm_warm_start"]

        self.assertIs(summary["enabled"], False)
        self.assertEqual(summary["warm_passes"], 0)
        self.assertEqual(summary["cold_passes"], summary["forward_passes"])
        self.assertIsNone(summary["mean_iterations_warm"])
        self.assertIsNotNone(summary["mean_iterations_cold"])

    def test_warm_start_summary_present_without_layer_support(self):
        """The run-card block exists even when no layer recorded anything."""
        from experiments.ra.planning.methods import admm as admm_method

        cfg = plan_config(
            self.dataset,
            planning={
                "method": "admm",
                "admm": {"machine": "cpu", "dtype": "float32", "warm_start_reset_every": 2},
            },
        )
        method = planning.make_method(cfg)
        ctx = types.SimpleNamespace(problem=types.SimpleNamespace(subproblems=[]))
        summary = admm_method.AdmmGradientMethod._warm_start_summary(method, ctx)
        self.assertEqual(summary["forward_passes"], 0)
        self.assertEqual(summary["warm_passes"], 0)
        self.assertEqual(summary["reset_every"], 2)
        self.assertIsNone(summary["mean_iterations_warm"])


# ===========================================================================
# Registry and solver validation (spec 7.1 test 19)
# ===========================================================================


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestValidation(unittest.TestCase):
    def test_unknown_method_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            planning.make_method({"planning": {"method": "quantum_annealing"}})
        self.assertIn("quantum_annealing", str(ctx.exception))

    def test_unknown_solver_rejected(self):
        import cvxpy as cp

        with self.assertRaises(ConfigError) as ctx:
            planning.require_solver("MOSEK")
        message = str(ctx.exception)
        self.assertIn("MOSEK", message)
        for solver in cp.installed_solvers():
            self.assertIn(solver, message)

    def test_registered_methods(self):
        """D-W2: three classes, five presets, and the presets share a class."""
        planning.make_method({"planning": {"method": "monolithic"}})
        self.assertEqual({"single_level", "gradient", "admm"}, set(planning.METHODS))
        for preset in ("monolithic", "stochastic", "relaxed"):
            self.assertEqual(planning.METHOD_PRESETS[preset], "single_level")
        self.assertEqual(planning.METHODS["single_level"].__name__, "SingleLevelMethod")
        self.assertEqual(planning.METHODS["gradient"].__name__, "GradientMethod")
        self.assertEqual(planning.METHODS["admm"].__name__, "AdmmGradientMethod")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
