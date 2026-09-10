"""Reporting / guardrail fixes S1-S5 (2026-09-09).

S1  the minibatch guard in ``AbstractPlanningProblem.solve`` compared
    ``batch_size`` against ``time_horizon`` (hours) instead of
    ``num_subproblems`` (blocks), so any ``batch_size`` above the block length
    silently became full-batch gradient descent.
S2  ``block_metrics`` splits generation cost by the sign of the marginal cost.
S3  the same split reaches ``design.json`` as ``opex_gross_raw`` /
    ``opex_credit_raw``.
S4  a design says whether its objective is a full-window value or an in-sample
    estimate from a sampled block set.
S5  ``best_sampled`` / ``best_rolling`` are refused under a minibatch.

Everything runs on tiny horizons (6-48 h, 1-4 blocks) per the project's unit
test rule; no dataset outside the synthetic tiny fixture is touched.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

from experiments.ra import blocks as blocks_mod
from experiments.ra import config as config_mod
from experiments.ra import metrics
from experiments.ra.config import ConfigError
from experiments.ra.planning import base as planning_base
from experiments.ra.planning import objectives
from zap.devices import Generator, Load
from zap.devices.transporter import DirectedLine
from zap.importers.wy_store import HourWindow, LoadOptions, load_system
from zap.network import DispatchOutcome, PowerNetwork
from zap.tests.test_ra_planning_methods import (
    EXTENDABLE_GENERATORS,
    VOLL,
    deep_merge,
    write_planning_dataset,
)

# ===========================================================================
# Shared config helper
# ===========================================================================


def tiny_plan_config(dataset: Path, n_hours: int, **overrides) -> dict:
    cfg = {
        "mode": "plan",
        "dataset": {
            "dir": str(dataset),
            "years": [2020],
            "window": {"start": 0, "stop": n_hours},
        },
        "heuristics": {"name": "none", "ucap_derate": False, "outage_draws": []},
        "selection": {
            "strategy": "all",
            "block_size": None,
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
        "system": {"storage_soc_mode": "cyclic_free"},
    }
    for key, value in overrides.items():
        cfg[key] = deep_merge(cfg[key], value) if isinstance(cfg.get(key), dict) else value
    return cfg


# ===========================================================================
# S1 -- the minibatch guard counts blocks, not hours
# ===========================================================================


class TestMinibatchGuardCountsBlocks(unittest.TestCase):
    """``batch_size`` is a number of subproblems; the guard used to test hours."""

    _tmp: tempfile.TemporaryDirectory
    dataset: Path

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        # 8 hours so a 2 h block size gives four subproblems; the six-hour
        # window below is a slice of the same store.
        cls.dataset = write_planning_dataset(Path(cls._tmp.name) / "tiny", n_hours=8)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _problem(self, n_hours: int):
        loaded = load_system(
            self.dataset,
            LoadOptions(years=(2020,), window=HourWindow(0, n_hours), voll=VOLL),
        )
        cfg = tiny_plan_config(
            self.dataset,
            n_hours,
            selection={"strategy": "all", "block_size": 2},
            planning={"method": "gradient", "optimizer": {"num_iterations": 1}},
        )
        ctx = planning_base.make_method(cfg).build(loaded)
        return ctx.problem

    def _batch_size(self, problem, batch_size):
        problem.solve(
            num_iterations=1,
            batch_size=batch_size,
            batch_strategy="fixed",
            init_full_loss=False,
            verbosity=0,
        )
        return len(problem.batch)

    def test_three_subproblems_two_hour_horizon(self):
        """Spec S1: 3 blocks of 2 h -- 3 stays 3, 5 collapses to 3, 2 stays 2."""
        problem = self._problem(6)
        self.assertEqual(problem.num_subproblems, 3)
        self.assertEqual(problem.time_horizon, 2)

        self.assertEqual(self._batch_size(problem, 3), 3)
        self.assertEqual(self._batch_size(problem, 5), 3)
        self.assertEqual(self._batch_size(problem, 2), 2)

    def test_batch_size_above_block_length_is_not_collapsed(self):
        """The regression itself: 4 blocks of 2 h keep a batch of 3.

        With the old guard (``batch_size > self.time_horizon``) 3 > 2 collapsed
        the run to full-batch gradient descent over all four blocks.  The
        three-subproblem case above cannot see this, because there the collapse
        target (``num_subproblems`` = 3) equals the requested batch size.
        """
        problem = self._problem(8)
        self.assertEqual(problem.num_subproblems, 4)
        self.assertEqual(problem.time_horizon, 2)

        self.assertEqual(self._batch_size(problem, 3), 3)
        self.assertEqual(self._batch_size(problem, 4), 4)
        self.assertEqual(self._batch_size(problem, 9), 4)
        self.assertEqual(self._batch_size(problem, 0), 4)


# ===========================================================================
# S2 -- gross / credit split of the block dispatch cost
# ===========================================================================


def signed_cost_system(n_hours: int = 4):
    """One bus, two generators at +10 and -5 $/MWh, and a load that takes both.

    Nothing is solved: the dispatch is written by hand so every metric has a
    value that can be checked by arithmetic.
    """
    net = PowerNetwork(num_nodes=1)
    gen = Generator(
        num_nodes=1,
        name=np.array(["ccgt", "ptc_wind"]),
        terminal=np.array([0, 0]),
        nominal_capacity=np.array([100.0, 100.0]),
        dynamic_capacity=np.ones((2, n_hours)),
        # +10 $/MWh of gross cost, -5 $/MWh of credit (a PTC-priced row).
        linear_cost=np.array([[10.0], [-5.0]]),
        emission_rates=np.array([[0.4], [0.0]]),
    )
    load = Load(
        num_nodes=1,
        name=np.array(["l1"]),
        terminal=np.array([0]),
        load=np.full((1, n_hours), 90.0),
        linear_cost=np.array([[1000.0]]),
    )
    devices = [gen, load]
    power = [
        [np.vstack([np.full((1, n_hours), 30.0), np.full((1, n_hours), 60.0)])],
        [np.full((1, n_hours), -90.0)],
    ]
    outcome = DispatchOutcome(
        phase_duals=None,
        local_equality_duals=None,
        local_inequality_duals=None,
        local_variables=[None, None],
        power=power,
        angle=[[None], [None]],
        prices=np.full((1, n_hours), 10.0),
        global_angle=None,
    )
    index = SimpleNamespace(
        carrier={"Generator": np.array(["CCGT", "onwind"])},
        vre_mask=np.array([False, True]),
    )
    loaded = SimpleNamespace(network=net, index=index, meta={})
    return loaded, devices, outcome


class TestGenerationCostSplit(unittest.TestCase):
    N_HOURS = 4

    def test_split_signs_and_identity(self):
        loaded, devices, outcome = signed_cost_system(self.N_HOURS)
        block = blocks_mod.Block(index=0, year=2020, start=0, stop=self.N_HOURS)
        m = metrics.block_metrics(loaded, devices, outcome, block)

        gross = 10.0 * 30.0 * self.N_HOURS
        credit = -5.0 * 60.0 * self.N_HOURS

        self.assertAlmostEqual(m["generation_cost_gross"], gross)
        self.assertAlmostEqual(m["generation_credit"], credit)
        self.assertAlmostEqual(m["generation_cost"], gross + credit)
        # Signs: the credit is a revenue, the gross cost is a cost, and the net
        # is much smaller than either -- the whole reason for reporting both.
        self.assertLess(m["generation_credit"], 0.0)
        self.assertGreater(m["generation_cost_gross"], 0.0)
        self.assertLess(abs(m["generation_cost"]), m["generation_cost_gross"])

    def test_split_is_identity_without_negative_costs(self):
        loaded, devices, outcome = signed_cost_system(self.N_HOURS)
        devices[0].linear_cost = np.array([[10.0], [5.0]])
        block = blocks_mod.Block(index=0, year=2020, start=0, stop=self.N_HOURS)
        m = metrics.block_metrics(loaded, devices, outcome, block)

        self.assertEqual(m["generation_credit"], 0.0)
        self.assertAlmostEqual(m["generation_cost_gross"], m["generation_cost"])

    def test_split_is_registered_everywhere(self):
        for key in ("generation_cost_gross", "generation_credit"):
            self.assertIn(key, metrics.ADDITIVE_METRICS)
            self.assertIn(key, metrics._MONEY_METRICS)

    def test_split_scales_with_cost_and_power_units(self):
        """`_to_physical_units` undoes the solver scaling for both halves."""
        loaded, devices, outcome = signed_cost_system(self.N_HOURS)
        block = blocks_mod.Block(index=0, year=2020, start=0, stop=self.N_HOURS)
        plain = metrics.block_metrics(loaded, devices, outcome, block)

        loaded_scaled, devices_scaled, outcome_scaled = signed_cost_system(self.N_HOURS)
        loaded_scaled.meta = {"power_unit": 10.0, "cost_unit": 2.0}
        for device in devices_scaled:
            device.scale_costs(2.0)
            device.scale_power(10.0)
        outcome_scaled.power = [
            [p / 10.0 for p in terminals] for terminals in outcome_scaled.power
        ]
        scaled = metrics.block_metrics(loaded_scaled, devices_scaled, outcome_scaled, block)

        for key in ("generation_cost", "generation_cost_gross", "generation_credit"):
            self.assertAlmostEqual(scaled[key], plain[key], places=6, msg=key)

    def test_deviation_vs_reference_reports_both_halves(self):
        rows = []
        for start, stop in ((0, 2), (2, 4)):
            rows.append(
                {
                    "method": "lp",
                    "block_size": "2",
                    "start": start,
                    "stop": stop,
                    "status": "ok",
                    "generation_cost": 60.0,
                    "generation_cost_gross": 100.0,
                    "generation_credit": -40.0,
                }
            )
        rows.append(
            {
                "method": "lp",
                "block_size": "reference",
                "start": 0,
                "stop": 4,
                "status": "ok",
                "generation_cost": 100.0,
                "generation_cost_gross": 180.0,
                "generation_credit": -80.0,
            }
        )
        out = metrics.deviation_vs_reference(pd.DataFrame(rows))
        reported = dict(zip(out["metric"], out["value"]))
        reference = dict(zip(out["metric"], out["reference"]))

        self.assertIn("generation_cost_gross", reported)
        self.assertIn("generation_credit", reported)
        self.assertAlmostEqual(reported["generation_cost_gross"], 200.0)
        self.assertAlmostEqual(reported["generation_credit"], -80.0)
        self.assertAlmostEqual(reference["generation_cost_gross"], 180.0)


# ===========================================================================
# S3 / S4 -- the split and the basis flags reach design.json
# ===========================================================================

N_HOURS = 48
#: The row given a negative marginal cost, standing in for a PTC-priced wind row.
CREDIT_ROW = "z2 onwind"
CREDIT_COST = -12.76


class PlanningFixture(unittest.TestCase):
    """The 48 h planning fixture with one negative-marginal-cost generator."""

    _tmp: tempfile.TemporaryDirectory
    dataset: Path
    loaded: object

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dataset = write_planning_dataset(Path(cls._tmp.name) / "tiny", n_hours=N_HOURS)
        gens = pd.read_csv(cls.dataset / "static" / "generators.csv", index_col=0)
        assert CREDIT_ROW in gens.index and CREDIT_ROW in EXTENDABLE_GENERATORS
        gens.loc[CREDIT_ROW, "marginal_cost"] = CREDIT_COST
        gens.to_csv(cls.dataset / "static" / "generators.csv")
        cls.loaded = load_system(
            cls.dataset,
            LoadOptions(years=(2020,), window=HourWindow(0, N_HOURS), voll=VOLL),
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def plan(self, **overrides):
        cfg = tiny_plan_config(self.dataset, N_HOURS, **overrides)
        return planning_base.plan(self.loaded, cfg)


class TestOpexSplitInDesign(PlanningFixture):
    def _assert_split(self, result):
        obj = result.objective
        self.assertIsNotNone(obj["opex_gross_raw"])
        self.assertIsNotNone(obj["opex_credit_raw"])
        self.assertAlmostEqual(
            obj["opex_gross_raw"] + obj["opex_credit_raw"],
            obj["opex_raw"],
            delta=abs(obj["opex_raw"]) * 1e-9 + 1e-6,
        )
        # The negative-cost row runs, so the credit is strictly negative and the
        # gross cost strictly exceeds the net.
        self.assertLess(obj["opex_credit_raw"], 0.0)
        self.assertGreater(obj["opex_gross_raw"], obj["opex_raw"])
        return obj

    def test_single_level_opex_splits(self):
        result = self.plan()
        obj = self._assert_split(result)
        af = result.annualization["annualization_factor"]
        self.assertAlmostEqual(obj["opex_gross_annual"], obj["opex_gross_raw"] * af)
        self.assertAlmostEqual(obj["opex_credit_annual"], obj["opex_credit_raw"] * af)

    def test_gradient_opex_splits(self):
        result = self.plan(
            selection={"strategy": "all", "block_size": 24},
            planning={
                "method": "gradient",
                "warm_start": {"enabled": False},
                "optimizer": {"num_iterations": 2, "batch_size": 0},
            },
        )
        self._assert_split(result)

    def test_admm_opex_splits(self):
        """Verifier V-S3-1: the ADMM path torchifies the block devices.

        ``negative_cost_credit`` re-evaluates ``operation_cost`` in numpy, so it
        has to numpy-ify the device as well as the state; before the fix this
        raised ``TypeError`` inside the caller's ``except`` and every ADMM
        design reported ``opex_gross_raw = opex_credit_raw = None``.
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
                "warm_start": {"enabled": False},
                "optimizer": {"num_iterations": 1, "batch_size": 0},
            },
        )
        obj = result.objective
        self.assertIsNotNone(obj["opex_credit_raw"], "ADMM design lost the opex split")
        self.assertTrue(np.isfinite(obj["opex_credit_raw"]))
        self.assertTrue(np.isfinite(obj["opex_gross_raw"]))
        self.assertLess(obj["opex_credit_raw"], 0.0)
        self.assertAlmostEqual(
            obj["opex_gross_raw"] + obj["opex_credit_raw"],
            obj["opex_raw"],
            delta=abs(obj["opex_raw"]) * 1e-6 + 1e-3,
        )

    def test_credit_is_generator_only_and_export_revenue_is_its_own_line(self):
        """Verifier V-S3-b: `opex_credit_raw` means what `generation_credit` means.

        With `export_mode: sink` the export links carry a large negative cost.
        That is export revenue, not a production credit, so it stays out of
        `opex_credit_raw` and is reported separately -- as a component *of*
        `opex_gross_raw`, which keeps the two-term identity exact.
        """
        result = self.plan(system={"export_mode": "sink"})
        obj = result.objective
        self.assertAlmostEqual(
            obj["opex_gross_raw"] + obj["opex_credit_raw"],
            obj["opex_raw"],
            delta=abs(obj["opex_raw"]) * 1e-9 + 1e-6,
        )
        self.assertIn("opex_export_revenue_raw", obj)
        self.assertIsNotNone(obj["opex_export_revenue_raw"])
        self.assertLessEqual(obj["opex_export_revenue_raw"], 0.0)
        self.assertLess(obj["opex_credit_raw"], 0.0)

    def test_planning_metrics_surface_the_split(self):
        row = metrics.planning_metrics(self.plan())
        for key in (
            "opex_gross_raw",
            "opex_credit_raw",
            "opex_gross_annual",
            "opex_credit_annual",
        ):
            self.assertIn(key, row)
            self.assertIsNotNone(row[key])
        self.assertAlmostEqual(
            row["opex_gross_raw"] + row["opex_credit_raw"],
            row["opex_raw"],
            delta=abs(row["opex_raw"]) * 1e-9 + 1e-6,
        )


class TestCreditPartition(unittest.TestCase):
    """`credit` and `export_revenue` partition the all-device negative-price cost."""

    N_HOURS = 3

    def _devices(self):
        gen = Generator(
            num_nodes=2,
            name=np.array(["ptc_wind"]),
            terminal=np.array([0]),
            nominal_capacity=np.array([100.0]),
            dynamic_capacity=np.ones((1, self.N_HOURS)),
            linear_cost=np.array([[-5.0]]),
        )
        line = DirectedLine(
            num_nodes=2,
            name=np.array(["exp"]),
            source_terminal=np.array([0]),
            sink_terminal=np.array([1]),
            nominal_capacity=np.array([50.0]),
            min_power=np.array([[0.0]]),
            max_power=np.array([[1.0]]),
            linear_cost=np.array([[-178.0]]),
            efficiency=np.array([[1.0]]),
        )
        power = [
            [np.full((1, self.N_HOURS), 60.0)],
            [np.full((1, self.N_HOURS), -20.0), np.full((1, self.N_HOURS), 20.0)],
        ]
        angle = [[None], [None, None]]
        local = [None, None]
        return [gen, line], power, angle, local

    def test_generator_and_export_parts_sum_to_the_whole(self):
        devices, power, angle, local = self._devices()
        both = objectives.dispatch_cost_credit(devices, power, angle, local)
        credit = objectives.dispatch_cost_credit(
            devices, power, angle, local, classes=objectives.CREDIT_CLASSES
        )
        export = objectives.dispatch_cost_credit(
            devices, power, angle, local, classes=objectives.EXPORT_CLASSES
        )
        self.assertAlmostEqual(credit, -5.0 * 60.0 * self.N_HOURS)
        self.assertAlmostEqual(export, -178.0 * 20.0 * self.N_HOURS)
        self.assertAlmostEqual(both, credit + export)
        # The point of the split: the generator credit excludes export revenue.
        self.assertNotAlmostEqual(credit, both)


class _FakeLayer:
    def __init__(self, devices):
        self.devices = devices


class _FakeSub:
    """The three attributes `problem_cost_credit` reads off a subproblem."""

    def __init__(self, devices, state, snapshot_weight=1.0):
        self.layer = _FakeLayer(devices)
        self.state = state
        self.snapshot_weight = snapshot_weight


class _FakeProblem:
    def __init__(self, subproblems, weights):
        self.subproblems = subproblems
        self.weights = weights


class TestCreditWeighting(unittest.TestCase):
    """`problem_cost_credit` weights exactly as `StochasticPlanningProblem.op_cost`.

    `problem_cvx.py` applies `snapshot_weight` to `sub.cost` only, never to
    `sub.op_cost`, and `op_cost` is `sum_i w_i * sub_i.get_op_cost()`.  A credit
    weighted by `w_i * snapshot_weight_i` would therefore break
    `opex == gross + credit` the moment snapshot weights stopped being 1.0
    (verifier V-S3-a).
    """

    N_HOURS = 2

    def _problem(self, weights, snapshot_weights):
        subs = []
        for snapshot_weight in snapshot_weights:
            gen = Generator(
                num_nodes=1,
                name=np.array(["ptc_wind"]),
                terminal=np.array([0]),
                nominal_capacity=np.array([100.0]),
                dynamic_capacity=np.ones((1, self.N_HOURS)),
                linear_cost=np.array([[-5.0]]),
            )
            state = SimpleNamespace(
                power=[[np.full((1, self.N_HOURS), 60.0)]],
                angle=[[None]],
                local_variables=[None],
            )
            subs.append(_FakeSub([gen], state, snapshot_weight=snapshot_weight))
        return _FakeProblem(subs, weights)

    def test_credit_ignores_snapshot_weight(self):
        per_block = -5.0 * 60.0 * self.N_HOURS
        unit = objectives.problem_cost_credit(self._problem([1.0, 1.0], [1.0, 1.0]))
        self.assertAlmostEqual(unit, 2.0 * per_block)

        # `op_cost` does not scale with snapshot_weight, so neither may `credit`.
        weighted = objectives.problem_cost_credit(self._problem([1.0, 1.0], [3.0, 7.0]))
        self.assertAlmostEqual(weighted, unit)

    def test_credit_scales_with_the_subproblem_weight(self):
        per_block = -5.0 * 60.0 * self.N_HOURS
        scaled = objectives.problem_cost_credit(self._problem([2.0, 0.5], [1.0, 1.0]))
        self.assertAlmostEqual(scaled, 2.5 * per_block)

    def test_missing_state_reports_no_credit(self):
        problem = self._problem([1.0, 1.0], [1.0, 1.0])
        problem.subproblems[1].state = None
        self.assertIsNone(objectives.problem_cost_credit(problem))


class TestObjectiveBasis(PlanningFixture):
    def test_full_window_objective_is_not_in_sample(self):
        result = self.plan(selection={"strategy": "all", "block_size": None})
        obj = result.objective
        self.assertEqual(result.annualization["coverage"], 1.0)
        self.assertEqual(obj["basis"], "full_window")
        self.assertIs(obj["in_sample"], False)
        # `block_size: null` means "one block covering the window", and the
        # record says so in hours instead of repeating the null.
        self.assertEqual(obj["operational_model"]["block_size"], N_HOURS)
        self.assertEqual(obj["operational_model"]["n_blocks"], 1)
        self.assertEqual(obj["operational_model"]["storage_soc_mode"], "cyclic_free")

    def test_sampled_objective_is_flagged_in_sample(self):
        result = self.plan(
            selection={"strategy": "random", "block_size": 24, "num_blocks": 1, "seed": 0}
        )
        obj = result.objective
        self.assertEqual(result.annualization["coverage"], 0.5)
        self.assertEqual(obj["basis"], "sampled")
        self.assertIs(obj["in_sample"], True)
        # An explicit block size is reported as given, which is how a reader
        # tells a 24 h sample apart from a whole-window solve.
        self.assertEqual(obj["operational_model"]["block_size"], 24)
        self.assertEqual(obj["operational_model"]["n_blocks"], 1)

    def test_metrics_and_design_record_carry_the_flags(self):
        result = self.plan(
            selection={"strategy": "random", "block_size": 24, "num_blocks": 1, "seed": 0}
        )
        row = metrics.planning_metrics(result)
        self.assertEqual(row["objective_basis"], "sampled")
        self.assertIs(row["objective_in_sample"], True)
        self.assertEqual(row["objective_block_size"], 24)
        self.assertEqual(row["objective_n_blocks"], 1)

        record = result.to_record()
        self.assertEqual(record["objective"]["basis"], "sampled")
        self.assertEqual(record["objective"]["operational_model"]["block_size"], 24)


# ===========================================================================
# S5 -- best_sampled / best_rolling are refused under a minibatch
# ===========================================================================


class TestDesignSelectionGuard(unittest.TestCase):
    @staticmethod
    def _cfg(**optimizer) -> dict:
        cfg = config_mod.base_config()
        cfg["name"] = "guard_test"
        cfg["mode"] = "plan"
        cfg["planning"]["method"] = "gradient"
        cfg["planning"]["optimizer"].update(optimizer)
        return cfg

    def test_minibatch_rejects_best_sampled(self):
        for rule in ("best_sampled", "best_rolling"):
            with self.subTest(rule=rule):
                with self.assertRaises(ConfigError) as caught:
                    config_mod.validate(self._cfg(design_selection=rule, batch_size=4))
                message = str(caught.exception)
                self.assertIn("best_checkpointed", message)
                self.assertIn("1 B$", message)

    def test_full_batch_allows_best_sampled(self):
        cfg = config_mod.validate(self._cfg(design_selection="best_sampled", batch_size=0))
        self.assertEqual(cfg["planning"]["optimizer"]["design_selection"], "best_sampled")

    def test_minibatch_allows_best_checkpointed(self):
        cfg = config_mod.validate(
            self._cfg(
                design_selection="best_checkpointed",
                batch_size=4,
                checkpoint_every=20,
            )
        )
        self.assertEqual(cfg["planning"]["optimizer"]["batch_size"], 4)

    def test_shipped_campaign_configs_still_validate(self):
        from experiments.ra import paths

        root = paths.config_root() / "experiments"
        shipped = sorted(root.glob("plan_z4_2020_c*.yaml"))
        # Cells 1-5, plus the step-rule reruns c4' / c5'
        # (`memory/plans/2026-09-10-step-rule-spec.md`); the assertion is on the
        # five original cells being present, not on the campaign never growing.
        names = {path.stem for path in shipped}
        for cell in (
            "plan_z4_2020_c1_lp_full",
            "plan_z4_2020_c2_lp_weeks52",
            "plan_z4_2020_c3_lp_weeks12",
            "plan_z4_2020_c4_grad_det",
            "plan_z4_2020_c5_sgd_b4",
        ):
            self.assertIn(cell, names)
        self.assertGreaterEqual(len(shipped), 5)
        for path in shipped:
            with self.subTest(config=path.name):
                cfg = config_mod.load_config(path)
                self.assertIn("name", cfg)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
