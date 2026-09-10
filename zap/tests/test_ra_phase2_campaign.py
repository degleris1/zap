"""The four phase-2 campaign fixes (`memory/plans/2026-09-09-phase2-campaign.md`).

Every test runs on the 48-hour planning fixture of ``test_ra_planning_methods``
(two 24 h blocks, or six 8 h blocks where a longer minibatch sequence is
needed), so the whole module is seconds of HiGHS/CLARABEL work.

Covered here:

1. ``planning.optimizer.max_seconds`` -- the *soft* wall-clock cap: the loop
   finishes the iteration it is in, keeps the design and the history, and says
   why it stopped;
2. ``planning.optimizer.design_selection`` -- the design is an iterate chosen
   post hoc (``best_sampled`` from the loss history, ``best_checkpointed`` from
   full-block-set forward passes taken every ``checkpoint_every`` iterations);
3. the batch size in the plan task/design id, so a deterministic and a
   stochastic gradient run over the same pool do not collide;
4. ``selection.align_blocks`` -- random block starts drawn from the block grid,
   so "12 of the 52 weeks" is literally true;

plus the seedable minibatch RNG the replicates need, and the config-surface
validation of all of it.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

from experiments.ra import config
from experiments.ra import tasks as tasks_mod
from experiments.ra.planning import history as history_mod
from experiments.ra.planning.methods.gradient import select_iterate
from zap.tests.test_ra_planning_harness import write_config
from zap.tests.test_ra_planning_methods import (
    PLANNING_AVAILABLE,
    TASK_A,
    PlanningFixtureMixin,
    deep_merge,
)

#: The gradient preset every test here starts from.  CLARABEL on the fixture,
#: like the rest of the planning tests; the campaign itself runs HiGHS.
GRADIENT = {
    "method": "gradient",
    "dispatch_solver": "CLARABEL",
    # A small learning rate and no convergence tests: these tests are about the
    # harness (the wall-clock cap, the design-selection rule, the task id), not
    # about the step rule, and a run that stops on a tolerance at iteration 1
    # exercises none of it.  `step_size` is Adam's, in MW, on a fixture whose
    # largest row is 700 MW.
    "optimizer": {
        "num_iterations": 4,
        "batch_size": 0,
        "step_size": 5.0,
        "stopping": {"tol_rel_objective": None, "tol_stationarity": None},
    },
}


# ===========================================================================
# 1. The soft wall-clock cap
# ===========================================================================


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestMaxSeconds(PlanningFixtureMixin, unittest.TestCase):
    def test_max_seconds_breaks_gracefully(self):
        """A cap of ~0 s still returns a design, a history and a reason."""
        result = self.plan(
            planning=deep_merge(
                GRADIENT, {"optimizer": {"num_iterations": 50, "max_seconds": 1e-6}}
            )
        )
        self.assertEqual(result.solver["stop_reason"], "wall_clock")
        self.assertEqual(result.solver["stopped_by"], "max_seconds")
        self.assertEqual(result.solver["max_seconds"], 1e-6)

        completed = result.solver["num_iterations_completed"]
        self.assertGreaterEqual(completed, 1)
        self.assertLess(completed, 50)
        # The history is complete for the iterations that ran: index 0 is the
        # state before the first step.
        self.assertEqual(len(result.history["loss"]), completed + 1)
        self.assertTrue(np.all(np.isfinite([float(x) for x in result.history["loss"]])))

        # A design came back, inside its bounds, with a finite objective.
        self.assertTrue(result.parameters)
        self.assert_within_bounds(result)
        self.assertTrue(np.isfinite(result.objective["raw"]))

        # ... and the iteration tables are non-empty.
        tables = history_mod.build_iteration_tables(result)
        self.assertEqual(len(tables["iterations"]), completed + 1)

    def test_no_cap_runs_every_iteration(self):
        result = self.plan(
            planning=deep_merge(GRADIENT, {"optimizer": {"num_iterations": 3}}),
        )
        self.assertEqual(result.solver["stop_reason"], "num_iterations")
        self.assertEqual(result.solver["stopped_by"], "iterations")
        self.assertIsNone(result.solver["max_seconds"])
        self.assertEqual(result.solver["num_iterations_completed"], 3)
        self.assertEqual(len(result.history["loss"]), 4)


# ===========================================================================
# 2. Design = the selected iterate
# ===========================================================================


class TestSelectIterate(unittest.TestCase):
    """``select_iterate`` on synthetic histories with a known argmin."""

    HISTORY: ClassVar[dict] = {
        "loss": [10.0, 7.0, 4.0, 9.0],
        "rolling_loss": [10.0, 8.5, 7.0, 6.5],
        "param": [
            {"generator_capacity": np.array([0.0])},
            {"generator_capacity": np.array([1.0])},
            {"generator_capacity": np.array([2.0])},
            {"generator_capacity": np.array([3.0])},
        ],
    }

    def test_best_sampled_is_the_loss_argmin(self):
        params, index, rule = select_iterate(self.HISTORY, "best_sampled")
        self.assertEqual((index, rule), (2, "best_sampled"))
        self.assertEqual(float(params["generator_capacity"][0]), 2.0)

    def test_best_rolling_uses_the_rolling_series(self):
        params, index, rule = select_iterate(self.HISTORY, "best_rolling")
        self.assertEqual((index, rule), (3, "best_rolling"))
        self.assertEqual(float(params["generator_capacity"][0]), 3.0)

    def test_final_selects_nothing(self):
        self.assertEqual(select_iterate(self.HISTORY, "final"), (None, None, "final"))

    def test_missing_param_history_falls_back_to_final(self):
        history = {k: v for k, v in self.HISTORY.items() if k != "param"}
        with self.assertLogs("experiments.ra.planning.methods.gradient", "WARNING"):
            params, index, rule = select_iterate(history, "best_sampled")
        self.assertEqual((params, index, rule), (None, None, "final"))

    def test_non_finite_losses_fall_back_to_final(self):
        history = dict(self.HISTORY, loss=[float("nan")] * 4)
        with self.assertLogs("experiments.ra.planning.methods.gradient", "WARNING"):
            self.assertEqual(select_iterate(history, "best_sampled")[2], "final")

    def test_nan_losses_are_skipped_not_selected(self):
        history = dict(self.HISTORY, loss=[10.0, float("nan"), 4.0, 9.0])
        _, index, _ = select_iterate(history, "best_sampled")
        self.assertEqual(index, 2)

    def test_unknown_rule_raises(self):
        with self.assertRaises(ValueError):
            select_iterate(self.HISTORY, "best_guess")

    def test_returned_parameters_are_a_copy(self):
        params, _, _ = select_iterate(self.HISTORY, "best_sampled")
        params["generator_capacity"][0] = -1.0
        self.assertEqual(float(self.HISTORY["param"][2]["generator_capacity"][0]), 2.0)


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestDesignSelectionEndToEnd(PlanningFixtureMixin, unittest.TestCase):
    def test_best_sampled_returns_the_argmin_iterate(self):
        """At ``batch_size: 0`` the sampled loss *is* the full objective.

        So the reported objective of the returned design must equal the loss of
        the iteration the selection named -- which is the strongest available
        statement that the design really is that iterate and that the full
        forward pass was re-run at it.
        """
        result = self.plan(
            planning=deep_merge(
                GRADIENT, {"optimizer": {"num_iterations": 4, "design_selection": "best_sampled"}}
            )
        )
        losses = [float(x) for x in result.history["loss"]]
        index = result.objective["design_iteration"]
        self.assertEqual(result.objective["design_selection"], "best_sampled")
        self.assertEqual(index, int(np.argmin(losses)))
        self.assertAlmostEqual(result.objective["raw"] / losses[index], 1.0, places=6)
        self.assertAlmostEqual(
            result.objective["best_sampled_objective_raw"], min(losses), places=6
        )
        self.assertAlmostEqual(result.objective["final_sampled_objective_raw"], losses[-1])
        self.assertIsNotNone(result.objective["best_rolling_objective_raw"])

        # The returned parameters are the iterate, not the last one.
        for param, values in result.parameters.items():
            np.testing.assert_allclose(
                np.asarray(values, dtype=float).reshape(-1),
                np.asarray(result.history["param"][index][param], dtype=float).reshape(-1),
            )
        self.assert_within_bounds(result)

    def test_final_rule_keeps_the_last_iterate(self):
        result = self.plan(
            planning=deep_merge(GRADIENT, {"optimizer": {"design_selection": "final"}})
        )
        self.assertEqual(result.objective["design_selection"], "final")
        self.assertIsNone(result.objective["design_iteration"])
        last = result.history["param"][-1]
        for param, values in result.parameters.items():
            np.testing.assert_allclose(
                np.asarray(values, dtype=float).reshape(-1),
                np.asarray(last[param], dtype=float).reshape(-1),
            )
        # The three candidate objectives are recorded whatever the rule.
        for key in (
            "final_sampled_objective_raw",
            "best_sampled_objective_raw",
            "best_rolling_objective_raw",
        ):
            self.assertIsNotNone(result.objective[key], key)

    def test_best_checkpointed_picks_the_best_full_objective(self):
        """Cell 5's rule: a minibatch run judged by full-block-set forward passes."""
        result = self.plan(
            selection={"strategy": "all", "block_size": 8},
            planning=deep_merge(
                GRADIENT,
                {
                    "optimizer": {
                        "num_iterations": 4,
                        "batch_size": 2,
                        "batch_strategy": "random",
                        "design_selection": "best_checkpointed",
                        "checkpoint_every": 2,
                    }
                },
            ),
        )
        checkpoints = result.objective["checkpoints"]
        # iterations 0, 2, 4 -- the first and the last iterate are always
        # candidates, and 4 is both a multiple of 2 and the final one (recorded
        # once).
        self.assertEqual([c["iteration"] for c in checkpoints], [0, 2, 4])
        self.assertEqual(result.solver["checkpoint_every"], 2)
        self.assertGreater(result.timing["checkpoint_s"], 0.0)

        objectives = [c["objective_raw"] for c in checkpoints]
        best = int(np.argmin(objectives))
        self.assertEqual(result.objective["design_selection"], "best_checkpointed")
        self.assertEqual(result.objective["design_iteration"], checkpoints[best]["iteration"])
        # The post-solve forward pass at the design reproduces the checkpoint.
        self.assertAlmostEqual(result.objective["raw"] / objectives[best], 1.0, places=6)
        # A minibatch loss is not the full objective, so the checkpoint must not
        # be read off the loss history.
        self.assertEqual(len(result.history["batch"][-1]), 2)

        # The checkpoints land in the column reserved for a full-horizon
        # objective, and nowhere else.
        table = history_mod.build_iteration_tables(result)["iterations"]
        filled = table.dropna(subset=["estimated_full_objective_annual"])
        self.assertEqual(sorted(filled["iteration"].tolist()), [0, 2, 4])
        af = float(result.annualization["annualization_factor"])
        self.assertAlmostEqual(
            float(filled.iloc[0]["estimated_full_objective_annual"]),
            objectives[0] * af,
            places=6,
        )

    def test_checkpoints_without_the_rule_are_recorded_not_used(self):
        result = self.plan(
            planning=deep_merge(
                GRADIENT,
                {"optimizer": {"design_selection": "final", "checkpoint_every": 2}},
            )
        )
        self.assertTrue(result.objective["checkpoints"])
        self.assertEqual(result.objective["design_selection"], "final")


# ===========================================================================
# 3. The batch size in the design id
# ===========================================================================


class TestPlanTaskId(unittest.TestCase):
    def test_batch_size_distinguishes_two_gradient_runs(self):
        common = {
            "preset": "gradient",
            "strategy": "all",
            "block_size": 168,
            "seed": 42,
            "draw": None,
        }
        deterministic = tasks_mod.make_plan_task_id(**common, batch_size=0)
        stochastic = tasks_mod.make_plan_task_id(**common, batch_size=4)
        self.assertEqual(deterministic, "plan-gradient-all-b168-s42")
        self.assertEqual(stochastic, "plan-gradient-all-b168-s42-B4")
        self.assertNotEqual(deterministic, stochastic)

    def test_zero_batch_size_leaves_existing_ids_unchanged(self):
        self.assertEqual(
            tasks_mod.make_plan_task_id(
                preset="monolithic", strategy="all", block_size=None, seed=42, draw=None
            ),
            "plan-monolithic-all-bfull-s42",
        )

    def test_draw_suffix_still_comes_last(self):
        self.assertEqual(
            tasks_mod.make_plan_task_id(
                preset="gradient", strategy="all", block_size=168, seed=42, draw=3, batch_size=4
            ),
            "plan-gradient-all-b168-s42-B4-d3",
        )


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestPlanTaskEnumeration(unittest.TestCase):
    """``enumerate_plan_tasks`` reads the batch size out of the config."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ra-phase2-id-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _plan_cfg(self, name: str, extra: dict) -> dict:
        base = {
            "mode": "plan",
            "name": name,
            "dataset": {"years": [2020], "window": {"start": 0, "stop": 48}},
            "planning": {"method": "gradient"},
            "selection": {"strategy": "all", "block_size": 24},
        }
        path = write_config(self.tmp / f"{name}.yaml", config.deep_merge(base, extra))
        return config.load_config(path)

    def test_two_batch_rules_enumerate_distinct_design_ids(self):
        det = self._plan_cfg("c4", {"planning": {"optimizer": {"batch_size": 0}}})
        sgd = self._plan_cfg(
            "c5",
            {
                "planning": {
                    "optimizer": {
                        "batch_size": 4,
                        "batch_strategy": "random",
                        # A minibatch forbids `best_sampled` (config.validate),
                        # so this mirrors the shipped c5 preset.
                        "design_selection": "best_checkpointed",
                        "checkpoint_every": 20,
                    }
                }
            },
        )
        det_ids = [t.design_id for t in tasks_mod.enumerate_tasks(det)]
        sgd_ids = [t.design_id for t in tasks_mod.enumerate_tasks(sgd)]
        self.assertEqual(det_ids, ["plan-gradient-all-b24-s42"])
        self.assertEqual(sgd_ids, ["plan-gradient-all-b24-s42-B4"])
        self.assertEqual(len(set(det_ids + sgd_ids)), 2)


# ===========================================================================
# 4. Aligned random block selection
# ===========================================================================


def legacy_random_blocks(total_hours, block_size, num_blocks, seed, step=1):
    """The ported ``_sample_random`` loop, reimplemented to pin the RNG contract."""
    rng = np.random.default_rng(seed)
    available = set(range(0, total_hours - block_size + 1, step))
    blocks = []
    while len(blocks) < num_blocks and available:
        start = rng.choice(list(available))
        blocks.append((start, start + block_size))
        for s in range(max(0, start - block_size + 1), start + block_size):
            available.discard(s)
    return sorted(blocks)


def _bare_sampler(total_hours: int, num_years: int = 1):
    """A ``SystemBlockSampler`` with only the horizon geometry set.

    ``__init__`` needs a ``LoadedSystem``; the ``_sample_*`` methods need only
    these four attributes, so the sampling contract is testable without data.
    """
    from experiments.ra.planning.sampler import SystemBlockSampler

    sampler = SystemBlockSampler.__new__(SystemBlockSampler)
    per_year = total_hours // num_years
    sampler.total_hours = total_hours
    sampler.num_years = num_years
    sampler.hours_per_year = [per_year] * num_years
    sampler.year_boundaries = np.cumsum([0] + sampler.hours_per_year).tolist()
    return sampler


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestAlignBlocks(unittest.TestCase):
    TOTAL = 8736
    SIZE = 168

    def sampler(self):
        return _bare_sampler(self.TOTAL)

    def test_aligned_starts_are_a_subset_of_all_blocks(self):
        sampler = self.sampler()
        blocks = sampler.sample_blocks(
            block_size=self.SIZE, num_blocks=12, strategy="random", seed=42, align=True
        )
        self.assertEqual(len(blocks), 12)
        for start, stop in blocks:
            self.assertEqual(start % self.SIZE, 0, f"start {start} is off the grid")
            self.assertEqual(stop - start, self.SIZE)
        grid = set(sampler._sample_all(self.SIZE))
        self.assertTrue(set(blocks) <= grid)
        self.assertEqual(len(set(blocks)), 12)

    def test_unaligned_is_the_pre_change_behaviour(self):
        """``align=False`` must reproduce the ported RNG draw exactly."""
        sampler = self.sampler()
        blocks = sampler.sample_blocks(
            block_size=self.SIZE, num_blocks=12, strategy="random", seed=42, align=False
        )
        self.assertEqual(blocks, legacy_random_blocks(self.TOTAL, self.SIZE, 12, 42, step=1))
        # ... and it is a different, off-grid selection.
        self.assertFalse(all(start % self.SIZE == 0 for start, _ in blocks))

    def test_aligned_matches_the_grid_stepped_reference(self):
        sampler = self.sampler()
        blocks = sampler.sample_blocks(
            block_size=self.SIZE, num_blocks=12, strategy="random", seed=42, align=True
        )
        self.assertEqual(
            blocks, legacy_random_blocks(self.TOTAL, self.SIZE, 12, 42, step=self.SIZE)
        )

    def test_aligned_stratified_stays_on_the_grid(self):
        sampler = _bare_sampler(2 * self.TOTAL, num_years=2)
        blocks = sampler.sample_blocks(
            block_size=self.SIZE,
            num_blocks=8,
            strategy="stratified",
            seed=42,
            align=True,
        )
        self.assertEqual(len(blocks), 8)
        for start, _ in blocks:
            self.assertEqual(start % self.SIZE, 0)

    def test_selection_spec_carries_the_flag(self):
        from experiments.ra.planning.selection import selection_spec

        spec = selection_spec(
            {"selection": {"strategy": "random", "block_size": 168, "num_blocks": 12}}
        )
        self.assertFalse(spec.align_blocks)
        spec = selection_spec(
            {
                "selection": {
                    "strategy": "random",
                    "block_size": 168,
                    "num_blocks": 12,
                    "align_blocks": True,
                }
            }
        )
        self.assertTrue(spec.align_blocks)
        self.assertTrue(spec.to_dict()["align_blocks"])


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestAlignedSelectionEndToEnd(PlanningFixtureMixin, unittest.TestCase):
    def test_aligned_selection_is_recorded_and_on_the_grid(self):
        result = self.plan(
            selection={
                "strategy": "random",
                "block_size": 8,
                "num_blocks": 3,
                "align_blocks": True,
            },
            planning={"method": "monolithic"},
        )
        blocks = result.selection["blocks"]
        self.assertEqual(len(blocks), 3)
        self.assertTrue(result.selection["align_blocks"])
        for start, stop in blocks:
            self.assertEqual(start % 8, 0)
            self.assertEqual(stop - start, 8)


# ===========================================================================
# The seedable minibatch RNG
# ===========================================================================


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestBatchSeed(PlanningFixtureMixin, unittest.TestCase):
    def batches(self, seed: int):
        result = self.plan(
            selection={"strategy": "all", "block_size": 8, "seed": seed},
            planning=deep_merge(
                GRADIENT,
                {"optimizer": {"num_iterations": 4, "batch_size": 2, "batch_strategy": "random"}},
            ),
        )
        self.assertEqual(result.solver["batch_seed"], seed)
        return [[int(b) for b in batch] for batch in result.history["batch"]]

    def test_same_seed_reproduces_the_batch_sequence(self):
        self.assertEqual(self.batches(42), self.batches(42))

    def test_different_seed_changes_the_batch_sequence(self):
        self.assertNotEqual(self.batches(42), self.batches(7))


# ===========================================================================
# Config surface
# ===========================================================================


class TestCampaignConfigSurface(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ra-phase2-cfg-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _cfg(self, extra: dict) -> Path:
        return write_config(self.tmp / "cfg.yaml", config.deep_merge({"mode": "plan"}, extra))

    def test_base_yaml_carries_the_new_keys(self):
        base = config.base_config()
        self.assertIn("align_blocks", base["selection"])
        self.assertFalse(base["selection"]["align_blocks"])
        opt = base["planning"]["optimizer"]
        self.assertIsNone(opt["max_seconds"])
        # `best_checkpointed` / 20 became the defaults on 2026-09-10 alongside
        # `rule: adam`: it is the only selection rule that is correct for a
        # minibatch as well as a full batch, so the two kinds of cell are not
        # confounded by it.
        self.assertEqual(opt["design_selection"], "best_checkpointed")
        self.assertEqual(opt["checkpoint_every"], 20)

    def test_max_seconds_must_be_positive(self):
        with self.assertRaises(config.ConfigError):
            config.load_config(self._cfg({"planning": {"optimizer": {"max_seconds": 0}}}))

    def test_max_seconds_must_be_below_the_hard_timeout(self):
        with self.assertRaisesRegex(config.ConfigError, "timeout_s"):
            config.load_config(
                self._cfg({"planning": {"timeout_s": 600, "optimizer": {"max_seconds": 600}}})
            )
        cfg = config.load_config(
            self._cfg({"planning": {"timeout_s": 600, "optimizer": {"max_seconds": 300}}})
        )
        self.assertEqual(cfg["planning"]["optimizer"]["max_seconds"], 300.0)

    def test_unknown_design_selection_is_rejected(self):
        with self.assertRaisesRegex(config.ConfigError, "design_selection"):
            config.load_config(
                self._cfg({"planning": {"optimizer": {"design_selection": "best_guess"}}})
            )

    def test_best_checkpointed_needs_checkpoints(self):
        # `checkpoint_every: 0` has to be set explicitly now that the default is
        # 20; the guard is that the pair is inconsistent, not that it is unset.
        with self.assertRaisesRegex(config.ConfigError, "checkpoint_every"):
            config.load_config(
                self._cfg(
                    {
                        "planning": {
                            "optimizer": {
                                "design_selection": "best_checkpointed",
                                "checkpoint_every": 0,
                            }
                        }
                    }
                )
            )
        cfg = config.load_config(
            self._cfg(
                {
                    "planning": {
                        "optimizer": {
                            "design_selection": "best_checkpointed",
                            "checkpoint_every": 20,
                        }
                    }
                }
            )
        )
        self.assertEqual(cfg["planning"]["optimizer"]["checkpoint_every"], 20)

    def test_history_rules_need_the_param_history(self):
        # `best_sampled` / `best_rolling` read the chosen iterate back out of the
        # parameter history; the default rule is `best_checkpointed`, which does
        # not, so the rule has to be named for this guard to apply.
        with self.assertRaisesRegex(config.ConfigError, "save_param_history"):
            config.load_config(
                self._cfg(
                    {
                        "planning": {
                            "optimizer": {
                                "design_selection": "best_sampled",
                                "save_param_history": False,
                            }
                        }
                    }
                )
            )
        # ... and `best_checkpointed` does not need it.
        cfg = config.load_config(
            self._cfg({"planning": {"optimizer": {"save_param_history": False}}})
        )
        self.assertFalse(cfg["planning"]["optimizer"]["save_param_history"])

    def test_align_blocks_is_rejected_where_it_means_nothing(self):
        with self.assertRaisesRegex(config.ConfigError, "align_blocks"):
            config.load_config(self._cfg({"selection": {"align_blocks": True}}))
        cfg = config.load_config(
            self._cfg(
                {
                    "selection": {
                        "strategy": "random",
                        "num_blocks": 12,
                        "align_blocks": True,
                    }
                }
            )
        )
        self.assertTrue(cfg["selection"]["align_blocks"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
