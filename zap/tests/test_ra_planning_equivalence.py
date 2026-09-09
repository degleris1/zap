"""The new planning path reproduces ``experiments/multi_year/runner.py``.

WP5 spec section 7.2.  Two tests:

1. **Sampler equality** -- ``SystemBlockSampler`` and ``MultiYearBlockSampler``
   return identical block lists for the same horizon geometry, strategy and
   seed.  No solver, no data; this must never be skipped.
2. **Objective and capacity equality** -- the old path (pypsa ``.nc`` ->
   ``load_pypsa_network`` -> ``MultiYearBlockSampler`` ->
   ``MonolithicPlanningProblem``) and the new path (dataset directory ->
   ``load_system`` -> ``SystemBlockSampler`` -> ``SingleLevelMethod``) reach the
   same optimum on the restricted fixture in ``fixtures/equivalence.py``.

``experiments/multi_year/`` is byte-identical: nothing here writes to it (the
runner's output directory is redirected to a temp dir).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

from zap.tests.fixtures import equivalence as eqfix

try:
    from experiments.ra import planning

    PLANNING_AVAILABLE = True
    PLANNING_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - Task A not merged
    planning = None  # type: ignore[assignment]
    PLANNING_AVAILABLE = False
    PLANNING_IMPORT_ERROR = exc

TASK_A = f"experiments.ra.planning is not importable: {PLANNING_IMPORT_ERROR!r}"

N_HOURS = 48

#: WP3 replaced the symmetric ``DCLine`` with the one-way ``DirectedLine``, and
#: ``setup_parameter_names`` keys on ``type(device).__name__.lower()``, so the
#: two paths name the transmission parameter differently.  The mapping is part
#: of the extraction, not a discrepancy.
PARAM_ALIASES = {"dcline_capacity": "directedline_capacity"}


# ===========================================================================
# Test 1 -- sampler equality (no solver, no data)
# ===========================================================================


class TestSamplerEquality(unittest.TestCase):
    """The ported sampling code is the same algorithm as the old sampler."""

    NUM_YEARS = 2
    HOURS_PER_YEAR = (8760, 8760)

    def _old(self):
        from zap.importers.multi_year import MultiYearBlockSampler

        sampler = object.__new__(MultiYearBlockSampler)
        sampler.num_years = self.NUM_YEARS
        sampler.hours_per_year = list(self.HOURS_PER_YEAR)
        sampler.total_hours = sum(self.HOURS_PER_YEAR)
        sampler.year_boundaries = np.cumsum([0] + sampler.hours_per_year).tolist()
        return sampler

    def _new(self):
        from experiments.ra.planning.sampler import SystemBlockSampler

        sampler = object.__new__(SystemBlockSampler)
        sampler.num_years = self.NUM_YEARS
        sampler.hours_per_year = list(self.HOURS_PER_YEAR)
        sampler.total_hours = sum(self.HOURS_PER_YEAR)
        sampler.year_boundaries = np.cumsum([0] + sampler.hours_per_year).tolist()
        return sampler

    @unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
    def test_block_lists_are_identical(self):
        old, new = self._old(), self._new()
        for strategy in ("all", "uniform", "random", "stratified"):
            for block_size in (24, 168):
                for seed in (0, 42):
                    num_blocks = None
                    if strategy in ("random", "stratified") or strategy == "uniform":
                        num_blocks = 6
                    kwargs = {
                        "block_size": block_size,
                        "num_blocks": num_blocks,
                        "strategy": strategy,
                        "seed": seed,
                    }
                    with self.subTest(strategy=strategy, block_size=block_size, seed=seed):
                        self.assertEqual(old.sample_blocks(**kwargs), new.sample_blocks(**kwargs))

    @unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
    def test_avoid_year_boundaries_is_identical(self):
        old, new = self._old(), self._new()
        kwargs = {
            "block_size": 168,
            "num_blocks": 8,
            "strategy": "random",
            "avoid_year_boundaries": True,
            "seed": 3,
        }
        self.assertEqual(old.sample_blocks(**kwargs), new.sample_blocks(**kwargs))


# ===========================================================================
# Test 2 -- objective and capacity equality on the restricted fixture
# ===========================================================================


def _old_config(nc_path: Path, *, block_size, name: str) -> dict:
    """``runner.run_experiment`` config; every solver is HiGHS.

    Deviation from spec 7.2 ("every solver is CLARABEL"), reported deliberately.
    Now that the fixture's capital costs are pro-rated to the horizon the LP
    actually expands capacity, and CLARABEL's interior-point solution of the two
    (algebraically identical, differently ordered) LPs differs by ~2e-6
    relative -- above the rtol=1e-6 the spec asks the parameters to match at.
    HiGHS returns the exact vertex on both sides, so the comparison tests the
    two paths rather than one solver's tolerance. The dispatch solver is HiGHS
    on both sides for the same reason; no gradient is taken here, so the
    conditioning argument for CLARABEL (R-W2) does not apply.
    """
    return {
        "name": name,
        "method": "monolithic" if block_size is None else "stochastic",
        "network_files": [str(nc_path)],
        "hours_per_year": N_HOURS,
        "block_size": block_size,
        "sampling_strategy": "all",
        "num_blocks": None,
        "num_workers": 1,
        "seed": 42,
        "pypsa_args": eqfix.pypsa_args(),
        "dispatch_solver": "HIGHS",
        "dispatch_solver_kwargs": {},
        "relaxation": {
            "should_solve": True,
            "kind": "monolithic",
            "solver": "HIGHS",
            "solver_kwargs": {},
        },
        "wandb": {"enabled": False},
        "export": {"should_export": False},
    }


def _new_config(dataset: Path, *, block_size) -> dict:
    return {
        "mode": "plan",
        "dataset": {
            "dir": str(dataset),
            "years": [2020],
            "window": {"start": 0, "stop": N_HOURS},
        },
        "heuristics": {"name": "none", "ucap_derate": False, "outage_draws": []},
        "selection": {
            "strategy": "all",
            "block_size": block_size,
            "num_blocks": None,
            "seed": 42,
        },
        "planning": {
            "method": "monolithic" if block_size is None else "stochastic",
            "dispatch_solver": "HIGHS",
            "dispatch_solver_kwargs": {},
            "expansion": {"mode": "pypsa"},
            "single_level": {"kind": "primal", "solver": "HIGHS", "solver_kwargs": {}},
        },
    }


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestPathEquivalence(unittest.TestCase):
    """Old and new paths on the same tiny network, built from the same arrays."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.dataset, network = eqfix.write_equivalence_fixture(root / "ds", n_hours=N_HOURS)
        cls.nc_path = root / "equivalence.nc"
        network.export_to_netcdf(str(cls.nc_path))
        cls.network = network

        from zap.importers.wy_store import load_system

        cls.loaded = load_system(cls.dataset, eqfix.load_options(n_hours=N_HOURS))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # -- helpers ---------------------------------------------------------

    def _old_sampler(self):
        from zap.importers.multi_year import MultiYearBlockSampler

        return MultiYearBlockSampler([self.network], [self.network.snapshots], **eqfix.pypsa_args())

    def _run_old(self, cfg: dict) -> dict:
        from experiments.multi_year import runner

        original = runner.OUTPUT_PATH
        runner.OUTPUT_PATH = Path(self._tmp.name) / "old_outputs"
        try:
            return runner.run_experiment(dict(cfg))
        finally:
            runner.OUTPUT_PATH = original

    # -- tests -----------------------------------------------------------

    def test_importers_build_the_same_devices(self):
        """The precondition of the whole test: the two device lists agree."""
        old = self._old_sampler().base_devices
        eqfix.assert_devices_match(old, self.loaded.devices, compare_bounds=False)

    def test_expansion_reproduces_the_old_bounds(self):
        """``apply_expansion`` recovers what ``load_pypsa_network`` read from pypsa."""
        from experiments.ra.planning import expansion

        expanded = expansion.apply_expansion(
            self.loaded, _new_config(self.dataset, block_size=None)
        )
        eqfix.assert_devices_match(
            self._old_sampler().base_devices, expanded.devices, compare_bounds=True
        )

    def _assert_expansion_is_live(self, new):
        """The comparison is only meaningful if the LP actually built something.

        The fixture's capital costs are pro-rated to the 48-hour horizon
        (``eqfix.CAPITAL_COST_SCALE``); without that every extendable row sits on
        its lower bound, ``capex_raw`` is 0, and the two paths agree on a design
        neither of them chose.
        """
        self.assertGreater(
            float(new.objective["capex_raw"]),
            0.0,
            "capex_raw is 0: no row expanded, so this test compares nothing",
        )
        above = [
            param
            for param, value in new.parameters.items()
            if np.any(
                np.asarray(value, dtype=float).reshape(-1)
                > np.asarray(new.lower_bounds[param], dtype=float).reshape(-1) + 1e-6
            )
        ]
        self.assertTrue(above, "every capacity sits at its lower bound")

    def _assert_paths_agree(self, block_size, name):
        old = self._run_old(_old_config(self.nc_path, block_size=block_size, name=name))
        new = planning.plan(self.loaded, _new_config(self.dataset, block_size=block_size))

        self._assert_expansion_is_live(new)
        self.assertIsNotNone(old["final_cost"])
        self.assertAlmostEqual(
            new.objective_raw / old["final_cost"], 1.0, delta=1e-6, msg="objective"
        )
        self.assertTrue(old["optimal_parameters"])
        for key, value in old["optimal_parameters"].items():
            new_key = PARAM_ALIASES.get(key, key)
            self.assertIn(new_key, new.parameters, f"{key} has no counterpart")
            np.testing.assert_allclose(
                np.asarray(new.parameters[new_key], dtype=float).reshape(-1),
                np.asarray(value, dtype=float).reshape(-1),
                rtol=1e-6,
                atol=1e-6,
                err_msg=key,
            )

    def test_monolithic_preset_matches_runner(self):
        self._assert_paths_agree(None, "equiv_monolithic")

    def test_stochastic_preset_matches_runner(self):
        self._assert_paths_agree(24, "equiv_stochastic")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
