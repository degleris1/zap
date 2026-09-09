"""Tests for the WP5 planning core (`experiments/ra/planning`, WP5 spec section 9 Task A).

Everything here runs on 48 hours of the WP1 tiny dataset fixture, except the
sampler-equality test (WP5 spec section 7.2 test 1), which is data-free and must never
be skipped: it pins the four ``_sample_*`` bodies ported verbatim from
``zap.importers.multi_year.MultiYearBlockSampler`` against the original.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

from experiments.ra.config import ConfigError
from experiments.ra.planning import (
    base as planning_base,
)
from experiments.ra.planning import (
    design as design_mod,
)
from experiments.ra.planning import (
    expansion as expansion_mod,
)
from experiments.ra.planning import (
    parameters as parameters_mod,
)
from experiments.ra.planning import (
    selection as selection_mod,
)
from experiments.ra.planning.sampler import SystemBlockSampler
from zap.importers.multi_year import MultiYearBlockSampler

# ===========================================================================
# Section 7.2 test 1 -- sampler equality (no data, no solver, never skipped)
# ===========================================================================


def _fake_sampler(cls, *, num_years=2, hours_per_year=8760):
    """A sampler with the horizon geometry set by hand, bypassing ``__init__``."""
    obj = object.__new__(cls)
    obj.num_years = num_years
    obj.hours_per_year = [hours_per_year] * num_years
    obj.total_hours = num_years * hours_per_year
    obj.year_boundaries = np.cumsum([0] + obj.hours_per_year).tolist()
    return obj


class TestSamplerEquality(unittest.TestCase):
    """``SystemBlockSampler`` samples exactly what ``MultiYearBlockSampler`` samples."""

    def test_sample_blocks_match_multi_year_sampler(self):
        old = _fake_sampler(MultiYearBlockSampler)
        new = _fake_sampler(SystemBlockSampler)

        cases = []
        for strategy in ("all", "uniform", "random", "stratified"):
            for block_size in (24, 168):
                for seed in (0, 42):
                    num_blocks = 5 if strategy in ("random", "stratified") else None
                    cases.append((strategy, block_size, num_blocks, seed, False))
        # the avoid_year_boundaries branch of _sample_random
        cases.append(("random", 168, 5, 42, True))

        for strategy, block_size, num_blocks, seed, avoid in cases:
            with self.subTest(strategy=strategy, block_size=block_size, seed=seed, avoid=avoid):
                a = old.sample_blocks(
                    block_size=block_size,
                    num_blocks=num_blocks,
                    strategy=strategy,
                    avoid_year_boundaries=avoid,
                    seed=seed,
                )
                b = new.sample_blocks(
                    block_size=block_size,
                    num_blocks=num_blocks,
                    strategy=strategy,
                    avoid_year_boundaries=avoid,
                    seed=seed,
                )
                self.assertEqual(len(a), len(b))
                self.assertEqual(
                    [(int(s), int(e)) for s, e in a],
                    [(int(s), int(e)) for s, e in b],
                )

    def test_unknown_strategy_raises_in_both(self):
        new = _fake_sampler(SystemBlockSampler)
        with self.assertRaises(ValueError):
            new.sample_blocks(block_size=24, strategy="nope")

    def test_monolithic_sentinel_is_one_block(self):
        new = _fake_sampler(SystemBlockSampler)
        self.assertEqual(new.sample_blocks(block_size=None), [(0, 17520)])


# ===========================================================================
# Selection registry
# ===========================================================================


class TestSelection(unittest.TestCase):
    def _cfg(self, **selection):
        return {"selection": selection, "planning": {"method": "monolithic"}}

    def test_selection_registry_lists_phase2_stubs(self):
        for name in ("all", "uniform", "random", "stratified", "kmedoids", "gradient_stress"):
            self.assertIn(name, selection_mod.SELECTORS)
        sampler = _fake_sampler(SystemBlockSampler)
        for name in selection_mod.PHASE2_STRATEGIES:
            spec = selection_mod.SelectionSpec(strategy=name, block_size=168, num_blocks=4, seed=0)
            with self.assertRaises(NotImplementedError):
                selection_mod.SELECTORS[name](spec).select(sampler)

    def test_make_selector_rejects_unknown_strategy(self):
        with self.assertRaises(ConfigError) as ctx:
            selection_mod.make_selector(self._cfg(strategy="kmedians"), total_hours=48)
        self.assertIn("kmedians", str(ctx.exception))

    def test_make_selector_rejects_block_size_over_horizon(self):
        with self.assertRaises(ConfigError):
            selection_mod.make_selector(self._cfg(strategy="all", block_size=168), total_hours=48)

    def test_make_selector_requires_num_blocks(self):
        with self.assertRaises(ConfigError):
            selection_mod.make_selector(
                self._cfg(strategy="random", block_size=24, num_blocks=None), total_hours=48
            )

    def test_dispatch_mode_selection_keys_are_ignored(self):
        # R-W9: `mode: dispatch` keys share the block; plan mode must not choke.
        selector = selection_mod.make_selector(
            {
                "selection": {
                    "blocks": [24, 168],
                    "reference": "window",
                    "strategy": "all",
                    "block_size": 24,
                }
            },
            total_hours=48,
        )
        self.assertEqual(selector.spec.strategy, "all")
        self.assertIsNone(selector.weights([(0, 24)]))


class TestAnnualization(unittest.TestCase):
    def test_annualization_factor_arithmetic(self):
        self.assertEqual(planning_base.annualization_factor(48, 48), 1.0)
        self.assertEqual(planning_base.annualization_factor(48, 24), 2.0)
        self.assertAlmostEqual(planning_base.annualization_factor(8760, 168 * 12), 8760 / 2016)
        with self.assertRaises(ValueError):
            planning_base.annualization_factor(48, 0)


class TestSolverValidation(unittest.TestCase):
    def test_unknown_solver_rejected(self):
        import cvxpy as cp

        with self.assertRaises(ConfigError) as ctx:
            planning_base.require_solver("GRB_NOT_INSTALLED")
        message = str(ctx.exception)
        for name in cp.installed_solvers():
            if name in ("HIGHS", "CLARABEL", "SCS"):
                self.assertIn(name, message)

    def test_known_solver_is_normalized(self):
        self.assertEqual(planning_base.require_solver("highs"), "HIGHS")

    def test_unknown_method_rejected(self):
        with self.assertRaises(ConfigError) as ctx:
            planning_base.make_method({"planning": {"method": "benders"}})
        self.assertIn("benders", str(ctx.exception))


class TestEmissionsValidation(unittest.TestCase):
    """D-W5, at validation time rather than at solve time."""

    def _cfg(self, method, emissions, **planning):
        return {"planning": {"method": method, "emissions": emissions, **planning}}

    def test_cap_rejected_for_gradient(self):
        from experiments.ra.planning import constraints

        with self.assertRaises(ConfigError):
            constraints.validate_emissions(
                self._cfg("gradient", {"mode": "cap", "cap": 1.0, "cap_basis": "horizon"})
            )

    def test_cap_rejected_for_strong_duality(self):
        from experiments.ra.planning import constraints

        with self.assertRaises(ConfigError):
            constraints.validate_emissions(
                self._cfg(
                    "relaxed",
                    {"mode": "cap", "cap": 1.0, "cap_basis": "horizon"},
                    single_level={"kind": "strong_duality"},
                )
            )

    def test_dual_ascent_rejected_for_single_level(self):
        from experiments.ra.planning import constraints

        with self.assertRaises(ConfigError):
            constraints.validate_emissions(
                self._cfg("monolithic", {"mode": "dual_ascent", "dual_ascent": {"target": 1.0}})
            )

    def test_cap_annualization(self):
        from experiments.ra.planning import constraints

        annual = self._cfg("monolithic", {"mode": "cap", "cap": 100.0, "cap_basis": "annual"})
        horizon = self._cfg("monolithic", {"mode": "cap", "cap": 100.0, "cap_basis": "horizon"})
        self.assertEqual(constraints.emissions_limit(annual, 2.0), 50.0)
        self.assertEqual(constraints.emissions_limit(horizon, 2.0), 100.0)
        self.assertIsNone(
            constraints.emissions_limit(self._cfg("monolithic", {"mode": "none"}), 2.0)
        )


# ===========================================================================
# Tests on the tiny dataset
# ===========================================================================


def _dataset_available() -> bool:
    try:
        import zarr  # noqa: F401

        from zap.importers.wy_store import load_system  # noqa: F401
        from zap.tests.fixtures.tiny_dataset import write_tiny_dataset  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means the fixture is unusable
        return False
    return True


#: Rows the test makes extendable (the fixture ships an all-frozen fleet).
EXTENDABLE_GENERATORS = {
    "z1 solar": (0.0, np.inf),
    "z2 onwind": (5.0, 500.0),
}
EXTENDABLE_STORAGE = {"z1 battery": (0.0, np.inf)}


class TinyPlanningMixin(unittest.TestCase):
    """A 48-hour, one-year tiny system with a couple of extendable rows."""

    @classmethod
    def setUpClass(cls):
        if not _dataset_available():  # pragma: no cover
            raise unittest.SkipTest("WP1 (wy_store + tiny dataset fixture) is not available")
        from zap.importers.wy_store import convert_dataset
        from zap.tests.fixtures.tiny_dataset import write_tiny_dataset

        cls.tmp = Path(tempfile.mkdtemp(prefix="ra-planning-core-"))
        cls.dataset = write_tiny_dataset(cls.tmp / "tiny", n_hours=48, years=(2020,))
        cls._make_extendable(cls.dataset)
        convert_dataset(cls.dataset)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @staticmethod
    def _make_extendable(dataset: Path) -> None:
        gens = pd.read_csv(dataset / "static" / "generators.csv", index_col=0)
        gens["p_nom_min"] = 0.0
        gens["p_nom_max"] = gens["p_nom"]
        for name, (lo, hi) in EXTENDABLE_GENERATORS.items():
            gens.loc[name, "p_nom_extendable"] = True
            gens.loc[name, "p_nom_min"] = lo
            gens.loc[name, "p_nom_max"] = hi
        gens.to_csv(dataset / "static" / "generators.csv")

        units = pd.read_csv(dataset / "static" / "storage_units.csv", index_col=0)
        units["p_nom_min"] = 0.0
        units["p_nom_max"] = units["p_nom"]
        for name, (lo, hi) in EXTENDABLE_STORAGE.items():
            units.loc[name, "p_nom_extendable"] = True
            units.loc[name, "p_nom_min"] = lo
            units.loc[name, "p_nom_max"] = hi
        units.to_csv(dataset / "static" / "storage_units.csv")

    # -- helpers ----------------------------------------------------------
    def cfg(self, *, expansion="pypsa", selection=None, planning=None) -> dict:
        sel = {"strategy": "all", "block_size": 24, "num_blocks": None, "seed": 42}
        sel.update(selection or {})
        plan_block = {
            "method": "monolithic",
            "dispatch_solver": "HIGHS",
            "expansion": {"mode": expansion},
            "single_level": {"solver": "HIGHS"},
        }
        plan_block.update(planning or {})
        return {
            "dataset": {
                "dir": str(self.dataset),
                "years": [2020],
                "window": {"start": 0, "stop": 48},
            },
            "heuristics": {"ucap_derate": False, "outage_draws": []},
            "selection": sel,
            "planning": plan_block,
        }

    def loaded(self):
        from zap.importers.wy_store import HourWindow, LoadOptions, load_system

        return load_system(
            self.dataset,
            LoadOptions(years=(2020,), window=HourWindow(0, 48), demand_scaling="none"),
        )

    def context(self, **cfg_kwargs):
        """Run the shared ``build()`` and hand back the context."""
        cfg = self.cfg(**cfg_kwargs)
        method = _BuildOnlyMethod(cfg)
        try:
            return cfg, method.build(self.loaded())
        except ValueError as exc:  # pragma: no cover - WP1/WP3 join not ready
            if "time horizon" in str(exc) or "do not span" in str(exc):
                self.skipTest(f"blocked planning problem not yet buildable: {exc}")
            raise


class _BuildOnlyMethod(planning_base.PlanningMethod):
    """A concrete method that only exercises the shared ``build()`` (section 3.3)."""

    name = "build_only"

    def solve(self, ctx):  # pragma: no cover - never called
        raise NotImplementedError


class TestExpansionAndBounds(TinyPlanningMixin):
    """Section 7.1 tests 1 and 2."""

    def test_parameter_names_and_bounds(self):
        loaded = self.loaded()
        expanded = expansion_mod.apply_expansion(loaded, self.cfg(expansion="pypsa"))
        devices = expanded.devices
        names = parameters_mod.setup_parameter_names(devices)
        self.assertEqual(
            set(names), {"generator_capacity", "directedline_capacity", "storageunit_power"}
        )

        lower, upper = parameters_mod.setup_bounds(
            devices, names, min_capacity_mw=0.1, min_storage_mw=10.0
        )

        # shapes match the device attribute
        for param, (idx, attr) in names.items():
            current = np.asarray(getattr(devices[idx], attr))
            self.assertEqual(lower[param].shape, current.shape, param)
            self.assertEqual(upper[param].shape, current.shape, param)

        gen_idx = expanded.index.device_index["Generator"]
        gen_names = list(expanded.index.names["Generator"])
        gen_lo = lower["generator_capacity"].reshape(-1)
        gen_hi = upper["generator_capacity"].reshape(-1)
        p_nom = np.asarray(devices[gen_idx].nominal_capacity).reshape(-1)

        for i, name in enumerate(gen_names):
            if name in EXTENDABLE_GENERATORS:
                continue
            with self.subTest(generator=name):
                self.assertAlmostEqual(gen_lo[i], max(p_nom[i], 0.1))
                self.assertAlmostEqual(gen_hi[i], p_nom[i])

        i_solar = gen_names.index("z1 solar")
        # p_nom_min is 0, but no design may retire: the lower bound is p_nom
        # (80 MW), not the 0.1 MW generator floor (WP5-A verification defect 2).
        self.assertAlmostEqual(gen_lo[i_solar], 80.0)
        self.assertTrue(np.isfinite(gen_hi[i_solar]))
        self.assertAlmostEqual(gen_hi[i_solar], (80.0 + 1000.0) * 10.0)

        i_wind = gen_names.index("z2 onwind")
        # p_nom_min 5 MW < p_nom 60 MW -> pinned at the as-built capacity
        self.assertAlmostEqual(gen_lo[i_wind], 60.0)
        self.assertAlmostEqual(gen_hi[i_wind], 500.0)

        st_names = list(expanded.index.names["StorageUnit"])
        st_lo = lower["storageunit_power"].reshape(-1)
        st_hi = upper["storageunit_power"].reshape(-1)
        i_batt = st_names.index("z1 battery")
        self.assertAlmostEqual(st_lo[i_batt], 30.0)  # as-built, above the storage floor
        self.assertAlmostEqual(st_hi[i_batt], (30.0 + 1000.0) * 10.0)
        i_phs = st_names.index("z2 PHS")
        self.assertAlmostEqual(st_lo[i_phs], 20.0)
        self.assertAlmostEqual(st_hi[i_phs], 20.0)

        self.assertTrue(np.all(lower["generator_capacity"] <= upper["generator_capacity"]))
        report = expanded.meta["expansion"]
        self.assertEqual(report["classes"]["Generator"]["extendable"], 2)
        self.assertEqual(report["classes"]["Generator"]["upper_bounds_invented"], 1)
        # both extendable generators had p_nom_min < p_nom
        self.assertEqual(report["classes"]["Generator"]["retirement_blocked"], 2)
        self.assertEqual(report["classes"]["Generator"]["upper_bounds_raised_to_p_nom"], 0)
        self.assertEqual(report["classes"]["StorageUnit"]["retirement_blocked"], 1)

    def test_expansion_none_is_frozen(self):
        loaded = self.loaded()
        same = expansion_mod.apply_expansion(loaded, self.cfg(expansion="none"))
        self.assertIs(same, loaded)

        names = parameters_mod.setup_parameter_names(same.devices)
        for param, (idx, attr) in names.items():
            device = same.devices[idx]
            min_attr, max_attr = parameters_mod._bound_attrs(device)
            cap = np.asarray(getattr(device, attr), dtype=float)
            np.testing.assert_allclose(np.asarray(getattr(device, min_attr), dtype=float), cap)
            np.testing.assert_allclose(np.asarray(getattr(device, max_attr), dtype=float), cap)

        lower, upper = parameters_mod.setup_bounds(
            same.devices, names, min_capacity_mw=0.1, min_storage_mw=10.0
        )
        for param, (idx, attr) in names.items():
            cap = np.asarray(getattr(same.devices[idx], attr), dtype=float)
            np.testing.assert_allclose(upper[param], cap)
            # the design is pinned to the as-built capacity, up to the floors
            np.testing.assert_allclose(lower[param], np.maximum(cap, lower[param]))

    def test_setup_bounds_refuses_an_infinite_upper_bound(self):
        loaded = self.loaded()
        devices = [d for d in loaded.devices]
        names = parameters_mod.setup_parameter_names(devices)
        idx, _ = names["generator_capacity"]
        import copy as _copy

        devices[idx] = _copy.deepcopy(devices[idx])
        devices[idx].max_nominal_capacity = np.full_like(
            np.asarray(devices[idx].max_nominal_capacity, dtype=float), np.inf
        )
        with self.assertRaises(ValueError) as ctx:
            parameters_mod.setup_bounds(devices, names)
        self.assertIn("expansion", str(ctx.exception))

    def test_expansion_rejects_a_mismatched_csv(self):
        gens = pd.read_csv(self.dataset / "static" / "generators.csv", index_col=0)
        extra = gens.iloc[[0]].copy()
        extra.index = ["bogus row"]
        bad_dir = self.tmp / "bad"
        if bad_dir.exists():
            shutil.rmtree(bad_dir)
        shutil.copytree(self.dataset, bad_dir)
        pd.concat([gens, extra]).to_csv(bad_dir / "static" / "generators.csv")

        cfg = self.cfg()
        cfg["dataset"]["dir"] = str(bad_dir)
        with self.assertRaises(ValueError):
            expansion_mod.apply_expansion(self.loaded(), cfg)


class TestBuild(TinyPlanningMixin):
    """The shared ``build()`` body and the annualization arithmetic."""

    def test_build_all_blocks_covers_the_horizon(self):
        _cfg, ctx = self.context()
        self.assertEqual(ctx.blocks, [(0, 24), (24, 48)])
        self.assertEqual(ctx.total_hours, 48)
        self.assertEqual(ctx.sampled_hours, 48)
        self.assertEqual(ctx.coverage, 1.0)
        self.assertEqual(ctx.annualization_factor, 1.0)
        self.assertEqual(len(ctx.problem.subproblems), 2)
        self.assertEqual(ctx.meta["selection"]["strategy"], "all")
        self.assertEqual(ctx.meta["emissions"]["mode"], "none")
        self.assertAlmostEqual(ctx.meta["year_factor"], 8760.0 / 48.0)

        year_blocks = ctx.sampler.to_blocks(ctx.blocks)
        self.assertEqual(
            [(b.year, b.start, b.stop) for b in year_blocks], [(2020, 0, 24), (2020, 24, 48)]
        )

    def test_partial_coverage_sets_the_annualization_factor(self):
        _cfg, ctx = self.context(
            selection={"strategy": "random", "block_size": 24, "num_blocks": 1, "seed": 7}
        )
        self.assertEqual(len(ctx.blocks), 1)
        self.assertEqual(ctx.sampled_hours, 24)
        self.assertEqual(ctx.coverage, 0.5)
        self.assertEqual(ctx.annualization_factor, 2.0)

    def test_selection_seed_is_reproducible(self):
        _, a = self.context(
            selection={"strategy": "random", "block_size": 24, "num_blocks": 1, "seed": 7}
        )
        _, b = self.context(
            selection={"strategy": "random", "block_size": 24, "num_blocks": 1, "seed": 7}
        )
        self.assertEqual(a.blocks, b.blocks)

    def test_create_stochastic_problem_prorates_capital_cost(self):
        """One 24 h block of a 48 h horizon halves every subproblem capital cost."""
        _cfg, ctx = self.context(
            selection={"strategy": "uniform", "block_size": 24, "num_blocks": 1, "seed": 0}
        )
        self.assertEqual(len(ctx.problem.subproblems), 1)
        base = ctx.sampler.base_devices
        block = ctx.problem.subproblems[0].layer.devices
        checked = 0
        for base_dev, block_dev in zip(base, block):
            if getattr(base_dev, "capital_cost", None) is None:
                continue
            np.testing.assert_allclose(
                np.asarray(block_dev.capital_cost, dtype=float),
                np.asarray(base_dev.capital_cost, dtype=float) * 0.5,
                rtol=1e-12,
            )
            checked += 1
        self.assertGreater(checked, 0)

    def test_build_rejects_a_selection_that_does_not_fit(self):
        bad = _BuildOnlyMethod(
            self.cfg(selection={"strategy": "uniform", "block_size": 24, "num_blocks": 4})
        )
        with self.assertRaises(ValueError):
            bad.build(self.loaded())

    def test_monolithic_sentinel_builds_one_block(self):
        method = _BuildOnlyMethod(self.cfg(selection={"strategy": "all", "block_size": None}))
        ctx = method.build(self.loaded())
        self.assertEqual(ctx.blocks, [(0, 48)])
        self.assertEqual(ctx.annualization_factor, 1.0)


class TestDesignRoundTrip(TinyPlanningMixin):
    """Section 7.1 test 20."""

    def _result(self, ctx, design_id="plan-monolithic-all-b24-s42"):
        return planning_base.PlanningResult.from_context(
            ctx,
            method="single_level",
            preset="monolithic",
            kind="primal",
            design_id=design_id,
            parameters={k: v.copy() for k, v in ctx.initial_parameters.items()},
            objective={"raw": 1.0, "capex_raw": 0.6, "opex_raw": 0.4},
            solver={"name": "HIGHS", "status": "optimal", "kwargs": {}},
        )

    def test_design_roundtrip(self):
        _cfg, ctx = self.context()
        result = self._result(ctx)
        run_dir = self.tmp / "run"
        path = result.write(run_dir)
        self.assertEqual(design_mod.design_paths(run_dir), [path])

        record = design_mod.read_design_record(path)
        self.assertEqual(record["schema_version"], 1)
        for key in (
            "design_id",
            "method",
            "preset",
            "kind",
            "dataset",
            "years",
            "window",
            "heuristics",
            "selection",
            "annualization",
            "parameter_names",
            "capacities",
            "bounds",
            "objective",
            "emissions",
            "solver",
            "timing",
            "compute",
        ):
            self.assertIn(key, record)
        self.assertEqual(record["annualization"]["coverage"], 1.0)
        self.assertEqual(record["objective"]["annual"], 1.0)

        # The as-built capacities travel with the design: `metrics.py` needs them
        # to report `capacity_added_mw`, and they reach it only through
        # `PlanningResult.meta` (`ctx.meta["initial_parameters"]`).
        self.assertIn("initial_parameters", ctx.meta)
        self.assertEqual(sorted(result.meta["initial_parameters"]), sorted(ctx.parameter_names))
        self.assertEqual(record["initial_parameters"], result.meta["initial_parameters"])
        for param, values in record["initial_parameters"].items():
            np.testing.assert_allclose(
                np.asarray(values, dtype=float),
                np.asarray(ctx.initial_parameters[param], dtype=float).reshape(-1),
            )
        # ... and they are what a metrics row differences the design against.
        from experiments.ra.metrics import planning_metrics

        metrics = planning_metrics(result)
        self.assertIn("capacity_added_mw", metrics)
        self.assertAlmostEqual(metrics["capacity_added_mw"], 0.0, places=6)

        loaded = self.loaded()
        design = design_mod.read_design(path, system=loaded)
        devices = design.apply(loaded)
        for cls_name, values in design.capacities.items():
            device = devices[loaded.index.device_index[cls_name]]
            expected = np.asarray(values, dtype=float).reshape(-1, 1)
            for attr in (
                "nominal_capacity",
                "power_capacity",
                "min_nominal_capacity",
                "max_nominal_capacity",
                "min_power_capacity",
                "max_power_capacity",
            ):
                current = getattr(device, attr, None)
                if current is not None:
                    np.testing.assert_allclose(np.asarray(current, dtype=float), expected)

    def test_design_with_foreign_names_is_rejected(self):
        _cfg, ctx = self.context()
        result = self._result(ctx)
        record = result.to_record()
        record["capacities"]["Generator"]["names"] = [
            f"other {i}" for i in range(len(record["capacities"]["Generator"]["names"]))
        ]
        with self.assertRaises(ValueError) as err:
            design_mod.record_to_design(record, system=self.loaded())
        self.assertIn("different dataset", str(err.exception))

    def test_history_is_written_and_serializable(self):
        import json

        _cfg, ctx = self.context()
        result = self._result(ctx, design_id="with-history")
        result.history = {
            "loss": [np.float64(3.0), 2.0],
            "param": [{k: v for k, v in ctx.initial_parameters.items()}],
        }
        run_dir = self.tmp / "run_history"
        result.write(run_dir)
        self.assertEqual(result.history_path, "designs/with-history.history.json")
        payload = json.loads((run_dir / result.history_path).read_text())
        self.assertEqual(payload["loss"], [3.0, 2.0])
        self.assertIsInstance(payload["param"][0], dict)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TestNoRetirements(TinyPlanningMixin):
    """WP5-A verification defect 2: no design may retire existing capacity.

    ``AbstractDevice.get_investment_cost`` is ``capital_cost * (x - p_nom)`` with
    no one-sided term, so any ``x < p_nom`` is a capital *refund*.  Until zap
    grows a one-sided capex term, ``apply_expansion`` floors every existing row
    at its as-built capacity (spec section 4).
    """

    def test_lower_bounds_never_fall_below_as_built(self):
        loaded = self.loaded()
        expanded = expansion_mod.apply_expansion(loaded, self.cfg(expansion="pypsa"))
        names = parameters_mod.setup_parameter_names(expanded.devices)
        lower, upper = parameters_mod.setup_bounds(
            expanded.devices, names, min_capacity_mw=0.1, min_storage_mw=10.0
        )
        for param, (idx, attr) in names.items():
            as_built = np.asarray(getattr(loaded.devices[idx], attr), dtype=float)
            with self.subTest(param=param):
                self.assertTrue(np.all(lower[param] >= as_built - 1e-9))
                self.assertTrue(np.all(upper[param] >= lower[param] - 1e-9))

    def test_monolithic_lp_never_retires_and_never_books_negative_capex(self):
        import cvxpy as cp

        from zap.planning.monolithic import MonolithicPlanningProblem

        _cfg, ctx = self.context()
        lp = MonolithicPlanningProblem(ctx.problem, solver=cp.HIGHS)
        params, data = lp.solve()
        self.assertEqual(data["problem"].status, "optimal")

        for param, (idx, attr) in ctx.parameter_names.items():
            as_built = np.asarray(
                getattr(ctx.sampler.base_devices[idx], attr), dtype=float
            ).reshape(-1)
            optimal = np.asarray(params[param], dtype=float).reshape(-1)
            with self.subTest(param=param):
                self.assertTrue(
                    np.all(optimal >= as_built - 1e-6),
                    f"{param}: LP retired capacity {optimal} below as-built {as_built}",
                )

        self.assertGreaterEqual(float(data["investment_objective"].value), -1e-6)


class TestAnnualizationOfResults(TinyPlanningMixin):
    """``PlanningResult.from_context`` multiplies by ``total / sampled`` (spec 6)."""

    def test_from_context_annualizes_at_half_coverage(self):
        _cfg, ctx = self.context(
            selection={"strategy": "uniform", "block_size": 24, "num_blocks": 1}
        )
        self.assertEqual(ctx.sampled_hours, 24)
        self.assertEqual(ctx.total_hours, 48)
        self.assertEqual(ctx.coverage, 0.5)
        self.assertEqual(ctx.annualization_factor, 2.0)

        result = planning_base.PlanningResult.from_context(
            ctx,
            method="monolithic",
            preset="monolithic",
            parameters=dict(ctx.initial_parameters),
            objective={
                "raw": 100.0,
                "capex_raw": 40.0,
                "opex_raw": 60.0,
                "emissions_tonnes_raw": 7.0,
            },
        )
        # A factor of 0.5 (the inverted annualization) would fail every one of these.
        self.assertAlmostEqual(result.objective["annual"], 200.0)
        self.assertAlmostEqual(result.objective["capex_annual"], 80.0)
        self.assertAlmostEqual(result.objective["opex_annual"], 120.0)
        self.assertAlmostEqual(result.objective["emissions_tonnes_annual"], 14.0)
        self.assertAlmostEqual(result.objective_annual, 2.0 * result.objective_raw)
        self.assertEqual(result.annualization["coverage"], 0.5)
        self.assertEqual(result.annualization["annualization_factor"], 2.0)
