"""WP-P1: what a run writes (``memory/plans/2026-09-09-plots-spec.md`` section 5.1).

Everything here runs on the 48-hour synthetic export from
``zap/tests/fixtures/tiny_dataset.py`` with HiGHS.  Nothing reads ``data/``.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

from experiments.ra import blocks as blocks_mod
from experiments.ra import cli, config, dispatch, identity, metrics, paths, persist
from experiments.ra import system as system_mod
from experiments.ra import tasks as tasks_mod

EXTENDABLE_GENERATORS = {"z1 solar": 500.0, "z2 onwind": 400.0}
EXTENDABLE_STORAGE = {"z1 battery": 200.0}


def _fixture_available() -> bool:
    try:
        import zarr  # noqa: F401

        from zap.importers.wy_store import load_system  # noqa: F401
        from zap.tests.fixtures.tiny_dataset import write_tiny_dataset  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _fake_gradient_result():
    """A synthetic ``PlanningResult``-shaped object with a 3-entry history.

    ``grad_norm_l2`` deliberately carries NaN, 0.0 and a value above the clip.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        design_id="d",
        run_id="r",
        method="gradient",
        preset="gradient",
        emissions={"price": 0.0},
        capacities={},
        parameter_names={},
        years=[2020],
        window={"start": 0, "stop": 48},
        annualization={"total_hours": 48, "annualization_factor": 1.0},
        selection={"blocks": [[0, 24], [24, 48]]},
        meta={
            "optimizer": {"step_size": 0.2, "clip": 5.0e3, "save_param_history": False},
            "carrier_labels": {},
        },
        history={
            "loss": [1.0, 2.0, 3.0],
            "rolling_loss": [1.0, 2.0, 3.0],
            "grad_norm": [1.0, 1.0, 1.0],
            "grad_norm_l2": [float("nan"), 0.0, 1.0e4],
            "proj_grad_norm": [1.0, 1.0, 1.0],
            "suboptimality": [0.0, 0.0, 0.0],
            "time": [0.0, 1.0, 2.0],
            "batch": [[0], [1], [0]],
        },
    )


class PersistenceMixin(unittest.TestCase):
    """A converted 48-hour tiny dataset, shared by the whole module."""

    @classmethod
    def setUpClass(cls):
        if not _fixture_available():  # pragma: no cover
            raise unittest.SkipTest("WP1 (wy_store + tiny dataset fixture) is not available")
        from zap.importers.wy_store import convert_dataset
        from zap.tests.fixtures.tiny_dataset import write_tiny_dataset

        cls.class_tmp = Path(tempfile.mkdtemp(prefix="ra-persist-"))
        cls.dataset = write_tiny_dataset(cls.class_tmp / "tiny", n_hours=48, years=(2020,))
        cls._make_extendable(cls.dataset)
        convert_dataset(cls.dataset)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.class_tmp, ignore_errors=True)

    @staticmethod
    def _make_extendable(dataset: Path) -> None:
        gens = pd.read_csv(dataset / "static" / "generators.csv", index_col=0)
        gens["p_nom_min"] = 0.0
        gens["p_nom_max"] = gens["p_nom"]
        for name, cap in EXTENDABLE_GENERATORS.items():
            gens.loc[name, "p_nom_extendable"] = True
            gens.loc[name, "p_nom_max"] = cap
        gens.to_csv(dataset / "static" / "generators.csv")

        units = pd.read_csv(dataset / "static" / "storage_units.csv", index_col=0)
        units["p_nom_min"] = 0.0
        units["p_nom_max"] = units["p_nom"]
        for name, cap in EXTENDABLE_STORAGE.items():
            units.loc[name, "p_nom_extendable"] = True
            units.loc[name, "p_nom_max"] = cap
        units.to_csv(dataset / "static" / "storage_units.csv")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ra-persist-run-"))
        self.runs_root = self.tmp / "runs"
        system_mod.clear_system_cache()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- configs ----------------------------------------------------------
    def write_config(self, name: str, extra: dict | None = None) -> Path:
        cfg = {
            "includes": [str(paths.config_root() / "base.yaml")],
            "name": name,
            "dataset": {
                "dir": str(self.dataset),
                "years": [2020],
                "window": {"start": 0, "stop": 48},
            },
            "selection": {"blocks": [24], "reference": "none"},
            "methods": {"lp": {"enabled": True, "solver": "HIGHS", "timeout_s": 600}},
        }
        if extra:
            cfg = config.deep_merge(cfg, extra)
        path = self.tmp / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    def run_cli(self, path: Path) -> Path:
        code = cli.main(["run", "--config", str(path), "--runs-root", str(self.runs_root)])
        self.assertEqual(code, 0)
        return paths.run_dir(identity.run_id(config.load_config(path)), self.runs_root)


class TestHourlyStore(PersistenceMixin):
    def test_hourly_written_when_flag_on(self):
        """T1."""
        path = self.write_config("hourly_on", {"output": {"save_hourly": "carrier_bus"}})
        run_dir = self.run_cli(path)

        parts = sorted((run_dir / "hourly").glob("*.parquet"))
        task_ids = [t.task_id for t in tasks_mod.enumerate_tasks(config.load_config(path))]
        self.assertEqual([p.stem for p in parts], sorted(task_ids))
        combined = run_dir / "hourly.parquet"
        self.assertTrue(combined.exists())

        frame = pd.read_parquet(combined)
        self.assertEqual(tuple(frame.columns), persist.HOURLY_COLUMNS)
        self.assertTrue(set(frame["quantity"].astype(str)) <= set(persist.HOURLY_QUANTITIES))

        # Dtypes are declared, not inferred.
        schema = pq.read_schema(combined)
        declared = persist.hourly_schema()
        for field in declared:
            self.assertEqual(
                str(schema.field(field.name).type), str(field.type), msg=field.name
            )

        hours = sorted(frame["hour"].unique())
        self.assertEqual(hours, list(range(48)))
        for block_start in (0, 24):
            block = frame[(frame["hour"] >= block_start) & (frame["hour"] < block_start + 24)]
            self.assertEqual(sorted(block["hour"].unique()),
                             list(range(block_start, block_start + 24)))

        key = ["method", "block_size", "draw", "design_id", "year", "hour", "quantity",
               "carrier", "bus", "name"]
        self.assertFalse(frame.duplicated(subset=key).any())

    def test_hourly_absent_when_flag_off(self):
        """T2."""
        off = self.run_cli(self.write_config("hourly_off"))
        self.assertFalse((off / "hourly").exists())
        self.assertFalse((off / "hourly.parquet").exists())

        on = self.run_cli(
            self.write_config("hourly_on2", {"output": {"save_hourly": "carrier_bus"}})
        )
        self.assertEqual(
            set(pd.read_csv(off / "metrics.csv").columns),
            set(pd.read_csv(on / "metrics.csv").columns),
        )

    def test_hourly_reconciles_with_metrics(self):
        """T3: the real correctness gate."""
        path = self.write_config("reconcile", {"output": {"save_hourly": "carrier_bus"}})
        run_dir = self.run_cli(path)
        hourly = pd.read_parquet(run_dir / "hourly.parquet")
        rows = metrics.read_task_records(run_dir)
        self.assertTrue(rows)

        for record in rows:
            task_id = record["task_id"]
            block = hourly[hourly["task_id"].astype(str) == task_id]
            self.assertFalse(block.empty, task_id)
            m = record["metrics"]

            by_carrier = json.loads(m["generation_mwh_by_carrier"])
            dispatch_rows = block[block["quantity"] == "dispatch_mw"]
            got = dispatch_rows.groupby(dispatch_rows["carrier"].astype(str))["value"].sum()
            self.assertEqual(set(got.index), set(by_carrier))
            for carrier, value in by_carrier.items():
                self.assertAlmostEqual(got[carrier], value, delta=1e-9 * max(1.0, abs(value)))

            for quantity, metric in (
                ("unserved_mw", "unserved_energy_mwh"),
                ("curtailment_mw", "curtailment_mwh"),
            ):
                total = float(block[block["quantity"] == quantity]["value"].sum())
                self.assertAlmostEqual(
                    total, float(m[metric]), delta=1e-9 * max(1.0, abs(float(m[metric])))
                )

            # One membership for available capacity (2026-09-09): the hourly
            # rows and metrics.csv must agree exactly, and imports are in
            # neither.
            avail = block[block["quantity"] == "available_capacity_mw"]
            self.assertFalse(avail.empty, task_id)
            carriers = set(avail["carrier"].astype(str))
            self.assertTrue(carriers, task_id)
            self.assertFalse(
                {c for c in carriers if "import" in c},
                f"{task_id}: import carriers in available_capacity_mw: {carriers}",
            )
            per_hour = avail.groupby(avail["hour"].astype(int))["value"].sum()
            self.assertAlmostEqual(
                float(per_hour.min()),
                float(m["available_mw_min"]),
                delta=1e-6 * max(1.0, abs(float(m["available_mw_min"]))),
            )
            self.assertAlmostEqual(
                float(per_hour.sum()),
                float(m["available_mwh_total"]),
                delta=1e-6 * max(1.0, abs(float(m["available_mwh_total"]))),
            )

    def test_hourly_prices_are_load_buses_only(self):
        """T4 (D5)."""
        path = self.write_config("prices", {"output": {"save_hourly": "carrier_bus"}})
        run_dir = self.run_cli(path)
        hourly = pd.read_parquet(run_dir / "hourly.parquet")
        static = json.loads((run_dir / "system_static.json").read_text())

        buses = set(hourly[hourly["quantity"] == "price_usd_per_mwh"]["bus"].astype(str))
        self.assertEqual(buses, set(static["load_buses"]))
        self.assertNotIn("z1_imports", buses)
        self.assertNotIn("z1_exports", buses)

    def test_hourly_quantity_filter(self):
        """T5."""
        path = self.write_config(
            "filtered",
            {"output": {"save_hourly": "carrier_bus", "hourly_quantities": ["dispatch_mw"]}},
        )
        run_dir = self.run_cli(path)
        hourly = pd.read_parquet(run_dir / "hourly.parquet")
        self.assertEqual(set(hourly["quantity"].astype(str)), {"dispatch_mw"})

        bad = self.write_config(
            "bad_quantity",
            {"output": {"save_hourly": "carrier_bus", "hourly_quantities": ["nope"]}},
        )
        with self.assertRaises(config.ConfigError):
            config.load_config(bad)

        with self.assertRaises(config.ConfigError) as ctx:
            config.load_config(self.write_config("dev", {"output": {"save_hourly": "device"}}))
        self.assertIn("carrier x bus", str(ctx.exception))


class TestAdmmTrace(PersistenceMixin):
    def test_admm_trace_subsampling(self):
        """T6."""
        extra = {
            "output": {"admm_trace_every": 5},
            "methods": {
                "lp": {"enabled": False},
                "admm": {
                    "enabled": True,
                    "required": False,
                    "timeout_s": 600,
                    "solver_kwargs": {
                        "num_iterations": 32,
                        "rho_power": 1.0,
                        "minimum_iterations": 100,
                        "atol": 1e-8,
                        "rtol": 1e-8,
                    },
                },
            },
        }
        path = self.write_config("admm_trace", extra)
        cfg = config.load_config(path)
        run_dir = paths.run_dir(identity.run_id(cfg), self.runs_root)
        cli.main(["run", "--config", str(path), "--runs-root", str(self.runs_root)])

        parts = sorted((run_dir / "admm_trace").glob("*.parquet"))
        self.assertTrue(parts, "no ADMM trace was written")
        frame = pd.read_parquet(parts[0])
        self.assertEqual(tuple(frame.columns), persist.ADMM_TRACE_COLUMNS)
        iterations = list(frame["iteration"])
        self.assertEqual(iterations, sorted(set(iterations)))
        last = max(iterations)
        self.assertIn(last, iterations)
        self.assertEqual([i for i in iterations if i != last], [i for i in iterations
                                                                if i % 5 == 0 and i != last])
        self.assertTrue(all(i % 5 == 0 or i == last for i in iterations))

        off = self.write_config("admm_no_trace", config.deep_merge(extra, {"output":
                                                                           {"admm_trace_every": 0}}))
        cfg_off = config.load_config(off)
        run_dir_off = paths.run_dir(identity.run_id(cfg_off), self.runs_root)
        cli.main(["run", "--config", str(off), "--runs-root", str(self.runs_root)])
        self.assertFalse((run_dir_off / "admm_trace").exists())


class TestIterationTables(PersistenceMixin):
    def plan_config(self, name: str, extra: dict | None = None) -> Path:
        cfg = {
            "includes": [str(paths.config_root() / "base.yaml")],
            "name": name,
            "mode": "plan",
            "dataset": {
                "dir": str(self.dataset),
                "years": [2020],
                "window": {"start": 0, "stop": 48},
            },
            "selection": {"strategy": "all", "block_size": 24},
            "planning": {
                "method": "monolithic",
                "dispatch_solver": "HIGHS",
                "single_level": {"kind": "primal", "solver": "HIGHS"},
                "timeout_s": 900,
            },
        }
        if extra:
            cfg = config.deep_merge(cfg, extra)
        path = self.tmp / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    def test_iteration_tables(self):
        """T7."""
        from experiments.ra.planning import history as history_mod

        path = self.plan_config(
            "grad",
            {"planning": {"method": "gradient", "optimizer": {"num_iterations": 3}}},
        )
        run_dir = self.run_cli(path)

        # One file per task and table, so concurrent tasks cannot lose rows.
        self.assertEqual(
            [p.name for p in (run_dir / "iterations").glob("*.iterations.parquet")],
            ["plan-gradient-all-b24-s42.iterations.parquet"],
        )
        scalars = history_mod.read_table(run_dir, "iterations")
        self.assertEqual(tuple(scalars.columns), history_mod.ITERATIONS_COLUMNS)
        self.assertEqual(len(scalars), 4)  # num_iterations + 1
        self.assertEqual(list(scalars["iteration"]), [0, 1, 2, 3])
        self.assertTrue(((scalars["clip_fraction"] > 0) & (scalars["clip_fraction"] <= 1)).all())
        expected = scalars["step_size"] * np.minimum(scalars["grad_norm_l2"], scalars["clip"])
        np.testing.assert_allclose(scalars["step_norm_mw"], expected, rtol=1e-12)

        blocks = history_mod.read_table(run_dir, "iteration_blocks")
        self.assertEqual(tuple(blocks.columns), history_mod.ITERATION_BLOCKS_COLUMNS)
        self.assertEqual(len(blocks), int(scalars["n_batch"].sum()))

        capacity = history_mod.read_table(run_dir, "iteration_capacity")
        self.assertEqual(tuple(capacity.columns), history_mod.ITERATION_CAPACITY_COLUMNS)
        pairs = capacity[["param", "carrier"]].drop_duplicates()
        self.assertEqual(len(capacity), len(scalars) * len(pairs))

        mono = self.run_cli(self.plan_config("mono"))
        self.assertFalse((mono / "iterations").exists())


class TestIterationRegressions(TestIterationTables):
    """Regressions found by the verifier on the first WP-P1 diff."""

    def test_sampled_objective_annual_is_not_double_scaled(self):
        """A minibatch loss is already in whole-block-set units.

        ``StochasticPlanningProblem._get_batch_weights`` rescales a batch by
        ``total_weight / total_batch_weight``, so ``history["loss"]`` is in the
        units of every sampled block, not of the batch.  With two 24 h blocks
        covering the whole 48 h window the annualization factor is exactly 1.0,
        so the annual objective must equal the raw one even though each
        iteration only saw one of the two blocks.
        """
        from experiments.ra.planning import history as history_mod

        path = self.plan_config(
            "grad_batch",
            {
                "planning": {
                    "method": "gradient",
                    "optimizer": {
                        "num_iterations": 1,
                        "batch_size": 1,
                        "batch_strategy": "sequential",
                    },
                }
            },
        )
        run_dir = self.run_cli(path)
        scalars = history_mod.read_table(run_dir, "iterations")

        self.assertTrue((scalars["n_batch"] == 1).all())
        self.assertTrue((scalars["batch_hours"] == 24).all())
        self.assertTrue((scalars["annualization_factor"] == 1.0).all())
        np.testing.assert_allclose(
            scalars["sampled_objective_annual"], scalars["sampled_objective_raw"], rtol=1e-12
        )

    def test_iteration_files_are_per_task(self):
        """Two planning tasks in one run dir keep each other's rows.

        The tables used to be one file per run, rebuilt read-modify-write; two
        concurrent shards writing at once would then lose one of them.
        """
        from experiments.ra.planning import history as history_mod

        result = _fake_gradient_result()
        run_dir = self.tmp / "shared"
        for draw in (0, 1):
            task = tasks_mod.Task(
                task_id=f"plan-d{draw}",
                method="plan",
                block=blocks_mod.Block(index=0, year=2020, start=0, stop=48),
                block_size=24,
                draw=draw,
                design_id=f"plan-d{draw}",
            )
            written = history_mod.write_iteration_tables(result, run_dir, {}, task=task)
            self.assertTrue(written)

        names = sorted(p.name for p in (run_dir / "iterations").glob("*.iterations.parquet"))
        self.assertEqual(names, ["plan-d0.iterations.parquet", "plan-d1.iterations.parquet"])
        scalars = history_mod.read_table(run_dir, "iterations")
        self.assertEqual(sorted(set(scalars["task_id"])), ["plan-d0", "plan-d1"])
        self.assertEqual(len(scalars), 6)
        self.assertEqual(tuple(scalars.columns), history_mod.ITERATIONS_COLUMNS)

    def test_legacy_iteration_file_is_ignored_when_per_task_files_exist(self):
        """A re-run can leave both behind; reading both would double-count."""
        from experiments.ra.planning import history as history_mod

        result = _fake_gradient_result()
        run_dir = self.tmp / "legacy"
        task = tasks_mod.Task(
            task_id="plan-x",
            method="plan",
            block=blocks_mod.Block(index=0, year=2020, start=0, stop=48),
            block_size=24,
            design_id="plan-x",
        )
        history_mod.write_iteration_tables(result, run_dir, {}, task=task)
        per_task = history_mod.read_table(run_dir, "iterations")
        # The pre-change layout: one whole-run file holding the same task.
        per_task.to_parquet(run_dir / "iterations" / "iterations.parquet", index=False)

        again = history_mod.read_table(run_dir, "iterations")
        self.assertEqual(len(again), len(per_task))
        self.assertEqual(set(again["task_id"]), {"plan-x"})
        self.assertEqual(
            [p.name for p in history_mod.table_paths(run_dir, "iterations")],
            ["plan-x.iterations.parquet"],
        )

    def test_clip_fraction_is_nan_when_the_gradient_is_unmeasured(self):
        """A gradient we could not measure says nothing about the clip."""
        from experiments.ra.planning import history as history_mod

        table = history_mod.build_iteration_tables(_fake_gradient_result())["iterations"]
        self.assertTrue(np.isnan(table["clip_fraction"].iloc[0]))
        self.assertTrue(np.isnan(table["step_norm_mw"].iloc[0]))
        # A zero gradient is never clipped, and it moves nothing.
        self.assertEqual(table["clip_fraction"].iloc[1], 1.0)
        self.assertEqual(table["step_norm_mw"].iloc[1], 0.0)
        # The clip binds at 1e4 > 5e3.
        self.assertAlmostEqual(table["clip_fraction"].iloc[2], 0.5)
        self.assertAlmostEqual(table["step_norm_mw"].iloc[2], 0.2 * 5.0e3)


class TestOutputFlagsAndStatics(PersistenceMixin):
    def test_output_flags_do_not_change_the_run_id(self):
        """T8."""
        base = config.load_config(self.write_config("id_base"))
        expected = identity.run_id(base)
        variants = [
            {"save_hourly": "carrier_bus"},
            {"save_ens_profile": False},
            {"admm_trace_every": 7},
            {"save_iterations": False},
            {"combine_hourly": False},
            {"figures": True},
            {"hourly_quantities": ["dispatch_mw"]},
        ]
        for i, output in enumerate(variants):
            cfg = config.load_config(self.write_config(f"id_v{i}", {"output": output}))
            cfg["name"] = base["name"]
            self.assertEqual(identity.run_id(cfg), expected, msg=str(output))

        run_dir = self.run_cli(
            self.write_config("id_env", {"output": {"save_hourly": "carrier_bus"}})
        )
        env = json.loads((run_dir / "env.json").read_text())
        self.assertEqual(env["output"]["save_hourly"], "carrier_bus")

    def test_ens_profile_rows_are_shortfall_hours(self):
        """T9."""
        path = self.write_config(
            "shedding",
            {"system": {"demand_scaling": "fixed", "scale_load": 4.0}},
        )
        run_dir = self.run_cli(path)
        profile = pd.read_parquet(run_dir / "ens_profile.parquet")
        self.assertEqual(tuple(profile.columns), persist.ENS_PROFILE_COLUMNS)
        self.assertFalse(profile.empty, "the scaled run did not shed load")

        records = {r["task_id"]: r for r in metrics.read_task_records(run_dir)}
        total = float(profile["ens_mwh"].sum())
        expected = sum(float(r["metrics"]["unserved_energy_mwh"]) for r in records.values())
        self.assertAlmostEqual(total, expected, delta=1e-6 * max(1.0, expected))

        # The rows are exactly the (hour, bus) pairs above the tolerance.
        cfg = config.load_config(path)
        loaded = system_mod.build_system(cfg)
        block = blocks_mod.make_blocks(cfg, 24)[0]
        devices = dispatch.slice_devices(loaded, cfg, block)
        task = tasks_mod.Task(task_id="t", method="lp", block=block, block_size=24)
        outcome = loaded.network.dispatch(devices, time_horizon=block.hours, solver="HIGHS")
        power = metrics.numpyify(outcome.power)[: len(devices)]
        groups = metrics.device_groups(devices)
        names = persist.bus_name_table(loaded, devices)
        expected_pairs = set()
        for entry in persist.load_shortfall(devices, power, groups):
            for row, node in enumerate(entry.terminals):
                for t in np.nonzero(entry.shortfall[row, :] > metrics.SHORTFALL_TOL_MW)[0]:
                    expected_pairs.add((int(block.start + t), str(names[int(node)])))
        got = persist.build_ens_profile_frame(loaded, devices, outcome, block, task=task)
        self.assertEqual(set(zip(got["hour"], got["bus"].astype(str))), expected_pairs)

    def test_system_static_json(self):
        """T10."""
        from zap.tests.fixtures.tiny_dataset import (
            TINY_BUSES,
            TINY_CARRIERS,
            TINY_LINKS,
            TINY_STORAGE,
        )

        run_dir = self.run_cli(self.write_config("static"))
        payload = json.loads((run_dir / "system_static.json").read_text())
        for key in ("schema_version", "dataset", "buses", "load_buses", "carriers",
                    "lines", "storage"):
            self.assertIn(key, payload)
        self.assertEqual(payload["schema_version"], persist.SYSTEM_STATIC_SCHEMA_VERSION)
        self.assertEqual([b["name"] for b in payload["buses"]], list(TINY_BUSES))
        self.assertEqual(len(payload["lines"]), len(TINY_LINKS))
        self.assertEqual(len(payload["storage"]), len(TINY_STORAGE))
        self.assertEqual(set(payload["carriers"]), set(TINY_CARRIERS))
        for bus in payload["buses"]:
            self.assertIsNotNone(bus["x"], bus["name"])
            self.assertIsNotNone(bus["y"], bus["name"])
        self.assertEqual(sorted(payload["load_buses"]), ["z1", "z2"])

    def test_available_metrics(self):
        """T11."""
        def solve(power_unit: float):
            cfg = config.load_config(
                self.write_config(
                    f"avail_{power_unit:g}",
                    {"system": {"power_unit": power_unit, "cost_unit": 1.0}},
                )
            )
            system_mod.clear_system_cache()
            loaded = system_mod.build_system(cfg, cache=False)
            block = blocks_mod.make_blocks(cfg, 24)[0]
            devices = dispatch.slice_devices(loaded, cfg, block)
            task = tasks_mod.Task(task_id="t", method="lp", block=block, block_size=24)
            payload = dispatch.solve_block_lp(loaded, devices, task, cfg)
            return loaded, devices, payload["metrics"]

        loaded, devices, m = solve(1.0)

        generator = devices[loaded.index.device_index["Generator"]]
        available = np.asarray(generator.nominal_capacity, dtype=float) * np.asarray(
            generator.dynamic_capacity, dtype=float
        )
        available = np.broadcast_to(available, (available.shape[0], 24))
        in_state = ~np.asarray(loaded.index.import_mask, dtype=bool)
        per_hour = available[in_state, :].sum(axis=0)

        # One membership (2026-09-09): in-state generators PLUS storage power
        # capacity x power_availability.  Imports never count.
        storage = devices[loaded.index.device_index["StorageUnit"]]
        storage_available = np.asarray(storage.power_capacity, dtype=float).reshape(
            -1, 1
        ) * np.atleast_2d(np.asarray(storage.power_availability, dtype=float))
        storage_available = np.broadcast_to(storage_available, (storage_available.shape[0], 24))
        per_hour = per_hour + storage_available.sum(axis=0)
        self.assertGreater(storage_available.sum(), 0.0)

        self.assertAlmostEqual(m["available_mw_min"], float(per_hour.min()), places=6)
        self.assertAlmostEqual(m["available_mwh_total"], float(per_hour.sum()), places=4)

        _, _, scaled = solve(1000.0)
        for key in ("available_mw_min", "available_mwh_total"):
            self.assertAlmostEqual(scaled[key] / m[key], 1.0, places=9, msg=key)


class TestEvaluationTables(PersistenceMixin):
    """Regressions on ``evaluate.write_eval_tables`` (verifier items 2 and 3)."""

    def test_ens_profile_averages_over_every_scored_case(self):
        """The mean is over all cases, not only over the ones that shed.

        Four ok draws of one design, one of which sheds 10 MWh in a single
        hour: the expected value in that hour is 2.5 MWh, not 10.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        from experiments.ra import evaluate as evaluate_mod

        run_dir = self.tmp / "evalrun"
        (run_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "task_id": f"d1-lp-24-y2020-b00000-d{draw}",
                    "status": "ok",
                    "method": "lp",
                    "block_size": 24,
                    "year": 2020,
                    "block_index": 0,
                    "start": 0,
                    "stop": 24,
                    "hours": 24,
                    "draw": draw,
                    "design_id": "d1",
                    "operational_cost": 1.0,
                    "unserved_energy_mwh": 10.0 if draw == 0 else 0.0,
                    "lost_load_hours": 1 if draw == 0 else 0,
                }
                for draw in range(4)
            ]
        ).to_csv(run_dir / "metrics.csv", index=False)

        profile = pd.DataFrame(
            [
                {
                    "task_id": "d1-lp-24-y2020-b00000-d0",
                    "design_id": "d1",
                    "method": "lp",
                    "block_size": "24",
                    "year": 2020,
                    "draw": 0,
                    "hour": 5,
                    "day_of_year": 1,
                    "hour_of_day": 5,
                    "bus": "z1",
                    "ens_mwh": 10.0,
                }
            ]
        )
        (run_dir / "ens_profile").mkdir(exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(profile, schema=persist.ens_profile_schema(),
                                 preserve_index=False),
            run_dir / "ens_profile" / "d1-lp-24-y2020-b00000-d0.parquet",
        )

        evaluate_mod.write_eval_tables(run_dir)
        out = pd.read_parquet(run_dir / "eval_ens_profile.parquet")
        self.assertEqual(tuple(out.columns), evaluate_mod.EVAL_ENS_PROFILE_COLUMNS)
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(str(row["method"]), "lp")
        self.assertEqual(str(row["block_size"]), "24")
        self.assertAlmostEqual(float(row["mean_ens_mwh"]), 2.5)
        self.assertEqual(int(row["n_draws"]), 4)
        self.assertEqual(int(row["n_year_draws_with_ens"]), 1)

    def test_ens_profile_does_not_mix_block_sizes(self):
        """A 24 h-block score and a 168 h-block score are different dispatches.

        Summing them into one hour would report ENS the system never had, and
        would divide by a case count that collapsed the two series.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        from experiments.ra import evaluate as evaluate_mod

        run_dir = self.tmp / "twoblocks"
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "task_id": f"d1-lp-{size}-y2020-b00000-d{draw}",
                    "status": "ok",
                    "method": "lp",
                    "block_size": size,
                    "year": 2020,
                    "block_index": 0,
                    "start": 0,
                    "stop": 24,
                    "hours": 24,
                    "draw": draw,
                    "design_id": "d1",
                    "operational_cost": 1.0,
                }
                for size in (24, 48)
                for draw in range(2)
            ]
        ).to_csv(run_dir / "metrics.csv", index=False)

        profile = pd.DataFrame(
            [
                {
                    "task_id": f"d1-lp-{size}-y2020-b00000-d0",
                    "design_id": "d1",
                    "method": "lp",
                    "block_size": str(size),
                    "year": 2020,
                    "draw": 0,
                    "hour": 5,
                    "day_of_year": 1,
                    "hour_of_day": 5,
                    "bus": "z1",
                    "ens_mwh": 10.0,
                }
                for size in (24, 48)
            ]
        )
        (run_dir / "ens_profile").mkdir(exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(
                profile, schema=persist.ens_profile_schema(), preserve_index=False
            ),
            run_dir / "ens_profile" / "both.parquet",
        )

        evaluate_mod.write_eval_tables(run_dir)
        out = pd.read_parquet(run_dir / "eval_ens_profile.parquet")
        self.assertEqual(len(out), 2)
        self.assertEqual(sorted(out["block_size"].astype(str)), ["24", "48"])
        # Two cases per (method, block_size), one of which shed 10 MWh.
        for _, row in out.iterrows():
            self.assertAlmostEqual(float(row["mean_ens_mwh"]), 5.0)
            self.assertEqual(int(row["n_draws"]), 2)
            self.assertEqual(int(row["n_year_draws_with_ens"]), 1)

    def test_aggregate_writes_eval_tables_for_an_evaluation_run(self):
        """A sharded campaign only sees the whole ledger at aggregate time."""
        from experiments.ra import evaluate as evaluate_mod
        from experiments.ra.system import Design

        path = self.write_config("evalcampaign")
        cfg = config.load_config(path)
        run_dir = paths.run_dir(identity.run_id(cfg), self.runs_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        cli.touch_run_dir(cfg, run_dir)
        evaluate_mod.evaluate_designs([Design(design_id="d1")], cfg, run_dir)

        eval_path = run_dir / "eval.parquet"
        self.assertTrue(eval_path.exists())
        eval_path.unlink()

        code = cli.main(
            ["aggregate", "--run-id", run_dir.name, "--runs-root", str(self.runs_root)]
        )
        self.assertEqual(code, 0)
        self.assertTrue(eval_path.exists())
        table = pd.read_parquet(eval_path)
        self.assertEqual(tuple(table.columns), evaluate_mod.EVAL_COLUMNS)
        self.assertEqual(set(table["design_id"]), {"d1"})

        # A benchmark run (as-built only) gets no eval.parquet.
        plain = self.run_cli(self.write_config("plainbench"))
        self.assertFalse((plain / "eval.parquet").exists())

    def test_planning_run_gets_no_eval_tables(self):
        """A `mode: plan` ledger carries design ids too, and must be excluded.

        A planning task's row is named after the design it *produced*, so a
        design-id test alone fires on every planning run and writes a table
        whose "operational cost" is a planning objective -- which P6 then
        prefers over the design record.
        """
        from experiments.ra import evaluate as evaluate_mod

        path = self.tmp / "planeval.yaml"
        path.write_text(
            yaml.safe_dump(
                config.deep_merge(
                    {
                        "includes": [str(paths.config_root() / "base.yaml")],
                        "name": "planeval",
                        "mode": "plan",
                        "dataset": {
                            "dir": str(self.dataset),
                            "years": [2020],
                            "window": {"start": 0, "stop": 48},
                        },
                        "selection": {"strategy": "all", "block_size": 24},
                        "planning": {
                            "method": "gradient",
                            "dispatch_solver": "HIGHS",
                            "single_level": {"kind": "primal", "solver": "HIGHS"},
                            "optimizer": {"num_iterations": 1},
                            "timeout_s": 900,
                        },
                    },
                    {},
                )
            )
        )
        run_dir = self.run_cli(path)
        self.assertTrue((run_dir / "designs").is_dir())
        self.assertFalse((run_dir / "eval.parquet").exists())

        cfg = config.load_config(path)
        frame = pd.read_csv(run_dir / "metrics.csv")
        # Non-`asbuilt` design ids alone are not enough.
        self.assertNotEqual(set(frame["design_id"]), {"asbuilt"})
        self.assertFalse(evaluate_mod.is_evaluation_run(frame, cfg))
        self.assertFalse(evaluate_mod.is_evaluation_run(frame))  # ledger-only fallback

        self.assertEqual(
            cli.main(["aggregate", "--run-id", run_dir.name, "--runs-root", str(self.runs_root)]),
            0,
        )
        self.assertFalse((run_dir / "eval.parquet").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
