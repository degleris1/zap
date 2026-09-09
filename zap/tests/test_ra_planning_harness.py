"""Harness integration for ``mode: plan`` (WP5 spec section 7.3, Task C).

A planning run goes through the *same* machinery as a dispatch run (D-W1): the
same run id, the same ledger, the same atomic task JSON, the same
``metrics.csv``, the same ``CARD.md``, the same resume rule.  What it adds is
one task per outage draw, a ``designs/<design_id>.json`` artefact, and the
planning sections of the card.

Everything here runs on 48 hours of the WP1 tiny dataset fixture with HiGHS.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

from experiments.ra import cli, config, evaluate, identity, paths
from experiments.ra import tasks as tasks_mod

#: Rows the fixture is made extendable on (it ships an all-frozen fleet), so a
#: planning solve has something to decide.
EXTENDABLE_GENERATORS = {"z1 solar": 500.0, "z2 onwind": 400.0}
EXTENDABLE_STORAGE = {"z1 battery": 200.0}

#: Top-level keys every ``design.json`` must carry (spec section 3.4).
DESIGN_KEYS = (
    "schema_version",
    "design_id",
    "run_id",
    "zap_commit",
    "zap_dirty",
    "created_utc",
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
    "history_path",
)


def _fixture_available() -> bool:
    try:
        import zarr  # noqa: F401

        from zap.importers.wy_store import load_system  # noqa: F401
        from zap.tests.fixtures.tiny_dataset import write_tiny_dataset  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means the fixture is unusable
        return False
    return True


def write_config(path: Path, cfg: dict) -> Path:
    cfg = config.deep_merge({"includes": [str(paths.config_root() / "base.yaml")]}, cfg)
    path.write_text(yaml.safe_dump(cfg))
    return path


class PlanHarnessMixin(unittest.TestCase):
    """A converted 48-hour tiny dataset with a few extendable rows."""

    @classmethod
    def setUpClass(cls):
        if not _fixture_available():  # pragma: no cover
            raise unittest.SkipTest("WP1 (wy_store + tiny dataset fixture) is not available")
        from zap.importers.wy_store import convert_dataset
        from zap.tests.fixtures.tiny_dataset import write_tiny_dataset

        cls.class_tmp = Path(tempfile.mkdtemp(prefix="ra-plan-harness-"))
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
        self.tmp = Path(tempfile.mkdtemp(prefix="ra-plan-run-"))
        self.runs_root = self.tmp / "runs"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- configs ----------------------------------------------------------
    def plan_config(self, *, name: str = "plan_smoke", extra: dict | None = None) -> Path:
        cfg = {
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
                "timeout_s": 600,
            },
        }
        if extra:
            cfg = config.deep_merge(cfg, extra)
        return write_config(self.tmp / f"{name}.yaml", cfg)

    def dispatch_config(self, *, name: str = "eval_smoke") -> Path:
        return write_config(
            self.tmp / f"{name}.yaml",
            {
                "name": name,
                "mode": "dispatch",
                "dataset": {
                    "dir": str(self.dataset),
                    "years": [2020],
                    "window": {"start": 0, "stop": 24},
                },
                "selection": {"blocks": [24], "reference": "none"},
                "methods": {"lp": {"enabled": True, "solver": "HIGHS", "timeout_s": 600}},
            },
        )

    def run_dir_of(self, path: Path) -> Path:
        return paths.run_dir(identity.run_id(config.load_config(path)), self.runs_root)

    def run_plan(self, path: Path, *, argv_extra=()) -> Path:
        code = cli.main(
            ["run", "--config", str(path), "--runs-root", str(self.runs_root), *argv_extra]
        )
        self.assertEqual(code, 0)
        return self.run_dir_of(path)


class TestPlanTaskEnumeration(PlanHarnessMixin):
    """Section 7.3 test 1."""

    def test_plan_mode_task_enumeration(self):
        cfg = config.load_config(self.plan_config(extra={"heuristics": {"outage_draws": [0, 1]}}))
        plan_tasks = tasks_mod.enumerate_tasks(cfg)
        self.assertEqual(
            [t.task_id for t in plan_tasks],
            ["plan-monolithic-all-b24-s42-d0", "plan-monolithic-all-b24-s42-d1"],
        )
        for task in plan_tasks:
            self.assertEqual(task.method, tasks_mod.PLAN_METHOD)
            self.assertEqual(task.design_id, task.task_id)  # 3.4
            self.assertEqual((task.block.start, task.block.stop), (0, 48))

        # Deterministic across invocations.
        self.assertEqual(
            [t.task_id for t in tasks_mod.enumerate_tasks(cfg)],
            [t.task_id for t in plan_tasks],
        )

        # One task without draws; the monolithic sentinel is named "bfull".
        single = config.load_config(self.plan_config(name="p1"))
        self.assertEqual(
            [t.task_id for t in tasks_mod.enumerate_tasks(single)],
            ["plan-monolithic-all-b24-s42"],
        )
        full = config.load_config(
            self.plan_config(name="p2", extra={"selection": {"block_size": None}})
        )
        self.assertEqual(
            [t.task_id for t in tasks_mod.enumerate_tasks(full)],
            ["plan-monolithic-all-bfull-s42"],
        )

    def test_dispatch_enumeration_is_unaffected(self):
        # WP4's shipped benchmark still enumerates 364 + 52 + 1 tasks.
        cfg = config.load_config(paths.config_root() / "experiments" / "op_benchmark_z4_2020.yaml")
        counts: dict[str, int] = {}
        for task in tasks_mod.enumerate_tasks(cfg):
            counts[str(task.block_size)] = counts.get(str(task.block_size), 0) + 1
        self.assertEqual(counts, {"24": 364, "168": 52, "reference": 1})

    def test_shipped_plan_configs_enumerate_one_task(self):
        for name in (
            "plan_z4_smoke_48h",
            "plan_z4_smoke_48h_gradient",
            "plan_z4_2020_monolithic",
            "plan_z4_2020_stochastic_weeks",
            "plan_z4_2020_gradient_weeks",
            "plan_z4_2020_gradient_weeks_emissions",
        ):
            with self.subTest(config=name):
                cfg = config.load_config(paths.config_root() / "experiments" / f"{name}.yaml")
                self.assertEqual(cfg["mode"], "plan")
                self.assertEqual(len(tasks_mod.enumerate_tasks(cfg)), 1)


class TestPlanEndToEnd(PlanHarnessMixin):
    """Sections 7.3 tests 2, 3, 4 and 5."""

    def test_plan_end_to_end_two_days_highs(self):
        path = self.plan_config()
        run_dir = self.run_plan(path)

        for artefact in ("config.resolved.yaml", "env.json", "metrics.csv", "CARD.md"):
            self.assertTrue((run_dir / artefact).exists(), artefact)

        (task_file,) = sorted((run_dir / "tasks").glob("*.json"))
        record = json.loads(task_file.read_text())
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["task_id"], "plan-monolithic-all-b24-s42")
        self.assertEqual(record["method"], "plan")
        self.assertEqual(record["design_path"], "designs/plan-monolithic-all-b24-s42.json")

        design_path = run_dir / "designs" / "plan-monolithic-all-b24-s42.json"
        self.assertTrue(design_path.exists())
        design = json.loads(design_path.read_text())
        for key in DESIGN_KEYS:
            self.assertIn(key, design)
        self.assertEqual(design["schema_version"], 1)
        self.assertEqual(design["design_id"], "plan-monolithic-all-b24-s42")
        self.assertEqual(design["run_id"], identity.run_id(config.load_config(path)))
        self.assertEqual(design["method"], "single_level")
        self.assertEqual(design["preset"], "monolithic")
        self.assertEqual(design["kind"], "primal")
        self.assertEqual(design["window"], {"start": 0, "stop": 48})
        self.assertEqual(design["solver"]["name"], "HIGHS")
        self.assertEqual(design["solver"]["status"], "optimal")

        # Section 6: the block sample covers the horizon exactly here.
        ann = design["annualization"]
        self.assertEqual(ann["total_hours"], 48)
        self.assertEqual(ann["sampled_hours"], 48)
        self.assertAlmostEqual(ann["coverage"], 1.0)
        self.assertAlmostEqual(ann["annualization_factor"], 1.0)
        self.assertTrue(ann["capital_cost_prorated"])
        self.assertEqual(design["selection"]["blocks"], [[0, 24], [24, 48]])
        self.assertAlmostEqual(design["objective"]["annual"], design["objective"]["raw"])

        # Every capacity is inside its recorded bounds.
        for param, (device_index, attr) in design["parameter_names"].items():
            del device_index
            bounds = design["bounds"][param]
            values = next(
                entry[attr]
                for entry in design["capacities"].values()
                if attr in entry and len(entry[attr]) == len(bounds["lower"])
            )
            for lo, hi, value in zip(bounds["lower"], bounds["upper"], values):
                self.assertGreaterEqual(value, lo - 1e-9)
                self.assertLessEqual(value, hi + 1e-9)

        frame = pd.read_csv(run_dir / "metrics.csv")
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame["status"].iloc[0], "ok")
        self.assertTrue(pd.notna(frame["objective_annual"].iloc[0]))
        self.assertAlmostEqual(
            float(frame["objective_annual"].iloc[0]), design["objective"]["annual"]
        )
        self.assertAlmostEqual(float(frame["annualization_factor"].iloc[0]), 1.0)
        # `capacity_added_mw` needs the as-built capacities, which reach
        # `metrics.py` only through `ctx.meta["initial_parameters"]`.
        self.assertIn("capacity_added_mw", frame.columns)
        self.assertTrue(pd.notna(frame["capacity_added_mw"].iloc[0]))
        self.assertGreaterEqual(float(frame["capacity_added_mw"].iloc[0]), 0.0)

        card = (run_dir / "CARD.md").read_text()
        for heading in (
            "## Planning method",
            "## Period selection",
            "## Annualization",
            "## Objective",
            "## Emissions",
            "## Capacity bounds and floors",
            "## Designed capacity by carrier (MW)",
            "## Designs",
        ):
            self.assertIn(heading, card)
        # The annualization block states each quantity by name.
        for key in (
            "total_hours",
            "sampled_hours",
            "coverage",
            "annualization_factor",
            "year_factor",
            "snapshot_weight",
            "capital_cost_prorated",
        ):
            self.assertIn(key, card)
        self.assertIn("min_capacity_mw", card)
        self.assertIn("min_storage_mw", card)
        self.assertIn("rows_raised_by_floor", card)
        # The capacity-by-carrier table names carriers from the fixture.
        self.assertIn("| solar |", card)
        self.assertIn("**total**", card)

    def test_plan_run_resumes(self):
        """A finished plan task is skipped; `--set execution.force=true` re-runs it.

        The planning solve runs in a worker process (that is how
        `planning.timeout_s` is enforced), so "was it re-solved?" is answered by
        the artefacts rather than by monkeypatching the method in this process.
        """
        path = self.plan_config()
        run_dir = self.run_plan(path)
        task_file = run_dir / "tasks" / "plan-monolithic-all-b24-s42.json"
        design_file = run_dir / "designs" / "plan-monolithic-all-b24-s42.json"
        mtime = task_file.stat().st_mtime_ns
        design_created = json.loads(design_file.read_text())["created_utc"]

        self.run_plan(path)
        self.assertEqual(task_file.stat().st_mtime_ns, mtime)  # nothing re-solved
        self.assertEqual(json.loads(design_file.read_text())["created_utc"], design_created)

        self.run_plan(path, argv_extra=("--set", "execution.force=true"))
        self.assertNotEqual(task_file.stat().st_mtime_ns, mtime)
        self.assertNotEqual(json.loads(design_file.read_text())["created_utc"], design_created)

    def test_plan_task_failure_is_recorded(self):
        """A solve that raises inside the worker is recorded, and fails the run.

        The failure is injected through the config (an uninstalled solver) rather
        than by patching the method: the solve runs in another process.
        """
        path = self.plan_config(
            name="plan_boom", extra={"planning": {"single_level": {"solver": "MOSEK"}}}
        )
        code = cli.main(["run", "--config", str(path), "--runs-root", str(self.runs_root)])

        self.assertEqual(code, 1)  # planning is `required: true`
        run_dir = self.run_dir_of(path)
        (task_file,) = sorted((run_dir / "tasks").glob("*.json"))
        record = json.loads(task_file.read_text())
        self.assertEqual(record["status"], "failed")
        self.assertIn("MOSEK", record["error"])
        self.assertIn("Traceback", record["traceback"])
        self.assertIsNone(record["design_path"])
        self.assertFalse((run_dir / "designs").exists())

    def test_plan_timeout_is_enforced(self):
        """`planning.timeout_s` kills the solve; it is not a post-hoc label.

        With a 1 ms budget the worker cannot even import the planning core, so
        the task is recorded `timeout` with no design written. A timeout is not a
        required failure (spec R2), so the run still exits 0.
        """
        path = self.plan_config(name="plan_slow", extra={"planning": {"timeout_s": 0.001}})
        code = cli.main(["run", "--config", str(path), "--runs-root", str(self.runs_root)])
        self.assertEqual(code, 0)

        run_dir = self.run_dir_of(path)
        (task_file,) = sorted((run_dir / "tasks").glob("*.json"))
        record = json.loads(task_file.read_text())
        self.assertEqual(record["status"], "timeout")
        self.assertIn("timeout_s=0.001", record["error"])
        self.assertIsNone(record["design_path"])
        self.assertFalse((run_dir / "designs").exists())

    def test_plan_then_evaluate(self):
        """The evaluation seam: a design.json is scored by the dispatch path."""
        plan_dir = self.run_plan(self.plan_config(name="plan_for_eval"))

        designs = evaluate.designs_from_run(plan_dir)
        self.assertEqual(len(designs), 1)
        self.assertEqual(designs[0].design_id, "plan-monolithic-all-b24-s42")
        self.assertTrue(designs[0].capacities)

        eval_cfg = config.load_config(self.dispatch_config())
        eval_dir = self.runs_root / "evaluation"
        frame = evaluate.evaluate_designs(designs, eval_cfg, eval_dir)

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame["status"].iloc[0], "ok")
        self.assertEqual(frame["design_id"].iloc[0], "plan-monolithic-all-b24-s42")
        self.assertTrue(pd.notna(frame["operational_cost"].iloc[0]))
        self.assertTrue(
            float(frame["operational_cost"].iloc[0]) == float(frame["operational_cost"].iloc[0])
        )


class TestPlanIdentityAndModes(PlanHarnessMixin):
    """Sections 7.3 tests 6 and 7."""

    def test_run_id_includes_planning_config(self):
        base = config.load_config(self.plan_config())
        base_id = identity.run_id(base)

        step = config.load_config(
            self.plan_config(
                name="plan_smoke", extra={"planning": {"optimizer": {"step_size": 5.0e-3}}}
            )
        )
        self.assertNotEqual(identity.run_id(step), base_id)

        seed = config.load_config(
            self.plan_config(name="plan_smoke", extra={"selection": {"seed": 7}})
        )
        self.assertNotEqual(identity.run_id(seed), base_id)

        emissions = config.load_config(
            self.plan_config(
                name="plan_smoke",
                extra={"planning": {"emissions": {"mode": "price", "price": 50.0}}},
            )
        )
        self.assertNotEqual(identity.run_id(emissions), base_id)

        # Sharding and the runs directory are not part of the identity.
        shard = config.load_config(self.plan_config())
        shard["execution"]["shard"] = "1/4"
        shard["output"]["runs_root"] = "/tmp/somewhere-else"
        self.assertEqual(identity.run_id(shard), base_id)

    def test_plan_mode_warns_about_dispatch_only_keys(self):
        # R-W9: `mode: plan` ignores selection.blocks / reference /
        # reference_window. Setting them is accepted, and logged as ignored.
        path = self.plan_config(
            name="plan_dispatch_keys",
            extra={"selection": {"reference": "none", "blocks": [168]}},
        )
        run_dir = self.run_plan(path)
        log = (run_dir / "log.txt").read_text()
        self.assertIn("ignores these selection keys", log)
        self.assertIn("reference", log)
        self.assertIn("reads selection keys", log)
        self.assertIn("strategy", log)

    def test_dispatch_mode_warns_about_plan_only_keys(self):
        path = write_config(
            self.tmp / "dispatch_plan_keys.yaml",
            {
                "name": "dispatch_plan_keys",
                "dataset": {
                    "dir": str(self.dataset),
                    "years": [2020],
                    "window": {"start": 0, "stop": 24},
                },
                "selection": {
                    "blocks": [24],
                    "reference": "none",
                    "strategy": "random",
                    "seed": 3,
                    "num_blocks": 2,
                },
                "methods": {"lp": {"enabled": True, "solver": "STUB"}},
            },
        )
        code = cli.main(["run", "--config", str(path), "--runs-root", str(self.runs_root)])
        self.assertEqual(code, 0)
        log = (self.run_dir_of(path) / "log.txt").read_text()
        self.assertIn("ignores these selection keys", log)
        self.assertIn("strategy", log)


class TestPlanConfigSurface(unittest.TestCase):
    """The ``planning`` / ``selection`` / ``heuristics`` key space (section 8.1)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ra-plan-cfg-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _cfg(self, extra: dict) -> Path:
        return write_config(self.tmp / "cfg.yaml", config.deep_merge({"mode": "plan"}, extra))

    def test_base_yaml_carries_the_planning_key_space(self):
        base = config.base_config()
        self.assertEqual(base["mode"], "dispatch")
        self.assertEqual(base["heuristics"]["name"], "none")
        for key in ("strategy", "block_size", "num_blocks", "avoid_year_boundaries", "seed"):
            self.assertIn(key, base["selection"])
        plan = base["planning"]
        for key in (
            "method",
            "dispatch_solver",
            "dispatch_solver_kwargs",
            "regularize",
            "num_workers",
            "timeout_s",
            "required",
            "bounds",
            "expansion",
            "single_level",
            "warm_start",
            "optimizer",
            "admm",
            "emissions",
            "budget_constraints",
        ):
            self.assertIn(key, plan)
        self.assertEqual(plan["bounds"], {"min_capacity_mw": 0.1, "min_storage_mw": 10.0})
        self.assertEqual(plan["emissions"]["mode"], "none")
        self.assertEqual(plan["emissions"]["cap_basis"], "annual")

    def test_planning_defaults_match_the_core(self):
        from experiments.ra.planning import PLANNING_DEFAULTS, SELECTION_DEFAULTS

        base = config.base_config()
        for key, value in SELECTION_DEFAULTS.items():
            self.assertEqual(base["selection"][key], value, key)
        for key, value in PLANNING_DEFAULTS.items():
            self.assertEqual(base["planning"][key], value, key)

    def test_nested_unknown_planning_key_rejected(self):
        with self.assertRaises(config.ConfigError):
            config.load_config(self._cfg({"planning": {"single_level": {"kindd": "primal"}}}))
        with self.assertRaises(config.ConfigError):
            config.load_config(self._cfg({"planning": {"optimiser": {}}}))

    def test_solver_kwargs_are_opaque(self):
        cfg = config.load_config(
            self._cfg(
                {
                    "planning": {
                        "single_level": {"solver_kwargs": {"Method": 2, "Crossover": 0}},
                        "dispatch_solver_kwargs": {"anything": True},
                        "warm_start": {"solver_kwargs": {"whatever": 1}},
                        "admm": {"solver_kwargs": {"num_iterations": 10}},
                    }
                }
            )
        )
        self.assertEqual(cfg["planning"]["single_level"]["solver_kwargs"]["Method"], 2)
        # solver_kwargs blocks deep-merge onto their defaults, unvalidated.
        self.assertEqual(
            cfg["planning"]["dispatch_solver_kwargs"], {"verbose": False, "anything": True}
        )
        self.assertEqual(cfg["planning"]["warm_start"]["solver_kwargs"], {"whatever": 1})

    def test_invalid_enumerations_rejected(self):
        for extra in (
            {"mode": "planning"},
            {"planning": {"method": "sgd"}},
            {"planning": {"single_level": {"kind": "dual"}}},
            {"planning": {"expansion": {"mode": "guess"}}},
            {"planning": {"emissions": {"mode": "tax"}}},
            {"planning": {"emissions": {"cap_basis": "monthly"}}},
            {"planning": {"optimizer": {"batch_strategy": "spiral"}}},
            {"planning": {"admm": {"machine": "tpu"}}},
            {"selection": {"strategy": "kmeans"}},
            {"heuristics": {"name": "prm"}},
        ):
            with self.subTest(extra=extra), self.assertRaises(config.ConfigError):
                config.load_config(self._cfg(extra))

    def test_emissions_matrix_is_enforced_at_config_time(self):
        # D-W5: a hard cap is single-level primal only; dual ascent is gradient
        # or admm only. Both raise at validation, not at solve time.
        with self.assertRaises(config.ConfigError):
            config.load_config(
                self._cfg(
                    {"planning": {"method": "gradient", "emissions": {"mode": "cap", "cap": 1.0}}}
                )
            )
        with self.assertRaises(config.ConfigError):
            config.load_config(
                self._cfg(
                    {
                        "planning": {
                            "method": "relaxed",
                            "single_level": {"kind": "strong_duality"},
                            "emissions": {"mode": "cap", "cap": 1.0},
                        }
                    }
                )
            )
        with self.assertRaises(config.ConfigError):
            config.load_config(
                self._cfg(
                    {
                        "planning": {
                            "method": "monolithic",
                            "emissions": {"mode": "dual_ascent", "dual_ascent": {"target": 1.0}},
                        }
                    }
                )
            )
        # The valid corner is accepted.
        cfg = config.load_config(
            self._cfg(
                {"planning": {"method": "monolithic", "emissions": {"mode": "cap", "cap": 5.0}}}
            )
        )
        self.assertEqual(cfg["planning"]["emissions"]["cap"], 5.0)

    def test_mps_float64_rejected(self):
        with self.assertRaises(config.ConfigError):
            config.load_config(
                self._cfg({"planning": {"admm": {"machine": "mps", "dtype": "float64"}}})
            )

    def test_elcc_prm_axis_file_is_rejected(self):
        path = self.tmp / "elcc.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "includes": [
                        str(paths.config_root() / "base.yaml"),
                        str(paths.config_root() / "heuristics" / "elcc_prm.yaml"),
                    ]
                }
            )
        )
        with self.assertRaises(config.ConfigError):
            config.load_config(path)

    def test_strong_duality_axis_file_is_rejected(self):
        """`kind: strong_duality` is unrunnable and is rejected like `elcc_prm`.

        `RelaxedPlanningProblem` dualizes the device list through
        `zap.dual.dualize`, whose DUAL_CLASS table has no entry for WP3's
        `DirectedLine`, so every system this harness builds raises
        `KeyError: DirectedLine` deep inside zap. The preset file is kept, with a
        comment; loading it is a ConfigError that names the missing dual.
        """
        for extra in (
            {"planning": {"single_level": {"kind": "strong_duality"}}},
            {"planning": {"method": "relaxed", "single_level": {"kind": "strong_duality"}}},
        ):
            with self.subTest(extra=extra), self.assertRaises(config.ConfigError) as ctx:
                config.load_config(self._cfg(extra))
            self.assertIn("DirectedLine", str(ctx.exception))

        # ... including through the shipped preset file.
        path = self.tmp / "relaxed.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "includes": [
                        str(paths.config_root() / "base.yaml"),
                        str(paths.config_root() / "methods" / "plan_relaxed.yaml"),
                    ]
                }
            )
        )
        with self.assertRaises(config.ConfigError) as ctx:
            config.load_config(path)
        self.assertIn("DirectedLine", str(ctx.exception))

    def test_deprecated_optimizer_key_is_rejected(self):
        """`eval_final_full_loss` no longer does anything and says so.

        The planning objective is always the full forward pass at the final
        parameters; the key survives only so the key space is stable, and setting
        it is an error rather than a silent no-op.
        """
        cfg = config.load_config(self._cfg({"planning": {"optimizer": {}}}))
        self.assertFalse(cfg["planning"]["optimizer"]["eval_final_full_loss"])
        with self.assertRaises(config.ConfigError) as ctx:
            config.load_config(
                self._cfg({"planning": {"optimizer": {"eval_final_full_loss": True}}})
            )
        self.assertIn("deprecated", str(ctx.exception))

    def test_carbon_tax_and_emissions_price_are_exclusive(self):
        """Two ways to charge carbon at once is a double count, not a config."""
        for emissions in (
            {"mode": "price", "price": 50.0},
            {"mode": "dual_ascent", "dual_ascent": {"target": 1.0}},
        ):
            extra = {
                "system": {"carbon_tax": 10.0},
                "planning": {"method": "gradient", "emissions": emissions},
            }
            with self.subTest(mode=emissions["mode"]), self.assertRaises(config.ConfigError) as ctx:
                config.load_config(self._cfg(extra))
            self.assertIn("carbon_tax", str(ctx.exception))

        # Either one alone is fine.
        config.load_config(self._cfg({"system": {"carbon_tax": 10.0}}))
        config.load_config(self._cfg({"planning": {"emissions": {"mode": "price", "price": 50.0}}}))

    def test_shipped_axis_files_load(self):
        root = paths.config_root()
        for axis, name in [
            ("methods", "plan_monolithic"),
            ("methods", "plan_stochastic"),
            # methods/plan_relaxed.yaml is deliberately *not* loadable: see
            # test_strong_duality_axis_file_is_rejected.
            ("methods", "plan_gradient"),
            ("methods", "plan_admm"),
            ("selection", "full_horizon"),
            ("selection", "all_weeks"),
            ("selection", "all_days"),
            ("selection", "uniform_weeks"),
            ("selection", "random_weeks"),
            ("selection", "stratified_weeks"),
            ("selection", "kmedoids_weeks"),
            ("heuristics", "none"),
            ("heuristics", "ucap"),
            ("heuristics", "outage_draw0"),
            ("emissions", "none"),
            ("emissions", "price_50"),
        ]:
            with self.subTest(axis=axis, name=name):
                path = self.tmp / f"{axis}_{name}.yaml"
                path.write_text(
                    yaml.safe_dump(
                        {
                            "includes": [
                                str(root / "base.yaml"),
                                str(root / "methods" / "plan_stochastic.yaml"),
                                str(root / axis / f"{name}.yaml"),
                            ]
                        }
                    )
                )
                config.load_config(path)

        # The emissions axis files that constrain the method compose with it.
        for method, emissions in (
            ("plan_monolithic", "cap_annual"),
            ("plan_gradient", "dual_ascent"),
        ):
            with self.subTest(method=method, emissions=emissions):
                path = self.tmp / f"{method}_{emissions}.yaml"
                path.write_text(
                    yaml.safe_dump(
                        {
                            "includes": [
                                str(root / "base.yaml"),
                                str(root / "methods" / f"{method}.yaml"),
                                str(root / "emissions" / f"{emissions}.yaml"),
                            ]
                        }
                    )
                )
                config.load_config(path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
