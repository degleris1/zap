"""Tests for the RA experiment harness (`experiments/ra`, spec section 5.10).

Everything here runs on tiny horizons.  The pipeline tests use the ``STUB``
solver, which returns deterministic fake metrics without touching data or a
solver, so config -> run id -> tasks -> ledger -> metrics.csv -> CARD.md ->
resume is exercised end to end without WP1's weather store.  The tests that
need a real system are skipped until WP1's `zap.importers.wy_store` and its
tiny dataset fixture exist.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from zap.devices import Generator, Load, StorageUnit
from zap.devices.storage_unit import StorageUnitVariable
from zap.devices.transporter import DirectedLine
from zap.network import DispatchOutcome, PowerNetwork

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

from experiments.ra import blocks as blocks_mod
from experiments.ra import cli, config, dispatch, identity, metrics, paths, runcard
from experiments.ra import system as system_mod
from experiments.ra import tasks as tasks_mod

WP1_AVAILABLE = system_mod.WP1_AVAILABLE


def tiny_config(
    root: Path,
    *,
    name: str = "unit_test",
    window=(0, 48),
    block_sizes=(24,),
    reference="window",
    reference_window=None,
    extra: dict | None = None,
) -> Path:
    """Write a stub-solver config into ``root`` and return its path."""
    if reference_window is None:  # default: the whole window
        reference_window = (window[0], window[1] - window[0])
    cfg = {
        "includes": ["base.yaml"],
        "name": name,
        "dataset": {"years": [2020], "window": {"start": window[0], "stop": window[1]}},
        "selection": {
            "blocks": list(block_sizes),
            "reference": reference,
            "reference_window": {"start": reference_window[0], "hours": reference_window[1]},
        },
        "methods": {"lp": {"enabled": True, "solver": "STUB", "timeout_s": 60}},
    }
    if extra:
        cfg = config.deep_merge(cfg, extra)
    path = root / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


class TempRunMixin(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ra_test_"))
        self.runs_root = self.tmp / "runs"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestConfig(TempRunMixin):
    def test_include_expansion_and_override_precedence(self):
        # A three-level include chain: leaf <- middle <- top, plus --set.
        leaf = self.tmp / "leaf.yaml"
        middle = self.tmp / "middle.yaml"
        top = self.tmp / "top.yaml"
        with open(leaf, "w") as f:
            yaml.safe_dump({"system": {"voll": 1.0, "carbon_tax": 7.0}, "name": "leaf"}, f)
        with open(middle, "w") as f:
            yaml.safe_dump({"includes": ["leaf.yaml"], "system": {"voll": 2.0}, "name": "mid"}, f)
        with open(top, "w") as f:
            yaml.safe_dump({"includes": ["middle.yaml"], "system": {"voll": 3.0}, "name": "top"}, f)

        cfg = config.load_config(top, ["system.voll=4.0", "dataset.years=[2001, 2002]"])
        self.assertNotIn("includes", cfg)
        self.assertEqual(cfg["system"]["voll"], 4.0)  # --set wins
        self.assertEqual(cfg["system"]["carbon_tax"], 7.0)  # inherited from the leaf
        self.assertEqual(cfg["name"], "top")  # the file's own key wins over its includes
        self.assertEqual(cfg["dataset"]["years"], [2001, 2002])
        # Keys not mentioned anywhere come from base.yaml.
        self.assertEqual(cfg["system"]["export_mode"], "sink")

        cfg_mid = config.load_config(middle)
        self.assertEqual(cfg_mid["system"]["voll"], 2.0)
        self.assertEqual(cfg_mid["name"], "mid")

    def test_unknown_key_rejected(self):
        path = self.tmp / "bad.yaml"
        with open(path, "w") as f:
            yaml.safe_dump({"includes": ["base.yaml"], "systemm": {"voll": 1.0}}, f)
        with self.assertRaises(config.ConfigError):
            config.load_config(path)

        nested = self.tmp / "bad_nested.yaml"
        with open(nested, "w") as f:
            yaml.safe_dump({"includes": ["base.yaml"], "system": {"vol": 1.0}}, f)
        with self.assertRaises(config.ConfigError):
            config.load_config(nested)

    def test_solver_kwargs_are_opaque(self):
        path = self.tmp / "kwargs.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(
                {"includes": ["base.yaml"], "methods": {"lp": {"solver_kwargs": {"Threads": 4}}}}, f
            )
        cfg = config.load_config(path)
        self.assertEqual(cfg["methods"]["lp"]["solver_kwargs"], {"Threads": 4})

    def test_circular_include_raises(self):
        a, b = self.tmp / "a.yaml", self.tmp / "b.yaml"
        with open(a, "w") as f:
            yaml.safe_dump({"includes": ["b.yaml"]}, f)
        with open(b, "w") as f:
            yaml.safe_dump({"includes": ["a.yaml"]}, f)
        with self.assertRaises(config.ConfigError):
            config.load_config(a)

    def test_reference_window_must_align_to_every_block_size(self):
        # 5088 = 30*168 + 48 is a multiple of 24 but not of 168: the 168 h blocks
        # would cover only 504 h of the 672 h reference window.
        misaligned = tiny_config(
            self.tmp,
            name="misaligned",
            window=(0, 8760),
            block_sizes=(24, 168),
            reference="window",
            reference_window=(5088, 672),
        )
        with self.assertRaises(config.ConfigError) as ctx:
            config.load_config(misaligned)
        self.assertIn("not aligned to block size 168", str(ctx.exception))

        aligned = tiny_config(
            self.tmp,
            name="aligned",
            window=(0, 8760),
            block_sizes=(24, 168),
            reference="window",
            reference_window=(5040, 672),
        )
        cfg = config.load_config(aligned)
        self.assertEqual(cfg["selection"]["reference_window"], {"start": 5040, "hours": 672})

    def test_shipped_configs_are_valid(self):
        for path in sorted((paths.config_root() / "experiments").glob("*.yaml")):
            with self.subTest(config=path.name):
                cfg = config.load_config(path)
                self.assertIn("name", cfg)


class TestIdentity(TempRunMixin):
    def test_run_id_is_deterministic_and_sensitive(self):
        path = tiny_config(self.tmp)
        cfg_a = config.load_config(path)
        cfg_b = config.load_config(path)
        self.assertEqual(identity.run_id(cfg_a), identity.run_id(cfg_b))
        self.assertTrue(identity.run_id(cfg_a).startswith("unit_test-"))

        cfg_voll = config.load_config(path, ["system.voll=1234.0"])
        self.assertNotEqual(identity.run_id(cfg_a), identity.run_id(cfg_voll))

        cfg_shard = config.load_config(path, ["execution.shard=1/2"])
        cfg_runs = config.load_config(path, ["output.runs_root=/tmp/elsewhere"])
        self.assertEqual(identity.run_id(cfg_a), identity.run_id(cfg_shard))
        self.assertEqual(identity.run_id(cfg_a), identity.run_id(cfg_runs))

    def test_solver_kwargs_change_the_run_id(self):
        path = tiny_config(self.tmp, name="kwargsid")
        base = config.load_config(path)
        changed = config.load_config(path, ["methods.lp.solver_kwargs={'Threads': 4}"])
        self.assertNotEqual(identity.run_id(base), identity.run_id(changed))

    def test_canonical_json_is_sorted_and_compact(self):
        blob = identity.canonical_json({"b": 1, "a": {"d": 2, "c": 3}})
        self.assertEqual(blob, '{"a":{"c":3,"d":2},"b":1}')


class TestBlocks(TempRunMixin):
    def test_make_blocks_partitions_the_window(self):
        cfg = config.load_config(tiny_config(self.tmp, window=(0, 48), block_sizes=(24,)))
        blks = blocks_mod.make_blocks(cfg, 24)
        self.assertEqual([(b.index, b.start, b.stop) for b in blks], [(0, 0, 24), (1, 24, 48)])
        self.assertTrue(all(b.hours == 24 for b in blks))

    def test_make_blocks_drops_the_remainder(self):
        cfg = config.load_config(
            tiny_config(self.tmp, window=(0, 50), block_sizes=(24,), reference="none")
        )
        blks = blocks_mod.make_blocks(cfg, 24)
        self.assertEqual(len(blks), 2)
        self.assertEqual(blks[-1].stop, 48)

    def test_reference_block(self):
        cfg = config.load_config(
            tiny_config(self.tmp, window=(0, 48), reference="window", reference_window=(24, 24))
        )
        (blk,) = blocks_mod.make_blocks(cfg, "reference")
        self.assertEqual((blk.start, blk.stop), (24, 48))

        cfg_full = config.load_config(
            tiny_config(self.tmp, name="fullyear", window=(0, 48), reference="full_year")
        )
        (blk,) = blocks_mod.make_blocks(cfg_full, "reference")
        self.assertEqual((blk.start, blk.stop), (0, 48))

        cfg_none = config.load_config(
            tiny_config(self.tmp, name="noref", window=(0, 48), reference="none")
        )
        self.assertEqual(blocks_mod.make_blocks(cfg_none, "reference"), [])

    def test_block_time_periods_multi_year(self):
        cfg = config.load_config(
            tiny_config(
                self.tmp,
                window=(24, 72),
                reference_window=(24, 48),
                extra={"dataset": {"years": [2020, 2021]}},
            )
        )
        first = blocks_mod.Block(index=0, year=2020, start=24, stop=48)
        second = blocks_mod.Block(index=0, year=2021, start=24, stop=48)
        np.testing.assert_array_equal(dispatch.block_time_periods(cfg, first), np.arange(0, 24))
        np.testing.assert_array_equal(dispatch.block_time_periods(cfg, second), np.arange(48, 72))

    def test_prorate_uses_hours_per_year_not_the_window(self):
        loaded = SimpleNamespace(
            meta={"weather_store_attrs": {"hours_per_year": 8760}},
            devices=[],
            network=None,
        )
        self.assertEqual(dispatch.hours_per_year(loaded), 8760)
        self.assertEqual(dispatch.hours_per_year(SimpleNamespace(meta={})), 8760)
        self.assertEqual(dispatch.hours_per_year(SimpleNamespace(meta={"hours_per_year": 48})), 48)

        class Budgeted:
            def __init__(self):
                self.energy_budget_max = 8760.0

            def sample_time(self, periods, original):
                return self

            @property
            def time_horizon(self):
                return 0

        cfg = config.load_config(
            tiny_config(self.tmp, name="prorate", window=(0, 48), reference="none")
        )
        loaded = SimpleNamespace(
            meta={"weather_store_attrs": {"hours_per_year": 8760}},
            devices=[Budgeted()],
            network=None,
        )
        block = blocks_mod.make_blocks(cfg, 24)[0]
        (device,) = dispatch.slice_devices(loaded, cfg, block)
        # 24 h of a 8760 h year, NOT 24 h of the 48 h window.
        self.assertAlmostEqual(device.energy_budget_max, 8760.0 * 24 / 8760)

    def test_prorate_energy_budgets_is_noop(self):
        class Device:
            def __init__(self):
                self.nominal_capacity = np.ones((2, 1))

        devices = [Device(), Device()]
        out = blocks_mod.prorate_energy_budgets(devices, 24, 8760)
        self.assertIs(out, devices)
        for device in out:
            for attr in blocks_mod.ENERGY_BUDGET_ATTRS:
                self.assertFalse(hasattr(device, attr))

        class Budgeted:
            energy_budget_max = 100.0

        (scaled,) = blocks_mod.prorate_energy_budgets([Budgeted()], 24, 240)
        self.assertAlmostEqual(scaled.energy_budget_max, 10.0)


class TestTasks(TempRunMixin):
    def test_enumerate_tasks_is_deterministic(self):
        cfg = config.load_config(
            tiny_config(
                self.tmp,
                window=(0, 96),
                block_sizes=(24, 48),
                reference="window",
                reference_window=(0, 48),
            )
        )
        first = tasks_mod.enumerate_tasks(cfg)
        second = tasks_mod.enumerate_tasks(cfg)
        self.assertEqual([t.task_id for t in first], [t.task_id for t in second])
        self.assertEqual(len(first), 96 // 24 + 96 // 48 + 1)
        self.assertEqual(first[0].task_id, "asbuilt-lp-24-y2020-b00000")
        self.assertIn("asbuilt-lp-reference-y2020-b00000", [t.task_id for t in first])
        self.assertEqual(len({t.task_id for t in first}), len(first))

    def test_shipped_benchmark_task_counts(self):
        cfg = config.load_config(paths.config_root() / "experiments" / "op_benchmark_z4_2020.yaml")
        all_tasks = tasks_mod.enumerate_tasks(cfg)
        counts = {}
        for task in all_tasks:
            counts[str(task.block_size)] = counts.get(str(task.block_size), 0) + 1
        # dataset.window.stop = 8736 = 52*168 = 364*24, so the full-year
        # reference tiles both block sizes exactly (orchestrator, 2026-09-08).
        self.assertEqual(counts, {"24": 364, "168": 52, "reference": 1})
        self.assertEqual(len(all_tasks), 417)
        self.assertEqual(cfg["dataset"]["window"], {"start": 7, "stop": 8743})
        self.assertEqual(cfg["selection"]["reference"], "full_year")

        # The reference window -- here the whole window -- is covered exactly by
        # both block sizes, with no straddling or dropped blocks.
        bounds = blocks_mod.reference_bounds(cfg)
        self.assertEqual(bounds, (7, 8743))
        for size in cfg["selection"]["blocks"]:
            inside = [
                b
                for b in blocks_mod.make_blocks(cfg, size)
                if blocks_mod.block_is_inside(b, bounds)
            ]
            self.assertEqual(sum(b.hours for b in inside), bounds[1] - bounds[0])

    def test_enumerate_tasks_with_draws(self):
        cfg = config.load_config(
            tiny_config(self.tmp, window=(0, 48), extra={"heuristics": {"outage_draws": [0, 3]}})
        )
        ids = [t.task_id for t in tasks_mod.enumerate_tasks(cfg)]
        self.assertIn("asbuilt-lp-24-y2020-b00000-d0", ids)
        self.assertIn("asbuilt-lp-24-y2020-b00001-d3", ids)

    def test_shards_partition_tasks(self):
        cfg = config.load_config(tiny_config(self.tmp, window=(0, 240), block_sizes=(24,)))
        all_tasks = tasks_mod.enumerate_tasks(cfg)
        union = []
        for k in (1, 2, 3):
            union.extend(tasks_mod.select_shard(all_tasks, f"{k}/3"))
        self.assertEqual([t.task_id for t in union], [t.task_id for t in all_tasks])
        sizes = [len(tasks_mod.select_shard(all_tasks, f"{k}/3")) for k in (1, 2, 3)]
        self.assertEqual(sum(sizes), len(all_tasks))
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_atomic_task_write(self):
        path = self.tmp / "tasks" / "x.json"
        real_replace = os.replace

        def boom(src, dst):
            raise OSError("disk full")

        os.replace = boom
        try:
            with self.assertRaises(OSError):
                tasks_mod.write_json_atomic(path, {"a": 1})
        finally:
            os.replace = real_replace

        self.assertFalse(path.exists())
        self.assertFalse(path.with_suffix(".json.tmp").exists())

        # Unserializable payloads are stringified rather than half-written.
        tasks_mod.write_json_atomic(path, {"a": np.float64(1.5)})
        self.assertTrue(path.exists())

    def test_failed_task_is_recorded_not_raised(self):
        cfg = config.load_config(tiny_config(self.tmp, window=(0, 24)))
        task = tasks_mod.enumerate_tasks(cfg)[0]
        real = dispatch.solve_block

        def raiser(*args, **kwargs):
            raise RuntimeError("solver exploded")

        dispatch.solve_block = raiser
        try:
            record = tasks_mod.run_task(task, cfg, self.tmp / "run")
        finally:
            dispatch.solve_block = real

        self.assertEqual(record["status"], "failed")
        self.assertIn("solver exploded", record["error"])
        self.assertIn("Traceback", record["traceback"])
        self.assertTrue(tasks_mod.task_path(self.tmp / "run", task).exists())


class TestPipeline(TempRunMixin):
    def _run(self, config_path, extra_args=()):
        return cli.main(
            ["run", "--config", str(config_path), "--runs-root", str(self.runs_root), *extra_args]
        )

    def test_end_to_end_two_days_stub(self):
        path = tiny_config(self.tmp, window=(0, 48), block_sizes=(24,), reference_window=(0, 48))
        cfg = config.load_config(path)
        self.assertEqual(self._run(path), 0)

        run_dir = paths.run_dir(identity.run_id(cfg), self.runs_root)
        self.assertTrue((run_dir / "config.resolved.yaml").exists())
        self.assertTrue((run_dir / "env.json").exists())
        self.assertTrue((run_dir / "log.txt").exists())

        frame = pd.read_csv(run_dir / "metrics.csv")
        self.assertEqual(len(frame), 3)  # two 24 h blocks + one reference
        self.assertTrue((frame["status"] == "ok").all())

        card = (run_dir / "CARD.md").read_text()
        self.assertIn("Demand scaling", card)
        self.assertIn("Not exercised", card)
        self.assertIn(identity.run_id(cfg), card)

        blocked = frame[frame["block_size"] != "reference"]["operational_cost"].sum()
        reference = frame[frame["block_size"] == "reference"]["operational_cost"].sum()
        self.assertGreaterEqual(blocked, reference - 1e-6)

    def test_resume_skips_completed(self):
        path = tiny_config(
            self.tmp, name="resume", window=(0, 96), block_sizes=(24,), reference_window=(0, 48)
        )
        cfg = config.load_config(path)
        run_dir = paths.run_dir(identity.run_id(cfg), self.runs_root)

        calls = {"n": 0}
        real = dispatch.solve_block

        def counting(task, cfg_, design=None, **kwargs):
            calls["n"] += 1
            return real(task, cfg_, design=design, **kwargs)

        dispatch.solve_block = counting
        try:
            self.assertEqual(self._run(path, ["--shard", "1/2"]), 0)
            first_calls = calls["n"]
            done = sorted((run_dir / "tasks").glob("*.json"))
            self.assertTrue(done)
            mtimes = {p.name: p.stat().st_mtime_ns for p in done}

            self.assertEqual(self._run(path), 0)
            total_tasks = len(tasks_mod.enumerate_tasks(cfg))
            self.assertEqual(calls["n"], total_tasks)  # only the missing tasks were solved
            self.assertEqual(first_calls + (total_tasks - len(done)), calls["n"])
        finally:
            dispatch.solve_block = real

        for name, mtime in mtimes.items():
            self.assertEqual((run_dir / "tasks" / name).stat().st_mtime_ns, mtime)

        frame = pd.read_csv(run_dir / "metrics.csv")
        self.assertEqual(len(frame), len(tasks_mod.enumerate_tasks(cfg)))

    def test_force_reruns_completed(self):
        path = tiny_config(self.tmp, name="forced", window=(0, 24), reference="none")
        self.assertEqual(self._run(path), 0)
        calls = {"n": 0}
        real = dispatch.solve_block

        def counting(task, cfg_, design=None, **kwargs):
            calls["n"] += 1
            return real(task, cfg_, design=design, **kwargs)

        dispatch.solve_block = counting
        try:
            self.assertEqual(self._run(path, ["--force"]), 0)
        finally:
            dispatch.solve_block = real
        self.assertEqual(calls["n"], 1)

    def test_admm_failure_does_not_fail_run(self):
        path = tiny_config(
            self.tmp,
            name="admmfail",
            window=(0, 48),
            block_sizes=(24,),
            reference="none",
            extra={"methods": {"admm": {"enabled": True, "required": False, "timeout_s": 60}}},
        )
        cfg = config.load_config(path)
        real = dispatch.solve_block

        def flaky(task, cfg_, design=None, **kwargs):
            if task.method == "admm":
                raise RuntimeError("ADMM diverged")
            return real(task, cfg_, design=design, **kwargs)

        dispatch.solve_block = flaky
        try:
            self.assertEqual(self._run(path), 0)  # the run still exits 0
        finally:
            dispatch.solve_block = real

        run_dir = paths.run_dir(identity.run_id(cfg), self.runs_root)
        frame = pd.read_csv(run_dir / "metrics.csv")
        admm = frame[frame["method"] == "admm"]
        lp = frame[frame["method"] == "lp"]
        self.assertEqual(len(admm), 2)
        self.assertTrue((admm["status"] == "failed").all())
        self.assertTrue(admm["traceback"].str.contains("ADMM diverged").all())
        self.assertTrue((lp["status"] == "ok").all())

    def test_required_failure_fails_run(self):
        path = tiny_config(self.tmp, name="lpfail", window=(0, 24), reference="none")
        real = dispatch.solve_block

        def raiser(task, cfg_, design=None, **kwargs):
            raise RuntimeError("LP exploded")

        dispatch.solve_block = raiser
        try:
            self.assertEqual(self._run(path), 1)
        finally:
            dispatch.solve_block = real

    def test_run_dir_rejects_a_different_config(self):
        path = tiny_config(self.tmp, name="same", window=(0, 24), reference="none")
        cfg = config.load_config(path)
        run_dir = paths.run_dir(identity.run_id(cfg), self.runs_root)
        self.assertEqual(self._run(path), 0)

        # Hand-edit the stored config so its hash no longer matches.
        stored = config.load_config(run_dir / "config.resolved.yaml")
        stored["system"]["voll"] = 1.0
        config.dump_config(stored, run_dir / "config.resolved.yaml")
        with self.assertRaises(config.ConfigError):
            self._run(path)

    def test_plan_and_show_and_aggregate(self):
        path = tiny_config(
            self.tmp, name="planned", window=(0, 48), block_sizes=(24,), reference_window=(0, 48)
        )
        self.assertEqual(cli.main(["plan", "--config", str(path), "--limit", "-1"]), 0)
        self.assertEqual(self._run(path), 0)
        cfg = config.load_config(path)
        rid = identity.run_id(cfg)
        self.assertEqual(
            cli.main(["aggregate", "--run-id", rid, "--runs-root", str(self.runs_root)]), 0
        )
        self.assertEqual(cli.main(["show", "--run-id", rid, "--runs-root", str(self.runs_root)]), 0)

    def test_only_filters_tasks(self):
        path = tiny_config(
            self.tmp,
            name="onlylp",
            window=(0, 48),
            block_sizes=(24,),
            reference="none",
            extra={"methods": {"admm": {"enabled": True, "solver": "STUB", "timeout_s": 60}}},
        )
        self.assertEqual(self._run(path, ["--only", "method=lp"]), 0)
        cfg = config.load_config(path)
        run_dir = paths.run_dir(identity.run_id(cfg), self.runs_root)
        frame = pd.read_csv(run_dir / "metrics.csv")
        self.assertEqual(set(frame["method"]), {"lp"})

    def test_slow_task_is_marked_timeout(self):
        path = tiny_config(
            self.tmp,
            name="slow",
            window=(0, 24),
            reference="none",
            extra={"methods": {"lp": {"timeout_s": 0.0}}},
        )
        cfg = config.load_config(path)
        task = tasks_mod.enumerate_tasks(cfg)[0]
        record = tasks_mod.run_task(task, cfg, self.tmp / "slowrun")
        self.assertEqual(record["status"], "timeout")

    def test_subprocess_timeout_kills_a_slow_reference_solve(self):
        # A reference task with a deliberately slow stub runs in a child process
        # (the isolation path used by the real reference solve) and is killed.
        path = tiny_config(
            self.tmp,
            name="slowchild",
            window=(0, 24),
            block_sizes=(),
            reference="window",
            reference_window=(0, 24),
            extra={"methods": {"lp": {"timeout_s": 1.0, "solver_kwargs": {"stub_sleep_s": 30.0}}}},
        )
        cfg = config.load_config(path)
        (task,) = tasks_mod.enumerate_tasks(cfg)
        self.assertTrue(tasks_mod._use_subprocess(task, cfg))

        started = time.perf_counter()
        record = tasks_mod.run_task(task, cfg, self.tmp / "slowchild_run")
        elapsed = time.perf_counter() - started

        self.assertEqual(record["status"], "timeout")
        self.assertLess(elapsed, 20.0)  # the child was killed, not waited out
        self.assertIn("timeout_s=1.0", record["error"])

    def test_timeout_does_not_fail_the_run(self):
        path = tiny_config(
            self.tmp,
            name="timeoutok",
            window=(0, 24),
            block_sizes=(24,),
            reference="none",
            extra={"methods": {"lp": {"timeout_s": 0.0}}},
        )
        self.assertEqual(self._run(path), 0)  # a timeout is not a required failure
        cfg = config.load_config(path)
        run_dir = paths.run_dir(identity.run_id(cfg), self.runs_root)
        frame = pd.read_csv(run_dir / "metrics.csv")
        self.assertTrue((frame["status"] == "timeout").all())
        # The late solve's metrics are kept, not discarded.
        self.assertTrue(np.isfinite(frame["operational_cost"]).all())
        self.assertIn("timeout", (run_dir / "CARD.md").read_text())

    def test_stub_flag_overrides_the_solver(self):
        path = tiny_config(
            self.tmp,
            name="stubflag",
            window=(0, 24),
            reference="none",
            extra={"methods": {"lp": {"solver": "HIGHS"}}},
        )
        self.assertEqual(self._run(path, ["--stub"]), 0)
        cfg = config.load_config(path, list(cli.STUB_OVERRIDES))
        run_dir = paths.run_dir(identity.run_id(cfg), self.runs_root)
        frame = pd.read_csv(run_dir / "metrics.csv")
        self.assertTrue((frame["solver_status"] == "stub").all())


class TestMetricsAggregation(TempRunMixin):
    def _frame(self):
        rows = []
        # Reference block: hours [0, 48).
        rows.append(
            {
                "task_id": "asbuilt-lp-reference-y2020-b00000",
                "status": "ok",
                "method": "lp",
                "block_size": "reference",
                "start": 0,
                "stop": 48,
                "operational_cost": 100.0,
                "unserved_energy_mwh": 0.0,
                "wall_clock_s": 1.0,
            }
        )
        # 24 h blocks: two inside the window, one straddling, one outside.
        spans = [(0, 24), (24, 48), (36, 60), (72, 96)]
        for i, (start, stop) in enumerate(spans):
            rows.append(
                {
                    "task_id": f"asbuilt-lp-24-y2020-b{i:05d}",
                    "status": "ok",
                    "method": "lp",
                    "block_size": 24,
                    "start": start,
                    "stop": stop,
                    "operational_cost": 60.0,
                    "unserved_energy_mwh": 1.0,
                    "wall_clock_s": 0.5,
                }
            )
        return pd.DataFrame(rows)

    def test_deviation_vs_reference(self):
        dev = metrics.deviation_vs_reference(self._frame(), (0, 48))
        row = dev[dev["metric"] == "operational_cost"].iloc[0]
        self.assertEqual(row["n_blocks"], 2)  # only the two blocks inside the window
        self.assertEqual(row["n_blocks_excluded"], 1)  # the straddling block
        self.assertAlmostEqual(row["value"], 120.0)
        self.assertAlmostEqual(row["reference"], 100.0)
        self.assertAlmostEqual(row["dev_abs"], 20.0)
        self.assertAlmostEqual(row["dev_rel"], 0.2)

    def test_deviation_requires_full_coverage(self):
        # Aligned: two 24 h blocks cover the 48 h reference window exactly.
        dev = metrics.deviation_vs_reference(self._frame(), (0, 48))
        row = dev[dev["metric"] == "operational_cost"].iloc[0]
        self.assertEqual(row["ref_window_hours"], 48)
        self.assertEqual(row["ref_window_hours_covered"], 48)

        # Misaligned: a 168 h-style block that cannot tile the window leaves a
        # gap, which must be an error rather than a silent short comparison.
        frame = self._frame()
        frame.loc[frame["block_size"] == 24, "block_size"] = 168
        frame = frame[~((frame["start"] == 24) & (frame["block_size"] == 168))]
        with self.assertRaises(ValueError) as ctx:
            metrics.deviation_vs_reference(frame, (0, 48))
        self.assertIn("cover 24 h of the 48 h reference window", str(ctx.exception))

    def test_card_reports_a_coverage_failure(self):
        cfg = config.load_config(tiny_config(self.tmp, window=(0, 48), reference_window=(0, 48)))
        run_dir = self.tmp / "badcover"
        run_dir.mkdir()
        frame = self._frame()
        frame = frame[frame["start"] != 24]  # drop a block: the window is no longer covered
        card = runcard.write_card(run_dir, cfg, frame)
        text = card.read_text()
        self.assertIn("Not computed:", text)
        self.assertIn("reference window", text)

    def test_deviation_without_reference_is_empty(self):
        frame = self._frame()
        frame = frame[frame["block_size"] != "reference"]
        self.assertTrue(metrics.deviation_vs_reference(frame).empty)

    def test_aggregate_reads_task_files(self):
        run_dir = self.tmp / "run"
        (run_dir / "tasks").mkdir(parents=True)
        for i in range(3):
            record = {
                "task_id": f"t{i}",
                "status": "ok",
                "method": "lp",
                "block_size": 24,
                "start": 24 * i,
                "stop": 24 * (i + 1),
                "wall_clock_s": 1.0,
                "metrics": {"operational_cost": float(i)},
            }
            tasks_mod.write_json_atomic(run_dir / "tasks" / f"t{i}.json", record)
        frame = metrics.aggregate(run_dir)
        self.assertEqual(len(frame), 3)
        self.assertIn("operational_cost", frame.columns)
        self.assertTrue((run_dir / "metrics.csv").exists())

    def test_card_from_a_synthetic_frame(self):
        cfg = config.load_config(tiny_config(self.tmp, window=(0, 48), reference_window=(0, 48)))
        run_dir = self.tmp / "cardrun"
        run_dir.mkdir()
        frame = self._frame()
        frame.to_csv(run_dir / "metrics.csv", index=False)
        card = runcard.write_card(
            run_dir,
            cfg,
            frame,
            system_meta={
                "peak_load_mw": 81_940.0,
                "peak_available_mw": 126_800.0,
                "implied_scale": 1.315,
                "applied_scale": 1.0,
                "voll": 10_000.0,
                "weather_store_attrs": {"dataset": "ca2040_z4"},
            },
        )
        text = card.read_text()
        self.assertIn("peak_available_mw", text)
        self.assertIn("clipped to 1.0", text)
        self.assertIn("ca2040_z4", text)


class TestADMMGapVsLP(unittest.TestCase):
    """`runcard.admm_gap_vs_lp` pairs the two methods on the same blocks."""

    def _frame(self):
        rows = []
        for index, (lp_cost, admm_cost) in enumerate([(100.0, 101.0), (200.0, 210.0)]):
            rows.append(
                {
                    "method": "lp",
                    "block_size": 24,
                    "year": 2020,
                    "block_index": index,
                    "draw": None,
                    "design_id": "asbuilt",
                    "status": "ok",
                    "operational_cost": lp_cost,
                    "solve_wall_clock_s": 0.03,
                }
            )
            rows.append(
                {
                    "method": "admm",
                    "block_size": 24,
                    "year": 2020,
                    "block_index": index,
                    "draw": None,
                    "design_id": "asbuilt",
                    "status": "ok",
                    "operational_cost": admm_cost,
                    "solve_wall_clock_s": 100.0,
                    "admm_max_imbalance_mw": 0.2,
                }
            )
        return pd.DataFrame(rows)

    def test_gap_is_computed_on_shared_blocks(self):
        out = runcard.admm_gap_vs_lp(self._frame())
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["n_blocks"], 2)
        self.assertAlmostEqual(row["gap_rel"], 311.0 / 300.0 - 1.0)
        self.assertAlmostEqual(row["worst_block_gap_rel"], 0.05)
        self.assertAlmostEqual(row["max_imbalance_mw"], 0.2)

    def test_infeasible_admm_rows_are_excluded(self):
        frame = self._frame()
        frame.loc[(frame["method"] == "admm") & (frame["block_index"] == 1), "status"] = (
            "infeasible"
        )
        out = runcard.admm_gap_vs_lp(frame)
        self.assertEqual(int(out.iloc[0]["n_blocks"]), 1)
        self.assertAlmostEqual(out.iloc[0]["gap_rel"], 0.01)

    def test_lp_only_run_has_no_gap_table(self):
        frame = self._frame()
        self.assertTrue(runcard.admm_gap_vs_lp(frame[frame["method"] == "lp"]).empty)


class TestBlockMetrics(TempRunMixin):
    """`block_metrics` on a hand-built 2-bus system with a prescribed dispatch.

    Nothing is solved: the outcome is written by hand so every metric has a
    value that can be checked by arithmetic.
    """

    def _system(self, load_power=-90.0, reps=1, meta=None):
        """The same hourly pattern repeated ``reps`` times, so a 2 h and a 4 h
        block carry identical per-hour physics and any per-day metric must agree."""

        def tile(a):
            return np.tile(np.asarray(a, dtype=float), (1, reps))

        net = PowerNetwork(num_nodes=2)
        gen = Generator(
            num_nodes=2,
            name=np.array(["ccgt", "solar"]),
            terminal=np.array([0, 1]),
            nominal_capacity=np.array([100.0, 50.0]),
            dynamic_capacity=tile([[1.0, 1.0], [0.8, 0.8]]),
            linear_cost=np.array([[10.0], [0.0]]),
            emission_rates=np.array([[0.4], [0.0]]),
        )
        load = Load(
            num_nodes=2,
            name=np.array(["l1"]),
            terminal=np.array([1]),
            load=tile([[100.0, 100.0]]),
            linear_cost=np.array([[1000.0]]),
        )
        line = DirectedLine(
            num_nodes=2,
            name=np.array(["imp", "exp"]),
            source_terminal=np.array([0, 1]),
            sink_terminal=np.array([1, 0]),
            nominal_capacity=np.array([50.0, 50.0]),
            min_power=np.array([[0.0], [0.0]]),
            max_power=np.array([[1.0], [1.0]]),
            linear_cost=np.array([[1.0], [-5.0]]),
            efficiency=np.array([[1.0], [1.0]]),
        )
        storage = StorageUnit(
            num_nodes=2,
            name=np.array(["bat"]),
            terminal=np.array([1]),
            power_capacity=np.array([10.0]),
            duration=np.array([4.0]),
            linear_cost=np.array([0.0]),
        )
        devices = [gen, load, line, storage]

        power = [
            [tile([[60.0, 60.0], [30.0, 30.0]])],  # solar spills 10 MW/h of its 40 MW
            [np.full((1, 2 * reps), load_power)],  # 90 of 100 MW served
            [
                tile([[-20.0, -20.0], [-5.0, -5.0]]),
                tile([[20.0, 20.0], [5.0, 5.0]]),
            ],
            [tile([[-1.0, 2.0]])],
        ]
        local_variables = [
            None,
            None,
            None,
            StorageUnitVariable(
                energy=np.concatenate([np.array([[20.0]]), tile([[21.0, 19.0]])], axis=1),
                charge=tile([[1.0, 0.0]]),
                discharge=tile([[0.0, 2.0]]),
            ),
        ]
        outcome = DispatchOutcome(
            phase_duals=None,
            local_equality_duals=None,
            local_inequality_duals=None,
            local_variables=local_variables,
            power=power,
            angle=[[None], [None], [None, None], [None]],
            prices=tile([[10.0, 10.0], [20.0, 60.0]]),
            global_angle=None,
        )
        index = SimpleNamespace(
            carrier={
                "Generator": np.array(["CCGT", "solar"]),
                "DirectedLine": np.array(["imports", "exports"]),
            },
            vre_mask=np.array([False, True]),
        )
        loaded = SimpleNamespace(network=net, index=index, meta=dict(meta or {}))
        return loaded, devices, outcome

    def test_block_metrics_values(self):
        loaded, devices, outcome = self._system()
        block = blocks_mod.Block(index=0, year=2020, start=0, stop=2)
        m = metrics.block_metrics(loaded, devices, outcome, block)

        # ENS = (100 - 90) MW over 2 hours, and it is positive (sign check).
        self.assertAlmostEqual(m["unserved_energy_mwh"], 20.0)
        self.assertGreater(m["unserved_energy_mwh"], 0.0)
        self.assertEqual(m["lost_load_hours"], 2)
        self.assertAlmostEqual(m["voll_cost"], 1000.0 * 20.0)

        self.assertAlmostEqual(m["generation_cost"], 10.0 * 120.0)
        self.assertAlmostEqual(m["co2_tonnes"], 0.4 * 120.0)
        self.assertEqual(json.loads(m["generation_mwh_by_carrier"]), {"CCGT": 120.0, "solar": 60.0})
        # Curtailment is VRE only: solar has 50 * 0.8 = 40 MW available and
        # generates 30, so 10 MW/h. The CCGT's 40 MW/h of headroom is not counted.
        self.assertAlmostEqual(m["curtailment_mwh"], 20.0)

        self.assertAlmostEqual(m["imports_mwh"], 40.0)  # sink-end flow of the import link
        self.assertAlmostEqual(m["exports_mwh"], 10.0)  # source-end withdrawal of the export link
        self.assertAlmostEqual(m["export_revenue"], 50.0)  # -(-5 $/MWh * 5 MW * 2 h)

        self.assertAlmostEqual(m["storage_cycles"], 2.0 / (10.0 * 4.0))
        # Fleet start level 20 MWh of 10 MW * 4 h = 40 MWh of energy capacity.
        self.assertAlmostEqual(m["storage_start_soc_frac"], 0.5)
        self.assertAlmostEqual(m["mean_price"], 40.0)  # load-weighted: bus 1 only
        self.assertAlmostEqual(m["max_price"], 60.0)
        self.assertAlmostEqual(m["operational_cost"], 1200.0 + 20000.0 - 10.0)

    def test_max_price_ignores_non_load_buses(self):
        """Only load-carrying buses set the reported price extremes.

        On `ca2040_z4` the un-loaded `p6_imports` bus prices at exactly the
        60 $/MWh of its marginal unit and the export buses carry a degenerate
        -155 $/MWh dual; neither is a system price (benchmark review 1.3).
        """
        loaded, devices, outcome = self._system()
        block = blocks_mod.Block(index=0, year=2020, start=0, stop=2)

        # Bus 0 carries no load. Price it far above -- and then far below -- every
        # load bus; the reported max_price must not move either way.
        outcome.prices = np.array([[500.0, 500.0], [20.0, 60.0]])
        m = metrics.block_metrics(loaded, devices, outcome, block)
        self.assertAlmostEqual(m["max_price"], 60.0)
        self.assertAlmostEqual(m["max_price_all_buses"], 500.0)

        outcome.prices = np.array([[-155.0, -155.0], [20.0, 60.0]])
        m = metrics.block_metrics(loaded, devices, outcome, block)
        self.assertAlmostEqual(m["max_price"], 60.0)
        self.assertAlmostEqual(m["mean_price"], 40.0)

    def test_storage_cycles_per_day_is_block_invariant(self):
        """`storage_cycles` counts per block; the per-day column must not."""
        block2 = blocks_mod.Block(index=0, year=2020, start=0, stop=2)
        block4 = blocks_mod.Block(index=0, year=2020, start=0, stop=4)

        loaded2, devices2, outcome2 = self._system()
        loaded4, devices4, outcome4 = self._system(reps=2)

        m2 = metrics.block_metrics(loaded2, devices2, outcome2, block2)
        m4 = metrics.block_metrics(loaded4, devices4, outcome4, block4)

        # The raw column is *not* comparable: twice the block, twice the cycles.
        self.assertAlmostEqual(m4["storage_cycles"], 2.0 * m2["storage_cycles"])
        # The normalised one is.
        self.assertAlmostEqual(m2["storage_cycles_per_day"], m2["storage_cycles"] * 12.0)
        self.assertAlmostEqual(m4["storage_cycles_per_day"], m2["storage_cycles_per_day"])

    def test_metric_failures_are_counted_not_swallowed(self):
        """A device whose operation_cost raises must be named on the metrics row."""
        loaded, devices, outcome = self._system()
        block = blocks_mod.Block(index=0, year=2020, start=0, stop=2)

        clean = metrics.block_metrics(loaded, devices, outcome, block)
        self.assertEqual(clean["metric_failures"], 0)
        self.assertNotIn("metric_failure_detail", clean)

        class Boom:
            def operation_cost(self, *args, **kwargs):
                raise RuntimeError("no cost for you")

            def get_emissions(self, *args, **kwargs):
                return 0.0

        devices = list(devices) + [Boom()]
        outcome.power = list(outcome.power) + [[np.zeros((1, 2))]]
        outcome.angle = list(outcome.angle) + [[None]]
        outcome.local_variables = list(outcome.local_variables) + [None]
        loaded.network = None  # the network cannot price a device it does not know

        m = metrics.block_metrics(loaded, devices, outcome, block)
        self.assertEqual(m["metric_failures"], 1)
        self.assertIn("Boom", m["metric_failure_detail"])

    def test_metrics_are_reported_in_physical_units(self):
        """`power_unit` / `cost_unit` condition the solve; they must not leak to the card."""
        block = blocks_mod.Block(index=0, year=2020, start=0, stop=2)

        loaded, devices, outcome = self._system()
        base = metrics.block_metrics(loaded, devices, outcome, block)

        # The same system as the importer would build it at power_unit=1000,
        # cost_unit=10: every power and cost divided by its unit.
        scaled_loaded, scaled_devices, scaled_outcome = self._system(
            load_power=-90.0, meta={"power_unit": 1000.0, "cost_unit": 10.0}
        )
        for device in scaled_devices:
            device.scale_costs(10.0)
            device.scale_power(1000.0)
        scaled_outcome.power = [[p / 1000.0 for p in ps] for ps in scaled_outcome.power]
        scaled_outcome.prices = scaled_outcome.prices / 10.0
        scaled_outcome.local_variables[3] = StorageUnitVariable(
            *[np.asarray(v) / 1000.0 for v in scaled_outcome.local_variables[3]]
        )

        scaled = metrics.block_metrics(scaled_loaded, scaled_devices, scaled_outcome, block)

        for key in (
            "operational_cost",
            "generation_cost",
            "voll_cost",
            "co2_tonnes",
            "unserved_energy_mwh",
            "imports_mwh",
            "max_price",
            "storage_cycles",
        ):
            self.assertAlmostEqual(
                scaled[key], base[key], delta=1e-9 * max(1.0, abs(base[key])), msg=key
            )

    def test_ens_sign_and_zero(self):
        # Fully served demand: no unserved energy, no VOLL cost.
        loaded, devices, outcome = self._system(load_power=-100.0)
        block = blocks_mod.Block(index=0, year=2020, start=0, stop=2)
        served = metrics.block_metrics(loaded, devices, outcome, block)
        self.assertAlmostEqual(served["unserved_energy_mwh"], 0.0)
        self.assertEqual(served["lost_load_hours"], 0)

        # A flipped sign convention (load drawn as +power) would read as 200 MWh
        # of unserved energy instead of 0; pin that it does not.
        loaded, devices, outcome = self._system(load_power=+100.0)
        flipped = metrics.block_metrics(loaded, devices, outcome, block)
        self.assertAlmostEqual(flipped["unserved_energy_mwh"], 400.0)
        self.assertNotAlmostEqual(flipped["unserved_energy_mwh"], 0.0)

    def test_index_length_mismatch_raises(self):
        loaded, devices, outcome = self._system()
        loaded.index.carrier["Generator"] = np.array(["CCGT"])  # one row too few
        block = blocks_mod.Block(index=0, year=2020, start=0, stop=2)
        with self.assertRaises(ValueError):
            metrics.block_metrics(loaded, devices, outcome, block)


class TestADMMDispatchRecord(TempRunMixin):
    """`solve_block_admm` must report the iterate's imbalance and gate on it."""

    def _system(self):
        net = PowerNetwork(num_nodes=2)
        gen = Generator(
            num_nodes=2,
            name=np.array(["cheap", "peaker"]),
            terminal=np.array([0, 1]),
            nominal_capacity=np.array([80.0, 60.0]),
            dynamic_capacity=np.ones((2, 4)),
            linear_cost=np.array([[10.0], [90.0]]),
            emission_rates=np.array([[0.4], [0.6]]),
        )
        load = Load(
            num_nodes=2,
            name=np.array(["l1"]),
            terminal=np.array([1]),
            load=np.array([[60.0, 80.0, 100.0, 70.0]]),
            linear_cost=np.array([[1000.0]]),
        )
        line = DirectedLine(
            num_nodes=2,
            name=np.array(["ln"]),
            source_terminal=np.array([0]),
            sink_terminal=np.array([1]),
            nominal_capacity=np.array([100.0]),
            min_power=np.array([[0.0]]),
            max_power=np.array([[1.0]]),
            linear_cost=np.array([[0.0]]),
            efficiency=np.array([[1.0]]),
        )
        index = SimpleNamespace(
            carrier={"Generator": np.array(["CCGT", "OCGT"])},
            vre_mask=np.array([False, False]),
        )
        loaded = SimpleNamespace(network=net, index=index, meta={})
        return loaded, [gen, load, line]

    def _cfg(self, **admm):
        method = {
            "enabled": True,
            "required": False,
            "solver": "ADMM",
            "dtype": "float64",
            "timeout_s": 60,
            "max_imbalance_mw": 1.0,
            "solver_kwargs": {
                "num_iterations": 5000,
                "rho_power": 1.0,
                "adaptive_rho": False,
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
            },
        }
        method.update(admm)
        return {"methods": {"admm": method}}

    def _task(self):
        block = blocks_mod.Block(index=0, year=2020, start=0, stop=4)
        return tasks_mod.Task(task_id="t", method="admm", block=block, block_size=4)

    def test_ra_dispatch_admm_records_imbalance(self):
        loaded, devices = self._system()
        payload = dispatch.solve_block_admm(loaded, devices, self._task(), self._cfg())

        self.assertIn("admm_max_imbalance_mw", payload["metrics"])
        self.assertLess(payload["metrics"]["admm_max_imbalance_mw"], 1.0)
        self.assertTrue(np.isfinite(payload["metrics"]["admm_primal_tol"]))
        self.assertTrue(np.isfinite(payload["metrics"]["admm_dual_tol"]))
        self.assertIn("admm_converged", payload["metrics"])
        self.assertNotIn("status", payload)

    def test_ra_dispatch_admm_gates_an_under_iterated_solve(self):
        loaded, devices = self._system()
        cfg = self._cfg(
            solver_kwargs={
                "num_iterations": 2,
                "minimum_iterations": 1,
                "rho_power": 1.0,
                "adaptive_rho": False,
            }
        )
        payload = dispatch.solve_block_admm(loaded, devices, self._task(), cfg)

        self.assertGreater(payload["metrics"]["admm_max_imbalance_mw"], 1.0)
        self.assertEqual(payload["status"], "infeasible")
        self.assertIn("power balance", payload["error"])
        self.assertIn("admm_max_soc_residual_mwh", payload["metrics"])

    def test_ra_dispatch_admm_gates_a_storage_energy_violation(self):
        # Few battery inner iterations: nodal balance can pass while the projected
        # storage iterate breaks the SoC recursion; the second gate must catch it.
        loaded, devices = self._system()
        storage = StorageUnit(
            num_nodes=2,
            name=np.array(["bat"]),
            terminal=np.array([1]),
            power_capacity=np.array([40.0]),
            duration=np.array([2.0]),
            linear_cost=np.array([0.0]),
            charge_efficiency=np.array([0.9]),
            discharge_efficiency=np.array([0.9]),
        )
        devices = devices + [storage]
        cfg = self._cfg(
            solver_kwargs={
                "num_iterations": 300,
                "minimum_iterations": 10,
                "rho_power": 1.0,
                "adaptive_rho": False,
                "battery_inner_iterations": 1,
            }
        )
        cfg["methods"]["admm"]["max_imbalance_mw"] = 1e9
        cfg["methods"]["admm"]["max_soc_residual_mwh"] = 1e-6
        payload = dispatch.solve_block_admm(loaded, devices, self._task(), cfg)
        self.assertGreater(payload["metrics"]["admm_max_soc_residual_mwh"], 1e-6)
        self.assertEqual(payload["status"], "infeasible")
        self.assertIn("energy balance", payload["error"])

    def test_ra_dispatch_zero_gate_is_not_replaced_by_default(self):
        self.assertEqual(dispatch._gate_value({"max_imbalance_mw": 0.0}, "max_imbalance_mw", 1.0), 0.0)
        self.assertEqual(dispatch._gate_value({}, "max_imbalance_mw", 1.0), 1.0)

    def test_run_task_marks_an_infeasible_payload(self):
        cfg = config.load_config(tiny_config(self.tmp, window=(0, 48)))
        task = tasks_mod.enumerate_tasks(cfg)[0]

        real = dispatch.solve_block

        def infeasible(task, cfg, design=None, **kwargs):
            payload = real(task, cfg, design=design, **kwargs)
            payload["status"] = "infeasible"
            payload["error"] = "ADMM iterate violates nodal power balance by 113 MW"
            return payload

        dispatch.solve_block = infeasible
        try:
            record = tasks_mod.run_task(task, cfg, self.tmp / "run")
        finally:
            dispatch.solve_block = real

        self.assertEqual(record["status"], "infeasible")
        self.assertIn("113 MW", record["error"])
        # The metrics survive so the row can still be inspected.
        self.assertIn("operational_cost", record["metrics"])


class TestStubSolver(TempRunMixin):
    def test_stub_is_deterministic(self):
        cfg = config.load_config(tiny_config(self.tmp, window=(0, 48)))
        task = tasks_mod.enumerate_tasks(cfg)[0]
        a = dispatch.solve_block(task, cfg)
        b = dispatch.solve_block(task, cfg)
        for payload in (a, b):
            payload["metrics"].pop("solve_wall_clock_s")
        self.assertEqual(a, b)
        self.assertEqual(a["metrics"]["hours"], 24)


@unittest.skipIf(not WP1_AVAILABLE, "WP1 (zap.importers.wy_store) is not available yet")
class TestRealSystem(TempRunMixin):
    """Integration tests that need WP1's reader and its tiny dataset fixture."""

    def _dataset(self) -> Path:
        try:
            from zap.tests.fixtures.tiny_dataset import write_tiny_dataset
        except ImportError:  # pragma: no cover - fixture not built yet
            self.skipTest("WP1 tiny dataset fixture is not available yet")
        dataset = write_tiny_dataset(self.tmp / "tiny", n_hours=48, years=(2020,))
        from zap.importers.wy_store import convert_dataset

        convert_dataset(dataset)
        return dataset

    def _config(self) -> Path:
        dataset = self._dataset()
        return tiny_config(
            self.tmp,
            name="e2e_highs",
            window=(0, 48),
            block_sizes=(24,),
            reference_window=(0, 48),
            extra={
                "dataset": {"dir": str(dataset)},
                "methods": {"lp": {"solver": "HIGHS", "timeout_s": 300}},
            },
        )

    def _skip_if_devices_do_not_slice(self, cfg):
        """WP1/WP3 join: a device whose time-varying fields survive sample_time."""
        loaded = system_mod.build_system(cfg)
        block = blocks_mod.make_blocks(cfg, 24)[0]
        try:
            dispatch.slice_devices(loaded, cfg, block)
        except ValueError as exc:
            self.skipTest(f"blocked dispatch not yet buildable: {exc}")

    def test_end_to_end_two_days_highs(self):
        path = self._config()
        self._skip_if_devices_do_not_slice(config.load_config(path))
        cfg = config.load_config(path)
        code = cli.main(["run", "--config", str(path), "--runs-root", str(self.runs_root)])
        self.assertEqual(code, 0)

        run_dir = paths.run_dir(identity.run_id(cfg), self.runs_root)
        frame = pd.read_csv(run_dir / "metrics.csv")
        self.assertEqual(len(frame), 3)
        self.assertTrue((frame["status"] == "ok").all())
        self.assertTrue((run_dir / "CARD.md").exists())

        blocked = frame[frame["block_size"] != "reference"]["operational_cost"].sum()
        reference = frame[frame["block_size"] == "reference"]["operational_cost"].sum()
        self.assertGreaterEqual(blocked, reference - 1e-6)

        with open(run_dir / "system_meta.json") as f:
            meta = json.load(f)
        self.assertIn("peak_load_mw", meta)

    def test_power_unit_scaling_is_cost_invariant(self):
        """`system.power_unit` / `cost_unit` only condition the solve.

        They rescale the LP the importer builds; the reported cost is converted
        back, so the same block costs the same at any scaling (review 3.4).
        """
        dataset = self._dataset()

        def solve(power_unit, cost_unit):
            path = tiny_config(
                self.tmp,
                name=f"units_{power_unit:g}_{cost_unit:g}",
                window=(0, 48),
                block_sizes=(24,),
                reference="none",
                extra={
                    "dataset": {"dir": str(dataset)},
                    "system": {"power_unit": power_unit, "cost_unit": cost_unit},
                    "methods": {"lp": {"solver": "HIGHS", "timeout_s": 300}},
                },
            )
            cfg = config.load_config(path)
            self._skip_if_devices_do_not_slice(cfg)
            system_mod.clear_system_cache()
            loaded = system_mod.build_system(cfg, cache=False)
            block = blocks_mod.make_blocks(cfg, 24)[0]
            devices = dispatch.slice_devices(loaded, cfg, block)
            task = tasks_mod.Task(task_id="t", method="lp", block=block, block_size=24)
            return dispatch.solve_block_lp(loaded, devices, task, cfg)["metrics"]

        base = solve(1.0, 1.0)
        scaled = solve(1000.0, 10.0)

        for key in ("operational_cost", "generation_cost", "unserved_energy_mwh", "co2_tonnes"):
            denom = max(1.0, abs(base[key]))
            self.assertAlmostEqual(scaled[key] / denom, base[key] / denom, places=9, msg=key)

    def test_reference_only_run_records_demand_scaling(self):
        # The reference solve runs in a child process, so the provenance must
        # travel back in the task payload for the card to show it.
        dataset = self._dataset()
        path = tiny_config(
            self.tmp,
            name="refonly",
            window=(0, 48),
            block_sizes=(),
            reference="window",
            reference_window=(0, 48),
            extra={
                "dataset": {"dir": str(dataset)},
                "methods": {"lp": {"solver": "HIGHS", "timeout_s": 300}},
            },
        )
        cfg = config.load_config(path)
        (task,) = tasks_mod.enumerate_tasks(cfg)
        self.assertTrue(tasks_mod._use_subprocess(task, cfg))  # the isolated path

        system_mod.clear_system_cache()
        system_mod._LAST_META = None  # nothing in this process knows the meta
        self.assertEqual(
            cli.main(["run", "--config", str(path), "--runs-root", str(self.runs_root)]), 0
        )
        run_dir = paths.run_dir(identity.run_id(cfg), self.runs_root)
        self.assertTrue((run_dir / "system_meta.json").exists())
        card = (run_dir / "CARD.md").read_text()
        for key in ("peak_load_mw", "peak_available_mw", "implied_scale", "applied_scale"):
            self.assertIn(key, card)
            self.assertNotIn(f"**{key}:** -", card)

    def test_cyclic_soc_per_block(self):
        # This test pins the *fixed* boundary condition; the harness default is
        # cyclic_free (decision 2026-09-09), so select fixed explicitly.
        path = self._config()
        cfg = config.load_config(path)
        cfg["system"]["storage_soc_mode"] = "fixed"
        self._skip_if_devices_do_not_slice(cfg)
        loaded = system_mod.build_system(cfg)
        block = blocks_mod.make_blocks(cfg, 24)[1]
        devices = dispatch.slice_devices(loaded, cfg, block)

        storage = [d for d in devices if type(d).__name__ == "StorageUnit"]
        self.assertTrue(storage)
        for device in storage:
            np.testing.assert_allclose(np.asarray(device.initial_soc), 0.5)
            np.testing.assert_allclose(np.asarray(device.final_soc), 0.5)

        outcome = loaded.network.dispatch(devices, time_horizon=block.hours, solver="HIGHS")
        index = {id(d): i for i, d in enumerate(devices)}
        for device in storage:
            state = outcome.local_variables[index[id(device)]]
            energy = np.asarray(state.energy if hasattr(state, "energy") else state[0])
            expected = (
                0.5
                * np.asarray(device.power_capacity).ravel()
                * np.asarray(device.duration).ravel()
            )
            np.testing.assert_allclose(energy[:, 0], expected, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(energy[:, -1], expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
