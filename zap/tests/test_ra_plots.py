"""WP-P2: the ``ra plot`` catalogue (spec section 5.2).

A module-scoped fixture builds four run directories once on the 48-hour tiny
dataset: two dispatch runs with hourly data at different block sizes (so the
comparison plots have two inputs), one ADMM run with a convergence trace, and
one gradient planning run that supplies the designs P1 / P2 / P6 read.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

from experiments.ra import cli, config, identity, paths
from experiments.ra import plots as plots_mod
from experiments.ra.plots import loader as loader_mod
from experiments.ra.plots import style as style_mod

CATALOGUE_IDS = (
    tuple(f"O{i}" for i in range(1, 12))
    + tuple(f"P{i}" for i in range(1, 10))
    + tuple(f"R{i}" for i in range(1, 8))
)

PHASE_A_IDS = (
    "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10", "O11", "P1", "P2", "P6",
)

#: A second, independent copy of every phase-A CSV header (D9 / T14).  A change
#: to ``PlotSpec.columns`` alone must not pass the suite.
GOLDEN_COLUMNS = {
    "O1": ["run_id", "label", "method", "block_size", "year", "hour", "carrier",
           "available_gw"],
    "O2": ["run_id", "label", "year", "hour", "load_gw", "vre_available_gw", "net_load_gw",
           "net_load_rank", "net_load_duration_frac"],
    "O3": ["run_id", "label", "year", "hour", "available_gw", "net_load_gw", "headroom_gw",
           "headroom_frac", "ens_gw"],
    "O4": ["run_id", "label", "method", "block_size", "year", "hour", "series", "carrier",
           "value_gw"],
    "O5": ["label_a", "label_b", "year", "hour", "series", "carrier", "value_a_gw",
           "value_b_gw", "diff_gw"],
    "O6": ["run_id", "label", "method", "block_size", "year", "hour", "carrier", "soc_gwh",
           "soc_frac", "block_boundary"],
    "O7": ["run_id", "label", "method", "block_size", "line", "bus0", "bus1", "carrier",
           "capacity_gw", "mean_flow_gw", "peak_flow_gw", "mean_loading", "peak_loading",
           "hours_at_capacity"],
    "O8": ["run_id", "label", "method", "block_size", "bus", "quantile", "price_usd_per_mwh",
           "n_hours", "share_hours_at_voll"],
    "O9": ["run_id", "label", "method", "block_size", "year", "block_index", "start", "stop",
           "hours", "metric", "value", "cumulative_value", "reference_value",
           "cumulative_dev_rel"],
    "O10": ["run_id", "label", "block_size", "year", "block_index", "iteration", "objective",
            "primal_power", "primal_phase", "dual_power", "dual_phase", "primal_tol",
            "dual_tol"],
    "O11": ["run_id", "label", "method", "block_size", "hours_per_block", "n_blocks",
            "total_hours", "solve_s_total", "solve_s_mean", "solve_s_p95", "wall_s_total",
            "cores", "cpu_seconds", "n_variables_mean", "n_constraints_mean"],
    "P1": ["run_id", "label", "design_id", "device_class", "carrier", "unit", "as_built",
           "designed", "delta"],
    "P2": ["run_id", "label", "design_id", "formulation", "heuristic", "selection_strategy",
           "emissions_mode", "carrier", "unit", "designed", "delta"],
    "P6": ["run_id", "label", "design_id", "source", "component", "value_bn_usd"],
}

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


class PlotFixture(unittest.TestCase):
    """Four run directories, built once for the whole module."""

    @classmethod
    def setUpClass(cls):
        if not _fixture_available():  # pragma: no cover
            raise unittest.SkipTest("WP1 (wy_store + tiny dataset fixture) is not available")
        from zap.importers.wy_store import convert_dataset
        from zap.tests.fixtures.tiny_dataset import write_tiny_dataset

        cls.class_tmp = Path(tempfile.mkdtemp(prefix="ra-plots-"))
        cls.dataset = write_tiny_dataset(cls.class_tmp / "tiny", n_hours=48, years=(2020,))
        cls._make_extendable(cls.dataset)
        convert_dataset(cls.dataset)
        cls.runs_root = cls.class_tmp / "runs"

        cls.run_a = cls._dispatch_run(
            "run_a",
            {
                "selection": {"blocks": [24], "reference": "window",
                              "reference_window": {"start": 0, "hours": 48}},
                "output": {"save_hourly": "carrier_bus"},
            },
        )
        cls.run_b = cls._dispatch_run(
            "run_b",
            {
                "selection": {"blocks": [48], "reference": "none"},
                "output": {"save_hourly": "carrier_bus"},
            },
        )
        cls.run_admm = cls._dispatch_run(
            "run_admm",
            {
                "selection": {"blocks": [24], "reference": "none"},
                "output": {"admm_trace_every": 5},
                "methods": {
                    "lp": {"enabled": False},
                    "admm": {
                        "enabled": True,
                        "required": False,
                        "timeout_s": 900,
                        "solver_kwargs": {
                            "num_iterations": 22,
                            "rho_power": 1.0,
                            "minimum_iterations": 100,
                            "atol": 1e-8,
                            "rtol": 1e-8,
                        },
                    },
                },
            },
            expect_zero=False,
        )
        cls.run_plain = cls._dispatch_run(
            "run_plain", {"selection": {"blocks": [24], "reference": "none"}}
        )
        cls.run_plan = cls._plan_run()

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

    @classmethod
    def _write_config(cls, name: str, cfg: dict) -> Path:
        path = cls.class_tmp / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    @classmethod
    def _dispatch_run(cls, name: str, extra: dict, expect_zero: bool = True) -> str:
        cfg = {
            "includes": [str(paths.config_root() / "base.yaml")],
            "name": name,
            "dataset": {
                "dir": str(cls.dataset),
                "years": [2020],
                "window": {"start": 0, "stop": 48},
            },
            "selection": {"blocks": [24], "reference": "none"},
            "methods": {"lp": {"enabled": True, "solver": "HIGHS", "timeout_s": 600}},
        }
        cfg = config.deep_merge(cfg, extra)
        path = cls._write_config(name, cfg)
        code = cli.main(["run", "--config", str(path), "--runs-root", str(cls.runs_root)])
        if expect_zero and code != 0:  # pragma: no cover - a broken fixture
            raise RuntimeError(f"fixture run {name} exited {code}")
        return identity.run_id(config.load_config(path))

    @classmethod
    def _plan_run(cls) -> str:
        cfg = {
            "includes": [str(paths.config_root() / "base.yaml")],
            "name": "run_plan",
            "mode": "plan",
            "dataset": {
                "dir": str(cls.dataset),
                "years": [2020],
                "window": {"start": 0, "stop": 48},
            },
            "selection": {"strategy": "all", "block_size": 24},
            "planning": {
                "method": "gradient",
                "dispatch_solver": "HIGHS",
                "single_level": {"kind": "primal", "solver": "HIGHS"},
                "optimizer": {"num_iterations": 3},
                "timeout_s": 900,
            },
        }
        path = cls._write_config("run_plan", cfg)
        code = cli.main(["run", "--config", str(path), "--runs-root", str(cls.runs_root)])
        if code != 0:  # pragma: no cover - a broken fixture
            raise RuntimeError(f"fixture plan run exited {code}")
        return identity.run_id(config.load_config(path))

    # -- helpers ----------------------------------------------------------
    def handles(self, *run_ids, labels=None):
        return plots_mod.load_runs(list(run_ids), runs_root=self.runs_root, labels=labels)

    def runs_for(self, plot_id):
        if plot_id == "O10":
            return self.handles(self.run_admm)
        if plot_id in ("P1", "P2", "P6"):
            return self.handles(self.run_plan)
        if plot_id == "O5":
            return self.handles(self.run_a, self.run_b)
        return self.handles(self.run_a, self.run_b)

    def setUp(self):
        plt.close("all")
        self.out = Path(tempfile.mkdtemp(prefix="ra-plots-out-"))

    def tearDown(self):
        plt.close("all")
        shutil.rmtree(self.out, ignore_errors=True)


class TestCatalogue(PlotFixture):
    def test_catalogue_is_complete(self):
        """T12."""
        self.assertEqual(set(plots_mod.PLOTS), set(CATALOGUE_IDS))
        self.assertEqual(len(plots_mod.PLOTS), 27)
        for plot_id, spec in plots_mod.PLOTS.items():
            self.assertTrue(spec.title.strip(), plot_id)
            self.assertIn(spec.tier, ("debug", "report"), plot_id)
            self.assertIn(spec.phase, ("A", "B"), plot_id)
            self.assertTrue(spec.columns, plot_id)
        self.assertEqual(set(plots_mod.ids(phase="A")), set(PHASE_A_IDS))

    def test_phase_b_plots_raise(self):
        """T15."""
        listed = set(plots_mod.catalogue()["plot_id"])
        for plot_id, spec in plots_mod.PLOTS.items():
            if spec.phase != "B":
                continue
            self.assertIn(plot_id, listed)
            with self.assertRaises(NotImplementedError, msg=plot_id):
                spec.fn(self.handles(self.run_a))

    def test_carrier_colors_cover_the_dataset(self):
        """T21."""
        from matplotlib.colors import is_color_like

        for carrier in style_mod.DATASET_CARRIERS:
            self.assertIn(carrier, style_mod.CARRIER_COLORS, carrier)
            self.assertTrue(is_color_like(style_mod.CARRIER_COLORS[carrier]), carrier)
        style_mod._WARNED.discard("nonsense")
        with self.assertLogs(style_mod.logger, level="WARNING") as captured:
            first = style_mod.carrier_color("nonsense")
        self.assertTrue(is_color_like(first))
        self.assertEqual(len(captured.records), 1)
        # The warning is emitted once per carrier, not once per call.
        self.assertEqual(style_mod.carrier_color("nonsense"), first)


class TestPhaseAPlots(PlotFixture):
    def test_phase_a_plots_write_png_and_csv(self):
        """T13."""
        for plot_id in PHASE_A_IDS:
            with self.subTest(plot=plot_id):
                png, csv = plots_mod.render(plot_id, self.runs_for(plot_id), self.out)
                self.assertTrue(png.exists() and png.stat().st_size > 0, plot_id)
                self.assertTrue(csv.exists() and csv.stat().st_size > 0, plot_id)
                header = list(pd.read_csv(csv, nrows=0).columns)
                self.assertEqual(header, list(plots_mod.PLOTS[plot_id].columns), plot_id)
                self.assertEqual(plt.get_fignums(), [], f"{plot_id} leaked a figure")

    def test_csv_columns_are_stable(self):
        """T14."""
        self.assertEqual(set(GOLDEN_COLUMNS), set(PHASE_A_IDS))
        for plot_id, expected in GOLDEN_COLUMNS.items():
            with self.subTest(plot=plot_id):
                _png, csv = plots_mod.render(plot_id, self.runs_for(plot_id), self.out)
                self.assertEqual(list(pd.read_csv(csv, nrows=0).columns), expected)

    def test_o8_load_buses_only(self):
        """T16 (D5)."""
        _fig, table = plots_mod.plot("O8", self.handles(self.run_a))
        plt.close("all")
        static = self.handles(self.run_a)[0].system_static()
        allowed = set(static["load_buses"]) | {"__all__"}
        self.assertEqual(set(table["bus"]) - allowed, set())
        self.assertIn("z1", set(table["bus"]))

    def test_run_count_guards(self):
        """T17."""
        with self.assertRaises(ValueError) as ctx:
            plots_mod.plot("O5", self.handles(self.run_a))
        self.assertIn("exactly two runs", str(ctx.exception).replace("2", "two"))
        with self.assertRaises(ValueError):
            plots_mod.plot("O5", self.handles(self.run_a, self.run_b, self.run_plain))
        for plot_id in ("O4", "O9", "O11"):
            fig, table = plots_mod.plot(plot_id, self.handles(self.run_a, self.run_b))
            plt.close(fig)
            self.assertEqual(len(set(table["run_id"])), 2, plot_id)

    def test_missing_data_is_named(self):
        """T18."""
        run = self.handles(self.run_plain)
        with self.assertRaises(loader_mod.MissingDataError) as ctx:
            plots_mod.plot("O4", run)
        message = str(ctx.exception)
        self.assertIn("hourly.parquet", message)
        self.assertIn("output.save_hourly", message)


class TestPlotCli(PlotFixture):
    def test_cli_out_dir_resolution(self):
        """T19 (D1)."""
        code = cli.main(
            [
                "plot",
                "--run-id",
                self.run_a,
                "--runs-root",
                str(self.runs_root),
                "--plot",
                "O4",
            ]
        )
        self.assertEqual(code, 0)
        run_dir = paths.run_dir(self.run_a, self.runs_root)
        self.assertTrue(list((run_dir / "figures").glob("O4_*.png")))

        figures = self.out / "figs"
        import os

        os.environ["CH3_FIGURES_DIR"] = str(figures)
        try:
            code = cli.main(
                [
                    "plot",
                    "--run-id",
                    self.run_a,
                    "--runs-root",
                    str(self.runs_root),
                    "--plot",
                    "O8",
                    "--study",
                    "blocking",
                ]
            )
        finally:
            del os.environ["CH3_FIGURES_DIR"]
        self.assertEqual(code, 0)
        self.assertTrue(list((figures / "blocking").glob("O8_*.png")))

        code = cli.main(
            [
                "plot",
                "--run-id",
                self.run_a,
                "--run-id",
                self.run_b,
                "--runs-root",
                str(self.runs_root),
                "--plot",
                "O5",
            ]
        )
        self.assertEqual(code, 1)

    def test_cli_skips_missing_data_without_fail_fast(self):
        """T20."""
        code = cli.main(
            [
                "plot",
                "--run-id",
                self.run_plain,
                "--runs-root",
                str(self.runs_root),
                "--tier",
                "debug",
                "--out",
                str(self.out / "skip"),
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue(list((self.out / "skip").glob("O9_*.png")))

        code = cli.main(
            [
                "plot",
                "--run-id",
                self.run_plain,
                "--runs-root",
                str(self.runs_root),
                "--tier",
                "debug",
                "--out",
                str(self.out / "ff"),
                "--fail-fast",
            ]
        )
        self.assertEqual(code, 1)

    def test_cli_list(self):
        self.assertEqual(cli.main(["plot", "--list"]), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
