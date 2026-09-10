"""WP-P2: the ``ra plot`` catalogue (spec section 5.2).

A module-scoped fixture builds four run directories once on the 48-hour tiny
dataset: two dispatch runs with hourly data at different block sizes (so the
comparison plots have two inputs), one ADMM run with a convergence trace, and
one gradient planning run that supplies the designs P1 / P2 / P6 read.
"""

import itertools
import math
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
    tuple(f"O{i}" for i in range(1, 14))
    + tuple(f"P{i}" for i in range(1, 10))
    + tuple(f"R{i}" for i in range(1, 8))
)

PHASE_A_IDS = (
    "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10", "O11", "O12", "O13",
    "P1", "P2", "P6",
)

#: A second, independent copy of every phase-A CSV header (D9 / T14).  A change
#: to ``PlotSpec.columns`` alone must not pass the suite.
GOLDEN_COLUMNS = {
    "O1": ["run_id", "label", "method", "block_size", "year", "hour", "carrier",
           "available_gw"],
    "O2": ["run_id", "label", "year", "hour", "load_gw", "vre_available_gw", "net_load_gw",
           "net_load_rank", "net_load_duration_frac"],
    "O3": ["run_id", "label", "year", "hour", "available_gw", "load_gw", "dispatchable_gw",
           "net_load_gw", "headroom_gw", "headroom_frac", "ens_gw"],
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
    "O12": ["run_id", "label", "block_size", "block_start_hour", "bus", "hour",
            "delta_price", "delta_price_vs_block_lp"],
    "O13": ["run_id", "label", "block_size", "hour", "hour_of_day", "hours_into_block",
            "max_abs_delta_price", "mean_abs_delta_price",
            "max_abs_delta_price_vs_block_lp", "n_buses"],
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
        # ADMM *and* LP on the same blocks, with a reference solve and every bus's
        # price persisted: the input `price_error.parquet` (and O12 / O13) needs.
        # The gates are wide open on purpose -- 30 ADMM iterations on the tiny
        # fixture is nowhere near a dispatch, and the point here is the pipeline.
        cls.run_price = cls._dispatch_run(
            "run_price",
            {
                "selection": {
                    "blocks": [24],
                    "reference": "window",
                    "reference_window": {"start": 0, "hours": 48},
                },
                "output": {"save_hourly": "carrier_bus", "price_all_buses": True},
                "methods": {
                    "lp": {"enabled": True, "solver": "HIGHS", "timeout_s": 600},
                    "admm": {
                        "enabled": True,
                        "required": False,
                        "timeout_s": 900,
                        "max_imbalance_mw": 1.0e9,
                        "max_soc_residual_mwh": 1.0e9,
                        "solver_kwargs": {
                            "num_iterations": 30,
                            "rho_power": 1.0,
                            "minimum_iterations": 10,
                            "atol": 1e-8,
                            "rtol": 1e-8,
                        },
                    },
                },
            },
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
        if plot_id in ("O12", "O13"):
            return self.handles(self.run_price)
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
        self.assertEqual(len(plots_mod.PLOTS), 29)
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

    def test_o12_and_o13_read_the_price_error_decomposition(self):
        """The per-(bus, hour) table reaches both plots, load buses only."""
        runs = self.handles(self.run_price)
        table = runs[0].price_error()
        self.assertFalse(table.empty)
        for column in (
            "task_id", "method", "block_size", "block_start_hour", "bus", "hour",
            "is_load_bus", "lp_price", "admm_price", "delta_price", "block_lp_price",
            "delta_price_vs_block_lp",
        ):
            self.assertIn(column, table.columns)
        # Only ADMM blocks are compared, never the reference row against itself.
        self.assertEqual(set(table["method"]), {"admm"})
        self.assertNotIn("reference", set(table["block_size"]))
        # `price_all_buses` is on, so non-load buses are recorded and flagged.
        self.assertIn(False, set(table["is_load_bus"]))
        self.assertTrue((table["delta_price"] - (table["admm_price"] - table["lp_price"]))
                        .abs().max() < 1e-9)
        # The same-block LP is the control that separates the ADMM dual error from
        # the blocking error; this fixture solves both methods on every block.
        self.assertTrue(table["block_lp_price"].notna().all())
        self.assertTrue(
            (table["delta_price_vs_block_lp"]
             - (table["admm_price"] - table["block_lp_price"])).abs().max() < 1e-9
        )

        _fig, o12 = plots_mod.plot("O12", runs, block_size=24)
        self.assertTrue(set(o12["block_size"]) == {"24"})
        self.assertFalse(o12.empty)
        _fig, o13 = plots_mod.plot("O13", runs)
        self.assertFalse(o13.empty)
        self.assertTrue((o13["hours_into_block"] >= 0).all())
        # Load buses only in both: the flagged non-load rows are dropped.
        merged = table[table["is_load_bus"]]
        self.assertEqual(len(o12), len(merged[merged["block_size"] == "24"]))

    def test_price_error_summary_columns_reach_metrics_csv(self):
        run_dir = loader_mod.resolve_run_dir(self.run_price, runs_root=self.runs_root)
        frame = pd.read_csv(run_dir / "metrics.csv")
        for column in (
            "price_error_load_max_usd_per_mwh",
            "price_error_load_rms_usd_per_mwh",
            "price_error_all_bus_max_usd_per_mwh",
            "price_error_n_bus_hours",
            "price_error_load_max_vs_block_lp_usd_per_mwh",
            "price_error_load_rms_vs_block_lp_usd_per_mwh",
        ):
            self.assertIn(column, frame.columns)
        admm = frame[(frame["method"] == "admm") & (frame["block_size"].astype(str) == "24")]
        self.assertTrue(admm["price_error_load_max_usd_per_mwh"].notna().any())
        text = (run_dir / "CARD.md").read_text()
        self.assertIn("## ADMM dual accuracy vs the reference LP", text)
        self.assertIn("price_error.parquet", text)

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


# --- palette maths, so the separation test does not depend on the bundled skill --
# OKLab (Ottosson 2020) + the Machado, Oliveira & Fernandes (2009) CVD transforms
# at severity 1.0, i.e. the same computation the `dataviz` skill's
# validate_palette.py does; the thresholds below are that skill's gates.
MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
}
NORMAL_FLOOR = 15.0  # OKLab dE x100, unsimulated vision
CVD_FLOOR = 8.0      # OKLab dE x100, min(protan, deutan)


def _to_linear(hex_color):
    raw = str(hex_color).lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(raw[i:i + 2], 16) / 255
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return tuple(out)


def _oklab(rgb):
    r, g, b = rgb
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def _simulate(rgb, kind):
    matrix = MACHADO[kind]
    return tuple(
        min(1.0, max(0.0, sum(matrix[row][col] * rgb[col] for col in range(3))))
        for row in range(3)
    )


def delta_e(first, second, kind=None):
    """OKLab distance x100 between two hex colours, optionally CVD-simulated."""
    a, b = _to_linear(first), _to_linear(second)
    if kind is not None:
        a, b = _simulate(a, kind), _simulate(b, kind)
    return 100 * math.dist(_oklab(a), _oklab(b))


class TestFigureDefects(PlotFixture):
    """The nine figure-quality defects found by reading every phase-1 PNG."""

    def _titles(self, fig):
        return [ax.get_title() for ax in fig.axes if ax.get_title()]

    def test_o1_and_o7_keep_one_series_per_run(self):
        """Defect 1/3: available capacity and line loading do not depend on the solve.

        ``run_a`` solves 24 h blocks *and* a reference window, so before the fix
        O1 drew the same panel twice and O7 listed every line twice.
        """
        runs = self.handles(self.run_a)
        solves = runs[0].hourly(quantities=["available_capacity_mw"])
        self.assertGreater(len(set(zip(solves["method"], solves["block_size"]))), 1)

        for plot_id in ("O1", "O7"):
            with self.subTest(plot=plot_id):
                fig, table = plots_mod.plot(plot_id, runs)
                per_run = table.groupby("run_id")[["method", "block_size"]].nunique()
                self.assertTrue((per_run == 1).all().all(), f"{plot_id}: {per_run}")
                if plot_id == "O1":
                    self.assertEqual(len(fig.axes), len(runs))
                plt.close(fig)

        fig, table = plots_mod.plot("O1", self.handles(self.run_a, self.run_b))
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(len(table.drop_duplicates(["run_id", "method", "block_size"])), 2)
        plt.close(fig)

    def test_o10_has_no_twin_axes(self):
        """Defect 5: the objective gets its own panel, never a second y axis."""
        runs = self.handles(self.run_admm)
        fig, table = plots_mod.plot("O10", runs)
        blocks = table.drop_duplicates(["run_id", "block_size", "block_index"])
        self.assertEqual(len(fig.axes), 2 * len(blocks))
        # A twinx axes sits exactly on top of its parent; distinct panels do not.
        boxes = [tuple(round(v, 6) for v in ax.get_position().bounds) for ax in fig.axes]
        self.assertEqual(len(set(boxes)), len(boxes), boxes)
        for ax in fig.axes:
            self.assertEqual(len(ax.get_shared_x_axes().get_siblings(ax)), 2)
        objective = [ax for ax in fig.axes if "objective" in ax.get_ylabel()]
        self.assertEqual(len(objective), len(blocks))
        for ax in objective:
            self.assertEqual(ax.get_yscale(), "linear")
        plt.close(fig)

    def test_block_size_suffix_only_for_numeric_sizes(self):
        """Defect 7: a reference solve is not "reference h blocks"."""
        from experiments.ra.plots import operational

        self.assertEqual(operational._block_label(24), "24 h blocks")
        self.assertEqual(operational._block_label("168"), "168 h blocks")
        self.assertEqual(operational._block_label("reference"), "reference")
        self.assertEqual(operational._block_label("none"), "none")
        self.assertEqual(
            operational._facet_title("run_a", "lp", "reference"), "run_a - lp / reference"
        )

        for plot_id in ("O1", "O4", "O6"):
            with self.subTest(plot=plot_id):
                fig, _table = plots_mod.plot(
                    plot_id, self.handles(self.run_a), block_size="reference"
                )
                titles = self._titles(fig)
                self.assertTrue(titles, plot_id)
                for title in titles:
                    self.assertNotIn("reference h blocks", title)
                self.assertTrue(any("reference" in t for t in titles), titles)
                plt.close(fig)

    def test_battery_greens_are_distinguishable(self):
        """Defect 10: the three battery rows are a ramp, not three shades of one green."""
        family = [
            style_mod.CARRIER_COLORS[name]
            for name in ("battery", "4hr_battery_storage", "8hr_battery_storage")
        ]
        self.assertEqual(len(set(family)), 3, family)
        for first, second in itertools.combinations(family, 2):
            with self.subTest(pair=(first, second)):
                self.assertGreaterEqual(delta_e(first, second), NORMAL_FLOOR)
                for kind in MACHADO:
                    self.assertGreaterEqual(delta_e(first, second, kind), CVD_FLOOR)
        # The pre-fix values, kept as the regression this test exists for.
        self.assertLess(delta_e("#a4d600", "#b8ea04"), NORMAL_FLOOR)

    def test_o11_x_axis_is_logarithmic(self):
        """Defect 8: 24 h, 168 h and a full year are three decades apart."""
        fig, table = plots_mod.plot("O11", self.handles(self.run_a, self.run_b))
        self.assertGreater(table["hours_per_block"].max() / table["hours_per_block"].min(), 1)
        for ax in fig.axes:
            self.assertEqual(ax.get_xscale(), "log")
        plt.close(fig)

    def test_o2_switches_to_weekly_means_over_long_spans(self):
        """Defect 9: 8,736 hourly points in a 5-inch panel are not a figure."""
        from experiments.ra.plots import operational

        fig, table = plots_mod.plot("O2", self.handles(self.run_a))
        span = int(table["hour"].max() - table["hour"].min()) + 1
        self.assertLessEqual(span, operational.HOURLY_PANEL_HOURS)
        time_panel = fig.axes[0]
        self.assertIn("hourly", time_panel.get_title())
        # Colour is the quantity, so the three series are not one colour + dashes.
        colors = {line.get_color() for line in time_panel.get_lines()}
        self.assertEqual(colors, set(style_mod.SERIES_COLORS.values()))
        plt.close(fig)

        # A span longer than four weeks: 700 synthetic hours from hour 7 (the
        # shipped window start), so the bins are aligned to the window, not to
        # hour 0 of the year, and the 28-hour tail bin is dropped rather than
        # drawn as a mean over 28 hours.
        long_frame = pd.DataFrame({
            "hour": range(7, 707),
            "load_gw": [30.0 + (h % 24) for h in range(7, 707)],
        })
        self.assertGreater(
            operational._span_hours(long_frame), operational.HOURLY_PANEL_HOURS
        )
        weekly = operational._weekly_mean(long_frame, ["load_gw"])
        self.assertEqual(len(weekly), 4)
        self.assertEqual(list(weekly["hour"]), [7, 175, 343, 511])
        self.assertAlmostEqual(
            float(weekly["load_gw"].iloc[0]),
            float(long_frame["load_gw"].iloc[:168].mean()),
        )

    def test_suptitle_clears_the_first_facet_on_a_tall_figure(self):
        """Defect 6: reserve head room as a function of the facet count."""
        import matplotlib.pyplot as mpl

        for n in (1, 3, 9):
            with self.subTest(facets=n):
                fig, axes = mpl.subplots(n, 1, figsize=(9.5, 3.0 * n), squeeze=False)
                for ax in axes[:, 0]:
                    ax.set_title("facet")
                style_mod.finish(fig, "a figure-level title")
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                title = fig._suptitle.get_window_extent(renderer)
                top = axes[0, 0].title.get_window_extent(renderer)
                self.assertGreater(title.y0, top.y1, f"{n} facets: suptitle overlaps")
                mpl.close(fig)


class TestStackedPlots(PlotFixture):
    """The two standing rules for stacked-by-carrier plots (Kamran, 2026-09-09)."""

    def _year_table(self, plot_id):
        """A synthetic year-long table with the schema the plot's figure reads."""
        hours = list(range(7, 7 + 8736))
        carriers = ["solar", "CCGT", "nuclear", "battery"]
        rows = []
        for hour in hours:
            profile = 1.0 + (hour % 24) / 24.0
            for carrier in carriers:
                rows.append({
                    "run_id": "run", "label": "run", "method": "lp", "block_size": "24",
                    "year": 2020, "hour": hour, "carrier": carrier,
                    "value": profile * (1.0 + carriers.index(carrier)),
                })
        frame = pd.DataFrame(rows)
        if plot_id == "O1":
            return frame.rename(columns={"value": "available_gw"})
        frame = frame.rename(columns={"value": "value_gw"})
        frame["series"] = "generation"
        load = frame[frame["carrier"] == "solar"].assign(
            series="load", carrier="load", value_gw=10.0
        )
        charge = frame[frame["carrier"] == "solar"].assign(
            series="storage_charge", carrier="battery", value_gw=1.0
        )
        return pd.concat([frame, load, charge], ignore_index=True)

    def test_stack_order_is_pinned_and_warns_on_unknown_carriers(self):
        order = style_mod.CARRIER_STACK_ORDER
        self.assertEqual(len(set(order)), len(order))
        # baseload < renewables < batteries < thermal < trade
        self.assertLess(order.index("nuclear"), order.index("solar"))
        self.assertLess(order.index("PHS"), order.index("solar"))
        self.assertLess(order.index("offwind_floating"), order.index("battery"))
        self.assertLess(order.index("8hr_battery_storage"), order.index("CCGT"))
        self.assertLess(order.index("oil"), order.index("imports"))

        picked = style_mod.stack_order(["CCGT", "solar", "battery", "nuclear"])
        self.assertEqual(picked, ["nuclear", "solar", "battery", "CCGT"])
        # Order does not depend on the order the carriers arrive in.
        self.assertEqual(style_mod.stack_order(reversed(picked)), picked)

        style_mod._WARNED.discard("stack:mystery_fuel")
        with self.assertLogs(style_mod.logger, level="WARNING"):
            out = style_mod.stack_order(["CCGT", "mystery_fuel", "aardvark"])
        self.assertEqual(out, ["CCGT", "aardvark", "mystery_fuel"])

    def test_rendered_stacks_follow_the_pinned_order(self):
        """O1 and O4 stack and label in CARRIER_STACK_ORDER, not alphabetically."""
        for plot_id in ("O1", "O4"):
            with self.subTest(plot=plot_id):
                fig, _table = plots_mod.plot(plot_id, self.handles(self.run_a))
                labels = [
                    text.get_text()
                    for text in fig.axes[0].get_legend().get_texts()
                    if text.get_text() in style_mod.CARRIER_STACK_ORDER
                ]
                self.assertTrue(labels, plot_id)
                self.assertEqual(labels, style_mod.stack_order(labels), plot_id)
                self.assertNotEqual(labels, sorted(labels), plot_id)
                plt.close(fig)

    def test_a_year_long_stack_becomes_a_twelve_panel_monthly_grid(self):
        """Rule 1: no stacked plot draws a year of hourly bands."""
        from experiments.ra.plots import operational

        for plot_id, build in (("O1", operational._o1_figure), ("O4", operational._o4_figure)):
            with self.subTest(plot=plot_id):
                table = self._year_table(plot_id)
                self.assertTrue(operational.is_monthly_view(table))
                fig = build(table)
                self.assertEqual(len(fig.axes), 12)
                self.assertEqual(
                    [ax.get_title() for ax in fig.axes], list(operational.MONTH_LABELS)
                )
                for ax in fig.axes:
                    self.assertEqual(ax.get_xlim(), (0.0, 23.0))
                    self.assertEqual(len(ax.get_shared_y_axes().get_siblings(ax)), 12)
                plt.close(fig)

        # ...and a window of four weeks or less still gets the hourly stack.
        short = self._year_table("O1")
        short = short[short["hour"] < 7 + 3 * 168]
        self.assertFalse(operational.is_monthly_view(short))
        fig = operational._o1_figure(short)
        self.assertEqual(len(fig.axes), 1)
        self.assertNotIn(fig.axes[0].get_title(), operational.MONTH_LABELS)
        plt.close(fig)

    def test_monthly_profile_uses_local_pacific_hours_and_month_boundaries(self):
        from experiments.ra.plots import operational

        frame = pd.DataFrame({"hour": [7, 8, 30, 24 * 31 + 7, 8759], "year": [2020] * 5})
        out = operational.local_calendar(frame)
        # The shipped window starts at hour 7 = local midnight.
        self.assertEqual(list(out["hour_of_day"]), [0, 1, 23, 0, 16])
        # Hours 7, 8 and 30 are all 1-2 Jan; hour 24*31+7 is 1 Feb; 8759 is 31 Dec.
        self.assertEqual(list(out["month"]), [1, 1, 1, 2, 12])


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

    def test_cli_data_out_splits_pngs_from_tables(self):
        """``--data-out``: images in the figures dir, tables in raw_data/."""
        figures = self.out / "figs"
        tables = figures / "raw_data"
        code = cli.main(
            [
                "plot",
                "--run-id", self.run_a,
                "--runs-root", str(self.runs_root),
                "--plot", "O4",
                "--out", str(figures),
                "--data-out", str(tables),
            ]
        )
        self.assertEqual(code, 0)
        pngs = list(figures.glob("O4_*.png"))
        csvs = list(tables.glob("O4_*.csv"))
        self.assertEqual(len(pngs), 1)
        self.assertEqual(len(csvs), 1)
        self.assertEqual(pngs[0].stem, csvs[0].stem)
        self.assertFalse(list(figures.glob("*.csv")))

    def test_cli_list(self):
        self.assertEqual(cli.main(["plot", "--list"]), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
