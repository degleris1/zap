"""Tests for `zap.reliability.outages` (WP2).

Everything here is hermetic: the fixtures write synthetic `static/*.csv` tables
into a temporary directory, so the real `data/` tree is never required. All
tests use <= 48 hours and <= 4 draws except `test_long_run_availability_*`,
which needs 8760 hours for a statistical check but touches no solver.
"""

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import yaml
import zarr

from zap.reliability import outages as ox

GEN_COLUMNS = ["name", "bus", "p_nom", "carrier"]
SU_COLUMNS = ["name", "bus", "p_nom", "carrier", "max_hours"]

# A small dataset with the same *carrier shape* as a pypsa-usa export: thermal,
# VRE (excluded), an import carrier (excluded), storage, and a zero-p_nom row.
DEFAULT_GENERATORS = [
    ("z1 CCGT", "z1", 1000.0, "CCGT"),
    ("z2 CCGT", "z2", 250.0, "CCGT"),
    ("z1 OCGT", "z1", 0.0, "OCGT"),
    ("z1 solar", "z1", 5000.0, "solar"),
    ("z2 onwind", "z2", 2000.0, "onwind"),
    ("z1_imports unspecified_imports", "z1_imports", 3000.0, "unspecified_imports"),
]
DEFAULT_STORAGE = [
    ("z1 battery", "z1", 400.0, "battery", 4.0),
    ("z2 PHS", "z2", 1200.0, "PHS", 12.0),
]


def write_static(
    root: Path,
    generators=DEFAULT_GENERATORS,
    storage=DEFAULT_STORAGE,
) -> Path:
    """Write the two static tables `build_unit_pool` reads."""
    static = Path(root) / "static"
    static.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(generators, columns=GEN_COLUMNS).to_csv(static / "generators.csv", index=False)
    pd.DataFrame(storage, columns=SU_COLUMNS).to_csv(static / "storage_units.csv", index=False)
    return Path(root)


def params_dict(**overrides) -> dict:
    base = {
        "version": 1,
        "reviewed": False,
        "pool_multiplier": 3.0,
        "min_units_per_row": 3,
        "min_pool_capacity_mw": 0.0,
        "excluded_carriers": ["solar", "onwind", "offwind_floating", "unspecified_imports"],
        "carriers": {
            "CCGT": {
                "unit_size_mw": 250,
                "forced_outage_rate": 0.045,
                "mttr_h": 50,
                "source": "test",
            },
            "OCGT": {
                "unit_size_mw": 100,
                "forced_outage_rate": 0.040,
                "mttr_h": 40,
                "source": "test",
            },
            "battery": {
                "unit_size_mw": 50,
                "forced_outage_rate": 0.020,
                "mttr_h": 24,
                "source": "test",
            },
            "PHS": {
                "unit_size_mw": 200,
                "forced_outage_rate": 0.050,
                "mttr_h": 60,
                "source": "test",
            },
        },
    }
    base.update(overrides)
    return base


def make_params(tmp: Path, **overrides) -> ox.OutageParams:
    path = Path(tmp) / "params.yaml"
    path.write_text(yaml.safe_dump(params_dict(**overrides)))
    return ox.load_outage_params(path)


class OutageTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="zap_outages_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.dataset = write_static(self.tmp / "ds")
        self.params = make_params(self.tmp)


class TestParams(OutageTestCase):
    def test_shipped_params_load(self):
        """The checked-in parameter file parses and is still unreviewed."""
        p = ox.load_outage_params()
        self.assertEqual(p.version, 1)
        self.assertFalse(p.reviewed)
        self.assertEqual(len(p.sha256), 64)
        for name, cp in p.carriers.items():
            self.assertTrue(cp.source, f"{name} has no source")

    def test_mttf_and_stationary_probability(self):
        for cp in list(self.params.carriers.values()) + list(
            ox.load_outage_params().carriers.values()
        ):
            f = cp.forced_outage_rate
            self.assertAlmostEqual(cp.mttf_h, cp.mttr_h * (1 - f) / f, places=12)
            self.assertAlmostEqual(cp.mttf_h / (cp.mttf_h + cp.mttr_h), 1 - f, places=12)
            self.assertGreater(cp.p_fail, 0.0)
            self.assertLess(cp.p_fail, 1.0)
            self.assertGreater(cp.p_repair, 0.0)
            self.assertLessEqual(cp.p_repair, 1.0)
            # The chain's stationary unavailability must be *exactly* the FOR.
            # `1 - exp(-1/MTTF)` fails this by ~1 % of FOR; linear rates do not.
            self.assertAlmostEqual(cp.p_fail / (cp.p_fail + cp.p_repair), f, places=12)
            self.assertAlmostEqual(cp.p_fail, 1.0 / cp.mttf_h, places=15)
            self.assertAlmostEqual(cp.p_repair, 1.0 / cp.mttr_h, places=15)

    def test_missing_carrier_raises(self):
        """A dataset carrier in neither list is a hard error."""
        params = make_params(self.tmp, excluded_carriers=["solar", "onwind"])
        with self.assertRaises(KeyError) as ctx:
            ox.build_unit_pool(self.dataset, params)
        self.assertIn("unspecified_imports", str(ctx.exception))


class TestUnitPool(OutageTestCase):
    def test_pool_sizing_and_offsets(self):
        pool = ox.build_unit_pool(self.dataset, self.params)
        # 1000 MW CCGT at 250 MW/unit, x3 -> 12 units; 250 -> 3; 0 -> min 3
        self.assertEqual(pool.row_units["z1 CCGT"], 12)
        self.assertEqual(pool.row_units["z2 CCGT"], 3)
        self.assertEqual(pool.row_units["z1 OCGT"], 3)  # min_units_per_row for p_nom == 0
        self.assertEqual(pool.row_units["z1 battery"], math.ceil(3 * 400 / 50))
        self.assertEqual(pool.row_units["z2 PHS"], math.ceil(3 * 1200 / 200))

        # Offsets are the running cumulative sum, slices are disjoint and cover the axis.
        offset = 0
        for row, n in pool.row_units.items():
            self.assertEqual(pool.row_offset[row], offset)
            offset += n
        self.assertEqual(offset, pool.n_units)
        self.assertEqual(len(pool.table), sum(pool.row_units.values()))
        self.assertEqual(list(pool.table.columns), ox.UNIT_TABLE_COLUMNS)
        self.assertEqual(pool.table["unit_id"].iloc[0], "z1 CCGT#0")
        self.assertEqual(pool.table["unit_id"].nunique(), pool.n_units)

    def test_sidecar_follows_the_store_not_the_dataset(self):
        """A sensitivity store built with --out must not clobber the canonical sidecar."""
        canonical = ox.default_units_csv(self.dataset / "outages.zarr")
        self.assertEqual(canonical.name, "outage_units.csv")
        alt = self.tmp / "elsewhere" / "sensitivity.zarr"
        alt.parent.mkdir(parents=True, exist_ok=True)
        ox.init_store(
            self.dataset,
            out_path=alt,
            years=[2020],
            draws=1,
            base_seed=1,
            params=self.params,
            hours_per_year=24,
        )
        self.assertFalse(canonical.exists())
        self.assertTrue((alt.parent / "sensitivity_units.csv").exists())

    def test_pool_covers_storage_and_excludes_vre(self):
        pool = ox.build_unit_pool(self.dataset, self.params)
        rows = set(pool.row_offset)
        self.assertIn("z1 battery", rows)
        self.assertIn("z2 PHS", rows)
        for absent in ["z1 solar", "z2 onwind", "z1_imports unspecified_imports"]:
            self.assertNotIn(absent, rows)
        comps = set(pool.table["component"])
        self.assertEqual(comps, {"Generator", "StorageUnit"})


class TestSampling(OutageTestCase):
    def _uniform_pool(self, n_units: int, forced_outage_rate=0.05, mttr_h=50.0):
        ds = write_static(
            self.tmp / f"uni{n_units}",
            generators=[("g", "z1", float(n_units), "CCGT")],
            storage=[],
        )
        params = make_params(
            self.tmp / f"uni{n_units}",
            pool_multiplier=1.0,
            min_units_per_row=1,
            carriers={
                "CCGT": {
                    "unit_size_mw": 1.0,
                    "forced_outage_rate": forced_outage_rate,
                    "mttr_h": mttr_h,
                    "source": "test",
                }
            },
            excluded_carriers=[],
        )
        pool = ox.build_unit_pool(ds, params)
        self.assertEqual(pool.n_units, n_units)
        return pool, params

    def test_initial_state_is_stationary(self):
        """Hour 0 alone must already sit at 1 - FOR (catches a flipped comparison)."""
        n_units, forced_outage_rate = 20_000, 0.05
        pool, params = self._uniform_pool(n_units, forced_outage_rate=forced_outage_rate)
        hour0 = np.concatenate(
            [
                ox.sample_chunk(pool, params, year=2020, draw=d, n_hours=1, base_seed=555)[:, 0]
                for d in range(3)
            ]
        )
        n = hour0.size
        expected = 1.0 - forced_outage_rate
        sigma = math.sqrt(expected * forced_outage_rate / n)
        self.assertLess(abs(hour0.mean() - expected), 3.0 * sigma)

    def test_long_run_availability_matches_for(self):
        n_units, forced_outage_rate, mttr_h = 300, 0.05, 50.0
        n_hours = 8760
        pool, params = self._uniform_pool(
            n_units, forced_outage_rate=forced_outage_rate, mttr_h=mttr_h
        )
        up = ox.sample_chunk(pool, params, year=2020, draw=0, n_hours=n_hours, base_seed=12345)
        self.assertEqual(up.shape, (n_units, n_hours))

        # Standard error of the mean of an AR(1)-like two-state chain: the
        # independent-sample variance inflated by the integrated autocorrelation
        # time 2 / (p_fail + p_repair) - 1.
        cp = params.carriers["CCGT"]
        expected = 1.0 - forced_outage_rate
        tau = 2.0 / (cp.p_fail + cp.p_repair) - 1.0
        sigma = math.sqrt(expected * forced_outage_rate * tau / (n_units * n_hours))
        self.assertLess(abs(up.mean() - expected), 4.0 * sigma)
        self.assertLess(4.0 * sigma, 0.01, "tolerance is no longer tight")

    def test_transition_rates(self):
        pool, params = self._uniform_pool(300, forced_outage_rate=0.05)
        cp = params.carriers["CCGT"]
        up = ox.sample_chunk(pool, params, year=2020, draw=0, n_hours=8760, base_seed=12345)
        prev, nxt = up[:, :-1].astype(bool), up[:, 1:].astype(bool)
        p_fail = (prev & ~nxt).sum() / prev.sum()
        p_repair = (~prev & nxt).sum() / (~prev).sum()
        self.assertLess(abs(p_fail - cp.p_fail) / cp.p_fail, 0.15)
        self.assertLess(abs(p_repair - cp.p_repair) / cp.p_repair, 0.15)

    def test_chunk_is_reproducible_in_isolation(self):
        pool = ox.build_unit_pool(self.dataset, self.params)
        a = ox.sample_chunk(pool, self.params, year=2020, draw=3, n_hours=48, base_seed=7)
        b = ox.sample_chunk(pool, self.params, year=2020, draw=3, n_hours=48, base_seed=7)
        np.testing.assert_array_equal(a, b)
        # Different (year, draw) and different seeds give different chunks.
        c = ox.sample_chunk(pool, self.params, year=2021, draw=3, n_hours=48, base_seed=7)
        d = ox.sample_chunk(pool, self.params, year=2020, draw=4, n_hours=48, base_seed=7)
        e = ox.sample_chunk(pool, self.params, year=2020, draw=3, n_hours=48, base_seed=8)
        self.assertFalse(np.array_equal(a, c))
        self.assertFalse(np.array_equal(a, d))
        self.assertFalse(np.array_equal(a, e))

    def test_store_draw_slices_are_independent_of_batching(self):
        """Generating draws 0-3 together == generating draw 3 alone."""
        kw = {
            "years": [2020],
            "draws": 4,
            "base_seed": 11,
            "params": self.params,
            "hours_per_year": 48,
        }
        all_at_once = ox.generate(
            self.dataset, out_path=self.tmp / "all.zarr", init=True, verbose=False, **kw
        )
        single = self.tmp / "one.zarr"
        ox.init_store(self.dataset, out_path=single, **kw)
        # Shard the four jobs into four shards and run only the last one.
        ox.generate(self.dataset, out_path=single, chunk=(4, 4), init=False, verbose=False, **kw)
        a = zarr.open_group(str(all_at_once), mode="r")["available"][0, 3]
        b = zarr.open_group(str(single), mode="r")["available"][0, 3]
        np.testing.assert_array_equal(a, b)
        # Un-generated slices keep the "not generated" fill value.
        self.assertTrue(np.all(zarr.open_group(str(single), mode="r")["available"][0, 0] == 255))

    def test_pool_growth_preserves_streams(self):
        pool = ox.build_unit_pool(self.dataset, self.params)
        base = ox.sample_chunk(pool, self.params, year=2020, draw=1, n_hours=48, base_seed=99)

        grown_dir = write_static(
            self.tmp / "grown",
            generators=DEFAULT_GENERATORS,
            storage=DEFAULT_STORAGE + [("z2 battery new", "z2", 830.0, "battery", 4.0)],
        )
        grown = ox.build_unit_pool(grown_dir, self.params)
        self.assertEqual(grown.n_units, pool.n_units + 50)
        grown_up = ox.sample_chunk(grown, self.params, year=2020, draw=1, n_hours=48, base_seed=99)
        np.testing.assert_array_equal(grown_up[: pool.n_units], base)


class TestRowAvailability(OutageTestCase):
    def _pool_with(self, unit_size, p_nom, n_units):
        ds = write_static(self.tmp / "rm", generators=[("g", "z1", p_nom, "CCGT")], storage=[])
        params = make_params(
            self.tmp / "rm",
            pool_multiplier=1.0,
            min_units_per_row=n_units,
            excluded_carriers=[],
            carriers={
                "CCGT": {
                    "unit_size_mw": unit_size,
                    "forced_outage_rate": 0.05,
                    "mttr_h": 50,
                    "source": "test",
                }
            },
        )
        pool = ox.build_unit_pool(ds, params)
        self.assertEqual(pool.n_units, n_units)
        return pool

    def test_capacity_mapping_with_remainder(self):
        pool = self._pool_with(unit_size=100.0, p_nom=250.0, n_units=5)
        caps = pd.Series({"g": 250.0})

        up = np.zeros((5, 1), dtype=np.uint8)
        up[[0, 1], 0] = 1  # weights [1, 1, 0.5] -> (1 + 1 + 0) / 2.5
        self.assertAlmostEqual(ox.row_availability(up, pool, caps, ["g"])[0, 0], 0.8, places=12)

        up = np.zeros((5, 1), dtype=np.uint8)
        up[[1, 2], 0] = 1  # (0 + 1 + 0.5) / 2.5
        self.assertAlmostEqual(ox.row_availability(up, pool, caps, ["g"])[0, 0], 0.6, places=12)

        # Exact multiples give unit weights.
        self.assertAlmostEqual(
            ox.row_availability(
                np.ones((5, 1), dtype=np.uint8), pool, pd.Series({"g": 200.0}), ["g"]
            )[0, 0],
            1.0,
            places=12,
        )

        # Zero capacity contributes nothing and reads as fully available.
        zero = ox.row_availability(
            np.zeros((5, 1), dtype=np.uint8), pool, pd.Series({"g": 0.0}), ["g"]
        )
        self.assertEqual(zero[0, 0], 1.0)

        # A capacity too small to fill one unit still maps onto that one unit.
        tiny = ox.row_availability(
            np.ones((5, 1), dtype=np.uint8), pool, pd.Series({"g": 1e-12}), ["g"]
        )
        self.assertAlmostEqual(tiny[0, 0], 1.0, places=12)
        tiny_down = ox.row_availability(
            np.zeros((5, 1), dtype=np.uint8), pool, pd.Series({"g": 1e-12}), ["g"]
        )
        self.assertAlmostEqual(tiny_down[0, 0], 0.0, places=12)

        # A design that outgrew the pool is an error, not a silent clip.
        with self.assertRaises(ValueError):
            ox.row_availability(up, pool, pd.Series({"g": 10_000.0}), ["g"])

    def test_rows_absent_from_pool_are_available(self):
        pool = ox.build_unit_pool(self.dataset, self.params)
        up = np.zeros((pool.n_units, 3), dtype=np.uint8)
        avail = ox.row_availability(up, pool, pd.Series(dtype=float), ["z1 solar"])
        np.testing.assert_array_equal(avail, np.ones((3, 1)))

    def test_shape_and_range(self):
        pool = ox.build_unit_pool(self.dataset, self.params)
        up = ox.sample_chunk(pool, self.params, year=2020, draw=0, n_hours=24, base_seed=3)
        rows = ["z1 CCGT", "z2 CCGT", "z1 battery", "z1 solar"]
        caps = pd.Series({"z1 CCGT": 1000.0, "z2 CCGT": 250.0, "z1 battery": 400.0})
        avail = ox.row_availability(up, pool, caps, rows)
        self.assertEqual(avail.shape, (24, 4))
        self.assertTrue(np.all(avail >= 0.0) and np.all(avail <= 1.0))
        np.testing.assert_array_equal(avail[:, 3], 1.0)


class TestStore(OutageTestCase):
    def _gen(self, path=None, draws=2, chunk=(1, 1), **kw):
        kwargs = {
            "years": [2020, 2021],
            "draws": draws,
            "base_seed": 4242,
            "params": self.params,
            "hours_per_year": 48,
            "verbose": False,
        }
        kwargs.update(kw)
        return ox.generate(
            self.dataset, out_path=path or (self.tmp / "o.zarr"), chunk=chunk, **kwargs
        )

    def test_store_schema(self):
        path = self._gen(init=True)
        root = zarr.open_group(str(path), mode="r")
        pool = ox.build_unit_pool(self.dataset, self.params)

        self.assertEqual(root["available"].shape, (2, 2, 48, pool.n_units))
        self.assertEqual(root["available"].dtype, np.uint8)
        self.assertEqual(root["available"].chunks, (1, 1, 48, pool.n_units))
        self.assertEqual(root["available"].fill_value, 255)
        comp = root["available"].compressor
        self.assertEqual(comp.cname, "zstd")
        self.assertEqual(comp.clevel, 5)
        self.assertEqual(comp.shuffle, numcodecs.Blosc.BITSHUFFLE)

        for key in [
            "generator_version",
            "dataset",
            "created_utc",
            "zap_commit",
            "base_seed",
            "params_sha256",
            "params",
            "pool_multiplier",
            "min_units_per_row",
            "min_pool_capacity_mw",
            "weather_years",
            "n_draws",
            "hours_per_year",
            "completed",
        ]:
            self.assertIn(key, root.attrs, key)
        self.assertEqual(root.attrs["base_seed"], 4242)
        self.assertEqual(root.attrs["weather_years"], [2020, 2021])

        np.testing.assert_array_equal(root["weather_year"][:], [2020, 2021])
        np.testing.assert_array_equal(root["draw"][:], [0, 1])
        np.testing.assert_array_equal(root["hour"][:], np.arange(48))
        for name in ["unit_id", "unit_row", "unit_carrier", "unit_bus", "unit_component"]:
            self.assertEqual(len(root[name]), pool.n_units)
        np.testing.assert_array_equal(root["unit_slot"][:], pool.table["slot"].to_numpy())

        sidecar = pd.read_csv(ox.default_units_csv(path))
        self.assertEqual(list(sidecar["unit_id"]), list(np.asarray(root["unit_id"][:])))
        self.assertEqual(list(sidecar["carrier"]), list(np.asarray(root["unit_carrier"][:])))
        self.assertEqual(len(sidecar), pool.n_units)

        # Everything was generated; no fill values left.
        self.assertFalse(np.any(np.asarray(root["available"][:]) == 255))
        self.assertEqual(
            sorted(map(tuple, root.attrs["completed"])),
            [(2020, 0), (2020, 1), (2021, 0), (2021, 1)],
        )

    def test_shards_compose(self):
        whole = self._gen(path=self.tmp / "whole.zarr", init=True, chunk=(1, 1))
        sharded = self.tmp / "sharded.zarr"
        ox.init_store(
            self.dataset,
            out_path=sharded,
            years=[2020, 2021],
            draws=2,
            base_seed=4242,
            params=self.params,
            hours_per_year=48,
        )
        self._gen(path=sharded, chunk=(1, 2))
        self._gen(path=sharded, chunk=(2, 2))

        a = np.asarray(zarr.open_group(str(whole), mode="r")["available"][:])
        b = np.asarray(zarr.open_group(str(sharded), mode="r")["available"][:])
        np.testing.assert_array_equal(a, b)
        self.assertEqual(
            sorted(map(tuple, zarr.open_group(str(sharded), mode="r").attrs["completed"])),
            [(2020, 0), (2020, 1), (2021, 0), (2021, 1)],
        )

    def test_shards_partition_jobs(self):
        jobs = ox._shard_jobs([2020, 2021], 3, (1, 1))
        self.assertEqual(len(jobs), 6)
        union = []
        for k in range(1, 4):
            union += ox._shard_jobs([2020, 2021], 3, (k, 3))
        self.assertEqual(sorted(union), sorted(jobs))
        self.assertEqual(len(set(union)), len(union))

    def test_generate_refuses_overwrite(self):
        path = self._gen(init=True)
        with self.assertRaises(FileExistsError):
            self._gen(path=path, init=True)
        # ... but --overwrite recreates it.
        self._gen(path=path, init=True, overwrite=True)

    def test_generate_requires_init(self):
        with self.assertRaises(FileNotFoundError):
            self._gen(path=self.tmp / "missing.zarr", init=False)

    def test_resume_skips_completed(self):
        path = self.tmp / "resume.zarr"
        ox.init_store(
            self.dataset,
            out_path=path,
            years=[2020, 2021],
            draws=2,
            base_seed=4242,
            params=self.params,
            hours_per_year=48,
        )
        self._gen(path=path, chunk=(1, 2))
        first = np.asarray(zarr.open_group(str(path), mode="r")["available"][0, 0]).copy()

        calls = []
        real = ox.sample_chunk

        def counting(*args, **kwargs):
            calls.append((kwargs["year"], kwargs["draw"]))
            return real(*args, **kwargs)

        ox.sample_chunk = counting
        try:
            self._gen(path=path, chunk=(1, 1))
        finally:
            ox.sample_chunk = real

        self.assertEqual(sorted(calls), [(2021, 0), (2021, 1)])
        np.testing.assert_array_equal(
            np.asarray(zarr.open_group(str(path), mode="r")["available"][0, 0]), first
        )

    def test_seed_mismatch_is_rejected(self):
        path = self._gen(init=True)
        with self.assertRaises(ValueError):
            self._gen(path=path, base_seed=1)


class TestUcap(OutageTestCase):
    def test_ucap_csv(self):
        # A big row (many units) so the empirical mean concentrates on 1 - FOR.
        ds = write_static(
            self.tmp / "ucap",
            generators=[
                ("big CCGT a", "z1", 10_000.0, "CCGT"),
                ("big CCGT b", "z1", 5_000.0, "CCGT"),
                ("empty OCGT", "z1", 0.0, "OCGT"),
            ],
            storage=[("z1 battery", "z1", 400.0, "battery", 4.0)],
        )
        path = ox.generate(
            ds,
            out_path=ds / "outages.zarr",
            years=[2020],
            draws=2,
            base_seed=2024,
            params=self.params,
            hours_per_year=8760,
            init=True,
            verbose=False,
        )
        df = ox.write_ucap(ds, store_path=path)

        self.assertEqual(
            list(df.columns),
            [
                "component",
                "row",
                "carrier",
                "bus",
                "p_nom_mw",
                "unit_size_mw",
                "n_units_row",
                "forced_outage_rate",
                "ucap_analytic",
                "ucap_empirical",
                "n_samples",
                "ucap_carrier_bus_empirical",
            ],
        )
        self.assertTrue((ds / "ucap.csv").exists())

        big = df.set_index("row").loc["big CCGT a"]
        self.assertLess(abs(big["ucap_empirical"] - (1 - 0.045)), 0.03)
        self.assertEqual(big["n_samples"], 2 * 8760)

        # A zero-capacity row falls back to the analytic value with no samples.
        empty = df.set_index("row").loc["empty OCGT"]
        self.assertEqual(empty["n_samples"], 0)
        self.assertAlmostEqual(empty["ucap_empirical"], 1 - 0.040, places=12)

        # The (carrier, bus) aggregate is the capacity-weighted mean of its rows.
        ccgt = df[df["carrier"] == "CCGT"]
        expected = (ccgt["p_nom_mw"] * ccgt["ucap_empirical"]).sum() / ccgt["p_nom_mw"].sum()
        for v in ccgt["ucap_carrier_bus_empirical"]:
            self.assertAlmostEqual(v, expected, places=12)

        # The zero-capacity OCGT is alone in its (carrier, bus) group.
        self.assertAlmostEqual(
            empty["ucap_carrier_bus_empirical"], empty["ucap_empirical"], places=12
        )

    def test_ucap_requires_a_generated_store(self):
        with self.assertRaises(FileNotFoundError):
            ox.write_ucap(self.dataset, store_path=self.tmp / "nope.zarr")


class TestCli(OutageTestCase):
    def test_dry_run_writes_nothing(self):
        store = self.tmp / "cli.zarr"
        params_path = self.tmp / "params.yaml"
        rc = ox.main(
            [
                "init",
                "--dataset-dir",
                str(self.dataset),
                "--params",
                str(params_path),
                "--out",
                str(store),
                "--years",
                "2020",
                "--draws",
                "2",
                "--seed",
                "5",
                "--hours-per-year",
                "48",
                "--dry-run",
            ]
        )
        self.assertEqual(rc, 0)
        self.assertFalse(store.exists())

    def test_init_generate_ucap_roundtrip(self):
        store = self.tmp / "cli.zarr"
        params_path = self.tmp / "params.yaml"
        common = ["--dataset-dir", str(self.dataset), "--params", str(params_path)]
        self.assertEqual(
            ox.main(
                ["init", *common, "--out", str(store)]
                + ["--years", "2020", "--draws", "2", "--seed", "5", "--hours-per-year", "48"]
            ),
            0,
        )
        self.assertEqual(ox.main(["generate", *common, "--out", str(store), "--chunk", "1/1"]), 0)
        self.assertEqual(
            ox.main(["ucap", *common, "--out", str(store), "--out-csv", str(self.tmp / "u.csv")]),
            0,
        )
        self.assertTrue((self.tmp / "u.csv").exists())


if __name__ == "__main__":
    unittest.main()
