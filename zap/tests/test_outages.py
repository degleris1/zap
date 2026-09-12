"""Tests for `zap.reliability.outages` (outage-pool spec WP-O2).

Everything here is hermetic: the fixtures write synthetic `static/*.csv` tables
into a temporary directory, so the real `data/` tree is never required. All
tests use <= 168 hours except `TestMarginalLaw`, which needs a full year for the
E1 statistical check but touches no solver and no store.
"""

import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zap.reliability import keys as K
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


def write_static(root: Path, generators=DEFAULT_GENERATORS, storage=DEFAULT_STORAGE) -> Path:
    """Write the two static tables ``row_specs`` reads."""
    static = Path(root) / "static"
    static.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(generators, columns=GEN_COLUMNS).to_csv(static / "generators.csv", index=False)
    pd.DataFrame(storage, columns=SU_COLUMNS).to_csv(static / "storage_units.csv", index=False)
    return Path(root)


def params_dict(**overrides) -> dict:
    base = {
        "version": 2,
        "reviewed": False,
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
        self.addCleanup(ox.unit_cache_clear)
        ox.unit_cache_clear()
        self.dataset = write_static(self.tmp / "ds")
        self.params = make_params(self.tmp)
        self.rows, _ = ox.dataset_row_specs(self.dataset, self.params)
        self.by_name = {r.name: r for r in self.rows}


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class TestParams(OutageTestCase):
    def test_shipped_params_load_at_version_2(self):
        p = ox.load_outage_params()
        self.assertEqual(p.version, 2)
        self.assertFalse(p.reviewed)
        self.assertEqual(len(p.sha256), 64)
        for name, cp in p.carriers.items():
            self.assertTrue(cp.source, f"{name} has no source")

    def test_version_1_pool_keys_are_refused(self):
        """`pool_multiplier` et al. are gone; a v1 file must fail loudly."""
        raw = params_dict(pool_multiplier=3.0, min_units_per_row=3, min_pool_capacity_mw=1e4)
        with self.assertRaises(KeyError) as ctx:
            ox.OutageParams.from_dict(raw)
        self.assertIn("pool_multiplier", str(ctx.exception))

    def test_mttf_and_stationary_probability(self):
        for cp in list(self.params.carriers.values()) + list(
            ox.load_outage_params().carriers.values()
        ):
            f = cp.forced_outage_rate
            self.assertAlmostEqual(cp.mttf_h, cp.mttr_h * (1 - f) / f, places=12)
            self.assertAlmostEqual(cp.mttf_h / (cp.mttf_h + cp.mttr_h), 1 - f, places=12)
            # The chain's stationary unavailability must be *exactly* the FOR.
            self.assertAlmostEqual(cp.p_fail / (cp.p_fail + cp.p_repair), f, places=12)
            self.assertAlmostEqual(cp.p_fail, 1.0 / cp.mttf_h, places=15)
            self.assertAlmostEqual(cp.p_repair, 1.0 / cp.mttr_h, places=15)

    def test_missing_carrier_raises(self):
        params = make_params(self.tmp, excluded_carriers=["solar", "onwind"])
        with self.assertRaises(KeyError) as ctx:
            ox.dataset_row_specs(self.dataset, params)
        self.assertIn("unspecified_imports", str(ctx.exception))

    def test_fingerprint_is_the_canonical_artefact(self):
        fp = ox.params_fingerprint(self.params, "slot-v1", 20260908)
        self.assertEqual(fp["scheme"], "slot-v1")
        self.assertEqual(fp["base_seed"], 20260908)
        self.assertEqual(fp["params_sha256"], self.params.sha256)


# ---------------------------------------------------------------------------
# Slot arithmetic
# ---------------------------------------------------------------------------


class TestSlotArithmetic(OutageTestCase):
    def test_slot_count(self):
        self.assertEqual(ox.slot_count(0.0, 250.0), 0)
        self.assertEqual(ox.slot_count(1.0, 250.0), 1)
        self.assertEqual(ox.slot_count(250.0, 250.0), 1)  # exact multiple, no extra unit
        self.assertEqual(ox.slot_count(250.1, 250.0), 2)
        self.assertEqual(ox.slot_count(1000.0, 250.0), 4)

    def test_row_weights_partial_last_slot(self):
        w = ox._row_weights(300.0, 250.0)
        np.testing.assert_allclose(w, [1.0, 0.2])
        self.assertAlmostEqual(float(w.sum()), 300.0 / 250.0)

    def test_row_weights_zero_capacity(self):
        self.assertEqual(ox._row_weights(0.0, 250.0).size, 0)

    def test_no_pool_bound(self):
        """Nothing can overflow: a 30 GW row simply asks for 120 slots."""
        self.assertEqual(ox._row_weights(30_000.0, 250.0).size, 120)


# ---------------------------------------------------------------------------
# sample_units
# ---------------------------------------------------------------------------


class TestSampleUnits(OutageTestCase):
    def _ids(self, n=6):
        return K.SlotV1().unit_ids(self.by_name["z1 CCGT"], n)

    def test_shape_dtype_and_values(self):
        cp = self.params.carriers["CCGT"]
        up = ox.sample_units(
            self._ids(6),
            cp.p_fail,
            cp.p_repair,
            cp.forced_outage_rate,
            year=2020,
            draw=0,
            base_seed=123,
            n_hours=24,
        )
        self.assertEqual(up.shape, (6, 24))
        self.assertEqual(up.dtype, np.uint8)
        self.assertTrue(set(np.unique(up)).issubset({0, 1}))

    def test_slot_realisation_does_not_depend_on_n(self):
        """The D1.3 property, end to end: n = 3 is a prefix of n = 300."""
        cp = self.params.carriers["CCGT"]
        kwargs = {
            "p_fail": cp.p_fail,
            "p_repair": cp.p_repair,
            "fo_rate": cp.forced_outage_rate,
            "year": 2020,
            "draw": 1,
            "base_seed": 77,
            "n_hours": 48,
        }
        small = ox.sample_units(self._ids(3), **kwargs)
        big = ox.sample_units(self._ids(300), **kwargs)
        np.testing.assert_array_equal(small, big[:3])

    def test_realisation_does_not_depend_on_the_batch_it_was_asked_in(self):
        """A unit's stream is seeded on its own key, not on its position."""
        cp = self.params.carriers["CCGT"]
        ids = self._ids(4)
        kwargs = {
            "p_fail": cp.p_fail,
            "p_repair": cp.p_repair,
            "fo_rate": cp.forced_outage_rate,
            "year": 2020,
            "draw": 0,
            "base_seed": 5,
            "n_hours": 24,
        }
        together = ox.sample_units(ids, **kwargs)
        reversed_order = ox.sample_units(ids[::-1], **kwargs)
        np.testing.assert_array_equal(together, reversed_order[::-1])
        alone = ox.sample_units(ids[2:3], **kwargs)
        np.testing.assert_array_equal(together[2], alone[0])

    def test_short_horizon_is_a_prefix_of_the_long_one(self):
        """A 168 h block is the first 168 hours of the full-year realisation."""
        cp = self.params.carriers["CCGT"]
        kwargs = {
            "p_fail": cp.p_fail,
            "p_repair": cp.p_repair,
            "fo_rate": cp.forced_outage_rate,
            "year": 2020,
            "draw": 0,
            "base_seed": 11,
        }
        short = ox.sample_units(self._ids(2), n_hours=24, **kwargs)
        long = ox.sample_units(self._ids(2), n_hours=168, **kwargs)
        np.testing.assert_array_equal(short, long[:, :24])

    def test_different_year_or_draw_decorrelates(self):
        """Independence is tested per unit-hour, not on the fleet-mean series.

        The fleet-mean availability series is autocorrelated with an integrated
        time of ~95 h, so over 168 h it carries ~2 independent samples and the
        sample correlation of two *independent* fleets is routinely +-0.6. The
        per-unit-hour agreement rate has 200 x 168 nearly-independent terms
        across units and is the sharp test: for independent chains it is
        ``(1-FOR)^2 + FOR^2``, against 1.0 for identical ones.
        """
        cp = self.params.carriers["CCGT"]
        f = cp.forced_outage_rate
        ids = self._ids(200)
        common = {
            "p_fail": cp.p_fail,
            "p_repair": cp.p_repair,
            "fo_rate": f,
            "base_seed": 99,
            "n_hours": 168,
        }
        a = ox.sample_units(ids, year=2020, draw=0, **common)
        b = ox.sample_units(ids, year=2020, draw=1, **common)
        c = ox.sample_units(ids, year=2021, draw=0, **common)
        independent = (1.0 - f) ** 2 + f**2
        for other in (b, c):
            self.assertFalse(np.array_equal(a, other))
            agreement = float((a == other).mean())
            self.assertAlmostEqual(agreement, independent, delta=0.05)
            self.assertLess(agreement, 0.99)

    def test_empty_unit_list(self):
        up = ox.sample_units(
            np.zeros(0, dtype=np.uint64), 0.1, 0.1, 0.1,
            year=2020, draw=0, base_seed=1, n_hours=24,
        )
        self.assertEqual(up.shape, (0, 24))

    def test_zero_hours_raises(self):
        with self.assertRaises(ValueError):
            ox.sample_units(self._ids(1), 0.1, 0.1, 0.1, year=2020, draw=0, base_seed=1, n_hours=0)


# ---------------------------------------------------------------------------
# row_availability
# ---------------------------------------------------------------------------


class TestRowAvailability(OutageTestCase):
    def _avail(self, caps, window=(0, 48), **kwargs):
        return ox.row_availability(
            self.rows,
            caps,
            self.params,
            year=2020,
            draw=0,
            base_seed=4242,
            scheme="slot-v1",
            window=window,
            hours_per_year=168,
            **kwargs,
        )

    def test_shape_and_range(self):
        caps = {r.name: 1000.0 for r in self.rows}
        avail = self._avail(caps)
        self.assertEqual(avail.shape, (48, len(self.rows)))
        self.assertTrue(np.all(avail >= 0.0) and np.all(avail <= 1.0))

    def test_zero_capacity_row_is_fully_available(self):
        caps = {r.name: 0.0 for r in self.rows}
        avail = self._avail(caps)
        np.testing.assert_allclose(avail, 1.0)

    def test_partial_last_slot_weighting_against_a_hand_value(self):
        """A 300 MW CCGT row is 1 full slot + a 0.2-weight slot."""
        row = self.by_name["z1 CCGT"]
        cp = self.params.carriers["CCGT"]
        caps = {r.name: 0.0 for r in self.rows}
        caps["z1 CCGT"] = 300.0
        avail = self._avail(caps)[:, self.rows.index(row)]

        ids = K.SlotV1().unit_ids(row, 2)
        up = ox.sample_units(
            ids,
            cp.p_fail,
            cp.p_repair,
            cp.forced_outage_rate,
            year=2020,
            draw=0,
            base_seed=4242,
            n_hours=168,
        ).astype(float)[:, :48]
        expected = (1.0 * up[0] + 0.2 * up[1]) / 1.2
        np.testing.assert_allclose(avail, expected, rtol=1e-12)
        # And the realised values really are the three possible weighted means.
        self.assertTrue(set(np.round(np.unique(avail), 12)).issubset({0.0, 1.0, 1 / 1.2, 0.2 / 1.2}))

    def test_growing_a_row_leaves_the_other_rows_bit_identical(self):
        """The property the accreditation perturbation loop is built on."""
        caps = {r.name: 500.0 for r in self.rows}
        base = self._avail(caps)
        grown = dict(caps)
        grown["z1 CCGT"] = 11_000.0
        after = self._avail(grown)
        j = self.rows.index(self.by_name["z1 CCGT"])
        others = [i for i in range(len(self.rows)) if i != j]
        np.testing.assert_array_equal(base[:, others], after[:, others])

    def test_growing_a_row_keeps_its_own_first_slots(self):
        row = self.by_name["z1 CCGT"]
        cp = self.params.carriers["CCGT"]
        ids_small = K.SlotV1().unit_ids(row, 2)
        ids_big = K.SlotV1().unit_ids(row, 44)
        np.testing.assert_array_equal(ids_small, ids_big[:2])
        del cp

    def test_window_slicing_matches_the_full_year(self):
        caps = {r.name: 800.0 for r in self.rows}
        full = self._avail(caps, window=(0, 168))
        sliced = self._avail(caps, window=(100, 168))
        np.testing.assert_array_equal(full[100:168], sliced)

    def test_bad_window_raises(self):
        caps = {r.name: 100.0 for r in self.rows}
        with self.assertRaises(ValueError):
            self._avail(caps, window=(0, 400))
        with self.assertRaises(ValueError):
            self._avail(caps, window=(50, 50))

    def test_unknown_scheme_raises(self):
        caps = {r.name: 100.0 for r in self.rows}
        with self.assertRaises(KeyError):
            ox.row_availability(
                self.rows, caps, self.params,
                year=2020, draw=0, scheme="nope", window=(0, 24), hours_per_year=168,
            )

    def test_slot_counts(self):
        caps = {"z1 CCGT": 1000.0, "z1 battery": 0.0, "z2 PHS": 300.0}
        counts = ox.slot_counts(self.rows, caps, self.params)
        self.assertEqual(counts["z1 CCGT"], 4)
        self.assertEqual(counts["z1 battery"], 0)
        self.assertEqual(counts["z2 PHS"], 2)


# ---------------------------------------------------------------------------
# The in-process slot cache (spec D3.2)
# ---------------------------------------------------------------------------


class TestUnitCache(OutageTestCase):
    def _avail(self, caps, *, draw=0):
        return ox.row_availability(
            self.rows, caps, self.params,
            year=2020, draw=draw, base_seed=7, scheme="slot-v1",
            window=(0, 24), hours_per_year=168,
        )

    def test_hits_and_misses(self):
        caps = {r.name: 500.0 for r in self.rows}
        ox.unit_cache_clear()
        self.assertEqual(ox.unit_cache_info()["misses"], 0)

        self._avail(caps)
        first = ox.unit_cache_info()
        self.assertGreater(first["misses"], 0)
        self.assertEqual(first["hits"], 0)
        self.assertEqual(first["units"], first["misses"])

        self._avail(caps)  # same (scheme, seed, year, draw): every slot is a hit
        second = ox.unit_cache_info()
        self.assertEqual(second["misses"], first["misses"])
        self.assertEqual(second["hits"], first["misses"])

    def test_a_perturbed_design_misses_only_on_its_extra_slots(self):
        caps = {r.name: 0.0 for r in self.rows}
        caps["z1 CCGT"] = 1000.0  # 4 slots
        ox.unit_cache_clear()
        self._avail(caps)
        self.assertEqual(ox.unit_cache_info()["misses"], 4)

        grown = dict(caps)
        grown["z1 CCGT"] = 1250.0  # 5 slots: 4 hits, 1 miss
        self._avail(grown)
        info = ox.unit_cache_info()
        self.assertEqual(info["misses"], 5)
        self.assertEqual(info["hits"], 4)

    def test_changing_year_or_draw_drops_the_cache(self):
        caps = {r.name: 500.0 for r in self.rows}
        ox.unit_cache_clear()
        self._avail(caps, draw=0)
        units = ox.unit_cache_info()["units"]
        self._avail(caps, draw=1)
        info = ox.unit_cache_info()
        self.assertEqual(info["units"], units)  # dropped wholesale, refilled
        self.assertEqual(info["hits"], 0)
        self.assertEqual(info["key"][3], 1)  # (scheme, seed, year, draw, hours)

    def test_cache_capacity_is_configurable(self):
        import os

        os.environ[ox.CACHE_UNITS_ENV] = "2"
        self.addCleanup(os.environ.pop, ox.CACHE_UNITS_ENV, None)
        self.assertEqual(ox.cache_capacity(), 2)
        caps = {r.name: 0.0 for r in self.rows}
        caps["z1 CCGT"] = 1000.0  # 4 slots > capacity 2
        ox.unit_cache_clear()
        self._avail(caps)
        self.assertLessEqual(ox.unit_cache_info()["units"], 2)

    def test_cached_rows_are_read_only(self):
        caps = {r.name: 500.0 for r in self.rows}
        ox.unit_cache_clear()
        self._avail(caps)
        row = next(iter(ox._UNIT_CACHE.values()))
        with self.assertRaises(ValueError):
            row[0] = 1


# ---------------------------------------------------------------------------
# E1: the marginal law is unchanged by construction -- tested anyway
# ---------------------------------------------------------------------------


class TestMarginalLaw(unittest.TestCase):
    """Spec D5 E1: mean availability and lag-1 autocorrelation of one carrier."""

    N_SLOTS = 2000
    N_HOURS = 8760

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="zap_outage_law_"))
        params = make_params(cls.tmp)
        cls.cp = params.carriers["CCGT"]
        row = K.RowSpec("law", "Generator", "CCGT", "z1", 0)
        ids = K.SlotV1().unit_ids(row, cls.N_SLOTS)
        cls.up = ox.sample_units(
            ids,
            cls.cp.p_fail,
            cls.cp.p_repair,
            cls.cp.forced_outage_rate,
            year=2020,
            draw=0,
            base_seed=20260908,
            n_hours=cls.N_HOURS,
        ).astype(np.float64)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_mean_availability_is_one_minus_for(self):
        target = 1.0 - self.cp.forced_outage_rate
        per_slot = self.up.mean(axis=1)
        mean = float(per_slot.mean())
        # SE over slots: the hours of one slot are correlated, so the slot mean
        # is the independent replicate, not the hour.
        se = float(per_slot.std(ddof=1) / math.sqrt(self.N_SLOTS))
        self.assertLess(abs(mean - target), 3 * se, f"mean {mean} vs {target} (SE {se})")

    def test_lag_one_autocorrelation_matches_the_chain(self):
        """The chain's memory, ``rho_1 = 1 - p_fail - p_repair``.

        Estimated from the realised transition counts rather than from the
        sample autocorrelation: with a demeaned series of 8,760 hours and an
        integrated autocorrelation time of ~95 h, the sample ACF1 is biased low
        by ~4e-3 (measured), which is ~12 SE here and would make a 3-SE test
        fail on correct draws. ``1 - p_hat_fail - p_hat_repair`` is the same
        quantity with a clean binomial error.
        """
        up = self.up.astype(bool)
        prev, nxt = up[:, :-1], up[:, 1:]
        n_up = int(prev.sum())
        n_down = int((~prev).sum())
        fails = int((prev & ~nxt).sum())
        repairs = int((~prev & nxt).sum())

        p_fail = fails / n_up
        p_repair = repairs / n_down
        se_fail = math.sqrt(p_fail * (1 - p_fail) / n_up)
        se_repair = math.sqrt(p_repair * (1 - p_repair) / n_down)
        self.assertLess(
            abs(p_fail - self.cp.p_fail), 3 * se_fail, f"p_fail {p_fail} vs {self.cp.p_fail}"
        )
        self.assertLess(
            abs(p_repair - self.cp.p_repair),
            3 * se_repair,
            f"p_repair {p_repair} vs {self.cp.p_repair}",
        )

        rho = 1.0 - p_fail - p_repair
        target = 1.0 - self.cp.p_fail - self.cp.p_repair
        se = math.hypot(se_fail, se_repair)
        self.assertLess(abs(rho - target), 3 * se, f"rho {rho} vs {target} (SE {se})")


# ---------------------------------------------------------------------------
# UCAP
# ---------------------------------------------------------------------------


class TestUcap(OutageTestCase):
    def test_write_ucap(self):
        out = self.tmp / "ucap.csv"
        df = ox.write_ucap(
            self.dataset,
            years=[2020],
            draws=3,
            base_seed=31337,
            scheme="slot-v1",
            hours_per_year=168,
            out_csv=out,
            params=self.params,
            model_year=None,
            verbose=False,
        )
        self.assertTrue(out.exists())
        self.assertEqual(list(df.columns), ox.UCAP_COLUMNS)
        self.assertEqual(set(df["row"]), {r.name for r in self.rows})
        self.assertTrue((df["outage_scheme"] == "slot-v1").all())
        self.assertTrue((df["base_seed"] == 31337).all())
        self.assertTrue((df["n_draws"] == 3).all())
        self.assertTrue(((df["ucap_empirical"] >= 0.0) & (df["ucap_empirical"] <= 1.0)).all())

        # A zero-capacity row derates nothing and falls back to the analytic value.
        zero = df[df["row"] == "z1 OCGT"].iloc[0]
        self.assertEqual(zero["n_units_row"], 0)
        self.assertEqual(zero["n_samples"], 0)
        self.assertAlmostEqual(zero["ucap_empirical"], zero["ucap_analytic"])

        # The analytic column is scheme-independent.
        ccgt = df[df["row"] == "z1 CCGT"].iloc[0]
        self.assertAlmostEqual(ccgt["ucap_analytic"], 1 - 0.045)
        self.assertEqual(ccgt["n_units_row"], 4)

    def test_ucap_is_reproducible(self):
        kwargs = {
            "years": [2020],
            "draws": 2,
            "base_seed": 5,
            "scheme": "slot-v1",
            "hours_per_year": 168,
            "params": self.params,
            "verbose": False,
        }
        a = ox.write_ucap(self.dataset, out_csv=self.tmp / "a.csv", **kwargs)
        ox.unit_cache_clear()
        b = ox.write_ucap(self.dataset, out_csv=self.tmp / "b.csv", **kwargs)
        np.testing.assert_allclose(a["ucap_empirical"], b["ucap_empirical"], rtol=0, atol=0)

    def test_ucap_rejects_empty_years_or_draws(self):
        with self.assertRaises(ValueError):
            ox.write_ucap(self.dataset, years=[], draws=1, params=self.params, verbose=False)
        with self.assertRaises(ValueError):
            ox.write_ucap(self.dataset, years=[2020], draws=0, params=self.params, verbose=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli(OutageTestCase):
    def test_rows_command(self):
        rc = ox.main(
            [
                "rows",
                "--dataset-dir",
                str(self.dataset),
                "--params",
                str(self.tmp / "params.yaml"),
            ]
        )
        self.assertEqual(rc, 0)

    def test_ucap_command(self):
        rc = ox.main(
            [
                "ucap",
                "--dataset-dir",
                str(self.dataset),
                "--params",
                str(self.tmp / "params.yaml"),
                "--years",
                "2020",
                "--draws",
                "1",
                "--hours-per-year",
                "168",
                "--out-csv",
                str(self.tmp / "cli_ucap.csv"),
            ]
        )
        self.assertEqual(rc, 0)
        self.assertTrue((self.tmp / "cli_ucap.csv").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
