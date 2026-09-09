"""Tests for the multi-year time-varying attribute registry (phase-1 spec WP3.3)."""

import unittest

import numpy as np

import zap
from zap.devices import DirectedLine
from zap.devices.storage_unit import StorageUnit
from zap.importers.multi_year import (
    TIME_VARYING_ATTRS,
    concatenate_time_varying_attrs,
)


HOURS = 24


def make_storage(availability, names=("battery",), terminals=(1,)):
    n = len(terminals)
    return StorageUnit(
        num_nodes=2,
        name=np.array(names),
        terminal=np.array(terminals),
        power_capacity=np.full(n, 10.0),
        duration=np.full(n, 4.0),
        linear_cost=np.full(n, 0.01),
        power_availability=availability,
    )


def make_directed_line(max_power):
    return DirectedLine(
        num_nodes=2,
        name=np.array(["link"]),
        source_terminal=np.array([0]),
        sink_terminal=np.array([1]),
        min_power=np.zeros(1),
        max_power=max_power,
        linear_cost=np.zeros(1),
        nominal_capacity=np.array([100.0]),
        efficiency=np.array([0.95]),
    )


class TestTimeVaryingRegistry(unittest.TestCase):
    def test_registry_lists_new_devices(self):
        self.assertEqual(TIME_VARYING_ATTRS[StorageUnit], ["power_availability", "linear_cost"])
        self.assertEqual(
            TIME_VARYING_ATTRS[DirectedLine], ["max_power", "min_power", "linear_cost"]
        )
        # Store stays out of the registry (decision D4)
        self.assertNotIn(zap.Store, TIME_VARYING_ATTRS)

    def test_time_varying_attrs_cover_new_devices(self):
        """Concatenating two synthetic years extends the registered attributes."""
        year0_avail = np.full((1, HOURS), 1.0)
        year1_avail = np.full((1, HOURS), 0.25)

        base_storage = make_storage(year0_avail)
        year_storage = make_storage(year1_avail)

        touched = concatenate_time_varying_attrs(
            base_storage,
            year_storage,
            expected_base_hours=HOURS,
            expected_year_hours=HOURS,
        )

        self.assertIn("power_availability", touched)
        self.assertEqual(base_storage.power_availability.shape, (1, 2 * HOURS))
        np.testing.assert_allclose(
            base_storage.power_availability[:, :HOURS], year0_avail, rtol=1e-12
        )
        np.testing.assert_allclose(
            base_storage.power_availability[:, HOURS:], year1_avail, rtol=1e-12
        )
        self.assertEqual(base_storage.time_horizon, 2 * HOURS)

        # linear_cost is constant (shape (1, 1)) and must be left alone
        self.assertNotIn("linear_cost", touched)
        self.assertEqual(base_storage.linear_cost.shape, (1, 1))

        # ---- DirectedLine ----
        year0_max = np.full((1, HOURS), 1.0)
        year1_max = np.full((1, HOURS), 0.5)

        base_line = make_directed_line(year0_max)
        year_line = make_directed_line(year1_max)

        touched = concatenate_time_varying_attrs(
            base_line,
            year_line,
            expected_base_hours=HOURS,
            expected_year_hours=HOURS,
        )

        self.assertIn("max_power", touched)
        self.assertEqual(base_line.max_power.shape, (1, 2 * HOURS))
        np.testing.assert_allclose(base_line.max_power[:, :HOURS], year0_max, rtol=1e-12)
        np.testing.assert_allclose(base_line.max_power[:, HOURS:], year1_max, rtol=1e-12)

        # Constant min_power / linear_cost are filtered out
        self.assertNotIn("min_power", touched)
        self.assertNotIn("linear_cost", touched)
        self.assertEqual(base_line.min_power.shape, (1, 1))

    def test_concatenation_applies_device_reordering(self):
        base = make_storage(
            np.vstack([np.zeros(HOURS), np.ones(HOURS)]),
            names=("a", "b"),
            terminals=(0, 1),
        )
        year = make_storage(
            np.vstack([np.full(HOURS, 3.0), np.full(HOURS, 2.0)]),
            names=("b", "a"),
            terminals=(0, 1),
        )

        concatenate_time_varying_attrs(
            base,
            year,
            expected_base_hours=HOURS,
            expected_year_hours=HOURS,
            reorder_idx=np.array([1, 0]),
        )

        np.testing.assert_allclose(
            base.power_availability[:, HOURS:],
            np.vstack([np.full(HOURS, 2.0), np.full(HOURS, 3.0)]),
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
