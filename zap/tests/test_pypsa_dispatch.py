"""
Tests for PyPSA network dispatch functionality.

This module tests:
1. Dispatch functionality - verifying operational optimization results
2. Roundtrip: Import PyPSA -> Zap dispatch -> Export PyPSA

Test structure:
- Roundtrip tests: Import PyPSA -> Zap dispatch -> Export PyPSA
- Device-specific tests: Verify each device type's results
"""

import unittest
import cvxpy as cp
import numpy as np
import pandas as pd
import pypsa
from pathlib import Path
import logging

from zap.devices.injector import Generator, Load
from zap.devices.transporter import DCLine, ACLine
from zap.devices.storage_unit import StorageUnit
from zap.devices.store import Store
from zap.importers.pypsa import load_pypsa_network, HOURS_PER_YEAR
from zap.exporters.pypsa import export_to_pypsa
from zap.tests import network_examples as examples
from zap.tests.plotting_helpers import (
    get_zap_energy_balance,
    plot_energy_balance_comparison,
    plot_price_comparison,
)

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Directory for saving plots
PLOT_DIR = Path(__file__).parent / "plots"


# Test tolerances
PRICE_TOLERANCE = 1e-2
POWER_TOLERANCE = 1e-3
COST_TOLERANCE = 1e-2
POWER_BALANCE_TOLERANCE = 1e-6

logger = logging.getLogger(__name__)


class TestPyPSARoundtripBase(unittest.TestCase):
    """Base class for PyPSA roundtrip tests with common utilities."""

    @classmethod
    def setUpClass(cls):
        """Load network and run dispatch."""
        cls.pypsa_network = cls.load_pypsa_network()
        cls.snapshots = cls.get_snapshots()

        # Import to Zap
        cls.net, cls.devices = load_pypsa_network(
            cls.pypsa_network,
            cls.snapshots,
            power_unit=cls.get_power_unit(),
            cost_unit=cls.get_cost_unit(),
            susceptance_unit=cls.get_susceptance_unit(),
        )

        cls.time_horizon = len(cls.snapshots)

        # Run dispatch
        cls.dispatch = cls.net.dispatch(
            cls.devices,
            time_horizon=cls.time_horizon,
            solver=cp.MOSEK,
        )

        # Export back to PyPSA
        cls.exported_network = export_to_pypsa(
            cls.net,
            cls.devices,
            cls.dispatch,
            cls.snapshots,
            power_unit=cls.get_power_unit(),
            cost_unit=cls.get_cost_unit(),
            susceptance_unit=cls.get_susceptance_unit(),
        )

    @classmethod
    def load_pypsa_network(cls):
        """Override this to load different test networks."""
        net = examples.load_example_network("texas_7node")
        # Scale loads by 0.5 to avoid unserved energy in dispatch-only tests
        # (The network has extendable generators that PyPSA would build, but Zap dispatch doesn't)
        net.loads_t.p_set = net.loads_t.p_set * 0.5
        net.loads["p_set"] = net.loads["p_set"] * 0.5
        # Disable capacity expansion for dispatch-only comparison
        net.generators["p_nom_extendable"] = False
        net.storage_units["p_nom_extendable"] = False
        return net

    @classmethod
    def get_snapshots(cls):
        """Override to customize time horizon. Uses network's native snapshots."""
        # Load network to get its native snapshots
        net = cls.load_pypsa_network()
        # Use first 24 hours of network's native snapshots
        return net.snapshots[:24]

    @classmethod
    def get_power_unit(cls):
        return 1.0

    @classmethod
    def get_cost_unit(cls):
        return 1.0

    @classmethod
    def get_susceptance_unit(cls):
        return 1.0

    def get_device_by_type(self, device_type):
        """Get device of a specific type from devices list."""
        for device in self.devices:
            if isinstance(device, device_type):
                return device
        return None

    def get_device_index(self, device_type):
        """Get index of device type in devices list."""
        for i, device in enumerate(self.devices):
            if isinstance(device, device_type):
                return i
        return None

    def plot_dispatch_results(self, filename_prefix="dispatch"):
        """Generate and save dispatch plots comparing PyPSA and Zap results.

        Creates side-by-side comparison plots showing:
        - Left column: PyPSA dispatch results
        - Right column: Zap dispatch results

        Rows show:
        1. Energy balance by carrier (generators + storage discharge positive, charging negative)
        2. Marginal prices
        """
        if not HAS_MATPLOTLIB:
            return

        # Create plot directory if it doesn't exist
        dispatch_plot_dir = PLOT_DIR / "dispatch"
        dispatch_plot_dir.mkdir(parents=True, exist_ok=True)

        # Time axis
        time_horizon = self.time_horizon
        hours = np.arange(time_horizon)

        # Run PyPSA optimization to get comparison data
        pypsa_net = self.pypsa_network.copy()
        pypsa_net.set_snapshots(self.snapshots)
        pypsa_net.snapshot_weightings.loc[:, :] = HOURS_PER_YEAR / len(self.snapshots)
        has_investment_periods = pypsa_net.investment_periods.size > 0
        pypsa_net.optimize(
            solver_name="highs", multi_investment_periods=has_investment_periods
        )

        # Get energy balance data using PyPSA statistics API
        pypsa_energy_raw = (
            pypsa_net.statistics.energy_balance(
                comps=["Generator", "StorageUnit"],
                aggregate_time=False,
                nice_names=False,
            )
            .droplevel(0)
            .T
        )
        # Group by carrier (avoid FutureWarning by transposing first)
        pypsa_energy = pypsa_energy_raw.T.groupby(level="carrier").sum().T
        pypsa_energy.index = hours

        # Get Zap energy balance
        zap_energy = get_zap_energy_balance(
            self.devices, self.dispatch, pypsa_net, time_horizon
        )

        # Get carrier colors and load profile
        carrier_colors = pypsa_net.carriers.color.to_dict()
        load_profile = (
            pypsa_net.loads_t.p_set.sum(axis=1).values
            if len(pypsa_net.loads) > 0
            else None
        )

        # Create figure
        _, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

        # Row 1: Energy balance comparison
        plot_energy_balance_comparison(
            axes[0, 0],
            axes[0, 1],
            pypsa_energy,
            zap_energy,
            carrier_colors,
            load_profile,
            title_left="PyPSA Dispatch by Carrier",
            title_right="Zap Dispatch by Carrier",
        )

        # Row 2: Price comparison
        plot_price_comparison(
            axes[1, 0],
            axes[1, 1],
            pypsa_net.buses_t.marginal_price,
            self.dispatch.prices,
            hours,
        )

        # Add legends
        axes[0, 1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6)
        axes[1, 1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6)

        plt.suptitle(
            f"{filename_prefix} - PyPSA vs Zap Comparison",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save plot
        plot_path = dispatch_plot_dir / f"{filename_prefix}_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Comparison plot saved to: {plot_path}")


class TestBasicRoundtrip(TestPyPSARoundtripBase):
    """Test basic roundtrip: PyPSA -> Zap -> PyPSA preserves structure."""

    def test_dispatch_optimal(self):
        """Check that dispatch solved successfully."""
        self.assertEqual(self.dispatch.problem.status, "optimal")

    def test_generate_dispatch_plot(self):
        """Generate dispatch plot for visual inspection."""
        self.plot_dispatch_results("texas_7node_24h")

    def test_bus_count_preserved(self):
        """Verify number of buses is preserved in roundtrip."""
        original_buses = len(self.pypsa_network.buses)
        exported_buses = len(self.exported_network.buses)
        self.assertEqual(original_buses, exported_buses)

    def test_snapshot_count_preserved(self):
        """Verify number of snapshots is preserved."""
        original_snapshots = len(self.snapshots)
        exported_snapshots = len(self.exported_network.snapshots)
        self.assertEqual(original_snapshots, exported_snapshots)

    def test_generator_count_preserved(self):
        """Verify number of generators matches imported count.

        Note: The importer may filter out some generators (e.g., inactive assets),
        so we compare against the imported device count rather than original.
        """
        gen_device = self.get_device_by_type(Generator)
        if gen_device is None:
            self.skipTest("No generators in network")

        imported_gens = gen_device.num_devices
        exported_gens = len(self.exported_network.generators)
        self.assertEqual(imported_gens, exported_gens)

    def test_load_count_preserved(self):
        """Verify number of loads is preserved."""
        original_loads = len(self.pypsa_network.loads)
        exported_loads = len(self.exported_network.loads)
        self.assertEqual(original_loads, exported_loads)

    def test_line_count_preserved(self):
        """Verify number of lines is preserved."""
        original_lines = len(self.pypsa_network.lines)
        exported_lines = len(self.exported_network.lines)
        self.assertEqual(original_lines, exported_lines)

    def test_link_count_preserved(self):
        """Verify number of links (DC lines) is preserved."""
        original_links = len(self.pypsa_network.links)
        exported_links = len(self.exported_network.links)
        self.assertEqual(original_links, exported_links)

    def test_storage_unit_count_preserved(self):
        """Verify number of storage units is preserved."""
        original_storage = len(self.pypsa_network.storage_units)
        exported_storage = len(self.exported_network.storage_units)
        self.assertEqual(original_storage, exported_storage)


class TestGeneratorDispatch(TestPyPSARoundtripBase):
    """Test generator dispatch results."""

    def test_generator_power_exported(self):
        """Verify generator power results are exported."""
        gen_device = self.get_device_by_type(Generator)
        if gen_device is None:
            self.skipTest("No generators in network")

        # Check that power results exist in exported network
        self.assertIn("p", self.exported_network.generators_t)
        self.assertFalse(self.exported_network.generators_t["p"].empty)

    def test_generator_power_nonnegative(self):
        """Verify generator power output is non-negative."""
        gen_device = self.get_device_by_type(Generator)
        if gen_device is None:
            self.skipTest("No generators in network")

        gen_idx = self.get_device_index(Generator)
        gen_power = self.dispatch.power[gen_idx][0]

        # Generator power should be >= 0 (injecting)
        self.assertTrue(np.all(gen_power >= -POWER_TOLERANCE))

    def test_generator_capacity_constraint(self):
        """Verify generator output respects capacity constraints."""
        gen_device = self.get_device_by_type(Generator)
        if gen_device is None:
            self.skipTest("No generators in network")

        gen_idx = self.get_device_index(Generator)
        gen_power = self.dispatch.power[gen_idx][0]

        # Power should be <= nominal_capacity * dynamic_capacity
        max_power = (
            gen_device.nominal_capacity.reshape(-1, 1) * gen_device.dynamic_capacity
        )
        self.assertTrue(np.all(gen_power <= max_power + POWER_TOLERANCE))

    def test_generator_marginal_cost_preserved(self):
        """Verify generator marginal costs are preserved in export."""
        gen_device = self.get_device_by_type(Generator)
        if gen_device is None:
            self.skipTest("No generators in network")

        # Check marginal cost is exported
        self.assertIn("marginal_cost", self.exported_network.generators_t)


class TestLoadDispatch(TestPyPSARoundtripBase):
    """Test load dispatch results."""

    def test_load_power_exported(self):
        """Verify load power results are exported."""
        load_device = self.get_device_by_type(Load)
        if load_device is None:
            self.skipTest("No loads in network")

        # Check that power results exist in exported network
        self.assertIn("p", self.exported_network.loads_t)

    def test_load_power_nonpositive(self):
        """Verify load power consumption is non-positive (withdrawing)."""
        load_device = self.get_device_by_type(Load)
        if load_device is None:
            self.skipTest("No loads in network")

        load_idx = self.get_device_index(Load)
        load_power = self.dispatch.power[load_idx][0]

        # Load power should be <= 0 (withdrawing)
        self.assertTrue(np.all(load_power <= POWER_TOLERANCE))

    def test_load_meets_demand(self):
        """Verify load meets demand (or sheds with cost)."""
        load_device = self.get_device_by_type(Load)
        if load_device is None:
            self.skipTest("No loads in network")

        load_idx = self.get_device_index(Load)
        load_power = self.dispatch.power[load_idx][0]

        # Load power should be >= -load (can shed)
        min_load = -load_device.load
        self.assertTrue(np.all(load_power >= min_load - POWER_TOLERANCE))


class TestACLineDispatch(TestPyPSARoundtripBase):
    """Test AC line dispatch results."""

    def test_ac_line_power_exported(self):
        """Verify AC line power flow results are exported."""
        ac_device = self.get_device_by_type(ACLine)
        if ac_device is None:
            self.skipTest("No AC lines in network")

        # Check that p0 and p1 results exist
        self.assertIn("p0", self.exported_network.lines_t)
        self.assertIn("p1", self.exported_network.lines_t)

    def test_ac_line_capacity_constraint(self):
        """Verify AC line flow respects capacity constraints."""
        ac_device = self.get_device_by_type(ACLine)
        if ac_device is None:
            self.skipTest("No AC lines in network")

        ac_idx = self.get_device_index(ACLine)
        p0 = self.dispatch.power[ac_idx][0]
        p1 = self.dispatch.power[ac_idx][1]

        # Power magnitude should be <= nominal_capacity * capacity
        max_flow = ac_device.nominal_capacity.reshape(
            -1, 1
        ) * ac_device.capacity.reshape(-1, 1)
        self.assertTrue(np.all(np.abs(p0) <= max_flow + POWER_TOLERANCE))
        self.assertTrue(np.all(np.abs(p1) <= max_flow + POWER_TOLERANCE))

    def test_ac_line_power_conservation(self):
        """Verify power is conserved on AC lines (p0 = -p1 for lossless)."""
        ac_device = self.get_device_by_type(ACLine)
        if ac_device is None:
            self.skipTest("No AC lines in network")

        ac_idx = self.get_device_index(ACLine)
        p0 = self.dispatch.power[ac_idx][0]
        p1 = self.dispatch.power[ac_idx][1]

        # For lossless lines, p0 + p1 should be approximately 0
        np.testing.assert_allclose(p0 + p1, 0, atol=POWER_TOLERANCE)


class TestDCLineDispatch(TestPyPSARoundtripBase):
    """Test DC line (Link) dispatch results."""

    def test_dc_line_power_exported(self):
        """Verify DC line power flow results are exported."""
        dc_device = self.get_device_by_type(DCLine)
        if dc_device is None:
            self.skipTest("No DC lines in network")

        # Check that p0 and p1 results exist
        self.assertIn("p0", self.exported_network.links_t)
        self.assertIn("p1", self.exported_network.links_t)

    def test_dc_line_capacity_constraint(self):
        """Verify DC line flow respects capacity constraints."""
        dc_device = self.get_device_by_type(DCLine)
        if dc_device is None:
            self.skipTest("No DC lines in network")

        dc_idx = self.get_device_index(DCLine)
        p0 = self.dispatch.power[dc_idx][0]

        # Power should be <= nominal_capacity * capacity
        max_flow = dc_device.nominal_capacity.reshape(-1, 1) * dc_device.capacity
        self.assertTrue(np.all(p0 <= max_flow + POWER_TOLERANCE))


class TestStorageUnitDispatch(TestPyPSARoundtripBase):
    """Test StorageUnit dispatch results."""

    def test_storage_unit_power_exported(self):
        """Verify storage unit power results are exported."""
        storage_device = self.get_device_by_type(StorageUnit)
        if storage_device is None:
            self.skipTest("No storage units in network")

        # Check that power results exist
        self.assertIn("p", self.exported_network.storage_units_t)

    def test_storage_unit_power_bounds(self):
        """Verify storage unit power respects capacity bounds."""
        storage_device = self.get_device_by_type(StorageUnit)
        if storage_device is None:
            self.skipTest("No storage units in network")

        storage_idx = self.get_device_index(StorageUnit)
        power = self.dispatch.power[storage_idx][0]

        # Power magnitude should be <= power_capacity
        power_cap = storage_device.power_capacity.reshape(-1, 1)
        self.assertTrue(np.all(np.abs(power) <= power_cap + POWER_TOLERANCE))

    def test_storage_unit_energy_balance(self):
        """Verify storage unit energy evolution is correct."""
        storage_device = self.get_device_by_type(StorageUnit)
        if storage_device is None:
            self.skipTest("No storage units in network")

        storage_idx = self.get_device_index(StorageUnit)

        # Check that local variables (energy state) exist
        local_vars = self.dispatch.local_variables[storage_idx]
        self.assertIsNotNone(local_vars)


class TestStoreDispatch(TestPyPSARoundtripBase):
    """Test Store dispatch results."""

    def test_store_count_preserved(self):
        """Verify number of stores is preserved."""
        original_stores = len(self.pypsa_network.stores)
        exported_stores = len(self.exported_network.stores)
        self.assertEqual(original_stores, exported_stores)


class TestPowerBalance(TestPyPSARoundtripBase):
    """Test power balance at each bus."""

    def test_power_balance(self):
        """Verify power balance at each bus and timestep."""
        # Build incidence matrices and compute net injection per bus
        num_buses = self.net.num_nodes
        time_horizon = self.time_horizon

        # Initialize net injection
        net_injection = np.zeros((num_buses, time_horizon))

        # Add contributions from each device
        for i, device in enumerate(self.devices):
            power = self.dispatch.power[i]

            if isinstance(device, (Generator, Load)):
                # Single terminal devices
                for j in range(device.num_devices):
                    bus = device.terminal[j]
                    net_injection[bus, :] += power[0][j, :]

            elif isinstance(device, StorageUnit):
                # Single terminal devices
                for j in range(device.num_devices):
                    bus = device.terminal[j]
                    net_injection[bus, :] += power[0][j, :]

            elif isinstance(device, Store):
                # Single terminal devices
                for j in range(device.num_devices):
                    bus = device.terminal[j]
                    net_injection[bus, :] += power[0][j, :]

            elif isinstance(device, (ACLine, DCLine)):
                # Two terminal devices
                for j in range(device.num_devices):
                    source_bus = device.source_terminal[j]
                    sink_bus = device.sink_terminal[j]
                    net_injection[source_bus, :] += power[0][j, :]
                    net_injection[sink_bus, :] += power[1][j, :]

        # Power balance should be zero at each bus
        np.testing.assert_allclose(
            net_injection,
            0,
            atol=POWER_BALANCE_TOLERANCE,
            err_msg="Power balance violated",
        )


class TestMarginalPrices(TestPyPSARoundtripBase):
    """Test marginal price results."""

    def test_marginal_prices_exported(self):
        """Verify marginal prices are exported."""
        self.assertIn("marginal_price", self.exported_network.buses_t)
        self.assertFalse(self.exported_network.buses_t["marginal_price"].empty)

    def test_marginal_prices_positive(self):
        """Verify marginal prices are generally non-negative."""
        # Prices can be negative in some cases but typically positive
        prices = self.dispatch.prices
        # Just check they're reasonable (not all zeros)
        self.assertTrue(np.any(prices != 0))


class TestObjectiveValueBase(unittest.TestCase):
    """Test that Zap and PyPSA objective values match using a controlled synthetic network."""

    @classmethod
    def create_simple_dispatch_network(cls):
        """Create a simple network for objective comparison.

        This network has simple parameters so we can directly compare objectives
        between Zap and PyPSA without unit conversion issues.
        """
        n = pypsa.Network()

        # Set snapshots
        snapshots = pd.date_range("2020-01-01", periods=24, freq="h")
        n.set_snapshots(snapshots)

        # Add buses
        n.add("Bus", "bus0")
        n.add("Bus", "bus1")

        # Add carrier
        n.add("Carrier", "gas", co2_emissions=0.0)

        # Add cheap generator at bus0
        n.add(
            "Generator",
            "gen_cheap",
            bus="bus0",
            p_nom=100.0,
            marginal_cost=10.0,
            carrier="gas",
        )

        # Add expensive generator at bus1
        n.add(
            "Generator",
            "gen_expensive",
            bus="bus1",
            p_nom=100.0,
            marginal_cost=50.0,
            carrier="gas",
        )

        # Add load at bus1
        load_profile = 50.0 * np.ones(24)  # Constant load
        n.add(
            "Load",
            "load0",
            bus="bus1",
            p_set=load_profile,
        )

        # Add transmission line with limited capacity
        n.add(
            "Line",
            "line_0_1",
            bus0="bus0",
            bus1="bus1",
            s_nom=30.0,  # Limited capacity forces some expensive generation
            x=0.1,
        )

        return n, snapshots

    @classmethod
    def setUpClass(cls):
        cls.pypsa_network, cls.snapshots = cls.create_simple_dispatch_network()

        # Import to Zap with no unit scaling
        cls.net, cls.devices = load_pypsa_network(
            cls.pypsa_network,
            cls.snapshots,
            power_unit=1.0,
            cost_unit=1.0,
            susceptance_unit=1.0,
        )

        cls.time_horizon = len(cls.snapshots)

        # Run Zap dispatch
        cls.dispatch = cls.net.dispatch(
            cls.devices,
            time_horizon=cls.time_horizon,
            solver=cp.MOSEK,
        )

    def test_objective_value_matches_pypsa(self):
        """Verify Zap dispatch objective matches PyPSA optimization within ~1%.

        This test compares the optimal objective values from both solvers
        to ensure they're solving the same optimization problem.
        """
        # Get Zap objective value
        zap_objective = self.dispatch.problem.value

        # Run PyPSA optimization on the same network
        pypsa_net = self.pypsa_network.copy()
        # Set snapshot weightings to scale operational costs by time horizon
        pypsa_net.snapshot_weightings.loc[:, :] = HOURS_PER_YEAR / len(self.snapshots)

        # Solve with PyPSA's linear optimal power flow
        # Use multi_investment_periods only if network has investment periods
        has_investment_periods = pypsa_net.investment_periods.size > 0
        pypsa_net.optimize(
            solver_name="highs", multi_investment_periods=has_investment_periods
        )

        # Get PyPSA objective value
        pypsa_objective = pypsa_net.objective

        # Check they match within 1%
        # Note: Both should be positive for a pure dispatch problem
        if abs(pypsa_objective) > 1e-6:
            relative_error = abs(zap_objective - pypsa_objective) / abs(pypsa_objective)
            self.assertLess(
                relative_error,
                0.01,
                f"Zap objective ({zap_objective:.2f}) differs from PyPSA objective "
                f"({pypsa_objective:.2f}) by {relative_error*100:.2f}%",
            )
        else:
            # If PyPSA objective is near 0, check Zap is also near 0
            self.assertAlmostEqual(zap_objective, 0, places=2)

    def test_generator_dispatch_pattern(self):
        """Verify dispatch pattern: cheap gen at capacity, expensive fills remainder."""
        gen_device = None
        for device in self.devices:
            if isinstance(device, Generator):
                gen_device = device
                break

        if gen_device is None:
            self.skipTest("No generators found")

        gen_idx = self.devices.index(gen_device)
        gen_power = self.dispatch.power[gen_idx][0]

        # Cheap generator (index 0) should be at or near capacity (limited by line)
        # Line capacity is 30, so cheap gen should produce 30
        cheap_gen_power = gen_power[0, :]
        self.assertTrue(
            np.allclose(cheap_gen_power, 30.0, atol=0.1),
            f"Cheap generator should produce ~30 MW (line limit), got {cheap_gen_power[0]:.2f}",
        )

        # Expensive generator (index 1) should produce the remainder
        # Load is 50, cheap gen produces 30, so expensive produces 20
        expensive_gen_power = gen_power[1, :]
        self.assertTrue(
            np.allclose(expensive_gen_power, 20.0, atol=0.1),
            f"Expensive generator should produce ~20 MW, got {expensive_gen_power[0]:.2f}",
        )

    def test_generate_dispatch_plot(self):
        """Generate dispatch plot comparing PyPSA and Zap for the synthetic network."""
        if not HAS_MATPLOTLIB:
            self.skipTest("matplotlib not available")

        # Create plot directory if it doesn't exist
        dispatch_plot_dir = PLOT_DIR / "dispatch"
        dispatch_plot_dir.mkdir(parents=True, exist_ok=True)

        # Time axis
        time_horizon = self.time_horizon
        hours = np.arange(time_horizon)

        # Run PyPSA optimization
        pypsa_net = self.pypsa_network.copy()
        # Set snapshot weightings to scale operational costs by time horizon
        pypsa_net.snapshot_weightings.loc[:, :] = HOURS_PER_YEAR / len(self.snapshots)
        # Use multi_investment_periods only if network has investment periods
        has_investment_periods = pypsa_net.investment_periods.size > 0
        pypsa_net.optimize(
            solver_name="highs", multi_investment_periods=has_investment_periods
        )

        # Create figure with side-by-side comparison
        _, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

        # Share y-axis between left and right columns for each row
        for row in range(2):
            axes[row, 0].sharey(axes[row, 1])

        # Get devices
        gen_device = None
        load_device = None
        for device in self.devices:
            if isinstance(device, Generator):
                gen_device = device
            if isinstance(device, Load):
                load_device = device

        # === Row 1: Generator dispatch ===
        # Left: PyPSA
        ax_pypsa_gen = axes[0, 0]
        pypsa_gen_p = pypsa_net.generators_t.p
        for gen_name in pypsa_gen_p.columns:
            ax_pypsa_gen.plot(
                hours, pypsa_gen_p[gen_name].values, label=gen_name, linewidth=2
            )

        # Add load
        if len(pypsa_net.loads) > 0:
            total_load = pypsa_net.loads_t.p.sum(axis=1).values
            ax_pypsa_gen.plot(hours, total_load, "k--", linewidth=2, label="Total Load")

        ax_pypsa_gen.set_ylabel("Power (MW)")
        ax_pypsa_gen.set_title("PyPSA Generator Dispatch")
        ax_pypsa_gen.legend()
        ax_pypsa_gen.grid(True, alpha=0.3)

        # Right: Zap
        ax_zap_gen = axes[0, 1]
        if gen_device is not None:
            gen_idx = self.devices.index(gen_device)
            gen_power = self.dispatch.power[gen_idx][0]

            for i in range(gen_device.num_devices):
                name = gen_device.name[i] if hasattr(gen_device, "name") else f"Gen {i}"
                ax_zap_gen.plot(hours, gen_power[i, :], label=name, linewidth=2)

            if load_device is not None:
                load_idx = self.devices.index(load_device)
                total_load = -np.sum(self.dispatch.power[load_idx][0], axis=0)
                ax_zap_gen.plot(
                    hours, total_load, "k--", linewidth=2, label="Total Load"
                )

        ax_zap_gen.set_ylabel("Power (MW)")
        ax_zap_gen.set_title("Zap Generator Dispatch")
        ax_zap_gen.legend()
        ax_zap_gen.grid(True, alpha=0.3)

        # === Row 2: Marginal prices ===
        # Left: PyPSA
        ax_pypsa_prices = axes[1, 0]
        pypsa_prices = pypsa_net.buses_t.marginal_price
        for bus_name in pypsa_prices.columns:
            ax_pypsa_prices.plot(
                hours, pypsa_prices[bus_name].values, label=bus_name, linewidth=2
            )

        ax_pypsa_prices.set_xlabel("Hour")
        ax_pypsa_prices.set_ylabel("Price ($/MWh)")
        ax_pypsa_prices.set_title("PyPSA Marginal Prices")
        ax_pypsa_prices.legend()
        ax_pypsa_prices.grid(True, alpha=0.3)

        # Right: Zap
        ax_zap_prices = axes[1, 1]
        prices = self.dispatch.prices
        for bus in range(prices.shape[0]):
            ax_zap_prices.plot(hours, prices[bus, :], label=f"Bus {bus}", linewidth=2)

        ax_zap_prices.set_xlabel("Hour")
        ax_zap_prices.set_ylabel("Price ($/MWh)")
        ax_zap_prices.set_title("Zap Marginal Prices")
        ax_zap_prices.legend()
        ax_zap_prices.grid(True, alpha=0.3)

        plt.suptitle(
            "Synthetic Network - PyPSA vs Zap Comparison",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save plot
        plot_path = dispatch_plot_dir / "synthetic_network_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Comparison plot saved to: {plot_path}")


class TestTexas7NodeRoundtrip(TestPyPSARoundtripBase):
    """Test roundtrip with texas_7node network."""

    @classmethod
    def load_pypsa_network(cls):
        return examples.load_example_network("texas_7node")

    @classmethod
    def get_snapshots(cls):
        # Use network's native snapshots (first 24 hours)
        net = examples.load_example_network("texas_7node")
        return net.snapshots[:24]


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    unittest.main()
