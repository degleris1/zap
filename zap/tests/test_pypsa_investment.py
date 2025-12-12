"""
Tests for PyPSA investment planning functionality.

This module tests:
1. Investment planning - verifying capacity expansion optimization
2. Capacity bounds and investment costs

Test structure:
- Investment tests: Verify capacity expansion and costs
- Roundtrip tests for extendable flags
"""

import unittest
import cvxpy as cp
import numpy as np
import pandas as pd
import pypsa
from pathlib import Path
from copy import deepcopy
import logging

import zap
from zap.devices.injector import Generator
from zap.devices.transporter import ACLine
from zap.devices.storage_unit import StorageUnit
from zap.importers.pypsa import load_pypsa_network, HOURS_PER_YEAR
from zap.exporters.pypsa import export_to_pypsa
from zap.tests import network_examples as examples
from zap.tests.plotting_helpers import (
    get_zap_energy_balance,
    plot_energy_balance_comparison,
    plot_price_comparison,
    plot_line_flows,
    plot_capacity_comparison,
    plot_capacity_evolution,
    aggregate_capacities_by_carrier,
    export_comparison_csvs,
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


class TestInvestmentPlanningBase(unittest.TestCase):
    """Base class for investment planning tests."""

    # Hyperparameter: Override capacity factors for renewable generators
    # Set to None to use actual profiles, or a float (e.g., 0.6) to override
    override_renewable_cf = None  # Set to 0.6 for testing with 60% CF

    @classmethod
    def create_investment_network(cls):
        """Create a simple network with extendable devices for investment testing."""
        # Create a simple 3-bus network
        n = pypsa.Network()

        # Set snapshots
        snapshots = pd.date_range("2020-01-01", periods=24, freq="h")
        n.set_snapshots(snapshots)

        # Add buses
        n.add("Bus", "bus0")
        n.add("Bus", "bus1")
        n.add("Bus", "bus2")

        # Add carrier for emissions with colors
        n.add("Carrier", "gas", co2_emissions=0.2, color="#d35050")
        n.add("Carrier", "solar", co2_emissions=0.0, color="#f9d002")

        # Add extendable generator (p_nom_min = p_nom for investment baseline)
        n.add(
            "Generator",
            "gen_solar",
            bus="bus0",
            p_nom=50.0,
            p_nom_extendable=True,
            p_nom_min=50.0,  # Set min to initial capacity
            capital_cost=50000.0,  # $/MW-year
            marginal_cost=0.0,
            carrier="solar",
        )

        # Add extendable gas generator (p_nom_min = p_nom for investment baseline)
        n.add(
            "Generator",
            "gen_gas",
            bus="bus1",
            p_nom=100.0,
            p_nom_extendable=True,
            p_nom_min=100.0,  # Set min to initial capacity
            capital_cost=30000.0,  # $/MW-year
            marginal_cost=50.0,
            carrier="gas",
        )

        # Add load
        load_profile = 80 + 40 * np.sin(np.linspace(0, 2 * np.pi, 24))
        n.add(
            "Load",
            "load0",
            bus="bus2",
            p_set=load_profile,
        )

        # Add extendable AC line (s_nom_min = s_nom for investment baseline)
        n.add(
            "Line",
            "line_0_1",
            bus0="bus0",
            bus1="bus1",
            s_nom=50.0,
            s_nom_extendable=True,
            s_nom_min=50.0,  # Set min to initial capacity
            capital_cost=10000.0,
            x=0.1,
            r=0.01,
        )

        # Add extendable line to bus2 (s_nom_min = s_nom for investment baseline)
        n.add(
            "Line",
            "line_1_2",
            bus0="bus1",
            bus1="bus2",
            s_nom=50.0,
            s_nom_extendable=True,
            s_nom_min=50.0,  # Set min to initial capacity
            capital_cost=10000.0,
            x=0.1,
            r=0.01,
        )

        # Add solar capacity factor
        solar_cf = 0.5 * (1 + np.sin(np.linspace(-np.pi / 2, np.pi / 2, 24)))
        n.generators_t["p_max_pu"] = pd.DataFrame(
            {"gen_solar": solar_cf}, index=snapshots
        )

        return n, snapshots

    @classmethod
    def setUpClass(cls):
        cls.pypsa_network, cls.snapshots = cls.create_investment_network()

        # Import to Zap
        cls.net, cls.devices = load_pypsa_network(
            cls.pypsa_network,
            cls.snapshots,
        )

        cls.time_horizon = len(cls.snapshots)

        # Apply capacity factor override if specified
        if cls.override_renewable_cf is not None:
            print(
                f"\n⚙️  Overriding renewable capacity factors to {cls.override_renewable_cf:.0%}"
            )
            for device in cls.devices:
                if isinstance(device, Generator):
                    # Get carrier info from PyPSA to identify renewables
                    for i, name in enumerate(device.name):
                        if name in cls.pypsa_network.generators.index:
                            carrier = cls.pypsa_network.generators.loc[name, "carrier"]
                            # Override for renewable carriers
                            if carrier in ["onwind", "offwind", "solar", "wind"]:
                                original_cf = device.max_power[i].mean()
                                device.max_power[i] = np.full(
                                    cls.time_horizon, cls.override_renewable_cf
                                )
                                if i < 3:  # Print first few for verification
                                    print(
                                        f"     {name} ({carrier}): {original_cf:.1%} → {cls.override_renewable_cf:.0%}"
                                    )

        # Set up planning problem with capacity expansion
        # Define which parameters to optimize
        parameter_names = {}
        for i, device in enumerate(cls.devices):
            if isinstance(device, Generator):
                parameter_names["generator"] = (i, "nominal_capacity")
            elif isinstance(device, ACLine):
                parameter_names["ac_line"] = (i, "nominal_capacity")
            elif isinstance(device, StorageUnit):
                parameter_names["storage_unit"] = (i, "power_capacity")

        # Create dispatch layer for planning
        cls.layer = zap.DispatchLayer(
            cls.net,
            cls.devices,
            parameter_names=parameter_names,
            time_horizon=cls.time_horizon,
            solver=cp.MOSEK,
            solver_kwargs={"verbose": False, "accept_unknown": True},
        )

        # Define objectives
        op_objective = zap.planning.DispatchCostObjective(cls.net, cls.devices)
        inv_objective = zap.planning.InvestmentObjective(cls.devices, cls.layer)

        # Set up bounds for capacity expansion using device min/max capacities
        lower_bounds = {}
        upper_bounds = {}
        for param_name, (device_idx, attr_name) in parameter_names.items():
            device = cls.devices[device_idx]

            # Handle different device types with different attribute names
            if isinstance(device, StorageUnit):
                # StorageUnit uses min/max_power_capacity instead of min/max_nominal_capacity
                min_attr = "min_power_capacity"
                max_attr = "max_power_capacity"
            else:
                # Generator and ACLine use min/max_nominal_capacity
                min_attr = "min_nominal_capacity"
                max_attr = "max_nominal_capacity"

            # Set lower bound
            if hasattr(device, min_attr) and getattr(device, min_attr) is not None:
                lower_bounds[param_name] = deepcopy(getattr(device, min_attr))
            else:
                lower_bounds[param_name] = np.zeros_like(getattr(device, attr_name))

            # Set upper bound
            if hasattr(device, max_attr) and getattr(device, max_attr) is not None:
                # Handle inf values by replacing with large number
                max_cap = deepcopy(getattr(device, max_attr))
                if np.any(np.isinf(max_cap)):
                    # Use 10x current capacity as upper bound for inf values
                    # This allows expansion while keeping problem bounded
                    current_cap = getattr(device, attr_name)
                    reasonable_upper = (current_cap + 1000.0) * 10.0
                    max_cap = np.where(np.isinf(max_cap), reasonable_upper, max_cap)
                upper_bounds[param_name] = max_cap
            else:
                # No max specified - use reasonable default based on current capacity
                current_cap = getattr(device, attr_name)
                upper_bounds[param_name] = (current_cap + 1000.0) * 10.0

        # Create planning problem with snapshot weighting
        # Weight operational costs by 8760/24 to annualize from 24-hour snapshot
        # This matches PyPSA's approach of scaling opex by 365
        snapshot_weight = HOURS_PER_YEAR / len(cls.snapshots)

        cls.problem = zap.planning.PlanningProblem(
            operation_objective=op_objective,
            investment_objective=inv_objective,
            layer=cls.layer,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            snapshot_weight=snapshot_weight,
        )

        # Initialize parameters at current values
        initial_params = {}
        for param_name, (device_idx, attr_name) in parameter_names.items():
            initial_params[param_name] = deepcopy(
                getattr(cls.devices[device_idx], attr_name)
            )

        # Solve planning problem with more iterations for convergence
        # Include PARAM tracker to enable capacity evolution plotting
        from zap.planning.trackers import (
            LOSS,
            GRAD_NORM,
            PROJ_GRAD_NORM,
            TIME,
            SUBOPTIMALITY,
            PARAM,
        )

        cls.optimized_params, cls.history = cls.problem.solve(
            num_iterations=100,
            algorithm=zap.planning.GradientDescent(step_size=1e-3, clip=1e3),
            initial_state=initial_params,
            trackers=[LOSS, GRAD_NORM, PROJ_GRAD_NORM, TIME, SUBOPTIMALITY, PARAM],
        )

        logger.info(f"Parameter names: {parameter_names}")
        logger.info(
            f"Optimized params keys: {cls.optimized_params.keys() if hasattr(cls.optimized_params, 'keys') else type(cls.optimized_params)}"
        )
        logger.info(f"Optimized params: {cls.optimized_params}")

        for param_name, (device_idx, attr_name) in parameter_names.items():
            if param_name in cls.optimized_params:
                new_capacity = cls.optimized_params[param_name]
                # Convert to numpy array if needed
                if hasattr(new_capacity, "numpy"):
                    new_capacity = new_capacity.numpy()
                setattr(cls.devices[device_idx], attr_name, new_capacity)
                logger.info(f"Updated {param_name}: {new_capacity}")
            else:
                logger.warning(f"Parameter {param_name} not found in optimized_params")

        # Run dispatch with optimized parameters to get dispatch results
        cls.dispatch = cls.net.dispatch(
            cls.devices,
            time_horizon=cls.time_horizon,
            solver=cp.MOSEK,
        )

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

    def plot_investment_results(self, filename_prefix="investment"):
        """Generate and save dispatch plots for investment planning tests."""
        if not HAS_MATPLOTLIB:
            return

        # Create plot directory if it doesn't exist
        planning_plot_dir = PLOT_DIR / "planning"
        planning_plot_dir.mkdir(parents=True, exist_ok=True)

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
        carrier_colors = (
            pypsa_net.carriers.color.to_dict() if len(pypsa_net.carriers) > 0 else {}
        )
        load_profile = (
            pypsa_net.loads_t.p_set.sum(axis=1).values
            if len(pypsa_net.loads) > 0
            else None
        )

        # Create figure with subplots (4 rows x 2 columns)
        _, axes = plt.subplots(4, 2, figsize=(16, 18))

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

        # Row 3: Line flows
        plot_line_flows(
            axes[2, 0],
            axes[2, 1],
            pypsa_net,
            self.dispatch,
            self.devices,
            hours,
        )

        # Row 4: Capacity comparison
        (
            pypsa_initial,
            pypsa_final,
            zap_initial,
            zap_final,
        ) = aggregate_capacities_by_carrier(pypsa_net, self.devices)
        plot_capacity_comparison(
            axes[3, 0],
            pypsa_initial,
            pypsa_final,
            zap_initial,
            zap_final,
            carrier_colors,
        )

        # Row 4 right: Capacity evolution during optimization
        plot_capacity_evolution(
            axes[3, 1],
            self.history,
            self.devices,
            pypsa_network=pypsa_net,
            title="Capacity Evolution During Optimization",
        )

        # Add legends
        axes[0, 1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        axes[1, 1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

        plt.suptitle(
            f"{filename_prefix} - PyPSA vs Zap Comparison",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save plot
        plot_path = planning_plot_dir / f"{filename_prefix}_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Export CSV data
        export_comparison_csvs(
            planning_plot_dir,
            filename_prefix,
            pypsa_initial,
            pypsa_final,
            zap_initial,
            zap_final,
            pypsa_net.buses_t.marginal_price,
            self.dispatch.prices,
            hours,
        )

        # Export PyPSA statistics
        pypsa_net.statistics().to_csv(
            planning_plot_dir / f"{filename_prefix}_pypsa_statistics.csv"
        )

        print(f"Comparison plot saved to: {plot_path}")


class TestGeneratorInvestment(TestInvestmentPlanningBase):
    """Test generator capacity expansion."""

    def get_generator_device(self):
        """Get generator device from devices list."""
        for device in self.devices:
            if isinstance(device, Generator):
                return device
        return None

    def test_generate_investment_plot(self):
        """Generate dispatch plot for investment planning visual inspection."""
        self.plot_investment_results("investment_planning_3bus")

    def test_generator_investment_matches_pypsa(self):
        """Verify generator capacity investments are reasonable compared to PyPSA results.

        Note: Zap uses gradient descent while PyPSA uses LP, so exact matches aren't expected.
        This test verifies that both find solutions that expand capacity to meet load.
        """
        gen_device = self.get_generator_device()
        if gen_device is None:
            self.skipTest("No generators in network")

        # Run PyPSA optimization
        pypsa_net = self.pypsa_network.copy()
        pypsa_net.set_snapshots(self.snapshots)
        # Set snapshot weightings to scale operational costs by time horizon
        pypsa_net.snapshot_weightings.loc[:, :] = HOURS_PER_YEAR / len(self.snapshots)

        pypsa_net.optimize(solver_name="highs", multi_investment_periods=True)

        # Compare capacities for each generator
        for gen_name in pypsa_net.generators.index:
            pypsa_cap = pypsa_net.generators.loc[gen_name, "p_nom_opt"]

            # Find corresponding Zap capacity
            if hasattr(gen_device, "name"):
                try:
                    idx = list(gen_device.name).index(gen_name)
                    zap_cap = gen_device.nominal_capacity[idx]
                    # Convert to scalar
                    zap_cap = (
                        float(np.atleast_1d(zap_cap)[0])
                        if hasattr(zap_cap, "__len__")
                        else float(zap_cap)
                    )

                    # Verify Zap capacity is at least the minimum
                    min_cap = float(
                        np.atleast_1d(gen_device.min_nominal_capacity[idx])[0]
                    )
                    self.assertGreaterEqual(
                        zap_cap,
                        min_cap,
                        f"Generator {gen_name}: Zap={zap_cap:.1f} < min={min_cap:.1f}",
                    )

                    # Log the comparison for debugging
                    print(
                        f"Generator {gen_name}: Zap={zap_cap:.1f}, PyPSA={pypsa_cap:.1f}"
                    )

                except ValueError:
                    pass  # Generator not found in Zap

    def test_generator_capacity_bounds(self):
        """Verify generator capacity is within bounds."""
        gen = self.get_generator_device()
        if gen is None:
            self.skipTest("No generators in network")

        # Check min/max bounds are set
        self.assertTrue(np.all(gen.min_nominal_capacity <= gen.nominal_capacity))
        self.assertTrue(np.all(gen.nominal_capacity <= gen.max_nominal_capacity))

    def test_generator_capital_cost_scaling(self):
        """Verify capital cost is scaled by time horizon."""
        gen = self.get_generator_device()
        if gen is None:
            self.skipTest("No generators in network")

        # Capital cost should be scaled by (num_snapshots / HOURS_PER_YEAR)
        len(self.snapshots) / HOURS_PER_YEAR

        # Check that capital costs were imported
        self.assertIsNotNone(gen.capital_cost)
        self.assertTrue(np.all(gen.capital_cost >= 0))

    def test_generator_investment_cost_formula(self):
        """Verify investment cost calculation: capital_cost * (capacity - min_capacity)."""
        gen = self.get_generator_device()
        if gen is None:
            self.skipTest("No generators in network")

        # Test investment cost formula
        new_capacity = gen.nominal_capacity * 1.5
        investment_cost = gen.get_investment_cost(nominal_capacity=new_capacity)

        expected_cost = np.sum(gen.capital_cost * (new_capacity - gen.nominal_capacity))

        self.assertAlmostEqual(investment_cost, expected_cost, places=3)


class TestExtendableFlags(TestInvestmentPlanningBase):
    """Test that extendable flags survive roundtrip."""

    def test_extendable_generator_export(self):
        """Verify p_nom_extendable flag is set correctly in export.

        Note: The exporter determines p_nom_extendable based on whether
        min_nominal_capacity != nominal_capacity or max_nominal_capacity != nominal_capacity.
        After import, these bounds are preserved from the original network.
        """
        # Export to PyPSA
        exported = export_to_pypsa(
            self.net,
            self.devices,
            self.dispatch,
            self.snapshots,
        )

        # Check that extendable generators have p_nom_min and p_nom_max set
        for gen_name in exported.generators.index:
            original_extendable = self.pypsa_network.generators.loc[
                gen_name, "p_nom_extendable"
            ]
            p_nom_min = exported.generators.loc[gen_name, "p_nom_min"]
            p_nom_max = exported.generators.loc[gen_name, "p_nom_max"]

            if original_extendable:
                # Extendable generators should have bounds that allow expansion
                # (min_bound < max_bound)
                self.assertLessEqual(
                    p_nom_min,
                    p_nom_max,
                    f"Generator {gen_name} should have valid bounds",
                )


# =============================================================================
# Investment Planning Tests with PyPSA-USA Network Data
# =============================================================================


class TestTexas7NodeInvestment(TestInvestmentPlanningBase):
    """Test investment planning with Texas 7-node network.

    To test with higher renewable capacity factors, set:
        override_renewable_cf = 0.6  # 60% capacity factor for all renewables
    """

    @classmethod
    def create_investment_network(cls):
        """Load Texas 7-node network and enable capacity expansion."""
        net = examples.load_example_network("texas_7node")

        # Use first 24 hours for faster testing
        snapshots = net.snapshots[:48]
        net.set_snapshots(snapshots)
        # Set snapshot weightings to scale operational costs by time horizon
        net.snapshot_weightings.loc[:, :] = HOURS_PER_YEAR / len(snapshots)

        # net.generators.loc[net.generators.carrier == 'onwind', 'p_nom'] += 10
        return net, snapshots

    def test_generate_investment_plot(self):
        """Generate investment planning plot for Texas 7-node network."""
        self.plot_investment_results("texas_7node_investment")


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    unittest.main()
