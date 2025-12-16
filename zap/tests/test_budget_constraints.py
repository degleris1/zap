"""Tests for budget constraints in the planning module.

This module tests budget constraint functionality using real zap networks
and devices imported from PyPSA, following the pattern from test_pypsa_investment.py.
"""

import unittest
import tempfile
import os
from copy import deepcopy

import cvxpy as cp
import numpy as np
import pandas as pd
import pypsa

import zap
from zap.devices.injector import Generator
from zap.devices.transporter import ACLine
from zap.importers.pypsa import load_pypsa_network, HOURS_PER_YEAR
from zap.planning.constraints import (
    BudgetConstraint,
    BudgetConstraintSet,
    ProjectionQP,
    _build_device_name_mapping,
)


def create_investment_network():
    """Create a simple network with extendable devices for testing.

    Based on TestInvestmentPlanningBase.create_investment_network() from test_pypsa_investment.py.
    """
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

    # Add extendable solar generator
    n.add(
        "Generator",
        "gen_solar",
        bus="bus0",
        p_nom=50.0,
        p_nom_extendable=True,
        p_nom_min=50.0,
        capital_cost=50000.0,
        marginal_cost=0.0,
        carrier="solar",
    )

    # Add extendable gas generator
    n.add(
        "Generator",
        "gen_gas",
        bus="bus1",
        p_nom=100.0,
        p_nom_extendable=True,
        p_nom_min=100.0,
        capital_cost=30000.0,
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

    # Add extendable AC lines
    n.add(
        "Line",
        "line_0_1",
        bus0="bus0",
        bus1="bus1",
        s_nom=50.0,
        s_nom_extendable=True,
        s_nom_min=50.0,
        capital_cost=10000.0,
        x=0.1,
        r=0.01,
    )

    n.add(
        "Line",
        "line_1_2",
        bus0="bus1",
        bus1="bus2",
        s_nom=50.0,
        s_nom_extendable=True,
        s_nom_min=50.0,
        capital_cost=10000.0,
        x=0.1,
        r=0.01,
    )

    # Add solar capacity factor
    solar_cf = 0.5 * (1 + np.sin(np.linspace(-np.pi / 2, np.pi / 2, 24)))
    n.generators_t["p_max_pu"] = pd.DataFrame({"gen_solar": solar_cf}, index=snapshots)

    return n, snapshots


class TestBudgetConstraint(unittest.TestCase):
    """Tests for BudgetConstraint dataclass."""

    def test_create_le_constraint(self):
        constraint = BudgetConstraint(
            name="test",
            coefficients={"param1": {0: 1.0, 1: 2.0}},
            rhs=100.0,
            sense="le",
        )
        self.assertEqual(constraint.name, "test")
        self.assertEqual(constraint.rhs, 100.0)
        self.assertEqual(constraint.sense, "le")

    def test_create_ge_constraint(self):
        constraint = BudgetConstraint(
            name="test",
            coefficients={"param1": {0: 1.0}},
            rhs=50.0,
            sense="ge",
        )
        self.assertEqual(constraint.sense, "ge")

    def test_invalid_sense_raises(self):
        with self.assertRaises(ValueError):
            BudgetConstraint(name="test", coefficients={}, rhs=0.0, sense="eq")


class TestBudgetConstraintsWithZapNetwork(unittest.TestCase):
    """Integration tests using real zap network and devices."""

    @classmethod
    def setUpClass(cls):
        """Set up test network and devices."""
        cls.pypsa_network, cls.snapshots = create_investment_network()
        cls.net, cls.devices = load_pypsa_network(cls.pypsa_network, cls.snapshots)
        cls.time_horizon = len(cls.snapshots)

        # Find generator device and set up parameter_names
        cls.gen_device = None
        cls.gen_device_idx = None
        cls.line_device = None
        cls.line_device_idx = None

        for i, device in enumerate(cls.devices):
            if isinstance(device, Generator):
                cls.gen_device = device
                cls.gen_device_idx = i
            elif isinstance(device, ACLine):
                cls.line_device = device
                cls.line_device_idx = i

        # Parameter names matching test_pypsa_investment.py pattern
        cls.parameter_names = {}
        if cls.gen_device_idx is not None:
            cls.parameter_names["generator"] = (cls.gen_device_idx, "nominal_capacity")
        if cls.line_device_idx is not None:
            cls.parameter_names["ac_line"] = (cls.line_device_idx, "nominal_capacity")

    def test_device_names_are_correct(self):
        """Verify device names match expected PyPSA generator names."""
        self.assertIsNotNone(self.gen_device)
        names = list(self.gen_device.name)
        self.assertIn("gen_solar", names)
        self.assertIn("gen_gas", names)

    def test_device_name_mapping(self):
        """Test device name mapping with real Generator device."""
        mapping = _build_device_name_mapping(self.devices, self.parameter_names)

        # Check generator mappings
        self.assertIn(("gen_solar", "nominal_capacity"), mapping)
        self.assertIn(("gen_gas", "nominal_capacity"), mapping)

        # Verify mapping structure
        param_name, idx = mapping[("gen_solar", "nominal_capacity")]
        self.assertEqual(param_name, "generator")
        self.assertEqual(idx, list(self.gen_device.name).index("gen_solar"))

    def test_csv_parsing_with_real_device_names(self):
        """Test CSV parsing referencing actual generator names."""
        csv_content = """constraint_name,attribute,device_name,multiplier,rhs_value,sense
total_gen_cap,nominal_capacity,gen_solar,1,,
total_gen_cap,nominal_capacity,gen_gas,1,,
total_gen_cap,rhs,,,200,le"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            csv_path = f.name

        try:
            constraint_set = BudgetConstraintSet.from_csv(
                csv_path, self.parameter_names, self.devices
            )

            self.assertEqual(len(constraint_set), 1)
            self.assertEqual(constraint_set.constraints[0].name, "total_gen_cap")
            self.assertEqual(constraint_set.constraints[0].rhs, 200.0)
            self.assertEqual(constraint_set.constraints[0].sense, "le")

            # Verify coefficients reference correct parameter
            coeffs = constraint_set.constraints[0].coefficients
            self.assertIn("generator", coeffs)
        finally:
            os.unlink(csv_path)

    def test_csv_parsing_unknown_device_raises(self):
        """Test that referencing unknown device name raises error."""
        csv_content = """constraint_name,attribute,device_name,multiplier,rhs_value,sense
bad,nominal_capacity,nonexistent_gen,1,,
bad,rhs,,,100,le"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            csv_path = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                BudgetConstraintSet.from_csv(csv_path, self.parameter_names, self.devices)
            self.assertIn("not found", str(ctx.exception))
        finally:
            os.unlink(csv_path)

    def test_constraint_matrix_building(self):
        """Test A, b matrix construction with real device indices."""
        # Create constraint: gen_solar + gen_gas <= 200
        solar_idx = list(self.gen_device.name).index("gen_solar")
        gas_idx = list(self.gen_device.name).index("gen_gas")

        constraint = BudgetConstraint(
            name="total_cap",
            coefficients={"generator": {solar_idx: 1.0, gas_idx: 1.0}},
            rhs=200.0,
            sense="le",
        )
        constraint_set = BudgetConstraintSet([constraint], self.parameter_names, self.devices)

        # Build matrices
        param_sizes = {"generator": self.gen_device.num_devices}
        if self.line_device_idx is not None:
            param_sizes["ac_line"] = self.line_device.num_devices

        param_offsets = {}
        offset = 0
        for p in self.parameter_names.keys():
            param_offsets[p] = offset
            offset += param_sizes[p]
        total_dim = offset

        A_le, b_le, A_ge, b_ge = constraint_set.build_constraint_matrices(
            param_sizes, param_offsets, total_dim
        )

        # Verify LE constraint
        self.assertEqual(A_le.shape[0], 1)
        self.assertEqual(b_le[0], 200.0)

        # Verify coefficients at correct positions
        A_dense = A_le.toarray()[0]
        gen_offset = param_offsets["generator"]
        self.assertEqual(A_dense[gen_offset + solar_idx], 1.0)
        self.assertEqual(A_dense[gen_offset + gas_idx], 1.0)

    def test_projection_le_constraint(self):
        """Test projection with sum of gen capacities <= limit."""
        solar_idx = list(self.gen_device.name).index("gen_solar")
        gas_idx = list(self.gen_device.name).index("gen_gas")

        # Constraint: gen_solar + gen_gas <= 100
        constraint = BudgetConstraint(
            name="cap_limit",
            coefficients={"generator": {solar_idx: 1.0, gas_idx: 1.0}},
            rhs=100.0,
            sense="le",
        )
        constraint_set = BudgetConstraintSet([constraint], self.parameter_names, self.devices)

        # Set up bounds
        lower_bounds = {"generator": np.zeros(self.gen_device.num_devices)}
        upper_bounds = {"generator": np.full(self.gen_device.num_devices, 1000.0)}

        if self.line_device_idx is not None:
            lower_bounds["ac_line"] = np.zeros(self.line_device.num_devices)
            upper_bounds["ac_line"] = np.full(self.line_device.num_devices, 1000.0)

        qp = ProjectionQP(self.parameter_names, lower_bounds, upper_bounds, constraint_set)

        # Test point that violates constraint: 80 + 80 = 160 > 100
        state = {"generator": np.array([80.0, 80.0])}
        if self.line_device_idx is not None:
            state["ac_line"] = np.array([50.0, 50.0])

        projected = qp.project(state)

        # Check constraint is satisfied
        gen_sum = projected["generator"][solar_idx] + projected["generator"][gas_idx]
        self.assertLessEqual(gen_sum, 100.0 + 1e-6)

    def test_projection_ge_constraint(self):
        """Test projection with sum of gen capacities >= minimum."""
        solar_idx = list(self.gen_device.name).index("gen_solar")
        gas_idx = list(self.gen_device.name).index("gen_gas")

        # Constraint: gen_solar + gen_gas >= 150
        constraint = BudgetConstraint(
            name="min_cap",
            coefficients={"generator": {solar_idx: 1.0, gas_idx: 1.0}},
            rhs=150.0,
            sense="ge",
        )
        constraint_set = BudgetConstraintSet([constraint], self.parameter_names, self.devices)

        lower_bounds = {"generator": np.zeros(self.gen_device.num_devices)}
        upper_bounds = {"generator": np.full(self.gen_device.num_devices, 1000.0)}

        if self.line_device_idx is not None:
            lower_bounds["ac_line"] = np.zeros(self.line_device.num_devices)
            upper_bounds["ac_line"] = np.full(self.line_device.num_devices, 1000.0)

        qp = ProjectionQP(self.parameter_names, lower_bounds, upper_bounds, constraint_set)

        # Test point that violates constraint: 30 + 40 = 70 < 150
        state = {"generator": np.array([30.0, 40.0])}
        if self.line_device_idx is not None:
            state["ac_line"] = np.array([50.0, 50.0])

        projected = qp.project(state)

        # Check constraint is satisfied
        gen_sum = projected["generator"][solar_idx] + projected["generator"][gas_idx]
        self.assertGreaterEqual(gen_sum, 150.0 - 1e-6)

    def test_projection_preserves_feasible(self):
        """Test that feasible points are unchanged."""
        solar_idx = list(self.gen_device.name).index("gen_solar")
        gas_idx = list(self.gen_device.name).index("gen_gas")

        # Constraint: gen_solar + gen_gas <= 200
        constraint = BudgetConstraint(
            name="cap_limit",
            coefficients={"generator": {solar_idx: 1.0, gas_idx: 1.0}},
            rhs=200.0,
            sense="le",
        )
        constraint_set = BudgetConstraintSet([constraint], self.parameter_names, self.devices)

        lower_bounds = {"generator": np.zeros(self.gen_device.num_devices)}
        upper_bounds = {"generator": np.full(self.gen_device.num_devices, 1000.0)}

        if self.line_device_idx is not None:
            lower_bounds["ac_line"] = np.zeros(self.line_device.num_devices)
            upper_bounds["ac_line"] = np.full(self.line_device.num_devices, 1000.0)

        qp = ProjectionQP(self.parameter_names, lower_bounds, upper_bounds, constraint_set)

        # Feasible point: 50 + 100 = 150 <= 200
        state = {"generator": np.array([50.0, 100.0])}
        if self.line_device_idx is not None:
            state["ac_line"] = np.array([50.0, 50.0])

        projected = qp.project(state)

        np.testing.assert_array_almost_equal(projected["generator"], state["generator"], decimal=5)

    def test_planning_problem_integration(self):
        """Test full PlanningProblem with budget_constraints parameter."""
        # Create dispatch layer
        layer = zap.DispatchLayer(
            self.net,
            self.devices,
            parameter_names=self.parameter_names,
            time_horizon=self.time_horizon,
            solver=cp.MOSEK,
            solver_kwargs={"verbose": False, "accept_unknown": True},
        )

        # Define objectives
        op_objective = zap.planning.DispatchCostObjective(self.net, self.devices)
        inv_objective = zap.planning.InvestmentObjective(self.devices, layer)

        # Set up bounds
        lower_bounds = {}
        upper_bounds = {}
        for param_name, (device_idx, attr_name) in self.parameter_names.items():
            device = self.devices[device_idx]
            current_cap = getattr(device, attr_name)
            lower_bounds[param_name] = np.zeros_like(current_cap)
            upper_bounds[param_name] = current_cap * 10.0

        # Create budget constraint CSV
        csv_content = """constraint_name,attribute,device_name,multiplier,rhs_value,sense
total_gen_cap,nominal_capacity,gen_solar,1,,
total_gen_cap,nominal_capacity,gen_gas,1,,
total_gen_cap,rhs,,,300,le"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            csv_path = f.name

        try:
            # Create planning problem with budget constraints
            snapshot_weight = HOURS_PER_YEAR / len(self.snapshots)

            problem = zap.planning.PlanningProblem(
                operation_objective=op_objective,
                investment_objective=inv_objective,
                layer=layer,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                budget_constraints=csv_path,
                snapshot_weight=snapshot_weight,
            )

            # Verify budget constraints were loaded
            self.assertIsNotNone(problem.budget_constraints)
            self.assertIsNotNone(problem._projection_qp)

            # Initialize parameters
            initial_params = {}
            for param_name, (device_idx, attr_name) in self.parameter_names.items():
                initial_params[param_name] = deepcopy(getattr(self.devices[device_idx], attr_name))

            # Run a few iterations
            from zap.planning.trackers import LOSS

            optimized_params, history = problem.solve(
                num_iterations=5,
                algorithm=zap.planning.GradientDescent(step_size=1e-4, clip=1e3),
                initial_state=initial_params,
                trackers=[LOSS],
                verbosity=0,
            )

            # Verify constraint is respected in final solution
            gen_cap = optimized_params["generator"]
            solar_idx = list(self.gen_device.name).index("gen_solar")
            gas_idx = list(self.gen_device.name).index("gen_gas")
            total_cap = gen_cap[solar_idx] + gen_cap[gas_idx]

            self.assertLessEqual(
                total_cap, 300.0 + 1e-6, f"Budget constraint violated: {total_cap} > 300"
            )

        finally:
            os.unlink(csv_path)


if __name__ == "__main__":
    unittest.main()
