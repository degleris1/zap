"""Tests for the one-way, lossy ``DirectedLine`` device (phase-1 spec WP3.2).

The assertions are on dispatch values, so the tests fail if the efficiency or the
one-way bound is dropped. Both the cvxpy path and the ADMM path are covered.
"""

import unittest

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import torch

import zap
from zap.admm import ADMMSolver
from zap.devices import DirectedLine


VOLL = 1000.0


def make_line(source, sink, capacity, linear_cost=0.0, num_nodes=2):
    n = np.size(source)
    return DirectedLine(
        num_nodes=num_nodes,
        name=np.array([f"line{i}" for i in range(n)]),
        source_terminal=np.asarray(source),
        sink_terminal=np.asarray(sink),
        min_power=np.zeros(n),
        max_power=np.ones(n),
        linear_cost=np.full(n, linear_cost, dtype=float),
        nominal_capacity=np.full(n, capacity, dtype=float),
    )


def make_line_eff(source, sink, capacity, efficiency, linear_cost=0.0, num_nodes=2):
    line = make_line(source, sink, capacity, linear_cost=linear_cost, num_nodes=num_nodes)
    line.efficiency = np.full((line.num_devices, 1), efficiency, dtype=float)
    return line


def two_bus_system(gen_bus, load_bus, load_value, efficiency=1.0, T=4):
    net = zap.PowerNetwork(num_nodes=2)
    generator = zap.Generator(
        num_nodes=2,
        name=np.array(["gen"]),
        terminal=np.array([gen_bus]),
        dynamic_capacity=np.full((1, T), 500.0),
        linear_cost=np.array([10.0]),
    )
    load = zap.Load(
        num_nodes=2,
        name=np.array(["load"]),
        terminal=np.array([load_bus]),
        load=np.full((1, T), load_value),
        linear_cost=np.array([VOLL]),
    )
    line = make_line_eff([0], [1], 500.0, efficiency)
    return net, [generator, load, line], T


class TestDirectedLine(unittest.TestCase):
    def test_rejects_negative_min_power(self):
        # Reverse flow through a lossy line would create energy (H1).
        with self.assertRaises(ValueError):
            DirectedLine(
                num_nodes=2,
                name=np.array(["line0"]),
                source_terminal=np.array([0]),
                sink_terminal=np.array([1]),
                min_power=np.array([-1.0]),
                max_power=np.array([1.0]),
                linear_cost=np.array([0.0]),
                nominal_capacity=np.array([100.0]),
            )

    def test_one_way_flow(self):
        # Generator at bus 0 (the line's source), load at bus 1 (the sink)
        net, devices, T = two_bus_system(gen_bus=0, load_bus=1, load_value=50.0)
        out = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS)

        line_power = out.power[2]
        np.testing.assert_allclose(line_power[1], np.full((1, T), 50.0), atol=1e-6)
        np.testing.assert_allclose(line_power[0], np.full((1, T), -50.0), atol=1e-6)

        # Load fully served: load power == -load
        np.testing.assert_allclose(out.power[1][0], np.full((1, T), -50.0), atol=1e-6)

        # Reversed: generator at the sink, load at the source. The line cannot
        # carry power backwards, so the load is entirely shed.
        net, devices, T = two_bus_system(gen_bus=1, load_bus=0, load_value=50.0)
        out = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS)

        np.testing.assert_allclose(out.power[2][1], np.zeros((1, T)), atol=1e-6)
        # Energy not served == full load (Load power is zero, ENS = load + power)
        ens = out.power[1][0] + 50.0
        np.testing.assert_allclose(ens, np.full((1, T), 50.0), atol=1e-6)
        self.assertAlmostEqual(out.problem.value, VOLL * 50.0 * T, delta=1e-3)

    def test_efficiency_losses(self):
        net, devices, T = two_bus_system(gen_bus=0, load_bus=1, load_value=90.0, efficiency=0.9)
        out = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS)

        # 90 MW delivered at the sink requires 100 MW withdrawn at the source
        np.testing.assert_allclose(out.power[2][1], np.full((1, T), 90.0), rtol=1e-6)
        np.testing.assert_allclose(out.power[2][0], np.full((1, T), -100.0), rtol=1e-6)
        np.testing.assert_allclose(out.power[0][0], np.full((1, T), 100.0), rtol=1e-6)

    def test_negative_cost_is_dcp_and_binds(self):
        T = 4
        capacity = 100.0
        net = zap.PowerNetwork(num_nodes=2)

        generator = zap.Generator(
            num_nodes=2,
            name=np.array(["gen"]),
            terminal=np.array([0]),
            dynamic_capacity=np.full((1, T), 500.0),
            linear_cost=np.array([10.0]),
        )
        # A free sink at bus 1 that can absorb anything the line delivers
        sink = zap.Injector(
            num_nodes=2,
            name=np.array(["sink"]),
            terminal=np.array([1]),
            min_power=np.full((1, T), -1.0),
            max_power=np.zeros((1, T)),
            linear_cost=np.zeros((1, T)),
            nominal_capacity=np.array([1000.0]),
        )
        line = make_line_eff([0], [1], capacity, 1.0, linear_cost=-100.0)

        devices = [generator, sink, line]
        out = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS)

        self.assertTrue(out.problem.is_dcp())
        self.assertIn(out.problem.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])

        # -100 $/MWh export revenue against 10 $/MWh generation: flow saturates
        np.testing.assert_allclose(out.power[2][1], np.full((1, T), capacity), rtol=1e-6)
        self.assertAlmostEqual(out.problem.value, (10.0 - 100.0) * capacity * T, delta=1e-3)

    def test_prox_matches_cvxpy(self):
        rho = 2.5
        rng = np.random.default_rng(0)
        n, T = 3, 4

        for efficiency in [1.0, 0.9]:
            for b in [-5.0, 0.0, 5.0]:
                for a in [None, 0.5]:
                    line = make_line_eff(
                        np.arange(n),
                        (np.arange(n) + 1) % n,
                        10.0,
                        efficiency,
                        linear_cost=b,
                        num_nodes=n,
                    )
                    if a is not None:
                        line.quadratic_cost = np.full((n, 1), a)

                    z0 = rng.normal(scale=8.0, size=(n, T))
                    z1 = rng.normal(scale=8.0, size=(n, T))

                    torch_line = line.torchify(machine="cpu", dtype=torch.float64)
                    power, angle, local = torch_line.admm_prox_update(
                        rho,
                        None,
                        [torch.tensor(z0), torch.tensor(z1)],
                        None,
                    )
                    self.assertIsNone(angle)
                    self.assertIsNone(local)

                    p0 = cp.Variable((n, T))
                    p1 = cp.Variable((n, T))
                    obj = (
                        cp.sum(cp.multiply(line.linear_cost, p1))
                        + (rho / 2) * cp.sum_squares(p0 - z0)
                        + (rho / 2) * cp.sum_squares(p1 - z1)
                    )
                    if a is not None:
                        obj = obj + a * cp.sum_squares(p1)
                    constraints = [
                        p1 + cp.multiply(line.efficiency, p0) == 0,
                        p1 >= cp.multiply(line.min_power, line.nominal_capacity),
                        p1 <= cp.multiply(line.max_power, line.nominal_capacity),
                    ]
                    problem = cp.Problem(cp.Minimize(obj), constraints)
                    problem.solve(
                        solver=cp.CLARABEL,
                        tol_gap_abs=1e-12,
                        tol_gap_rel=1e-12,
                        tol_feas=1e-12,
                    )

                    np.testing.assert_allclose(
                        power[0].numpy(),
                        p0.value,
                        atol=1e-6,
                        err_msg=f"eta={efficiency} b={b} a={a}",
                    )
                    np.testing.assert_allclose(
                        power[1].numpy(),
                        p1.value,
                        atol=1e-6,
                        err_msg=f"eta={efficiency} b={b} a={a}",
                    )

    def _three_bus_system(self, T=24):
        rng = np.random.default_rng(7)
        net = zap.PowerNetwork(num_nodes=3)

        generator = zap.Generator(
            num_nodes=3,
            name=np.array(["cheap", "expensive"]),
            terminal=np.array([0, 2]),
            dynamic_capacity=np.vstack([np.full(T, 60.0), np.full(T, 60.0)]),
            linear_cost=np.array([10.0, 40.0]),
        )
        load = zap.Load(
            num_nodes=3,
            name=np.array(["l1", "l2"]),
            terminal=np.array([1, 2]),
            load=np.vstack([30.0 + 10.0 * rng.random(T), 20.0 + 10.0 * rng.random(T)]),
            linear_cost=np.array([VOLL, VOLL]),
        )
        # Two directed lossy lines: 0 -> 1 and 1 -> 2
        line = DirectedLine(
            num_nodes=3,
            name=np.array(["a", "b"]),
            source_terminal=np.array([0, 1]),
            sink_terminal=np.array([1, 2]),
            min_power=np.zeros(2),
            max_power=np.ones(2),
            linear_cost=np.array([0.1, 0.1]),
            nominal_capacity=np.array([50.0, 50.0]),
            efficiency=np.array([0.95, 0.9]),
        )
        return net, [generator, load, line], T

    def test_admm_matches_cvx_on_small_network(self):
        net, devices, T = self._three_bus_system()

        out = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS)

        torch_devices = [d.torchify(machine="cpu", dtype=torch.float64) for d in devices]
        solver = ADMMSolver(
            num_iterations=5000,
            rho_power=1.0,
            dtype=torch.float64,
            machine="cpu",
            verbose=0,
            atol=1e-8,
            rtol=1e-8,
        )
        state, _ = solver.solve(net, torch_devices, T)

        self.assertAlmostEqual(float(state.objective) / out.problem.value, 1.0, delta=0.01)

        p_nom = devices[2].nominal_capacity
        for terminal in [0, 1]:
            admm_flow = np.asarray(state.power[2][terminal])
            np.testing.assert_allclose(
                admm_flow, out.power[2][terminal], atol=float(1e-2 * np.max(p_nom))
            )

        # The efficiency relation holds in the ADMM solution too
        np.testing.assert_allclose(
            np.asarray(state.power[2][1]),
            -devices[2].efficiency * np.asarray(state.power[2][0]),
            atol=1e-8,
        )

    def test_equality_matrices_match_autograd(self):
        n, T = 3, 5
        line = make_line_eff(np.arange(n), (np.arange(n) + 1) % n, 10.0, 0.87, num_nodes=n)
        # Distinct efficiencies per device
        line.efficiency = np.array([[0.87], [0.93], [1.0]])

        rng = np.random.default_rng(3)
        power = [rng.normal(size=(n, T)), rng.normal(size=(n, T))]

        eqs = line.equality_constraints(power, None, None)
        mats = line.equality_matrices(eqs, power, None, None)

        analytic = [sp.csr_matrix(mats[0].power[i]).toarray() for i in range(2)]

        # Numerical Jacobian of the (flattened) equality constraint
        eps = 1e-6
        for terminal in range(2):
            numerical = np.zeros((n * T, n * T))
            for j in range(n * T):
                perturbed = [p.copy() for p in power]
                perturbed[terminal].ravel()[j] += eps
                c_plus = line.equality_constraints(perturbed, None, None)[0].ravel()

                perturbed = [p.copy() for p in power]
                perturbed[terminal].ravel()[j] -= eps
                c_minus = line.equality_constraints(perturbed, None, None)[0].ravel()

                numerical[:, j] = (c_plus - c_minus) / (2 * eps)

            np.testing.assert_allclose(analytic[terminal], numerical, atol=1e-6)

        # And the analytic Jacobian is exactly diag(efficiency) / identity
        expected0 = np.diag(np.broadcast_to(line.efficiency, (n, T)).ravel())
        np.testing.assert_allclose(analytic[0], expected0, atol=1e-12)
        np.testing.assert_allclose(analytic[1], np.eye(n * T), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
