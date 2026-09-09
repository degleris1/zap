"""Tests for time-varying storage power availability (phase-1 spec WP3.1).

Every test must fail if the ``power_availability`` derate is dropped from the
charge/discharge bounds, so the assertions are on dispatch *values*, not merely
on the absence of exceptions. Both the cvxpy path and the ADMM path are covered.
"""

import unittest

import cvxpy as cp
import numpy as np
import torch

import zap
from zap.admm import ADMMSolver
from zap.devices.storage_unit import StorageUnit, get_ymin_ymax


VOLL = 1000.0
GEN_COST = 10.0

# Hours in which the storage unit is unavailable
OUTAGE_HOURS = list(range(5, 11))
STRESS_HOUR = 7

POWER_CAPACITY = 10.0
DURATION = 4.0


def build_system(power_availability=None, time_horizon=12):
    """A 2-bus system whose only way to survive hour 7 is to discharge storage.

    Bus 0 carries the generator, bus 1 the load and the storage unit. The
    generator is capacity-limited in the stress hour, so the load can only be
    served if the battery discharges then.
    """
    T = time_horizon
    net = zap.PowerNetwork(num_nodes=2)

    capacity = np.full((1, T), 30.0)
    capacity[0, STRESS_HOUR] = 20.0

    generator = zap.Generator(
        num_nodes=2,
        name=np.array(["gen"]),
        terminal=np.array([0]),
        dynamic_capacity=capacity,
        linear_cost=np.array([GEN_COST]),
    )
    load = zap.Load(
        num_nodes=2,
        name=np.array(["load"]),
        terminal=np.array([1]),
        load=np.full((1, T), 25.0),
        linear_cost=np.array([VOLL]),
    )
    line = zap.DCLine(
        num_nodes=2,
        name=np.array(["line"]),
        source_terminal=np.array([0]),
        sink_terminal=np.array([1]),
        capacity=np.array([100.0]),
        linear_cost=np.array([0.0]),
    )
    battery = StorageUnit(
        num_nodes=2,
        name=np.array(["battery"]),
        terminal=np.array([1]),
        power_capacity=np.array([POWER_CAPACITY]),
        duration=np.array([DURATION]),
        linear_cost=np.array([0.01]),
        power_availability=power_availability,
    )

    return net, [generator, load, line, battery], T


def outage_availability(time_horizon=12):
    avail = np.ones((1, time_horizon))
    avail[0, OUTAGE_HOURS] = 0.0
    return avail


def solve_cvx(devices, net, T):
    return net.dispatch(devices, time_horizon=T, solver=cp.HIGHS)


def solve_admm(devices, net, T, num_iterations=2000):
    # The battery prox returns the *unclipped* inner iterate, so the box
    # constraints are only satisfied to inner-solver accuracy. The default of 10
    # inner iterations leaves violations of order 1e-1 MW here; 200 drives them
    # to machine precision, which is what lets this test assert on bounds.
    torch_devices = [d.torchify(machine="cpu", dtype=torch.float64) for d in devices]
    solver = ADMMSolver(
        num_iterations=num_iterations,
        rho_power=1.0,
        dtype=torch.float64,
        machine="cpu",
        verbose=0,
        battery_inner_iterations=200,
        atol=1e-7,
        rtol=1e-7,
    )
    state, _ = solver.solve(net, torch_devices, T)
    return state


class TestStorageAvailability(unittest.TestCase):
    def test_default_availability_is_identity(self):
        """None and an explicit all-ones array must give the same dispatch."""
        net_a, devices_a, T = build_system(power_availability=None)
        net_b, devices_b, _ = build_system(power_availability=np.ones((1, 12)))

        out_a = solve_cvx(devices_a, net_a, T)
        out_b = solve_cvx(devices_b, net_b, T)

        self.assertAlmostEqual(out_a.problem.value, out_b.problem.value, places=6)
        for pa, pb in zip(out_a.power, out_b.power):
            for pa_i, pb_i in zip(pa, pb):
                np.testing.assert_allclose(pa_i, pb_i, rtol=1e-9, atol=1e-9)

        state_a = solve_admm(devices_a, net_a, T)
        state_b = solve_admm(devices_b, net_b, T)
        np.testing.assert_allclose(
            float(state_a.objective), float(state_b.objective), rtol=1e-9, atol=1e-9
        )

    def test_cvx_dispatch_respects_availability(self):
        avail = outage_availability()

        net, devices, T = build_system(power_availability=avail)
        out = solve_cvx(devices, net, T)

        charge = out.local_variables[3][1]
        discharge = out.local_variables[3][2]
        bound = POWER_CAPACITY * avail

        self.assertTrue(np.all(charge <= bound + 1e-6))
        self.assertTrue(np.all(discharge <= bound + 1e-6))

        # The derated unit cannot help in the stress hour
        self.assertLess(discharge[0, STRESS_HOUR], 1e-6)

        # ... but the identical system without the derate does discharge then,
        # so the test above cannot pass vacuously.
        net_free, devices_free, _ = build_system(power_availability=None)
        out_free = solve_cvx(devices_free, net_free, T)
        discharge_free = out_free.local_variables[3][2]
        self.assertGreater(discharge_free[0, STRESS_HOUR], 4.9)

        # Load shedding at VOLL therefore makes the derated system more expensive
        self.assertGreater(out.problem.value, out_free.problem.value + 1000.0)

    def test_admm_dispatch_respects_availability(self):
        avail = outage_availability()

        net, devices, T = build_system(power_availability=avail)
        cvx_out = solve_cvx(devices, net, T)
        state = solve_admm(devices, net, T)

        # ADMM reports (discharge - charge) as the device power; recover bounds
        # from the net power, which is bounded by the derated capacity either way.
        net_power = np.asarray(state.power[3][0])
        bound = POWER_CAPACITY * avail
        tol = 1e-3 * POWER_CAPACITY

        self.assertTrue(np.all(net_power <= bound + tol))
        self.assertTrue(np.all(net_power >= -bound - tol))

        # Specifically: no discharge in the stress hour
        self.assertLess(net_power[0, STRESS_HOUR], tol)

        self.assertAlmostEqual(float(state.objective) / cvx_out.problem.value, 1.0, delta=0.01)

        # Sanity: without the derate ADMM *does* discharge in the stress hour
        net_free, devices_free, _ = build_system(power_availability=None)
        state_free = solve_admm(devices_free, net_free, T)
        free_power = np.asarray(state_free.power[3][0])
        self.assertGreater(free_power[0, STRESS_HOUR], 4.0)

    def test_get_ymin_ymax_reshapes_per_scenario(self):
        """Guards the (N, S, 3T + 1, 1) reshape used by the battery prox."""
        N, T, S = 2, 3, 2
        pmax_t = torch.arange(N * S * T, dtype=torch.float64).reshape(N, S * T)
        smax = torch.tensor([[40.0], [80.0]], dtype=torch.float64)
        gamma1 = 0.5 * smax
        gammaT = 0.25 * smax

        ymin, ymax = get_ymin_ymax(T, S, pmax_t, smax, gamma1, gammaT, None, torch.float64)

        self.assertEqual(tuple(ymax.shape), (N, S, 3 * T + 1, 1))
        self.assertEqual(tuple(ymin.shape), (N, S, 3 * T + 1, 1))

        for n in range(N):
            for s in range(S):
                window = pmax_t[n, s * T : (s + 1) * T]
                # Charge and discharge blocks both carry the window's bound
                torch.testing.assert_close(ymax[n, s, :T, 0], window)
                torch.testing.assert_close(ymax[n, s, T : 2 * T, 0], window)
                # Interior energy slots carry smax
                torch.testing.assert_close(
                    ymax[n, s, 2 * T + 1 : 3 * T, 0],
                    smax[n, 0].expand(T - 1),
                )
                # SoC boundary conditions are equalities
                self.assertAlmostEqual(float(ymin[n, s, 2 * T, 0]), float(gamma1[n, 0]))
                self.assertAlmostEqual(float(ymax[n, s, 2 * T, 0]), float(gamma1[n, 0]))
                self.assertAlmostEqual(float(ymin[n, s, -1, 0]), float(gammaT[n, 0]))
                self.assertAlmostEqual(float(ymax[n, s, -1, 0]), float(gammaT[n, 0]))

        # Everything except the SoC boundaries has a zero lower bound
        interior = ymin[:, :, : 2 * T, 0]
        torch.testing.assert_close(interior, torch.zeros_like(interior))

    def test_admm_window_respects_availability(self):
        """The battery_window path must derate per (scenario, hour), not per device."""
        avail = outage_availability()
        net, devices, T = build_system(power_availability=avail)

        torch_devices = [d.torchify(machine="cpu", dtype=torch.float64) for d in devices]
        solver = ADMMSolver(
            num_iterations=3000,
            rho_power=1.0,
            dtype=torch.float64,
            machine="cpu",
            verbose=0,
            battery_inner_iterations=200,
            atol=1e-7,
            rtol=1e-7,
            battery_window=6,
        )
        state, _ = solver.solve(net, torch_devices, T)

        net_power = np.asarray(state.power[3][0])
        self.assertTrue(np.all(np.abs(net_power) <= POWER_CAPACITY * avail + 1e-6))

    def test_time_horizon_reports_availability_length(self):
        _, devices, _ = build_system(power_availability=None)
        self.assertEqual(devices[3].time_horizon, 0)

        _, devices, _ = build_system(power_availability=np.ones((1, 1)))
        self.assertEqual(devices[3].time_horizon, 0)

        _, devices, _ = build_system(power_availability=outage_availability(12))
        self.assertEqual(devices[3].time_horizon, 12)

    def test_sample_time_subsets_availability(self):
        avail = np.arange(48, dtype=float).reshape(1, 48) / 48.0
        battery = StorageUnit(
            num_nodes=2,
            name=np.array(["battery"]),
            terminal=np.array([1]),
            power_capacity=np.array([POWER_CAPACITY]),
            duration=np.array([DURATION]),
            linear_cost=np.array([0.01]),
            power_availability=avail,
        )

        periods = list(range(12, 24))
        sampled = battery.sample_time(periods, 48)

        self.assertEqual(sampled.power_availability.shape, (1, 12))
        np.testing.assert_allclose(sampled.power_availability, avail[:, periods], rtol=1e-12)
        self.assertEqual(sampled.time_horizon, 12)

        # Static attributes untouched
        np.testing.assert_allclose(sampled.power_capacity, battery.power_capacity)
        np.testing.assert_allclose(sampled.duration, battery.duration)
        np.testing.assert_allclose(sampled.initial_soc, battery.initial_soc)
        np.testing.assert_allclose(sampled.final_soc, battery.final_soc)

        # Original device unmodified
        self.assertEqual(battery.power_availability.shape, (1, 48))

    def test_energy_cap_not_derated(self):
        """Pins decision D8: availability derates power only, never energy."""
        T = 4
        battery = StorageUnit(
            num_nodes=2,
            name=np.array(["battery"]),
            terminal=np.array([1]),
            power_capacity=np.array([POWER_CAPACITY]),
            duration=np.array([DURATION]),
            power_availability=np.full((1, 1), 0.5),
        )

        energy_cap = POWER_CAPACITY * DURATION
        state = [
            np.full((1, T + 1), energy_cap),  # energy at the undegraded cap
            np.full((1, T), 0.5 * POWER_CAPACITY),  # charge at the derated cap
            np.full((1, T), 0.5 * POWER_CAPACITY),  # discharge at the derated cap
        ]
        power = [np.zeros((1, T))]

        ineqs = battery.inequality_constraints(power, None, state)

        # Energy at power_capacity * duration is exactly feasible (not derated)
        np.testing.assert_allclose(ineqs[1], np.zeros((1, T + 1)), atol=1e-12)
        # Charge / discharge at the derated cap are exactly at their bound
        np.testing.assert_allclose(ineqs[3], np.zeros((1, T)), atol=1e-12)
        np.testing.assert_allclose(ineqs[5], np.zeros((1, T)), atol=1e-12)

        # And one step above the derated power bound is infeasible
        state[1] = np.full((1, T), 0.5 * POWER_CAPACITY + 1.0)
        ineqs = battery.inequality_constraints(power, None, state)
        np.testing.assert_allclose(ineqs[3], np.ones((1, T)), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
