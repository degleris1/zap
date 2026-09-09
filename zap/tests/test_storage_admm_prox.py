"""The battery ADMM prox must solve the same problem as the StorageUnit LP model.

Regression tests for the defects found in the phase-1 benchmark review
(`memory/plans/2026-09-08-benchmark-review.md`, section 2.2): the prox dropped the
discharge efficiency, charged the linear cost to `charge` instead of `discharge`,
used row 0's parameters for the whole fleet, and returned the unclipped inner
iterate. Every test here fails on the pre-fix code.
"""

import unittest

import cvxpy as cp
import numpy as np
import torch

import zap
from zap.admm import ADMMSolver
from zap.devices import StorageUnit

torch.set_default_dtype(torch.float64)


def heterogeneous_battery(N=5, seed=0):
    """A fleet with distinct efficiencies, durations, capacities and costs."""
    rng = np.random.default_rng(seed)
    return StorageUnit(
        num_nodes=1,
        name=np.array([f"b{i}" for i in range(N)]),
        terminal=np.zeros(N, dtype=int),
        power_capacity=np.array([10.0, 25.0, 7.5, 100.0, 3.0])[:N],
        duration=np.array([4.0, 2.0, 8.0, 1.5, 6.0])[:N],
        charge_efficiency=np.array([0.95, 0.88, 1.0, 0.92, 0.80])[:N],
        discharge_efficiency=np.array([0.85, 0.99, 0.90, 1.0, 0.75])[:N],
        linear_cost=np.array([1.5, 0.0, 3.0, 0.25, 10.0])[:N],
        initial_soc=rng.uniform(0.3, 0.7, N),
        final_soc=rng.uniform(0.3, 0.7, N),
    )


def exact_prox(device: StorageUnit, z: np.ndarray, rho: float):
    """The prox the ADMM update is supposed to compute, solved by a conic solver.

    minimize   linear_cost . discharge + (rho / 2) ||p - z||^2
    subject to the StorageUnit LP model's own equality / inequality constraints
    """
    N, T = z.shape
    power = cp.Variable((N, T))
    energy = cp.Variable((N, T + 1))
    charge = cp.Variable((N, T))
    discharge = cp.Variable((N, T))
    state = [energy, charge, discharge]

    constraints = [c == 0 for c in device.equality_constraints([power], None, state, la=cp)]
    constraints += [c <= 0 for c in device.inequality_constraints([power], None, state, la=cp)]

    objective = device.operation_cost([power], None, state, la=cp)
    objective += 0.5 * rho * cp.sum_squares(power - z)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.CLARABEL)
    assert problem.status == cp.OPTIMAL, problem.status
    return power.value, energy.value, charge.value, discharge.value


def run_prox(device: StorageUnit, z: np.ndarray, rho: float, inner_iterations: int):
    torch_device = device.torchify(machine="cpu", dtype=torch.float64)
    zt = torch.tensor(z, dtype=torch.float64)
    power, _angle, state = torch_device.admm_prox_update(
        rho,
        rho,
        [zt],
        None,
        inner_iterations=inner_iterations,
        inner_over_relaxation=1.8,
    )
    return power[0].numpy(), state


class TestStorageProx(unittest.TestCase):
    def setUp(self):
        self.T = 12
        self.rho = 1.0
        self.device = heterogeneous_battery()
        rng = np.random.default_rng(7)
        pmax = np.asarray(self.device.power_capacity).reshape(-1, 1)
        self.z = rng.uniform(-1.0, 1.0, (self.device.num_devices, self.T)) * pmax

    def test_prox_matches_cvxpy_heterogeneous(self):
        p_exact, _, _, _ = exact_prox(self.device, self.z, self.rho)
        p_admm, _ = run_prox(self.device, self.z, self.rho, inner_iterations=2000)

        tol = 1e-3 * float(np.max(self.device.power_capacity))
        self.assertLess(float(np.max(np.abs(p_admm - p_exact))), tol)

    def test_prox_respects_own_bounds(self):
        p_admm, state = run_prox(self.device, self.z, self.rho, inner_iterations=10)
        pmax = np.asarray(self.device.power_capacity).reshape(-1, 1)

        charge = np.asarray(state.charge)
        discharge = np.asarray(state.discharge)
        self.assertTrue(np.all(np.abs(p_admm) <= pmax + 1e-9))
        self.assertTrue(np.all(charge >= -1e-9))
        self.assertTrue(np.all(discharge >= -1e-9))
        self.assertTrue(np.all(charge <= pmax + 1e-9))
        self.assertTrue(np.all(discharge <= pmax + 1e-9))

    def test_prox_local_variables_satisfy_soc(self):
        _, state = run_prox(self.device, self.z, self.rho, inner_iterations=5000)
        energy = np.asarray(state.energy)
        charge = np.asarray(state.charge)
        discharge = np.asarray(state.discharge)

        beta = np.asarray(self.device.charge_efficiency).reshape(-1, 1)
        eta = np.asarray(self.device.discharge_efficiency).reshape(-1, 1)
        emax = np.asarray(self.device.power_capacity).reshape(-1) * np.asarray(
            self.device.duration
        ).reshape(-1)

        residual = energy[:, 1:] - (energy[:, :-1] + beta * charge - discharge / eta)
        self.assertLess(float(np.max(np.abs(residual))), 1e-6)

        init = np.asarray(self.device.initial_soc).reshape(-1) * emax
        final = np.asarray(self.device.final_soc).reshape(-1) * emax
        np.testing.assert_allclose(energy[:, 0], init, atol=1e-9)
        np.testing.assert_allclose(energy[:, -1], final, atol=1e-9)

        self.assertTrue(np.all(energy >= -1e-9))
        self.assertTrue(np.all(energy <= emax.reshape(-1, 1) + 1e-9))

    def test_prox_cost_is_on_discharge(self):
        """A discharge cost must reduce discharge, not charge."""
        free = heterogeneous_battery()
        free.linear_cost = free.linear_cost * 0.0

        costly = heterogeneous_battery()
        costly.linear_cost = np.full_like(np.asarray(costly.linear_cost), 5.0)

        rng = np.random.default_rng(3)
        z = rng.uniform(-1.0, 1.0, (free.num_devices, self.T)) * np.asarray(
            free.power_capacity
        ).reshape(-1, 1)

        _, state_free = run_prox(free, z, self.rho, inner_iterations=2000)
        _, state_costly = run_prox(costly, z, self.rho, inner_iterations=2000)

        self.assertLess(
            float(np.asarray(state_costly.discharge).sum()),
            float(np.asarray(state_free.discharge).sum()) - 1e-3,
        )

    def test_prox_matches_cvxpy_with_quadratic_cost(self):
        """S4: the quadratic cost belongs on the discharge block, with a factor 2."""
        device = heterogeneous_battery()
        device.quadratic_cost = np.array([0.5, 0.0, 2.0, 0.1, 1.0]).reshape(-1, 1)

        p_exact, _, _, _ = exact_prox(device, self.z, self.rho)
        p_admm, _ = run_prox(device, self.z, self.rho, inner_iterations=2000)

        tol = 1e-3 * float(np.max(device.power_capacity))
        self.assertLess(float(np.max(np.abs(p_admm - p_exact))), tol)

    def test_prox_uses_every_units_parameters(self):
        """Making unit 3 lossless must not change unit 0's prox, and vice versa."""
        base = heterogeneous_battery()
        tweaked = heterogeneous_battery()
        eff = np.asarray(tweaked.discharge_efficiency).copy().reshape(-1)
        eff[3] = 0.5
        tweaked.discharge_efficiency = eff.reshape(-1, 1)

        p_base, _ = run_prox(base, self.z, self.rho, inner_iterations=2000)
        p_tweaked, _ = run_prox(tweaked, self.z, self.rho, inner_iterations=2000)

        self.assertGreater(float(np.max(np.abs(p_base[3] - p_tweaked[3]))), 1e-3)
        np.testing.assert_allclose(p_base[0], p_tweaked[0], atol=1e-6)


def small_system():
    """3 buses, 2 generators, 1 load, 1 line, 2 heterogeneous batteries, T = 12."""
    T = 12
    net = zap.PowerNetwork(3)

    # A load shape the cheap generator alone cannot serve in the peak hours, so
    # storage arbitrage (and therefore its round-trip efficiency) is priced.
    load_profile = np.array(
        [20.0, 20.0, 20.0, 25.0, 40.0, 70.0, 100.0, 110.0, 95.0, 60.0, 30.0, 20.0]
    )

    generators = zap.Generator(
        num_nodes=3,
        name=np.array(["g0", "g1"]),
        terminal=np.array([0, 1]),
        nominal_capacity=np.array([120.0, 55.0]),
        dynamic_capacity=np.vstack([np.ones(T), np.ones(T)]),
        linear_cost=np.array([180.0, 5.0]),
        emission_rates=np.array([0.4, 0.0]),
    )
    load = zap.Load(
        num_nodes=3,
        name=np.array(["l0"]),
        terminal=np.array([2]),
        load=load_profile.reshape(1, T),
        linear_cost=np.array([500.0]),
    )
    line1 = zap.DCLine(
        num_nodes=3,
        name=np.array(["ln0", "ln1"]),
        source_terminal=np.array([0, 1]),
        sink_terminal=np.array([2, 2]),
        capacity=np.array([200.0, 200.0]),
        linear_cost=np.array([0.0, 0.0]),
    )
    battery = StorageUnit(
        num_nodes=3,
        name=np.array(["b0", "b1"]),
        terminal=np.array([2, 1]),
        power_capacity=np.array([30.0, 15.0]),
        duration=np.array([4.0, 8.0]),
        charge_efficiency=np.array([0.90, 0.80]),
        discharge_efficiency=np.array([0.70, 1.00]),
        linear_cost=np.array([0.5, 2.0]),
    )
    return net, [generators, load, line1, battery], T


class TestADMMvsLP(unittest.TestCase):
    def test_admm_matches_lp_small_system(self):
        net, devices, T = small_system()

        lp = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS, add_ground=False)

        torch_devices = [d.torchify(machine="cpu", dtype=torch.float64) for d in devices]
        solver = ADMMSolver(
            machine="cpu",
            dtype=torch.float64,
            num_iterations=20000,
            rho_power=1.0,
            adaptive_rho=False,
            battery_inner_iterations=1000,
            battery_inner_over_relaxation=1.8,
            atol=1e-10,
            rtol=1e-10,
            verbose=0,
        )
        state, _ = solver.solve(net, torch_devices, T)

        admm_cost = float(state.objective)
        lp_cost = float(lp.problem.value)
        self.assertLess(abs(admm_cost - lp_cost) / abs(lp_cost), 1e-3)

        imbalance = float(torch.max(torch.abs(state.num_terminals * state.avg_power)).item())
        peak_load = float(np.max(np.asarray(devices[1].load)))
        self.assertLess(imbalance, 1e-4 * peak_load)


if __name__ == "__main__":
    unittest.main()
