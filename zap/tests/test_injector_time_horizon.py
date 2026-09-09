"""An injector with static bounds must report a static time horizon.

``make_dynamic`` reshapes 1-D attributes to ``(N, 1)``.  ``get_time_horizon``
used to report ``1`` for that shape, so any ``Injector`` built from scalar
bounds (an export sink, a fixed-price slack) claimed a one-hour horizon.
``PowerNetwork.dispatch`` asserts ``d.time_horizon in [0, time_horizon]``, so
such a device could only ever be dispatched over exactly one hour, and a device
sliced by ``sample_time`` kept the horizon of the array it came from.
"""

import cvxpy as cp
import numpy as np
import pytest

from zap.devices.abstract import get_time_horizon
from zap.devices.injector import Generator, Injector, Load
from zap.network import PowerNetwork

T = 6


def test_get_time_horizon_treats_single_column_as_static():
    assert get_time_horizon(np.zeros((3, 1))) == 0
    assert get_time_horizon(np.zeros(3)) == 0
    assert get_time_horizon(np.zeros((3, T))) == T


def _sink(n_nodes=2):
    return Injector(
        num_nodes=n_nodes,
        name=np.array(["sink"]),
        terminal=np.array([1]),
        nominal_capacity=np.array([50.0]),
        min_power=-np.ones(1),
        max_power=np.zeros(1),
        linear_cost=np.zeros(1),
    )


def test_static_injector_reports_zero_horizon():
    assert _sink().time_horizon == 0


def test_dynamic_injector_still_reports_its_horizon():
    device = _sink()
    device.min_power = -np.ones((1, T))
    device.max_power = np.zeros((1, T))
    assert device.time_horizon == T


def test_static_generator_matches_static_injector():
    """A Generator with static ``dynamic_capacity`` is the reference behaviour."""
    generator = Generator(
        num_nodes=2,
        name=np.array(["g"]),
        terminal=np.array([0]),
        nominal_capacity=np.array([100.0]),
        dynamic_capacity=np.ones(1),
        linear_cost=np.array([10.0]),
    )
    assert generator.time_horizon == _sink().time_horizon == 0


def test_static_injector_survives_sample_time_and_dispatch():
    network = PowerNetwork(2)
    generator = Generator(
        num_nodes=2,
        name=np.array(["g"]),
        terminal=np.array([0]),
        nominal_capacity=np.array([100.0]),
        dynamic_capacity=np.ones((1, T)),
        linear_cost=np.array([10.0]),
    )
    load = Load(
        num_nodes=2,
        name=np.array(["l"]),
        terminal=np.array([0]),
        load=np.full((1, T), 20.0),
        linear_cost=np.array([1000.0]),
    )
    sink = _sink()

    devices = [generator, load, sink]
    outcome = network.dispatch(devices, solver=cp.HIGHS)
    assert outcome.problem.status == cp.OPTIMAL

    # The whole point: a sliced block still dispatches.
    half = T // 2
    block = [d.sample_time(range(0, half), T) for d in devices]
    assert block[2].time_horizon == 0
    blocked = network.dispatch(block, solver=cp.HIGHS)
    assert blocked.problem.status == cp.OPTIMAL
    assert blocked.power[2][0].shape == (1, half)


def test_one_hour_dispatch_still_works():
    """time_horizon 0 is accepted for any T, including T = 1."""
    network = PowerNetwork(2)
    generator = Generator(
        num_nodes=2,
        name=np.array(["g"]),
        terminal=np.array([0]),
        nominal_capacity=np.array([100.0]),
        dynamic_capacity=np.ones((1, 1)),
        linear_cost=np.array([10.0]),
    )
    load = Load(
        num_nodes=2,
        name=np.array(["l"]),
        terminal=np.array([0]),
        load=np.full((1, 1), 20.0),
        linear_cost=np.array([1000.0]),
    )
    outcome = network.dispatch([generator, load, _sink()], time_horizon=1, solver=cp.HIGHS)
    assert outcome.problem.status == cp.OPTIMAL
    assert outcome.power[0][0] == pytest.approx(20.0, abs=1e-6)
