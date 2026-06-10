"""
Claim:
LineUtilizationObjective evaluates line stress as |flow| divided by the effective
thermal limit max_power * nominal_capacity, including planning-parameter overrides.

Plausible wrong implementations:
- Use nominal_capacity alone and ignore max_power/capacity.
- Use the wrong line terminal or flow sign.
- Ignore parameterized nominal_capacity values supplied by the planning layer.
- Aggregate the wrong level before applying the metric.
"""

import numpy as np
import pytest

import zap
from zap.network import DispatchOutcome
from zap.planning import LineUtilizationObjective


def _line_outcome(flow_terminal_1):
    flow_terminal_1 = np.asarray(flow_terminal_1, dtype=float)
    return DispatchOutcome(
        phase_duals=[None],
        local_equality_duals=[None],
        local_inequality_duals=[None],
        local_variables=[None],
        power=[[ -flow_terminal_1, flow_terminal_1]],
        angle=[None],
        prices=np.zeros((3, flow_terminal_1.shape[1])),
        global_angle=np.zeros((3, flow_terminal_1.shape[1])),
    )


def test_line_utilization_uses_flow_over_effective_limit_with_parameter_override():
    net = zap.PowerNetwork(num_nodes=3)
    line = zap.ACLine(
        num_nodes=3,
        source_terminal=np.array([0, 1]),
        sink_terminal=np.array([1, 2]),
        capacity=np.array([2.0, 4.0]),
        susceptance=np.ones(2),
        nominal_capacity=np.array([10.0, 5.0]),
    )
    objective = LineUtilizationObjective(net, [line], metric="quadratic", line_device_idx=[0])
    outcome = _line_outcome(np.array([[10.0, -20.0], [5.0, 15.0]]))

    u = objective.utilization(
        outcome,
        parameters=[{"nominal_capacity": np.array([5.0, 10.0])}],
        la=np,
    )

    expected = np.array([[1.0, 2.0], [0.125, 0.375]])
    np.testing.assert_allclose(u, expected, atol=1e-12)
    assert objective.forward(
        outcome,
        parameters=[{"nominal_capacity": np.array([5.0, 10.0])}],
        la=np,
    ) == pytest.approx(float(np.sum(expected**2)))


def test_line_utilization_threshold_metric_applies_per_line_time_before_sum():
    net = zap.PowerNetwork(num_nodes=2)
    line = zap.DCLine(
        num_nodes=2,
        source_terminal=np.array([0]),
        sink_terminal=np.array([1]),
        capacity=np.array([10.0]),
        nominal_capacity=np.array([1.0]),
    )
    objective = LineUtilizationObjective(
        net,
        [line],
        metric="threshold",
        tau=0.5,
        p=2.0,
        line_device_idx=[0],
    )
    outcome = _line_outcome(np.array([[2.0, 5.0, 8.0]]))

    assert objective.forward(outcome, la=np) == pytest.approx((0.8 - 0.5) ** 2)
