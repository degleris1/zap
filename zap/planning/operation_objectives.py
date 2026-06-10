import functools
import math

import numpy as np
import torch

from zap.devices.abstract import AbstractDevice
from zap.network import DispatchOutcome, PowerNetwork


def _np_logsumexp(x: np.ndarray, axis=None):
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    y = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    if axis is None:
        return y.reshape(())
    return np.squeeze(y, axis=axis)


def _logsumexp(x, *, axis=None, la=np):
    if la is torch:
        return torch.logsumexp(x, dim=axis)
    return _np_logsumexp(x, axis=axis)


def _softmax_max(x, *, alpha: float, axis=None, la=np):
    # Smooth approximation to max(x) using (1/alpha) logsumexp(alpha * x)
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    return _logsumexp(alpha * x, axis=axis, la=la) / alpha


def _relu(x, *, la=np):
    if la is torch:
        return torch.clamp_min(x, 0.0)
    return np.maximum(x, 0.0)


class AbstractOperationObjective:
    """Abstract implementation of operation objectives."""

    def __call__(self, y: DispatchOutcome, parameters=None, la=None):
        return self.forward(y, parameters=parameters, la=la)

    def forward(self, y: DispatchOutcome, parameters=None, la=None):
        raise NotImplementedError

    @property
    def is_convex(self):
        return False

    @property
    def is_linear(self):
        return False

    def __add__(self, other_objective):
        return MultiObjective([self, other_objective], [1.0, 1.0])

    def __mul__(self, weight):
        return MultiObjective([self], [weight])

    def __rmul__(self, weight):
        return self.__mul__(weight)


class MultiObjective(AbstractOperationObjective):
    """Weighted combination of multiple objectives."""

    def __init__(self, objectives: list[AbstractOperationObjective], weights: list[float]):
        self.objectives = objectives
        self.weights = weights

        # Merge multi-objectives
        new_objectives = []
        new_weights = []
        for obj, w in zip(objectives, weights):
            if isinstance(obj, MultiObjective):
                new_objectives.extend(obj.objectives)
                new_weights.extend([w * w_ for w_ in obj.weights])
            else:
                new_objectives.append(obj)
                new_weights.append(w)

        self.objectives = new_objectives
        self.weights = new_weights

        # Drop zero-weight objectives
        self.objectives = [obj for obj, w in zip(self.objectives, self.weights) if w > 0]
        self.weights = [w for w in self.weights if w > 0]

        # Check weights
        assert all([w >= 0 for w in weights])
        assert len(self.objectives) == len(self.weights)

    def forward(self, y: DispatchOutcome, parameters=None, la=np):
        return sum(
            w * obj(y, parameters=parameters, la=la)
            for w, obj in zip(self.weights, self.objectives)
        )

    @functools.cached_property
    def is_convex(self):
        return all(obj.is_convex for obj in self.objectives)

    @functools.cached_property
    def is_linear(self):
        return all(obj.is_linear for obj in self.objectives)


class DispatchCostObjective(AbstractOperationObjective):
    """Cost of the dispatch outcome."""

    def __init__(self, net: PowerNetwork, devices: list[AbstractDevice]):
        self.net = net
        self.devices = devices

        if getattr(devices[0], "torched", False):
            self.torch_devices = devices
            self.torched = True
        else:
            self.torch_devices = [d.torchify(machine="cpu") for d in devices]
            self.torched = False

    def forward(self, y: DispatchOutcome, parameters=None, la=None):
        if la is None:
            la = torch if self.torched else np

        devices = self.torch_devices if la == torch else self.devices
        return self.net.operation_cost(
            devices, y.power, y.angle, y.local_variables, parameters=parameters, la=la
        )

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return False


class EmissionsObjective(AbstractOperationObjective):
    """Total emissions of the dispatch outcome."""

    def __init__(self, devices: list[AbstractDevice]):
        self.devices = devices

        if getattr(devices[0], "torched", False):
            self.torch_devices = devices
            self.torched = True
        else:
            self.torch_devices = [d.torchify(machine="cpu") for d in devices]
            self.torched = False

    def forward(self, y: DispatchOutcome, parameters=None, la=None):
        if la is None:
            la = torch if self.torched else np

        devices = self.torch_devices if la == torch else self.devices
        emissions = [
            d.get_emissions(p, **param, la=la) for p, d, param in zip(y.power, devices, parameters)
        ]

        return sum(emissions)

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return True


class LMPObjective(AbstractOperationObjective):
    """Metric of dispatch LMPs."""

    def __init__(
        self,
        net: PowerNetwork,
        devices: list[AbstractDevice],
        lmp_metric: str = "meanmax",
        lmp_beta: float = 1.0,
    ):
        self.net = net
        self.devices = devices
        self.lmp_metric = lmp_metric
        self.lmp_beta = lmp_beta

        if getattr(devices[0], "torched", False):
            self.torch_devices = devices
            self.torched = True
        else:
            self.torch_devices = [d.torchify(machine="cpu") for d in devices]
            self.torched = False

    def forward(self, y: DispatchOutcome, parameters=None, la=None):
        if la is None:
            la = torch if self.torched else np

        devices = self.torch_devices if la == torch else self.devices
        lmps = y.prices
        if self.lmp_metric == "l2":
            return la.mean(lmps**2)
        elif self.lmp_metric == "l1":
            return la.sum(la.abs(lmps))
        elif self.lmp_metric == "max":
            return la.max(lmps)
        elif self.lmp_metric == "cvar":
            alpha = 0.95
            if la is torch:
                sorted_x, _ = torch.sort(lmps)  # ascending
                n = sorted_x.numel()
            else:
                sorted_x = np.sort(lmps)
                n = sorted_x.size

            # CVaR_alpha = mean of worst (1-alpha) tail
            k = int(math.floor(alpha * n))
            k = min(max(k, 0), n - 1)  # clamp to valid
            return sorted_x[k:].mean()
        elif self.lmp_metric == "meanmax":
            if la == torch:
                return lmps.max(dim=1).values.mean()
            else:
                return np.mean(np.max(lmps, axis=1))
        elif self.lmp_metric == "summax":
            if la == torch:
                return lmps.max(dim=1).values.sum()
            else:
                return np.sum(np.max(lmps, axis=1))
        elif self.lmp_metric == "meantopk":
            k = int(getattr(self, "topk", 5))
            if la == torch:
                return torch.topk(lmps, k, dim=1).values.mean()
            else:
                # sort along axis=1 and take last k
                return np.sort(lmps, axis=1)[:, -k:].mean()
        elif self.lmp_metric == "sumtopk":
            k = int(getattr(self, "topk", 5))
            if la == torch:
                return torch.topk(lmps, k, dim=1).values.sum()
            else:
                return np.sort(lmps, axis=1)[:, -k:].sum()
        elif self.lmp_metric == "meanpctl":
            q = float(getattr(self, "pctl", 0.95))
            if la == torch:
                # torch.quantile exists in recent versions; fallback: topk approximation below if needed
                return torch.quantile(lmps, q, dim=1).mean()
            else:
                return np.quantile(lmps, q, axis=1).mean()

        elif self.lmp_metric == "sumpctl":
            q = float(getattr(self, "pctl", 0.95))
            if la == torch:
                return torch.quantile(lmps, q, dim=1).sum()
            else:
                return np.quantile(lmps, q, axis=1).sum()

        elif self.lmp_metric == "meansmoothmax":
            alpha = float(getattr(self, "smooth_alpha", 20.0))
            if la == torch:
                sm = torch.logsumexp(alpha * lmps, dim=1) / alpha  # [N]
                return self.lmp_beta * sm.mean()
            else:
                x = alpha * lmps
                m = np.max(x, axis=1, keepdims=True)
                sm = (np.log(np.sum(np.exp(x - m), axis=1)) + m.squeeze(1)) / alpha  # [N]
                return self.lmp_beta * sm.mean()

        elif self.lmp_metric == "sumsmoothmax":
            alpha = float(getattr(self, "smooth_alpha", 20.0))
            if la == torch:
                sm = torch.logsumexp(alpha * lmps, dim=1) / alpha
                return self.lmp_beta * sm.sum()
            else:
                x = alpha * lmps
                m = np.max(x, axis=1, keepdims=True)
                sm = (np.log(np.sum(np.exp(x - m), axis=1)) + m.squeeze(1)) / alpha
                return self.lmp_beta * sm.sum()

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return False


class SCOPFLMPObjective(AbstractOperationObjective):
    """
    Security-Constrained OPF LMP objective.

    LMP objective that is contingency-aware. Supports both 2D prices
    (nodes, time) and 3D prices (nodes, time, scenarios), aggregating
    across scenarios when present.

    Args:
        net: PowerNetwork instance
        devices: List of devices
        lmp_metric: Metric to use ('meanmax', 'l2', 'meansmoothmax', etc.)
        lmp_beta: Scaling factor for objective
        aggregation: How to aggregate if scenario-specific prices available
    """

    def __init__(
        self,
        net: PowerNetwork,
        devices: list[AbstractDevice],
        lmp_metric: str = "sumsmoothmax",
        lmp_beta: float = 1.0,
        aggregation: str = "mean",
        node_idx: np.ndarray | list[int] | None = None,
    ):
        self.net = net
        self.devices = devices
        self.lmp_metric = lmp_metric
        self.lmp_beta = lmp_beta
        self.aggregation = aggregation
        self.node_idx = None if node_idx is None else np.asarray(node_idx, dtype=int)

        if getattr(devices[0], "torched", False):
            self.torch_devices = devices
            self.torched = True
        else:
            self.torch_devices = [d.torchify(machine="cpu") for d in devices]
            self.torched = False

    def forward(self, y: DispatchOutcome, parameters=None, la=None):
        if la is None:
            la = torch if self.torched else np

        devices = self.torch_devices if la == torch else self.devices

        lmps = y.prices

        # INVESTMENT_NODE_CANDS = [
        #     32,
        #     82,
        #     50,
        #     18,
        #     15,
        #     22,
        #     43,
        #     14,
        #     23,
        #     20,
        # ]
        # lmps = lmps[INVESTMENT_NODE_CANDS, :, :]
        if self.node_idx is not None:
            if la == torch:
                idx = torch.as_tensor(self.node_idx, device=lmps.device)
                lmps = lmps.index_select(0, idx)
            else:
                lmps = lmps[self.node_idx, :]

        def metric(lmp_2d):
            if self.lmp_metric == "l2":
                return la.mean(lmp_2d**2)
            if self.lmp_metric == "l1":
                return la.sum(la.abs(lmp_2d))
            if self.lmp_metric == "meanmax":
                if la == torch:
                    return lmp_2d.max(dim=1).values.mean()
                return np.mean(np.max(lmp_2d, axis=1))
            if self.lmp_metric == "meantopk":
                k = int(getattr(self, "topk", 5))
                if la == torch:
                    return torch.topk(lmp_2d, k, dim=1).values.mean()
                return np.sort(lmp_2d, axis=1)[:, -k:].mean()
            if self.lmp_metric == "meanpctl":
                q = float(getattr(self, "pctl", 0.95))
                if la == torch:
                    return torch.quantile(lmp_2d, q, dim=1).mean()
                return np.quantile(lmp_2d, q, axis=1).mean()
            if self.lmp_metric == "cvar":
                alpha = float(getattr(self, "cvar_alpha", 0.95))
                if la == torch:
                    sorted_x, _ = torch.sort(lmp_2d, dim=1)  # ascending
                    T = sorted_x.shape[1]
                    k0 = int(math.floor(alpha * T))
                    k0 = min(max(k0, 0), T - 1)
                    return sorted_x[:, k0:].mean()
                else:
                    sorted_x = np.sort(lmp_2d, axis=1)
                    T = sorted_x.shape[1]
                    k0 = int(math.floor(alpha * T))
                    k0 = min(max(k0, 0), T - 1)
                    return sorted_x[:, k0:].mean()
            if self.lmp_metric == "meansmoothmax":
                alpha = float(getattr(self, "smooth_alpha", 20.0))
                if la == torch:
                    sm = torch.logsumexp(alpha * lmp_2d, dim=1) / alpha
                    return self.lmp_beta * sm.mean()
                x = alpha * lmp_2d
                m = np.max(x, axis=1, keepdims=True)
                sm = (np.log(np.sum(np.exp(x - m), axis=1)) + m.squeeze(1)) / alpha
                return self.lmp_beta * sm.mean()
            if self.lmp_metric == "sumsmoothmax":
                alpha = float(getattr(self, "smooth_alpha", 20.0))
                if la == torch:
                    sm = torch.logsumexp(alpha * lmp_2d, dim=1) / alpha
                    return self.lmp_beta * sm.sum()
                else:
                    x = alpha * lmp_2d
                    m = np.max(x, axis=1, keepdims=True)
                    sm = (np.log(np.sum(np.exp(x - m), axis=1)) + m.squeeze(1)) / alpha
                    return self.lmp_beta * sm.sum()

        if lmps.ndim == 2:
            return metric(lmps)

        if lmps.ndim != 3:
            raise ValueError(f"Unexpected LMP shape: {lmps.shape}")

        num_scenarios = lmps.shape[2]
        per_scenario = [metric(lmps[:, :, s]) for s in range(num_scenarios)]
        if la == torch:
            per_scenario = torch.stack(per_scenario)
        else:
            per_scenario = np.array(per_scenario)

        if self.aggregation == "mean":
            return per_scenario.mean()
        if self.aggregation == "sum":
            return per_scenario.sum()
        if self.aggregation == "max":
            return per_scenario.max() if la == torch else np.max(per_scenario)
        raise ValueError(f"Unknown aggregation: {self.aggregation}")

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return False


# ===
# Line congestion / utilization objectives
# ===


def _is_scenario_list(x) -> bool:
    # Scenario layout: [scenario][terminal] (nested list-of-lists).
    # Base-case layout: [terminal] (list of tensors/arrays).
    return isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple))


class LineUtilizationObjective(AbstractOperationObjective):
    """
    Penalize approaching transmission limits using smooth, convex penalties on utilization.

    This objective is designed to create gradients even when no line constraints bind.

    Utilization is computed as u = |f| / f_max, where f is the line flow (terminal-1 power)
    and f_max is the line capacity (max_power * nominal_capacity).

    Supported metrics:
      - 'quadratic': sum_{l,t} u^2
      - 'smoothmax_time': sum_l (1/alpha) log sum_t exp(alpha * u_{l,t})
      - 'threshold': sum_{l,t} relu(u - tau)^p

    If scenario-specific flows are present (shape L x T x S), the metric is computed
    per-scenario and aggregated via scenario_aggregation ('mean'|'sum'|'max').
    """

    def __init__(
        self,
        net: PowerNetwork,
        devices: list[AbstractDevice],
        *,
        metric: str = "quadratic",
        beta: float = 1.0,
        alpha: float = 20.0,
        tau: float = 0.8,
        p: float = 2.0,
        scenario_aggregation: str = "max",
        include_slack_in_limit: bool = False,
        line_device_idx: list[int] | np.ndarray | None = None,
        eps: float = 1e-9,
    ):
        self.net = net
        self.devices = devices

        self.metric = str(metric)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.p = float(p)
        self.scenario_aggregation = str(scenario_aggregation)
        self.include_slack_in_limit = bool(include_slack_in_limit)
        self.line_device_idx = None if line_device_idx is None else np.asarray(line_device_idx)
        self.eps = float(eps)

        if getattr(devices[0], "torched", False):
            self.torch_devices = devices
            self.torched = True
        else:
            self.torch_devices = [d.torchify(machine="cpu") for d in devices]
            self.torched = False

    def _line_device_indices(self, devices):
        if self.line_device_idx is not None:
            return [int(i) for i in self.line_device_idx.ravel().tolist()]

        # Only include primal line devices (exclude dual formulations).
        from zap.devices.transporter.dc_line import PowerLine

        idx = []
        for i, d in enumerate(devices):
            if isinstance(d, PowerLine) and ".dual." not in getattr(d, "__module__", ""):
                idx.append(i)
        return idx

    def _extract_line_flows_and_limits(self, y: DispatchOutcome, parameters, la=np):
        devices = self.torch_devices if la is torch else self.devices
        line_dev_idx = self._line_device_indices(devices)
        if len(line_dev_idx) == 0:
            raise ValueError("No line devices found for LineUtilizationObjective.")

        # Determine number of scenarios S across included line devices.
        # We support two SCOPF layouts:
        #  - list-of-scenarios: power[i][s][terminal] -> (n_lines, T)
        #  - tensor scenario axis: power[i][terminal] -> (n_lines, T, S)
        scenario_count_by_dev: dict[int, int] = {}
        for i in line_dev_idx:
            power_i = y.power[i]
            if _is_scenario_list(power_i):
                scenario_count_by_dev[i] = len(power_i)
            else:
                f = power_i[1]
                scenario_count_by_dev[i] = int(f.shape[2]) if getattr(f, "ndim", 0) == 3 else 1
        S = int(max(scenario_count_by_dev.values()))

        flows_per_s = [[] for _ in range(S)]
        limits_list = []

        for i in line_dev_idx:
            d = devices[i]
            power_i = y.power[i]

            # Flow is terminal-1 power (terminal-0 is the negative copy for transporters).
            def get_flow_s(s: int):
                if _is_scenario_list(power_i):
                    return power_i[s][1]

                f = power_i[1]
                if getattr(f, "ndim", 0) == 3:
                    return f[:, :, s]
                return f

            # Effective limit: max_power * nominal_capacity (+ slack optionally).
            param_i = None if parameters is None else parameters[i]
            nominal = None if param_i is None else param_i.get("nominal_capacity", None)
            nominal = d.parameterize(nominal_capacity=nominal, la=la)

            fmax = d.max_power
            if la is torch and not torch.is_tensor(fmax):
                fmax = torch.as_tensor(fmax, device=nominal.device, dtype=nominal.dtype)
            if la is not torch:
                fmax = np.asarray(fmax)

            # Shape normalize fmax to (n_lines, T) or (n_lines, 1)
            if fmax.ndim == 1:
                fmax = fmax[:, None]
            elif fmax.ndim != 2:
                raise ValueError(f"Unexpected max_power shape for {type(d)}: {fmax.shape}")

            if nominal.ndim == 1:
                nominal_ = nominal[:, None]
            elif nominal.ndim == 2:
                nominal_ = nominal
            else:
                raise ValueError(f"Unexpected nominal_capacity shape for {type(d)}: {nominal.shape}")

            limit = fmax * nominal_

            # Ensure per-device limit time dimension matches the flow time dimension so we can concatenate.
            flow0 = get_flow_s(0)
            if getattr(flow0, "ndim", 0) != 2:
                raise ValueError(f"Unexpected line flow shape for {type(d)}: {getattr(flow0, 'shape', None)}")
            T = flow0.shape[1]
            if limit.shape[1] != T:
                if limit.shape[1] == 1:
                    if la is torch:
                        limit = limit.expand(-1, T)
                    else:
                        limit = np.repeat(limit, T, axis=1)
                else:
                    raise ValueError(
                        f"Line limit time dimension {limit.shape[1]} does not match flow time dimension {T} "
                        f"for {type(d)}."
                    )

            if self.include_slack_in_limit:
                slack = getattr(d, "slack", 0.0)
                if la is torch and not torch.is_tensor(slack):
                    slack = torch.as_tensor(slack, device=limit.device, dtype=limit.dtype)
                if la is not torch:
                    slack = np.asarray(slack)
                if getattr(slack, "ndim", 0) == 1:
                    slack = slack[:, None]
                limit = limit + slack

            limits_list.append(limit)

            for s in range(S):
                # If this device is not scenario-expanded, repeat its base-case flow across scenarios.
                fs = get_flow_s(min(s, scenario_count_by_dev[i] - 1))
                flows_per_s[s].append(fs)

        if la is torch:
            flows = [torch.cat(xs, dim=0) for xs in flows_per_s]
            limits = torch.cat(limits_list, dim=0)
        else:
            flows = [np.concatenate(xs, axis=0) for xs in flows_per_s]
            limits = np.concatenate(limits_list, axis=0)

        # Broadcast limits across time if needed.
        # flows[s]: (L, T), limits: (L, 1) or (L, T)
        if limits.shape[1] == 1 and flows[0].shape[1] != 1:
            if la is torch:
                limits = limits.expand(-1, flows[0].shape[1])
            else:
                limits = np.repeat(limits, flows[0].shape[1], axis=1)

        if S == 1:
            return flows[0], limits

        if la is torch:
            return torch.stack(flows, dim=2), limits
        return np.stack(flows, axis=2), limits

    def utilization(self, y: DispatchOutcome, parameters=None, la=None):
        if la is None:
            la = torch if self.torched else np

        flows, limits = self._extract_line_flows_and_limits(y, parameters, la=la)
        denom = limits + (limits.new_tensor(self.eps) if la is torch else self.eps)
        # If flows include a scenario axis (L,T,S), broadcast denom as (L,T,1).
        if getattr(flows, "ndim", None) == 3:
            denom = denom.unsqueeze(2) if la is torch else denom[:, :, None]
        return la.abs(flows) / denom

    def _metric_value(self, u, la=np):
        # u: (L, T) or (L, T, S)
        def metric_2d(u2):
            if self.metric == "quadratic":
                return la.sum(u2**2)
            if self.metric == "smoothmax_time":
                return la.sum(_softmax_max(u2, alpha=self.alpha, axis=1, la=la))
            if self.metric == "threshold":
                return la.sum(_relu(u2 - self.tau, la=la) ** self.p)
            raise ValueError(f"Unknown utilization metric: {self.metric}")

        if getattr(u, "ndim", None) == 2:
            return metric_2d(u)

        if getattr(u, "ndim", None) != 3:
            raise ValueError(f"Unexpected utilization shape: {getattr(u, 'shape', None)}")

        S = u.shape[2]
        per_s = [metric_2d(u[:, :, s]) for s in range(S)]
        if la is torch:
            per_s = torch.stack(per_s)
        else:
            per_s = np.asarray(per_s)

        if self.scenario_aggregation == "mean":
            return per_s.mean()
        if self.scenario_aggregation == "sum":
            return per_s.sum()
        if self.scenario_aggregation == "max":
            return per_s.max() if la is torch else np.max(per_s)
        raise ValueError(f"Unknown scenario_aggregation: {self.scenario_aggregation}")

    def forward(self, y: DispatchOutcome, parameters=None, la=None):
        if la is None:
            la = torch if self.torched else np

        u = self.utilization(y, parameters=parameters, la=la)
        return self.beta * self._metric_value(u, la=la)

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return False


class LineShadowPriceObjective(AbstractOperationObjective):
    """
    Congestion proxy based on the shadow prices of line flow constraints.

    For each line device, we sum the duals on its lower/upper flow limits
    (mu^- + mu^+), then aggregate over time and lines.
    """

    def __init__(
        self,
        net: PowerNetwork,
        devices: list[AbstractDevice],
        *,
        beta: float = 1.0,
        metric: str = "sum",
        alpha: float = 20.0,
        line_device_idx: list[int] | np.ndarray | None = None,
    ):
        self.net = net
        self.devices = devices
        self.beta = float(beta)
        self.metric = str(metric)
        self.alpha = float(alpha)
        self.line_device_idx = None if line_device_idx is None else np.asarray(line_device_idx)

        if getattr(devices[0], "torched", False):
            self.torch_devices = devices
            self.torched = True
        else:
            self.torch_devices = [d.torchify(machine="cpu") for d in devices]
            self.torched = False

    def _line_device_indices(self, devices):
        if self.line_device_idx is not None:
            return [int(i) for i in self.line_device_idx.ravel().tolist()]

        from zap.devices.transporter.dc_line import PowerLine

        idx = []
        for i, d in enumerate(devices):
            if isinstance(d, PowerLine) and ".dual." not in getattr(d, "__module__", ""):
                idx.append(i)
        return idx

    def forward(self, y: DispatchOutcome, parameters=None, la=None):
        if la is None:
            la = torch if self.torched else np

        if y.local_inequality_duals is None:
            raise ValueError(
                "LineShadowPriceObjective requires `DispatchOutcome.local_inequality_duals` (line flow-limit duals), "
                "but this outcome does not include them (e.g. ADMM outcomes set them to None)."
            )

        devices = self.torch_devices if la is torch else self.devices
        line_dev_idx = self._line_device_indices(devices)
        if len(line_dev_idx) == 0:
            raise ValueError("No line devices found for LineShadowPriceObjective.")

        mu_list = []
        for i in line_dev_idx:
            lamb = y.local_inequality_duals[i]
            if lamb is None or len(lamb) < 2:
                continue
            mu = lamb[0] + lamb[1]
            mu_list.append(mu)

        if len(mu_list) == 0:
            return (torch.tensor(0.0) if la is torch else 0.0) * self.beta

        if la is torch:
            mu_all = torch.cat(mu_list, dim=0)  # (L, T)
        else:
            mu_all = np.concatenate(mu_list, axis=0)

        if self.metric == "sum":
            val = mu_all.sum() if la is torch else float(np.sum(mu_all))
        elif self.metric == "smoothmax_time":
            val = la.sum(_softmax_max(mu_all, alpha=self.alpha, axis=1, la=la))
        else:
            raise ValueError(f"Unknown shadow price metric: {self.metric}")

        return self.beta * val

    def shadow_prices(self, y: DispatchOutcome, *, la=None):
        if la is None:
            la = torch if self.torched else np

        if y.local_inequality_duals is None:
            raise ValueError(
                "LineShadowPriceObjective.shadow_prices requires `DispatchOutcome.local_inequality_duals`, "
                "but this outcome does not include them."
            )

        devices = self.torch_devices if la is torch else self.devices
        line_dev_idx = self._line_device_indices(devices)

        mu_list = []
        for i in line_dev_idx:
            lamb = y.local_inequality_duals[i]
            if lamb is None or len(lamb) < 2:
                continue
            mu_list.append(lamb[0] + lamb[1])

        if len(mu_list) == 0:
            return torch.zeros(()) if la is torch else np.zeros(())
        return torch.cat(mu_list, dim=0) if la is torch else np.concatenate(mu_list, axis=0)

    @property
    def is_convex(self):
        return True

    @property
    def is_linear(self):
        return True


def line_utilization_metrics(u, *, eps: float = 1e-6, la=None):
    """
    Evaluation-only summary metrics for line utilization.

    Args:
        u: utilization array (L,T) or (L,T,S), values in [0, +inf)
    Returns:
        Dict with max/mean utilization and line counts above thresholds.
    """
    if la is None:
        la = torch if torch.is_tensor(u) else np

    if getattr(u, "ndim", None) == 3:
        # Worst-case over scenarios for reporting
        if la is torch:
            u_wc = u.max(dim=2).values
        else:
            u_wc = np.max(u, axis=2)
    else:
        u_wc = u

    if la is torch:
        max_u = u_wc.max()
        mean_u = u_wc.mean()
        line_max = u_wc.max(dim=1).values
        n90 = (line_max > 0.9).sum()
        n95 = (line_max > 0.95).sum()
        return {
            "max_line_utilization": max_u,
            "mean_line_utilization": mean_u,
            "lines_above_90pct": n90,
            "lines_above_95pct": n95,
        }

    u_wc = np.asarray(u_wc)
    return {
        "max_line_utilization": float(np.max(u_wc)),
        "mean_line_utilization": float(np.mean(u_wc)),
        "lines_above_90pct": int(np.sum(np.max(u_wc, axis=1) > 0.9 + eps)),
        "lines_above_95pct": int(np.sum(np.max(u_wc, axis=1) > 0.95 + eps)),
    }


def line_shadow_price_metrics(mu, *, eps: float = 1e-6, la=None):
    """
    Evaluation-only summary metrics for line shadow prices.

    Args:
        mu: shadow prices array (L,T), mu >= 0.
    Returns:
        Dict with total shadow price and count of binding lines.
    """
    if la is None:
        la = torch if torch.is_tensor(mu) else np

    if la is torch:
        total = mu.sum()
        binding = (mu.max(dim=1).values > eps).sum()
        return {"total_shadow_price": total, "binding_lines": binding}

    mu = np.asarray(mu)
    return {
        "total_shadow_price": float(np.sum(mu)),
        "binding_lines": int(np.sum(np.max(mu, axis=1) > eps)),
    }

# class DCTailPriceObjective(AbstractOperationObjective):
#     """
#     Penalize high LMPs *at the DC terminals* using a tail metric over time.

#     This makes "spreading" arise naturally (no caps) by making concentrated
#     capacity at a high-tail-price location expensive.

#     If weight_by_capacity:
#         sum_i dc_cap[i] * tail_t(price[terminal_i, :])
#     Else:
#         sum_i tail_t(price[terminal_i, :])

#     tail_t is controlled by lmp_metric (e.g. 'cvar', 'meantopk', 'meansmoothmax', 'meanmax').
#     """

#     def __init__(
#         self,
#         devices: list[AbstractDevice],
#         dc_device_idx: int,
#         lmp_metric: str = "cvar",
#         weight_by_capacity: bool = True,
#         aggregation_across_dcs: str = "sum",  # 'sum' | 'max' | 'mean'
#         cvar_alpha: float = 0.95,
#         topk: int = 5,
#         smooth_alpha: float = 20.0,
#     ):
#         self.devices = devices
#         self.dc_device_idx = int(dc_device_idx)
#         self.lmp_metric = lmp_metric
#         self.weight_by_capacity = bool(weight_by_capacity)
#         self.aggregation_across_dcs = aggregation_across_dcs
#         self.cvar_alpha = float(cvar_alpha)
#         self.topk = int(topk)
#         self.smooth_alpha = float(smooth_alpha)

#         if getattr(devices[0], "torched", False):
#             self.torch_devices = devices
#             self.torched = True
#         else:
#             self.torch_devices = [d.torchify(machine="cpu") for d in devices]
#             self.torched = False

#     def forward(self, y: DispatchOutcome, parameters=None, la=None):
#         if la is None:
#             la = torch if self.torched else np

#         devices = self.torch_devices if la == torch else self.devices
#         lmps = y.prices

#         dc_dev = devices[self.dc_device_idx]
#         terminals = dc_dev.terminals
#         if la == torch:
#             if not torch.is_tensor(terminals):
#                 terminals = torch.as_tensor(terminals, device=lmps.device)
#             else:
#                 terminals = terminals.to(device=lmps.device)
#         else:
#             terminals = np.asarray(terminals, dtype=int)

#         def tail_over_time(pr_2d):
#             # pr_2d: (n_dc, T)
#             if self.lmp_metric == "meanmax":
#                 if la == torch:
#                     return pr_2d.max(dim=1).values
#                 return np.max(pr_2d, axis=1)

#             if self.lmp_metric == "meantopk":
#                 k = max(1, int(self.topk))
#                 if la == torch:
#                     return torch.topk(pr_2d, k, dim=1).values.mean(dim=1)
#                 return np.sort(pr_2d, axis=1)[:, -k:].mean(axis=1)

#             if self.lmp_metric == "cvar":
#                 alpha = float(self.cvar_alpha)
#                 if la == torch:
#                     sorted_x, _ = torch.sort(pr_2d, dim=1)  # ascending
#                     T = sorted_x.shape[1]
#                     k0 = int(math.floor(alpha * T))
#                     k0 = min(max(k0, 0), T - 1)
#                     return sorted_x[:, k0:].mean(dim=1)
#                 sorted_x = np.sort(pr_2d, axis=1)
#                 T = sorted_x.shape[1]
#                 k0 = int(math.floor(alpha * T))
#                 k0 = min(max(k0, 0), T - 1)
#                 return sorted_x[:, k0:].mean(axis=1)

#             if self.lmp_metric == "meansmoothmax":
#                 alpha = float(self.smooth_alpha)
#                 if la == torch:
#                     return torch.logsumexp(alpha * pr_2d, dim=1) / alpha
#                 x = alpha * pr_2d
#                 m = np.max(x, axis=1, keepdims=True)
#                 return (np.log(np.sum(np.exp(x - m), axis=1)) + m.squeeze(1)) / alpha

#             raise ValueError(f"Unsupported lmp_metric for DCTailPriceObjective: {self.lmp_metric}")

#         def combine(dc_tail):
#             if self.aggregation_across_dcs == "sum":
#                 return dc_tail.sum() if la == torch else np.sum(dc_tail)
#             if self.aggregation_across_dcs == "mean":
#                 return dc_tail.mean() if la == torch else np.mean(dc_tail)
#             if self.aggregation_across_dcs == "max":
#                 return dc_tail.max() if la == torch else np.max(dc_tail)
#             raise ValueError(f"Unknown aggregation_across_dcs: {self.aggregation_across_dcs}")

#         if lmps.ndim == 2:
#             pr = lmps.index_select(0, terminals) if la == torch else lmps[terminals, :]
#             dc_tail = tail_over_time(pr)
#         elif lmps.ndim == 3:
#             num_scenarios = lmps.shape[2]
#             per_s = []
#             for s in range(num_scenarios):
#                 pr_s = lmps[:, :, s]
#                 pr_s = pr_s.index_select(0, terminals) if la == torch else pr_s[terminals, :]
#                 per_s.append(tail_over_time(pr_s))
#             if la == torch:
#                 dc_tail = torch.stack(per_s, dim=0).mean(dim=0)
#             else:
#                 dc_tail = np.mean(np.stack(per_s, axis=0), axis=0)
#         else:
#             raise ValueError(f"Unexpected prices shape: {lmps.shape}")

#         if self.weight_by_capacity:
#             dc_cap = None
#             if parameters is not None:
#                 dc_cap = parameters[self.dc_device_idx].get("nominal_capacity", None)
#             if dc_cap is None:
#                 dc_cap = getattr(dc_dev, "nominal_capacity", None)

#             if la == torch:
#                 if not torch.is_tensor(dc_cap):
#                     dc_cap = torch.as_tensor(dc_cap, device=dc_tail.device, dtype=dc_tail.dtype)
#                 dc_cap = dc_cap.reshape(-1)
#             else:
#                 dc_cap = np.asarray(dc_cap).reshape(-1)

#             return combine(dc_cap * dc_tail)

#         return combine(dc_tail)
