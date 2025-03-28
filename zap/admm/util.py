import torch
from typing import Optional

from zap.devices.abstract import AbstractDevice
from zap.util import DEFAULT_DTYPE, infer_machine


def unsqueeze_terminals_times_x(num_terminals, x):
    if x.dim() == 3:
        return num_terminals.unsqueeze(-1) * x
    else:
        return num_terminals * x


def scatter_sum(num_rows, index, source):
    # For 3d tensors (contingencies)
    if len(source.shape) == 3 and len(index.shape) == 2:
        index = index.unsqueeze(2).expand(-1, -1, source.shape[2])

    num_cols = source.shape[1:]
    result = torch.zeros((num_rows, *num_cols), device=source.device, dtype=source.dtype)

    return result.scatter_add(0, index, source)


def gather_sum(index, source):
    # For 3d tensors (contingencies)
    if len(source.shape) == 3 and len(index.shape) == 2:
        index = index.unsqueeze(2).expand(-1, -1, source.shape[2])

    result = torch.gather(source, 0, index)
    return result


def apply_incidence(device: AbstractDevice, x_list: list[torch.Tensor]):
    return apply_incidence_gpu(device, x_list)


def apply_incidence_transpose(device: AbstractDevice, x: torch.Tensor):
    return apply_incidence_transpose_gpu(device, x)


def apply_incidence_cpu(device: AbstractDevice, x_list: list[torch.Tensor]):
    return [A_dt @ x_dt for A_dt, x_dt in zip(device.incidence_matrix, x_list)]


def apply_incidence_gpu(device: AbstractDevice, x_list: list[torch.Tensor]):
    n, T = device.num_nodes, x_list[0].shape[1]
    machine = x_list[0].device
    return [
        scatter_sum(n, tau, x_dt) for tau, x_dt in zip(device.torch_terminals(T, machine), x_list)
    ]


def apply_incidence_transpose_cpu(device: AbstractDevice, x: torch.Tensor):
    return [A_dt.T @ x for A_dt in device.incidence_matrix]


def apply_incidence_transpose_gpu(device: AbstractDevice, x: torch.Tensor):
    T = x.shape[1]
    machine = x.device
    return [gather_sum(tau, x) for tau in device.torch_terminals(T, machine)]


def nested_map(f, *args, none_value=None):
    return [
        (none_value if arg_dev[0] is None else [f(*arg_dt) for arg_dt in zip(*arg_dev)])
        for arg_dev in zip(*args)
    ]


def nested_ax(x, a):
    return nested_map(lambda x: a * x, x)


def nested_a1bpa2x(x1, x2, a1, a2):
    return nested_map(lambda x, y: a1 * x + a2 * y, x1, x2)


def nested_bpax(x1, x2, a):
    return nested_map(lambda x, y: x + a * y, x1, x2)


def nested_add(x1, x2, alpha=None):
    if alpha is None:
        return nested_map(lambda x, y: x + y, x1, x2)
    else:
        return nested_map(lambda x, y: alpha * (x + y), x1, x2)


def nested_subtract(x1, x2, alpha=None):
    if alpha is None:
        return nested_map(lambda x, y: x - y, x1, x2)
    else:
        return nested_map(lambda x, y: alpha * (x - y), x1, x2)


def nested_norm(data, p=2):
    mini_norms = [
        (
            torch.tensor([0.0])
            if x_dev is None
            else torch.tensor([torch.linalg.vector_norm(x.ravel(), p) for x in x_dev])
        )
        for x_dev in data
    ]
    return torch.linalg.vector_norm(torch.concatenate(mini_norms), p)


def get_discrep(power1, power2):
    discreps = [
        (
            [0.0]
            if p_admm is None
            else [torch.linalg.norm(p1 - p2, 1) for p1, p2 in zip(p_admm, p_cvx)]
        )
        for p_admm, p_cvx in zip(power1, power2)
    ]
    return torch.sum(torch.tensor(discreps))


def get_num_terminals(net, devices, only_ac=False, machine=None, dtype=DEFAULT_DTYPE):
    if machine is None:
        machine = infer_machine()

    terminal_counts = torch.zeros(net.num_nodes, dtype=dtype, device=machine)
    for d in devices:
        if only_ac and (not d.is_ac):
            continue

        values, counts = torch.unique(d.terminals, return_counts=True)
        for t, c in zip(values, counts):
            terminal_counts[t] += c

    return torch.reshape(terminal_counts, (-1, 1))


def get_nodal_average(
    powers,
    net,
    devices,
    time_horizon,
    num_terminals=None,
    only_ac=False,
    check_connections=False,
    tol=1e-8,
    machine=None,
    dtype=DEFAULT_DTYPE,
    num_contingencies=0,
):
    if machine is None:
        machine = infer_machine()

    tol = torch.tensor(tol, device=machine)
    if num_terminals is None:
        num_terminals = get_num_terminals(
            net, devices, only_ac=only_ac, machine=machine, dtype=dtype
        )

    if check_connections:
        assert torch.all(num_terminals > 0)
    else:
        num_terminals = torch.maximum(num_terminals, tol)

    if num_contingencies > 0:
        average_x = torch.zeros(
            (net.num_nodes, time_horizon, num_contingencies + 1), device=machine, dtype=dtype
        )
    else:
        average_x = torch.zeros((net.num_nodes, time_horizon), device=machine, dtype=dtype)

    for dev, x_dev in zip(devices, powers):
        if x_dev is None:
            continue

        # Linear algebra version
        # for A_dt, x_dt in zip(dev.incidence_matrix, x_dev):
        #     average_x += A_dt @ x_dt

        # Torch version
        A_x = apply_incidence(dev, x_dev)
        for A_x_dt in A_x:
            if num_contingencies > 0 and len(A_x_dt.shape) == 2:
                A_x_dt = A_x_dt.unsqueeze(2)

            average_x += A_x_dt

    if num_contingencies > 0:
        num_terminals = num_terminals.unsqueeze(2)

    return torch.divide(average_x, num_terminals)


def get_terminal_residual(angles: list[Optional[list[torch.Tensor]]], average_angle, devices):
    AT_theta = [
        None if a_dev is None else apply_incidence_transpose(dev, average_angle)
        for a_dev, dev in zip(angles, devices)
    ]

    residuals = []
    for a_dev, AT_theta_dev in zip(angles, AT_theta):
        if a_dev is None:
            residuals += [None]

        else:
            a_dev: list[torch.Tensor]

            residuals += [
                [
                    a_dt.unsqueeze(2) - AT_theta_dt
                    if ((len(a_dt.shape) == 2) and (len(AT_theta_dt.shape) == 3))
                    else a_dt - AT_theta_dt
                    for a_dt, AT_theta_dt in zip(a_dev, AT_theta_dev)
                ]
            ]

    return residuals


def dc_average(
    x,
    net,
    devices,
    time_horizon,
    num_terminals,
    machine=None,
    dtype=DEFAULT_DTYPE,
    num_contingencies=0,
):
    return get_nodal_average(
        x,
        net,
        devices,
        time_horizon,
        num_terminals,
        only_ac=False,
        machine=machine,
        dtype=dtype,
        num_contingencies=num_contingencies,
    )


def ac_average(
    x,
    net,
    devices,
    time_horizon,
    num_ac_terminals,
    machine=None,
    dtype=DEFAULT_DTYPE,
    num_contingencies=0,
):
    return get_nodal_average(
        x,
        net,
        devices,
        time_horizon,
        num_ac_terminals,
        only_ac=True,
        check_connections=False,
        machine=machine,
        dtype=dtype,
        num_contingencies=num_contingencies,
    )
