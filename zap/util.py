import torch
import numpy as np
import cvxpy as cp

from numbers import Number

DEFAULT_DTYPE = torch.float32
TORCH_INTEGER_DTYPE = torch.int64
TORCH_INTEGER_TYPES = [torch.int8, torch.int16, torch.int32, torch.int64]


def infer_machine(machine=None):
    return machine
    # if machine is None:
    #     return "cuda" if torch.cuda.is_available() else "cpu"
    # else:
    #     return machine


def expand_params(params, devices):
    if params is None:
        params = [{} for _ in devices]

    assert len(params) == len(devices)
    return params


def replace_none(x, replacement):
    if x is None:
        return replacement
    else:
        return x


def grad_or_zero(x, to_numpy=False):
    if x is None:
        return None
    elif isinstance(x, list):
        return [grad_or_zero(xi, to_numpy=to_numpy) for xi in x]
    else:
        grad = torch.zeros_like(x) if x.grad is None else x.grad
        if to_numpy:
            return grad.numpy()
        else:
            return grad


def torchify(x, requires_grad=False, machine=None, dtype=torch.float64):
    if machine is None:
        machine = infer_machine()

    if isinstance(x, torch.Tensor):
        if requires_grad and (not x.requires_grad):
            x.requires_grad = True

        if x.device != machine or x.dtype != dtype:
            dtype = TORCH_INTEGER_DTYPE if dtype in TORCH_INTEGER_TYPES else dtype
            x = x.to(device=machine, dtype=dtype)

        return x

    elif x is None:
        return None

    elif isinstance(x, dict):
        return {
            k: torchify(v, requires_grad=requires_grad, machine=machine, dtype=dtype)
            for k, v in x.items()
        }

    elif isinstance(x, list):
        return [torchify(xi, requires_grad=requires_grad, machine=machine, dtype=dtype) for xi in x]

    elif isinstance(x, np.ndarray) or isinstance(x, Number):  # np array
        old_dtype = x.dtype if isinstance(x, np.ndarray) else type(x)

        if np.issubdtype(old_dtype, np.integer):
            dtype = TORCH_INTEGER_DTYPE

        return torch.tensor(x, requires_grad=requires_grad, device=machine, dtype=dtype)

    else:
        raise ValueError(f"Unknown type: {type(x)}. Cannot torchify.")


def torch_sparse(A):
    if isinstance(A, list):
        return [torch_sparse(Ai) for Ai in A]

    A = A.tocoo()
    values = A.data
    indices = np.vstack((A.row, A.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)

    return torch.sparse_coo_tensor(i, v, torch.Size(A.shape), dtype=torch.float64)


def choose_base_modeler(la):
    if la == torch:
        return torch
    elif la in [np, cp]:
        return np
    else:
        raise ValueError(f"Unknown la: {la}")


def envelope_variable(
    x, y, xmin, xmax, ymin, ymax, envelope_variables=None, envelope_constraints=None
):
    """Define a new variable `z = envelope(x * y)`.

    Here `envelope` is the convex (McCormick) envelope of the product `x * y`.
    """
    # Create product variable
    z_shape = np.maximum(x.shape, y.shape)
    z = cp.Variable(z_shape)

    # Add constraints
    #     z >= xmin * y + x * ymin - xmin * ymin,
    #     z >= xmax * y + x * ymax - xmax * ymax,
    #     z <= xmax * y + x * ymin - xmax * ymin,
    #     z <= xmin * y + x * ymax - xmin * ymax,
    constraints = [
        z >= cp.multiply(xmin, y) + cp.multiply(x, ymin) - cp.multiply(xmin, ymin),
        z >= cp.multiply(xmax, y) + cp.multiply(x, ymax) - cp.multiply(xmax, ymax),
        z <= cp.multiply(xmax, y) + cp.multiply(x, ymin) - cp.multiply(xmax, ymin),
        z <= cp.multiply(xmin, y) + cp.multiply(x, ymax) - cp.multiply(xmin, ymax),
    ]

    if envelope_variables is not None:
        envelope_variables.append(z)

    if envelope_constraints is not None:
        envelope_constraints += constraints

    return z


def use_envelope(envelope):
    if envelope is not None:
        env, lb, ub = envelope
        if len(lb) > 0:
            return True
    return False
