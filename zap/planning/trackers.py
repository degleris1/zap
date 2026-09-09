import numpy as np
import torch
import time

from copy import deepcopy

LOSS = "loss"
GRAD_NORM = "grad_norm"
PROJ_GRAD_NORM = "proj_grad_norm"
PARAM = "param"
TIME = "time"
SUBOPTIMALITY = "suboptimality"
GRAD = "grad"
ADMM_STATE = "admm_state"
BATCH = "batch"
GRAD_NORM_L2 = "grad_norm_l2"


def track_loss(J, grad, state, last_state, problem):
    return J.cpu().detach().numpy()


def track_grad_norm(J, grad: dict[str, torch.Tensor], state, last_state, problem):
    """Tracks the 1-norm of the gradient."""
    return sum([torch.linalg.vector_norm(g, ord=1).item() for g in grad.values()])


def track_grad_norm_l2(J, grad: dict[str, torch.Tensor], state, last_state, problem):
    """Tracks the 2-norm of the gradient.

    This is exactly the quantity ``solvers.GradientDescent.step`` compares to
    ``clip`` (it stacks the per-parameter 2-norms and takes their 2-norm), so a
    history carrying it can report the clip fraction of every iteration.
    ``track_grad_norm`` is the **1**-norm and is not comparable to ``clip``.
    """
    norms = []
    for g in grad.values():
        if not isinstance(g, torch.Tensor):
            g = torch.as_tensor(np.asarray(g, dtype=float))
        norms.append(torch.linalg.vector_norm(g, ord=2))
    if len(norms) == 0:
        return 0.0
    return torch.linalg.vector_norm(torch.stack(norms), ord=2).item()


def track_batch(J, grad, state, last_state, problem):
    """The subproblem indices this iteration's forward pass actually saw.

    ``AbstractPlanningProblem.solve`` stamps ``problem.batch`` immediately
    before every ``forward_and_back`` call; without it the minibatch is a local
    variable and nothing downstream can say which blocks a gradient step used.
    """
    return [int(i) for i in (getattr(problem, "batch", None) or [])]


def track_proj_grad_norm(J, grad, state, last_state, problem):
    """Tracks the 1-norm of the projected gradient, which is the difference between
    the current state and the previous state."""
    la = problem.la

    if last_state is None:
        return track_grad_norm(J, grad, state, last_state, problem)

    # Compute differences between states
    diffs = {k: state[k] - last_state[k] for k in state.keys()}

    return sum([la.linalg.norm(d, ord=1) for d in diffs.values()])


def track_param(J, grad, state, last_state, problem):
    return deepcopy(state)


def track_grad(J, grad, state, last_state, problem):
    return grad


def track_time(J, grad, state, last_state, problem):
    return time.time() - problem.start_time


def suboptimality(J, grad, state, last_state, problem):
    lb = 1.0 if problem.lower_bound is None else problem.lower_bound

    return (J.cpu().detach().numpy() / lb) - 1.0


def admm_state(J, grad, state, last_state, problem):
    # Do this by checkif if the state object has a copy function
    if hasattr(problem.layer, "state"):
        return problem.layer.state.copy()
    else:
        return None


TRACKER_MAPS = {
    LOSS: track_loss,
    GRAD_NORM: track_grad_norm,
    PROJ_GRAD_NORM: track_proj_grad_norm,
    PARAM: track_param,
    TIME: track_time,
    SUBOPTIMALITY: suboptimality,
    GRAD: track_grad,
    ADMM_STATE: admm_state,
    BATCH: track_batch,
    GRAD_NORM_L2: track_grad_norm_l2,
}

DEFAULT_TRACKERS = [LOSS, GRAD_NORM, PROJ_GRAD_NORM, TIME, SUBOPTIMALITY]
