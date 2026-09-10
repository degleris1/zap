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
FREE_GRAD_NORM_L2 = "free_grad_norm_l2"
N_FREE = "n_free"
N_AT_LOWER = "n_at_lower"
N_AT_UPPER = "n_at_upper"
STEP_NORM_MW_ACTUAL = "step_norm_mw_actual"
STEP_NORM_FREE_MW = "step_norm_free_mw"
OPT_DIAG = "opt_diag"
GRAD_SAMPLED = "grad_sampled"
OPT_MOMENTS = "opt_moments"

#: A row counts as "at" a bound when it is within this many MW of it.  The
#: projection clips exactly onto the bound, so an active row is bit-equal to
#: it; the tolerance only guards against float noise in the arithmetic.
BOUND_ATOL_MW = 1e-9


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


def _flat(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float).reshape(-1)


def active_set(state: dict, problem) -> dict[str, dict[str, np.ndarray]]:
    """Per-parameter boolean masks of the box-constraint active set.

    ``free`` rows are strictly inside their bounds; a row with
    ``lower == upper`` (structurally frozen, 127 of the 166 rows on
    ``ca2040_z4``) is at *both* bounds and free at neither, and is reported as
    ``at_lower`` only, so the three counts partition the rows.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    lowers = getattr(problem, "lower_bounds", None) or {}
    uppers = getattr(problem, "upper_bounds", None) or {}
    for param, value in state.items():
        eta = _flat(value)
        lo = _flat(lowers[param]) if param in lowers else np.full_like(eta, -np.inf)
        hi = _flat(uppers[param]) if param in uppers else np.full_like(eta, np.inf)
        at_lower = eta <= lo + BOUND_ATOL_MW
        at_upper = (eta >= hi - BOUND_ATOL_MW) & ~at_lower
        out[param] = {
            "at_lower": at_lower,
            "at_upper": at_upper,
            "free": ~(at_lower | at_upper),
        }
    return out


def track_free_grad_norm_l2(J, grad, state, last_state, problem):
    """2-norm of the gradient over rows strictly inside their bounds.

    ``grad_norm_l2`` is dominated by coordinates that cannot move: on
    ``ca2040_z4`` the free rows carry 0.7-1.3 % of it, so the clipped step the
    old rule normalised by that norm was ~75x smaller than it reported
    (spec section 1).  This is the norm that describes the step actually taken.
    """
    masks = active_set(state, problem)
    total = 0.0
    for param, mask in masks.items():
        g = _flat(grad[param])[mask["free"]]
        total += float(np.sum(g * g))
    return float(np.sqrt(total))


def _count(state, problem, key):
    masks = active_set(state, problem)
    return int(sum(int(np.count_nonzero(m[key])) for m in masks.values()))


def track_n_free(J, grad, state, last_state, problem):
    return _count(state, problem, "free")


def track_n_at_lower(J, grad, state, last_state, problem):
    return _count(state, problem, "at_lower")


def track_n_at_upper(J, grad, state, last_state, problem):
    return _count(state, problem, "at_upper")


def track_step_norm_mw_actual(J, grad, state, last_state, problem):
    """``||eta - eta_prev||_2`` in MW: the step the design actually took.

    The ``step_norm_mw`` column of the iteration table used to be
    ``step_size * min(||g||, clip)``, which is the step *before* the box
    projection and before the frozen coordinates are thrown away -- it reported
    1,000 MW where the realised movement was 13 MW (spec section 1).
    """
    if last_state is None:
        return 0.0
    total = 0.0
    for param in state:
        d = _flat(state[param]) - _flat(last_state[param])
        total += float(np.sum(d * d))
    return float(np.sqrt(total))


def track_step_norm_free_mw(J, grad, state, last_state, problem):
    """``||eta - eta_prev||_2`` over the rows that are free at the new iterate."""
    if last_state is None:
        return 0.0
    masks = active_set(state, problem)
    total = 0.0
    for param, mask in masks.items():
        d = (_flat(state[param]) - _flat(last_state[param]))[mask["free"]]
        total += float(np.sum(d * d))
    return float(np.sqrt(total))


def track_opt_diag(J, grad, state, last_state, problem):
    """``problem.algorithm.diagnostics()``: the step rule's own report."""
    algorithm = getattr(problem, "algorithm", None)
    diagnostics = getattr(algorithm, "diagnostics", None)
    if diagnostics is None:
        return {}
    try:
        return dict(diagnostics())
    except Exception:  # noqa: BLE001  # pragma: no cover - never fail a solve for a diagnostic
        return {}


def _grad_history_due(problem) -> bool:
    every = int(getattr(problem, "grad_history_every", 0) or 0)
    if every <= 0:
        return False
    return int(getattr(problem, "iteration", 0)) % every == 0


def track_grad_sampled(J, grad, state, last_state, problem):
    """The per-row gradient, but only every ``grad_history_every`` iterations.

    The full per-row gradient is what every number in the step-rule diagnosis
    had to be back-solved from, so it is worth recording -- but recording it at
    *every* iteration of a 6,000-iteration run would put ~100 MB of floats in
    ``designs/<id>.history.json``.  ``None`` on the iterations that are not due.
    """
    if not _grad_history_due(problem):
        return None
    return {k: _flat(v).copy() for k, v in grad.items()}


def track_opt_moments(J, grad, state, last_state, problem):
    """``{"m_hat": ..., "v_hat": ...}`` of the step rule, on recorded iterations."""
    if not _grad_history_due(problem):
        return None
    algorithm = getattr(problem, "algorithm", None)
    out: dict[str, dict] = {}
    for name, method in (("m_hat", "smoothed_gradient"), ("v_hat", "second_moment")):
        fn = getattr(algorithm, method, None)
        if fn is None:
            continue
        try:
            out[name] = {k: _flat(v).copy() for k, v in (fn() or {}).items()}
        except Exception:  # noqa: BLE001, S112  # pragma: no cover
            continue
    return out or None


def track_batch(J, grad, state, last_state, problem):
    """The subproblem indices this iteration's forward pass actually saw.

    ``AbstractPlanningProblem.solve`` stamps ``problem.batch`` immediately
    before every ``forward_and_back`` call; without it the minibatch is a local
    variable and nothing downstream can say which blocks a gradient step used.
    """
    return [int(i) for i in (getattr(problem, "batch", None) or [])]


def track_proj_grad_norm(J, grad, state, last_state, problem):
    """Tracks the 1-norm of the projected gradient.

    Despite the name this is **not** a gradient: it is ``||eta - eta_prev||_1``,
    the realised movement of the design in MW, and until the 2026-09-10 step-rule
    work it was the only honest step column the iteration table had (`step_norm_mw`
    was the pre-projection step and measured nothing).  The name is kept because
    every existing history and parquet carries it; :data:`STEP_NORM_MW_ACTUAL` is
    the 2-norm of the same quantity."""
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
    FREE_GRAD_NORM_L2: track_free_grad_norm_l2,
    N_FREE: track_n_free,
    N_AT_LOWER: track_n_at_lower,
    N_AT_UPPER: track_n_at_upper,
    STEP_NORM_MW_ACTUAL: track_step_norm_mw_actual,
    STEP_NORM_FREE_MW: track_step_norm_free_mw,
    OPT_DIAG: track_opt_diag,
    GRAD_SAMPLED: track_grad_sampled,
    OPT_MOMENTS: track_opt_moments,
}

DEFAULT_TRACKERS = [LOSS, GRAD_NORM, PROJ_GRAD_NORM, TIME, SUBOPTIMALITY]
