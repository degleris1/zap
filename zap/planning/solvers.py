"""Step rules for the planning descent loop.

``GradientDescent`` is the historical rule and is kept **behaviourally
byte-identical** -- it is the reproducibility baseline for the phase-2 campaign
cells c4 / c5 (``memory/plans/2026-09-10-step-rule-spec.md`` section 5, test 4).
The only edit it may ever receive is the ``**kwargs`` sink on ``step``, which
lets ``AbstractPlanningProblem.solve`` pass the same keyword arguments to every
rule without branching.

The other rules exist because the normalised-clipped step is not usable on this
problem (spec section 1):

* the clip normalises over *all* coordinates, and on ``ca2040_z4`` 127 of 166
  parameter rows are structurally frozen -- 98.7 % of the gradient norm belongs
  to coordinates that cannot move, so the realised step was 13 MW where the
  reported one was 1,000 MW;
* the step magnitude is proportional to the gradient magnitude in units nobody
  set (``step_size`` has units MW^2/$), so the same setting spikes by 1e6 MW
  under VOLL scarcity and crawls at 3 MW/iteration once the system is long.

:class:`AdamDescent` is the primary replacement: a **per-coordinate** rule whose
learning rate is in MW, so one iteration moves at most ``step_size`` MW per row
whatever the gradient scale is, and no global norm appears anywhere.
"""

import dataclasses
import math

import numpy as np
import torch

__all__ = [
    "AdagradDescent",
    "AdamDescent",
    "CapexScaledDescent",
    "GradientDescent",
    "TrustRegionDescent",
]


@dataclasses.dataclass(kw_only=True)
class GradientDescent:
    """Parameters for gradient descent."""

    step_size: float = 1e-3
    clip: float = 1e3

    def step(self, state: dict, grad: dict, **kwargs):
        grad_norms = [torch.linalg.vector_norm(grad[param], ord=2) for param in state.keys()]
        total_grad_norm = torch.linalg.vector_norm(torch.stack(grad_norms), ord=2)

        for param in state.keys():
            if total_grad_norm > self.clip:
                clipped_grad = (self.clip / total_grad_norm) * grad[param]
            else:
                clipped_grad = grad[param]

            if isinstance(state[param], torch.Tensor):
                state[param] = state[param].detach() - self.step_size * clipped_grad

            else:
                state[param] -= self.step_size * clipped_grad.numpy()

        return state

    def diagnostics(self) -> dict:
        return {"rule": "gradient", "lr_mw": float(self.step_size) * float(self.clip)}


# ---------------------------------------------------------------------------
# Helpers shared by the per-coordinate rules
# ---------------------------------------------------------------------------


def _as_tensor(x) -> torch.Tensor:
    """A detached float64 ``torch`` view of a numpy array or a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float64)
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


def _write(state: dict, param: str, delta: torch.Tensor) -> None:
    """``state[param] += delta``, preserving the container type of the state."""
    value = state[param]
    if isinstance(value, torch.Tensor):
        state[param] = value.detach() + delta.to(value.dtype).reshape(value.shape)
    else:
        arr = np.asarray(value, dtype=float)
        state[param] = arr + delta.numpy().reshape(arr.shape)


def _projected_gradient(g: torch.Tensor, eta, lower, upper, atol: float = 1e-9):
    """``g`` with the components that cannot move zeroed out.

    A coordinate is dropped when it is structurally frozen (``lower == upper``)
    or when it sits on a bound and the descent direction ``-g`` points out of the
    box.  What is left spans the directions the design can actually take, so a
    rule that normalises by ``||g||`` moves by the length it says it does.

    With no bounds handed in, this is the identity -- the caller then gets the
    unprojected behaviour, which is what a bare unit test of the rule wants.
    """
    if lower is None or upper is None:
        return g
    eta_t = _as_tensor(eta).reshape(g.shape)
    lo = _as_tensor(lower).reshape(g.shape)
    hi = _as_tensor(upper).reshape(g.shape)

    frozen = (hi - lo) <= atol
    at_lower = (eta_t <= lo + atol) & ~frozen
    at_upper = (eta_t >= hi - atol) & ~frozen
    # -g points down at a floor, or up at a ceiling: outward either way.
    outward = (at_lower & (g > 0)) | (at_upper & (g < 0))
    return torch.where(frozen | outward, torch.zeros_like(g), g)


def _decayed(
    step_size: float,
    decay: str,
    final_frac: float,
    iteration: int,
    num_iterations,
) -> float:
    """The learning rate of iteration ``iteration``, in MW.

    ``cosine`` decays from ``step_size`` to ``final_frac * step_size`` over
    ``num_iterations`` steps; ``inverse_sqrt`` decays as ``1/sqrt(1 + t)`` with
    the same floor.  Both need ``num_iterations``; with no horizon to decay over
    the rate is constant, which is the honest answer rather than a guess.
    """
    base = float(step_size)
    if decay in (None, "none"):
        return base
    frac = float(final_frac)
    if decay == "cosine":
        if not num_iterations:
            return base
        t = min(max(float(iteration) / float(num_iterations), 0.0), 1.0)
        return base * (frac + (1.0 - frac) * 0.5 * (1.0 + math.cos(math.pi * t)))
    if decay == "inverse_sqrt":
        return max(base / math.sqrt(1.0 + float(iteration)), base * frac)
    raise ValueError(f"unknown learning-rate decay {decay!r}")


class _PerCoordinateRule:
    """Bookkeeping shared by Adam / Adagrad / capex-scaled descent.

    Subclasses implement :meth:`_delta`, which returns the *uncapped* update of
    one parameter block; this class applies ``max_step_mw``, writes the state
    back and records the diagnostics the trackers report.
    """

    #: Set by ``reset``; the per-iteration diagnostics of the last ``step``.
    _diag: dict

    def reset(self) -> None:
        self._t = 0
        self._m: dict = {}
        self._v: dict = {}
        self._lr = float(self.step_size)
        self._diag = {
            "rule": self.rule,
            "lr_mw": float(self.step_size),
            "max_abs_step_mw": float("nan"),
            "mean_abs_step_mw": float("nan"),
            "n_capped": 0,
        }

    @property
    def max_step(self) -> float:
        cap = getattr(self, "max_step_mw", None)
        return 3.0 * float(self.step_size) if cap is None else float(cap)

    def _delta(self, param: str, g: torch.Tensor, lr: float) -> torch.Tensor:
        raise NotImplementedError

    def step(
        self,
        state: dict,
        grad: dict,
        *,
        iteration: int = 0,
        num_iterations=None,
        **kwargs,
    ) -> dict:
        if not hasattr(self, "_diag"):
            self.reset()
        self._t += 1
        lr = _decayed(
            self.step_size,
            getattr(self, "decay", "none"),
            getattr(self, "decay_final_frac", 0.05),
            iteration,
            num_iterations,
        )
        self._lr = lr
        cap = self.max_step

        max_abs = 0.0
        total_abs = 0.0
        count = 0
        n_capped = 0
        for param in state:
            g = _as_tensor(grad[param])
            delta = self._delta(param, g, lr)
            if math.isfinite(cap) and cap > 0:
                capped = torch.clamp(delta, -cap, cap)
                n_capped += int(torch.count_nonzero(capped != delta).item())
                delta = capped
            abs_delta = torch.abs(delta)
            if abs_delta.numel():
                max_abs = max(max_abs, float(torch.max(abs_delta).item()))
                total_abs += float(torch.sum(abs_delta).item())
                count += int(abs_delta.numel())
            _write(state, param, delta)

        self._diag = {
            "rule": self.rule,
            "lr_mw": float(lr),
            "max_abs_step_mw": max_abs,
            "mean_abs_step_mw": (total_abs / count) if count else float("nan"),
            "n_capped": n_capped,
        }
        return state

    def diagnostics(self) -> dict:
        if not hasattr(self, "_diag"):
            self.reset()
        return dict(self._diag)

    def smoothed_gradient(self) -> dict:
        """The bias-corrected first moment, keyed like ``state`` (numpy).

        The stationarity test consumes this rather than the raw gradient: under
        a minibatch the per-row gradient SNR is 0.2-0.4 and its sign is wrong on
        a third of iterations, while the EMA over ~10 iterations recovers the
        deterministic value (spec section 1).  Rules with no first moment fall
        back to the raw gradient in ``solve``.
        """
        return {}


@dataclasses.dataclass(kw_only=True)
class AdamDescent(_PerCoordinateRule):
    """Adam with the learning rate in MW and ``eps`` in $/MW-yr.

    ``Delta eta_j = -lr * m_hat_j / (sqrt(v_hat_j) + eps)``.

    Two units notes, both load-bearing:

    * ``step_size`` (the learning rate) is **MW per coordinate per iteration**.
      At ``t = 1`` the bias-corrected moments are ``m_hat = g`` and
      ``v_hat = g^2``, so the first step is exactly ``-lr * sign(g)`` however
      large ``||g||`` is: the spike of the normalised rule is impossible by
      construction, with no invented capacity bound.
    * ``eps`` is in the **units of the gradient**, $/MW-yr, not the ML default
      of 1e-8.  It is the gradient magnitude below which a row is treated as
      numerical dust: at 1e-8 a row whose gradient is rounding error still takes
      a full ``lr`` step (the ADMM-rho units trap again, LESSONS 2026-09-09).
    """

    rule = "adam"

    step_size: float = 200.0  # lr, MW per coordinate per iteration
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0  # $/MW-yr -- the same units as the gradient
    max_step_mw: float | None = None  # default 3 * step_size
    decay: str = "none"  # none | cosine | inverse_sqrt
    decay_final_frac: float = 0.05

    def __post_init__(self) -> None:
        self.reset()

    def _delta(self, param: str, g: torch.Tensor, lr: float) -> torch.Tensor:
        b1, b2 = float(self.beta1), float(self.beta2)
        m = self._m.get(param)
        v = self._v.get(param)
        if m is None:
            m = torch.zeros_like(g)
            v = torch.zeros_like(g)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        self._m[param] = m
        self._v[param] = v

        m_hat = m / (1.0 - b1**self._t)
        v_hat = v / (1.0 - b2**self._t)
        return -lr * m_hat / (torch.sqrt(v_hat) + float(self.eps))

    def smoothed_gradient(self) -> dict:
        if self._t <= 0:
            return {}
        correction = 1.0 - float(self.beta1) ** self._t
        return {k: (v / correction).numpy() for k, v in self._m.items()}

    def second_moment(self) -> dict:
        if self._t <= 0:
            return {}
        correction = 1.0 - float(self.beta2) ** self._t
        return {k: (v / correction).numpy() for k, v in self._v.items()}


@dataclasses.dataclass(kw_only=True)
class AdagradDescent(_PerCoordinateRule):
    """Adagrad: Adam without the first moment and with an accumulated second one.

    ``Delta eta_j = -lr * g_j / (sqrt(sum_s g_{j,s}^2) + eps)``.  One knob fewer
    than Adam (the 1/sqrt(t) decay is automatic), but no momentum, so it is the
    *fallback* rather than the primary rule: under the minibatch noise measured
    on c5 (per-row SNR 0.2-0.4) there is nothing left to average the sign errors
    away (spec section 2).
    """

    rule = "adagrad"

    step_size: float = 200.0
    eps: float = 1.0
    max_step_mw: float | None = None
    decay: str = "none"
    decay_final_frac: float = 0.05

    def __post_init__(self) -> None:
        self.reset()

    def _delta(self, param: str, g: torch.Tensor, lr: float) -> torch.Tensor:
        v = self._v.get(param)
        if v is None:
            v = torch.zeros_like(g)
        v = v + g * g
        self._v[param] = v
        return -lr * g / (torch.sqrt(v) + float(self.eps))


@dataclasses.dataclass(kw_only=True)
class CapexScaledDescent(_PerCoordinateRule):
    """``Delta eta_j = -alpha * g_j / gamma_j`` -- Kamran's own proposal (ablation A4).

    ``gamma`` is the annualised capital cost of row ``j`` in $/MW-yr, so the
    ratio is dimensionless and ``alpha`` (``step_size``) is in MW.  That fixes
    the units and much of the cross-technology conditioning, but it is *not*
    admissible as the primary rule: under VOLL scarcity ``g/gamma ~ -10``, so
    the first step is still ten times nominal, and it divides by zero at
    ``gamma = 0`` -- hence ``floor``.
    """

    rule = "capex_scaled"

    step_size: float = 200.0
    capex: dict = dataclasses.field(default_factory=dict)
    floor: float = 1.0
    max_step_mw: float | None = None
    decay: str = "none"
    decay_final_frac: float = 0.05

    def __post_init__(self) -> None:
        self.reset()

    def _delta(self, param: str, g: torch.Tensor, lr: float) -> torch.Tensor:
        gamma = self.capex.get(param)
        if gamma is None:
            scale = torch.full_like(g, float(self.floor))
        else:
            scale = torch.clamp(
                torch.abs(_as_tensor(gamma).reshape(g.shape)), min=float(self.floor)
            )
        return -lr * g / scale


@dataclasses.dataclass(kw_only=True)
class TrustRegionDescent:
    """Steepest descent with an adaptive trust radius, in MW (ablation A3).

    ``Delta eta = -radius * g / ||g||_2``, with the radius expanded when the
    realised decrease matches the first-order prediction and shrunk when it does
    not.  On this problem ``rho`` is 0.994-1.006 at every sampled iteration (F is
    an LP value function plus linear capex, i.e. convex piecewise linear, and at
    13 MW steps the curvature is under 0.5 % of the decrease), so the radius
    expands on essentially every deterministic iteration -- which is why this is
    the *deterministic upper bound* and not the primary rule.

    It is inadmissible under a minibatch for exactly that reason: ``rho`` needs
    the true decrease, and a 4-block estimate of the objective has sigma ~ 1 B$
    against a true decrease of 0.3 M$ (SNR 3e-4).  ``step`` therefore only
    updates the radius when it is handed a *finite* ``predicted_decrease``.

    The normalisation is over the **projected** gradient, not over every row:
    components belonging to structurally frozen rows (``lower == upper``) and to
    rows sitting on a bound whose descent direction points out of the box are
    zeroed before the norm is taken.  Normalising over all rows is precisely the
    defect the spec diagnosed in the ``gradient`` rule -- on ``ca2040_z4`` 98.7 %
    of the gradient norm sits on coordinates that cannot move, so a 50 MW radius
    would have produced a 0.7 MW step and the radius would have meant nothing.
    With the projection, ``||Delta eta||_2 == radius_mw`` exactly (before the box
    projection, which can only shorten it), so the radius *is* the step in MW.

    The step is never rejected: the caller has already paid for the forward pass
    at the new iterate, and this loop keeps every iterate it evaluates.
    """

    rule = "trust_region"

    initial_radius_mw: float = 50.0
    max_radius_mw: float = 5.0e3
    min_radius_mw: float = 1.0e-3
    eta_low: float = 0.1
    eta_high: float = 0.9
    expand: float = 2.0
    shrink: float = 0.5

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.radius_mw = float(self.initial_radius_mw)
        self.rho = float("nan")
        self._diag = {
            "rule": self.rule,
            "lr_mw": float(self.initial_radius_mw),
            "trust_radius_mw": float(self.initial_radius_mw),
            "rho_actual_pred": float("nan"),
            "max_abs_step_mw": float("nan"),
            "mean_abs_step_mw": float("nan"),
            "n_capped": 0,
        }

    def step(
        self,
        state: dict,
        grad: dict,
        *,
        iteration: int = 0,
        num_iterations=None,
        actual_decrease=None,
        predicted_decrease=None,
        lower_bounds=None,
        upper_bounds=None,
        **kwargs,
    ) -> dict:
        rho = float("nan")
        if (
            predicted_decrease is not None
            and actual_decrease is not None
            and math.isfinite(float(predicted_decrease))
            and abs(float(predicted_decrease)) > 0.0
        ):
            rho = float(actual_decrease) / float(predicted_decrease)
            if rho >= float(self.eta_high):
                self.radius_mw = min(self.radius_mw * float(self.expand), float(self.max_radius_mw))
            elif rho < float(self.eta_low):
                self.radius_mw = max(self.radius_mw * float(self.shrink), float(self.min_radius_mw))
        self.rho = rho

        tensors = {
            p: _projected_gradient(
                _as_tensor(grad[p]),
                state[p],
                None if lower_bounds is None else lower_bounds.get(p),
                None if upper_bounds is None else upper_bounds.get(p),
            )
            for p in state
        }
        norm = float(
            torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g, ord=2) for g in tensors.values()]),
                ord=2,
            ).item()
        )

        max_abs = 0.0
        total_abs = 0.0
        count = 0
        for param, g in tensors.items():
            delta = torch.zeros_like(g) if norm == 0.0 else -(self.radius_mw / norm) * g
            abs_delta = torch.abs(delta)
            if abs_delta.numel():
                max_abs = max(max_abs, float(torch.max(abs_delta).item()))
                total_abs += float(torch.sum(abs_delta).item())
                count += int(abs_delta.numel())
            _write(state, param, delta)

        self._diag = {
            "rule": self.rule,
            "lr_mw": float(self.radius_mw),
            "trust_radius_mw": float(self.radius_mw),
            "rho_actual_pred": rho,
            "max_abs_step_mw": max_abs,
            "mean_abs_step_mw": (total_abs / count) if count else float("nan"),
            "n_capped": 0,
        }
        return state

    def diagnostics(self) -> dict:
        return dict(self._diag)

    def smoothed_gradient(self) -> dict:
        return {}
