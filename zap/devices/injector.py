import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import torch
from attrs import define, field, Factory

from collections import namedtuple
from typing import Optional
from numpy.typing import NDArray

from .abstract import AbstractDevice, get_time_horizon, make_dynamic

#: Local variables of a committable :class:`Generator`, both **in MW** (not
#: PyPSA's dimensionless status): ``committed`` is committed capacity and
#: ``start_up`` is capacity started in the hour.  Flows through
#: ``model_variables`` / ``DispatchOutcome`` / ``torchify`` / ``nested_evaluate``
#: exactly as :class:`~zap.devices.storage_unit.StorageUnitVariable` does.
GeneratorCommitVariable = namedtuple("GeneratorCommitVariable", ["committed", "start_up"])

#: Boundary conditions on ``c_{g,-1}``; see :attr:`Generator.commitment_mode`.
VALID_COMMITMENT_MODES = ("cyclic_free", "pypsa")


def _np(value):
    """A numpy view of a possibly-torch device attribute."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _backend(value, la, like=None):
    """Put a device-owned *constant* on the backend ``la`` builds expressions in.

    ``la`` is ``numpy``, ``cvxpy`` or ``torch``.  The first two want numpy
    constants (cvxpy promotes them), the third wants a tensor on the same device
    and dtype as the dispatch variables.  Needed because
    ``AbstractDevice.lagrangian_gradients`` torchifies the *variables* of a
    numpy-side device and then evaluates the constraints with ``la=torch``.
    """
    if la is torch:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(_np(value))
        if isinstance(like, torch.Tensor):
            tensor = tensor.to(device=like.device, dtype=like.dtype)
        elif tensor.dtype not in (torch.float32, torch.float64):
            tensor = tensor.to(dtype=torch.get_default_dtype())
        return tensor
    return _np(value)


def previous_hour_matrix(time_horizon: int, commitment_mode: str, la=np, like=None):
    """``Sprev``: the ``(T, T)`` constant with ``(c @ Sprev)[:, t] == c[:, t-1]``.

    Its **first column** is the whole boundary condition (spec section 2):

    * ``cyclic_free`` wraps, ``Sprev[T-1, 0] = 1``, so ``c_{-1} = c_{T-1}`` and a
      block that starts committed pays a start only if it ended uncommitted;
    * ``pypsa`` leaves the first column zero, so ``c_{-1} = 0`` and every unit
      that is on in hour 0 pays a full start -- which is what PyPSA's linearised
      UC does when ``up_time_before == 0``.

    Note the *direction*: the wrap is ``c_{-1} = c_{T-1}``, not ``c_{T-1} = c_0``.
    For storage the cyclic boundary is an equality and the two readings coincide;
    the start transition is one-sided and they do not.
    """
    T = int(time_horizon)
    if T < 1:
        raise ValueError(f"time horizon must be positive, got {T}")
    if commitment_mode not in VALID_COMMITMENT_MODES:
        raise ValueError(
            f"commitment_mode must be one of {VALID_COMMITMENT_MODES}, got {commitment_mode!r}"
        )
    rows = np.arange(T - 1, dtype=int)
    cols = np.arange(1, T, dtype=int)
    if commitment_mode == "cyclic_free":
        rows = np.concatenate([rows, [T - 1]])
        cols = np.concatenate([cols, [0]])
    if la is torch:
        dense = np.zeros((T, T))
        dense[rows, cols] = 1.0
        return _backend(dense, la, like=like)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(T, T))


@define(kw_only=True, slots=False)
class AbstractInjector(AbstractDevice):
    """A single-node device that may deposit or withdraw power from the network. Abstract type that
    should not be instantiated but contains shared behavior among all subclasses."""

    num_nodes: int  # why do you re-instatiate this?
    name: str
    terminal: NDArray
    nominal_capacity: NDArray = field(
        default=Factory(lambda self: np.ones(self.num_devices), takes_self=True),
        converter=make_dynamic,
    )

    # These properties should be implemented by subclasses
    min_power: NDArray = field(init=False)
    max_power: NDArray = field(init=False)
    linear_cost: NDArray = field(init=False)
    quadratic_cost: Optional[NDArray] = field(init=False)
    capital_cost: Optional[NDArray] = field(init=False)
    emission_rates: Optional[NDArray] = field(init=False)

    @property
    def terminals(self):
        return self.terminal

    @property
    def time_horizon(self):
        return get_time_horizon(self.min_power)

    def scale_costs(self, scale):
        self.linear_cost /= scale
        if self.quadratic_cost is not None:
            self.quadratic_cost /= scale
        if self.capital_cost is not None:
            self.capital_cost /= scale

        # Emissions are in units of kg/MWh
        # but we scale them with costs so that prices can be stated in $/MWh
        if self.emission_rates is not None:
            self.emission_rates /= scale

    def scale_power(self, scale):
        self.nominal_capacity /= scale

        # Invert scaling because term is quadratic
        if self.quadratic_cost is not None:
            self.quadratic_cost *= scale

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def equality_constraints(
        self,
        power,
        angle,
        _,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
        envelope=None,
    ):
        return []

    def inequality_constraints(
        self,
        power,
        angle,
        _,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
        envelope=None,
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)
        max_power = self.parameterize(max_power=max_power, la=la)
        min_power = self.parameterize(min_power=min_power, la=la)
        power = power[0]

        return [
            la.multiply(min_power, nominal_capacity) - power,
            power - la.multiply(max_power, nominal_capacity),
        ]

    def operation_cost(
        self,
        power,
        angle,
        _,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
        envelope=None,
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)
        min_power = self.parameterize(min_power=min_power, la=la)
        linear_cost = self.parameterize(linear_cost=linear_cost, la=la)
        power = power[0] - la.multiply(min_power, nominal_capacity)

        cost = la.sum(la.multiply(linear_cost, power))
        if self.quadratic_cost is not None:
            cost += la.sum(la.multiply(self.quadratic_cost, la.square(power)))

        return cost

    # ====
    # PLANNING FUNCTIONS
    # ====

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)

        # Subsample linear cost
        if dev.linear_cost.shape[1] > 1:
            dev.linear_cost = dev.linear_cost[:, time_periods]

        return dev

    # ====
    # DIFFERENTIATION
    # ====

    def _equality_matrices(
        self,
        equalities,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
    ):
        return equalities

    def _inequality_matrices(
        self,
        inequalities,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
    ):
        size = inequalities[0].power[0].shape[1]
        inequalities[0].power[0] += -sp.eye(size)
        inequalities[1].power[0] += sp.eye(size)
        return inequalities

    def _hessian_power(
        self,
        hessians,
        power,
        angle,
        _,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
    ):
        if self.quadratic_cost is None:
            return hessians

        hessians[0] += 2 * sp.diags(
            (self.quadratic_cost * np.ones_like(power[0])).ravel()
        )
        return hessians

    # ====
    # ADMM FUNCTIONS
    # ====

    def admm_prox_update(
        self,
        rho_power,
        rho_angle,
        power,
        angle,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        power_weights=None,
        angle_weights=None,
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity)
        max_power = self.parameterize(max_power=max_power)
        min_power = self.parameterize(min_power=min_power)
        linear_cost = self.parameterize(linear_cost=linear_cost)

        # machine, dtype = power[0].device, power[0].dtype
        assert angle is None

        if self.has_changed:
            quadratic_cost = (
                0.0 * self.linear_cost
                if self.quadratic_cost is None
                else self.quadratic_cost
            )
            pmax = torch.multiply(max_power, nominal_capacity)
            pmin = torch.multiply(min_power, nominal_capacity)
            self.admm_data = (quadratic_cost, pmax, pmin)
            self.has_changed = False

        quadratic_cost, pmax, pmin = self.admm_data

        return _admm_prox_update(
            power, rho_power, self.linear_cost, quadratic_cost, pmin, pmax
        )

    def get_admm_power_weights(
        self,
        power,
        strategy: str,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity)
        linear_cost = self.parameterize(linear_cost=linear_cost)

        if strategy == "smart_cost":
            avg_cost = np.mean(linear_cost, axis=1).reshape((-1, 1))
            return [np.maximum(np.sqrt(1 / (avg_cost + 0.01)), 1.0)]

        if strategy == "smart_bounds":
            return [np.minimum(np.sqrt(1 / (nominal_capacity + 1.0)), 1.0)]

        else:
            return [np.ones_like(pi) for pi in power]


@define(kw_only=True, slots=False)
class Injector(AbstractInjector):
    """A single-node device that may deposit or withdraw power from the network."""

    min_power: NDArray = field(converter=make_dynamic)
    max_power: NDArray = field(converter=make_dynamic)
    linear_cost: NDArray = field(converter=make_dynamic)
    quadratic_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    capital_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    emission_rates: Optional[NDArray] = field(default=None, converter=make_dynamic)

    # TODO - Add dimension checks


@define(kw_only=True, slots=False)
class Generator(AbstractInjector):
    """An Injector that can only deposit power.

    Optionally carries the **minimal linearised unit-commitment** device of
    ``memory/plans/2026-09-14-uc-minimal-impl-spec.md``: a start-up cost and a
    minimum stable level, LP-relaxed, on a subset of rows.  It is fields on
    ``Generator`` rather than a subclass so that device order, ``design.json``
    keying, outage slot keys and accreditation row indexing all keep seeing one
    ``Generator`` row block.

    Parameters
    ----------
    committable : Optional[NDArray]
        ``(N,)`` bool; which rows carry commitment.  **Not** ``make_dynamic``.
        ``None`` (the default) is "no commitment anywhere" and every method below
        delegates to the plain injector path.
    min_power_fraction : Optional[NDArray]
        ``p_min_pu``, per unit of *committed* MW, on committable rows.  Ignored on
        non-committable rows (``Generator.min_power`` is identically zero there).
    start_up_cost_per_mw : Optional[NDArray]
        ``k_g``, **$ per MW started** -- the file's ``start_up_cost`` divided by
        the file's ``p_nom``.  This is the one deliberate semantic difference
        from PyPSA, which charges a flat ``$`` per start; the two coincide
        whenever the row is dispatched at its file capacity.
    commitment_mode : {"cyclic_free", "pypsa"}
        Boundary condition on ``c_{g,-1}``; see :func:`previous_hour_matrix`.

    Notes
    -----
    If an outage or UCAP derate pushes ``a_{g,t}`` below ``rho_g`` the two power
    families force ``c_{g,t} = 0``: the unit cannot be committed that hour and
    pays a restart afterwards.  That is intended, and it is the only place in
    this chapter where an outage draw has a multi-hour consequence.

    The commitment fields are expected to be set at **construction**.  The
    derived index (:attr:`commit_rows`, :attr:`commit_scatter`, ...) is cached in
    ``__attrs_post_init__``; mutating ``committable`` afterwards requires a call
    to :meth:`rebuild_commitment`.
    """

    dynamic_capacity: NDArray = field(converter=make_dynamic)
    linear_cost: NDArray = field(converter=make_dynamic)
    quadratic_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    capital_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    emission_rates: Optional[NDArray] = field(default=None, converter=make_dynamic)
    min_nominal_capacity: Optional[NDArray] = field(
        default=None, converter=make_dynamic
    )
    max_nominal_capacity: Optional[NDArray] = field(
        default=None, converter=make_dynamic
    )
    #: Unit commitment (all optional; ``None`` everywhere == today's device).
    committable: Optional[NDArray] = field(default=None)
    min_power_fraction: Optional[NDArray] = field(default=None, converter=make_dynamic)
    start_up_cost_per_mw: Optional[NDArray] = field(default=None, converter=make_dynamic)
    commitment_mode: str = field(default="cyclic_free")

    # TODO - Add dimension checks

    def __attrs_post_init__(self):
        self.commitment_mode = str(self.commitment_mode)
        if self.commitment_mode not in VALID_COMMITMENT_MODES:
            raise ValueError(
                f"commitment_mode must be one of {VALID_COMMITMENT_MODES}, "
                f"got {self.commitment_mode!r}"
            )
        if self.committable is not None:
            self.committable = np.asarray(self.committable, dtype=bool).reshape(-1)
            if self.committable.size != self.num_devices:
                raise ValueError(
                    f"committable has {self.committable.size} entries for "
                    f"{self.num_devices} generator rows"
                )
        self.rebuild_commitment()

    # ====
    # UNIT COMMITMENT
    # ====

    @property
    def is_committable(self) -> bool:
        committable = getattr(self, "committable", None)
        return committable is not None and bool(np.asarray(committable).any())

    def rebuild_commitment(self) -> None:
        """(Re)build the cached committable sub-index.

        ``commit_rows`` are the committable row numbers; ``commit_scatter`` is the
        dense ``(N, N_c)`` matrix with ``commit_scatter[commit_rows[j], j] = 1``,
        so ``commit_scatter @ c`` scatters a committable-only array back over all
        rows (zero elsewhere).  Stored as plain float arrays so ``torchify``
        converts them with everything else.
        """
        if not self.is_committable:
            self.commit_rows = np.zeros(0, dtype=int)
            self.commit_scatter = np.zeros((self.num_devices, 0))
            self.commit_min_power = np.zeros((self.num_devices, 1))
            self.commit_free_mask = np.ones((self.num_devices, 1))
            self.commit_start_up_cost = np.zeros((0, 1))
            return

        committable = np.asarray(self.committable, dtype=bool).reshape(-1)
        rows = np.flatnonzero(committable)
        scatter = np.zeros((self.num_devices, rows.size))
        scatter[rows, np.arange(rows.size)] = 1.0

        rho = np.zeros((self.num_devices, 1))
        if self.min_power_fraction is not None:
            values = _np(self.min_power_fraction).reshape(self.num_devices, -1)[:, :1]
            rho[rows, :] = values[rows, :]
        if np.any(rho < 0.0):
            raise ValueError("min_power_fraction must be non-negative")

        k = np.zeros((self.num_devices, 1))
        if self.start_up_cost_per_mw is not None:
            values = _np(self.start_up_cost_per_mw).reshape(self.num_devices, -1)[:, :1]
            k[rows, :] = values[rows, :]
        if np.any(k < 0.0):
            raise ValueError("start_up_cost_per_mw must be non-negative")

        self.commit_rows = rows
        self.commit_scatter = scatter
        self.commit_min_power = rho
        #: 0 on committable rows, 1 elsewhere -- the mask that removes the
        #: availability bound ``p <= a * P_bar`` on a committable row, so the dual
        #: of the max-load family never splits with the dual of ``c <= P_bar``.
        self.commit_free_mask = (~committable).astype(float).reshape(-1, 1)
        self.commit_start_up_cost = k[rows, :]

    def _commit_state(self, state):
        """Defensive re-wrap: ``nested_evaluate`` returns a plain list."""
        if state is None or isinstance(state, GeneratorCommitVariable):
            return state
        return GeneratorCommitVariable(*state)

    def model_local_variables(self, time_horizon: int):
        if not self.is_committable:
            return None
        n_c = int(_np(self.commit_rows).size)
        return GeneratorCommitVariable(
            cp.Variable((n_c, time_horizon)),
            cp.Variable((n_c, time_horizon)),
        )

    @property
    def min_power(self):
        return 0.0 * self.dynamic_capacity

    @property
    def max_power(self):
        return self.dynamic_capacity

    def scale_costs(self, scale):
        if self.start_up_cost_per_mw is not None:
            self.start_up_cost_per_mw = self.start_up_cost_per_mw / scale
            self.commit_start_up_cost = self.commit_start_up_cost / scale
        return super().scale_costs(scale)

    def scale_power(self, scale):
        if self.min_nominal_capacity is not None:
            self.min_nominal_capacity /= scale
        if self.max_nominal_capacity is not None:
            self.max_nominal_capacity /= scale
        # `k` is deliberately NOT scaled: `AbstractInjector.scale_power` divides
        # capacities and powers by `scale` but leaves `linear_cost` alone, so the
        # whole objective is in $ / power_unit afterwards (which is why
        # `metrics._to_physical_units` multiplies money by cost_unit * power_unit).
        # `k * su'` with `su' = su / scale` is already in those units; scaling `k`
        # as well would price a start `scale` times too high relative to fuel
        # (verifier, 2026-09-14: toy objective 5700 -> 6240 at power_unit 10).
        return super().scale_power(scale)

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def inequality_constraints(
        self,
        power,
        angle,
        state,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
        envelope=None,
    ):
        """Six ``expr <= 0`` families when committable, today's two otherwise.

        ``0`` and ``1`` **replace** the plain injector's min/max power families
        (same index, same ``(N, T)`` shape); they are not added alongside::

            0.  rho * (S @ c) - p                       <= 0
            1.  p - a * (a * m_nc * P_bar + S @ c)      <= 0
            2.  c - P_bar[committable]                  <= 0
            3.  -c                                      <= 0
            4.  c - c @ Sprev - su                      <= 0
            5.  -su                                     <= 0

        On a non-committable row ``S @ c = 0`` and ``m_nc = 1``, so 0 and 1 are
        exactly ``0 - p <= 0`` and ``p - a * P_bar <= 0``.
        """
        if not self.is_committable:
            return super().inequality_constraints(
                power,
                angle,
                state,
                nominal_capacity=nominal_capacity,
                max_power=max_power,
                min_power=min_power,
                linear_cost=linear_cost,
                la=la,
                envelope=envelope,
            )

        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)
        availability = self.parameterize(max_power=max_power, la=la)
        min_power_param = _np(self.parameterize(min_power=min_power, la=np))
        if np.any(min_power_param != 0.0):
            raise ValueError(
                "a committable Generator must keep `min_power` identically zero: the "
                "minimum stable level lives in `min_power_fraction`, and a nonzero "
                "`min_power` would also offset `operation_cost`"
            )

        state = self._commit_state(state)
        if state is None:
            raise ValueError(
                "a committable Generator needs its local variables; the caller passed None"
            )

        p = power[0]
        T = p.shape[1]
        scatter = _backend(self.commit_scatter, la, like=p)
        rho = _backend(self.commit_min_power, la, like=p)
        free = _backend(self.commit_free_mask, la, like=p)
        prev = previous_hour_matrix(T, self.commitment_mode, la=la, like=p)
        rows = _np(self.commit_rows)

        committed = scatter @ state.committed
        # Installed-or-committed MW, *before* the availability derate: the
        # nameplate on a non-committable row (`m_nc == 1`) and the committed MW
        # on a committable one (`m_nc == 0`).  `a` multiplies it exactly once, in
        # family 1 below -- the spec's section 1.3 code block writes
        # `multiply(multiply(a, m_nc), P_bar)` here, which would square the derate
        # on every non-committable row and contradicts both its own prose ("on a
        # non-committable row ... p - a*P_bar <= 0") and its KKT table.
        cap_full = la.multiply(free, nominal_capacity) + committed

        return [
            la.multiply(rho, committed) - p,
            p - la.multiply(availability, cap_full),
            state.committed - nominal_capacity[rows, :],
            -state.committed,
            state.committed - state.committed @ prev - state.start_up,
            -state.start_up,
        ]

    def operation_cost(
        self,
        power,
        angle,
        state,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
        envelope=None,
    ):
        cost = super().operation_cost(
            power,
            angle,
            state,
            nominal_capacity=nominal_capacity,
            max_power=max_power,
            min_power=min_power,
            linear_cost=linear_cost,
            la=la,
            envelope=envelope,
        )
        if not self.is_committable:
            return cost

        state = self._commit_state(state)
        if state is None:
            return cost

        k = _backend(self.commit_start_up_cost, la, like=power[0])
        return cost + la.sum(la.multiply(k, state.start_up))

    # ====
    # PLANNING FUNCTIONS
    # ====

    def get_investment_cost(self, nominal_capacity=None, la=np):
        if self.capital_cost is None or nominal_capacity is None:
            return 0.0

        pnom_min = self.nominal_capacity
        capital_cost = self.capital_cost

        return la.sum(la.multiply(capital_cost, (nominal_capacity - pnom_min)))

    def get_emissions(self, power, nominal_capacity=None, la=np):
        if self.emission_rates is None:
            return 0.0
        else:
            return la.sum(la.multiply(self.emission_rates, power[0]))

    def sample_time(self, time_periods, original_time_horizon):
        if self.is_committable:
            periods = np.asarray(time_periods).reshape(-1)
            if periods.size and not np.array_equal(periods, np.arange(periods[0], periods[-1] + 1)):
                raise ValueError(
                    "a committable Generator can only be sampled onto a contiguous, "
                    "increasing block of hours: the start transition and the "
                    f"{self.commitment_mode!r} boundary are statements about adjacent "
                    "hours. Got a non-contiguous selection of "
                    f"{periods.size} periods spanning [{periods[0]}, {periods[-1]}]."
                )

        dev = super().sample_time(time_periods, original_time_horizon)

        if dev.dynamic_capacity.shape[1] > 1:
            dev.dynamic_capacity = dev.dynamic_capacity[:, time_periods]

        # The committable sub-index, rho and k are static: nothing to slice, but
        # rebuild so that a device whose fields were edited in place is coherent.
        dev.rebuild_commitment()

        return dev

    # ====
    # DIFFERENTIATION
    # ====

    def _inequality_matrices(
        self,
        inequalities,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
    ):
        """``d(family) / d(variable)`` for the six families of
        :meth:`inequality_constraints`, in the same order and with the same sign.
        """
        if not self.is_committable:
            return super()._inequality_matrices(
                inequalities,
                nominal_capacity=nominal_capacity,
                max_power=max_power,
                min_power=min_power,
                linear_cost=linear_cost,
                la=la,
            )

        size = inequalities[0].power[0].shape[1]
        N = self.num_devices
        T = size // N
        n_c = int(_np(self.commit_rows).size)
        csize = n_c * T

        scatter = sp.csr_matrix(_np(self.commit_scatter))
        # Row-major flattening throughout, matching the existing `sp.eye(size)`.
        E = sp.kron(scatter, sp.eye(T), format="csr")
        rho = np.broadcast_to(_np(self.commit_min_power), (N, T))
        availability = _np(self.parameterize(max_power=max_power, la=np))
        # d(family 1)/dc = -a * d(S @ c)/dc: the `m_nc` mask sits on the `P_bar`
        # term, which is not a local variable, so it does not appear here.
        a_full = np.broadcast_to(availability, (N, T))
        prev = previous_hour_matrix(T, self.commitment_mode, la=np)
        # `vec_r(c @ Sprev) = kron(I_{N_c}, Sprev.T) vec_r(c)`.
        Pprev = sp.kron(sp.eye(n_c), sp.csr_matrix(prev).T, format="csr")

        inequalities[0].power[0] += -sp.eye(size)
        inequalities[0].local_variables[0] += sp.diags(np.ascontiguousarray(rho).ravel()) @ E

        inequalities[1].power[0] += sp.eye(size)
        inequalities[1].local_variables[0] += -sp.diags(np.ascontiguousarray(a_full).ravel()) @ E

        inequalities[2].local_variables[0] += sp.eye(csize)
        inequalities[3].local_variables[0] += -sp.eye(csize)

        inequalities[4].local_variables[0] += sp.eye(csize) - Pprev
        inequalities[4].local_variables[1] += -sp.eye(csize)

        inequalities[5].local_variables[1] += -sp.eye(csize)

        return inequalities

    # ====
    # ADMM FUNCTIONS
    # ====

    def admm_prox_update(self, *args, **kwargs):
        if self.is_committable:
            raise NotImplementedError(
                "Unit commitment has no closed-form ADMM prox: the per-unit subproblem "
                "is an LP over (p, c, su) coupled across time by the start transition. "
                "Use an LP dispatch layer (`methods.lp` / `planning.dispatch_solver`), "
                "or run with `heuristics.commitment.mode: off`. See "
                "`memory/plans/2026-09-14-uc-in-zap-feasibility.md` section 4."
            )
        return super().admm_prox_update(*args, **kwargs)


#: Class-level fallbacks, mirroring :attr:`StorageUnit.soc_mode`: ``attrs``
#: deletes the class attribute of every field it generates an ``__init__`` for,
#: so a subclass that builds its state by hand (``zap.devices.dual``) would
#: otherwise ``AttributeError`` on any of these.  An attrs-built instance always
#: shadows them from its own ``__dict__``.
Generator.committable = None
Generator.min_power_fraction = None
Generator.start_up_cost_per_mw = None
Generator.commitment_mode = "cyclic_free"
Generator.commit_rows = np.zeros(0, dtype=int)
Generator.commit_scatter = np.zeros((0, 0))
Generator.commit_min_power = np.zeros((0, 1))
Generator.commit_free_mask = np.ones((0, 1))
Generator.commit_start_up_cost = np.zeros((0, 1))


@define(kw_only=True, slots=False)
class Load(AbstractInjector):
    """An Injector that can only withdraw power."""

    load: NDArray = field(converter=make_dynamic)
    linear_cost: NDArray = field(converter=make_dynamic)
    quadratic_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)

    @property
    def min_power(self):
        return -self.load

    @property
    def max_power(self):
        return 0.0 * self.load

    @property
    def capital_cost(self):
        return None

    @property
    def emission_rates(self):
        return None

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)

        if dev.load.shape[1] > 1:
            dev.load = dev.load[:, time_periods]

        return dev


@torch.jit.script
def _admm_prox_update(
    power: list[torch.Tensor], rho: float, lin_cost, quad_cost, pmin, pmax
):
    # Problem is
    #     min_p    a (p - pmin)^2 + b (p - pmin) + (rho / 2) || (p - power) ||_2^2 + {box constraints}
    # Objective derivative is
    #     2 a (p - pmin) + b +  rho (p - power) = 0
    # Which is solved by
    #     p = (rho power + 2 a pmin - b) / (2 a + rho )
    num = rho * power[0] + 2 * quad_cost * pmin - lin_cost
    denom = 2 * quad_cost + rho
    p = torch.divide(num, denom)

    # Finally, we project onto the box constraints
    p = torch.clip(p, pmin, pmax)

    return [p], None, None
