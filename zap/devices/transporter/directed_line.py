import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import torch

from dataclasses import dataclass
from typing import Optional
from numpy.typing import NDArray

from zap.devices.abstract import make_dynamic
from zap.util import replace_none
from .transporter import Transporter


@dataclass(kw_only=True)
class DirectedLine(Transporter):
    """A one-way transporter with a conversion efficiency and a signed linear cost.

    ``power[1] = -efficiency * power[0]``; bounds and cost apply to ``power[1]``.
    Unlike ``Transporter`` / ``PowerLine`` the cost is linear in ``power[1]``
    (no ``abs``), so a negative ``linear_cost`` (e.g. export revenue) stays DCP.

    This is the zap analogue of a PyPSA ``Link``: PyPSA links are directional
    (``p_min_pu`` defaults to zero) and carry an efficiency, neither of which the
    symmetric ``DCLine`` can express.
    """

    name: Optional[NDArray] = None
    efficiency: NDArray = None  # (N, 1); defaults to ones in __post_init__

    #: Optional **aggregate interface limit** (import-limit spec, 2026-09-14).
    #: ``group[i]`` is the group index of row ``i``, ``-1`` meaning ungrouped;
    #: ``group_limit[g]`` caps ``sum_{i in g} power[1][i, t]`` -- the *delivered*
    #: MW at the sink end -- in every hour.  Both fields are given together or
    #: not at all, and the device is bit-identical to the ungrouped one when they
    #: are absent.  Groups are general so that a future per-interface or seasonal
    #: limit needs no new device code; ``wy_store`` builds exactly one, from
    #: ``carrier == "imports"``.
    group: Optional[NDArray] = None  # (N,) int; -1 = ungrouped
    #: ``(G,)``, ``(G, 1)`` or ``(G, T)`` MW.  A ``(G, T)`` limit is the
    #: **hour-of-day import profile** (import-limit profile spec P1, 2026-09-17)
    #: and lives on the *same hour grid* as this device's ``min_power`` /
    #: ``max_power`` / ``linear_cost``; :meth:`sample_time` slices it alongside
    #: them.  It is a right-hand side only: ``_inequality_matrices`` builds the
    #: family-2 Jacobian from :attr:`group_matrix` alone, so an hourly limit
    #: changes no differentiation code.
    group_limit: Optional[NDArray] = None  # (G,), (G, 1) or (G, T) MW
    group_name: Optional[NDArray] = None  # (G,) str; provenance / reporting only
    #: ``(G, N)`` 0/1 incidence, built in ``__post_init__``.  A dense float
    #: ndarray **field**, not a ``cached_property``, so that ``torchify`` finds it
    #: in ``__dict__`` and moves it with the rest of the device; G and N are tiny.
    group_matrix: Optional[NDArray] = None

    def __post_init__(self):
        super().__post_init__()
        self.efficiency = make_dynamic(replace_none(self.efficiency, np.ones(self.num_devices)))
        # A lossy line with reverse flow allowed (min_power < 0) would create
        # energy: p0 = -p1/eta withdraws less than |p1| when p1 < 0. Flow must
        # be one-way, which is the point of this device.
        if np.any(np.asarray(self.min_power) < 0):
            raise ValueError(
                "DirectedLine requires min_power >= 0 (one-way flow); "
                "model reverse capacity as a second DirectedLine."
            )
        self._init_groups()

    def _init_groups(self):
        """Validate and build the aggregate-limit fields (import-limit spec 3.1)."""
        if (self.group is None) != (self.group_limit is None):
            raise ValueError(
                "DirectedLine.group and DirectedLine.group_limit must be given together "
                f"(got group={'set' if self.group is not None else 'None'}, "
                f"group_limit={'set' if self.group_limit is not None else 'None'})"
            )
        if self.group is None:
            self.group_matrix = None
            return

        group = np.asarray(self.group).reshape(-1).astype(int)
        if group.size != self.num_devices:
            raise ValueError(
                f"DirectedLine.group has {group.size} entries but the device has "
                f"{self.num_devices} rows"
            )
        # ``(G,)``, ``(G, 1)`` and ``(G, T)`` are all legal; only the first
        # dimension is the group count.  A flat cap is widened to ``(G, 1)`` so
        # that everything downstream -- the constraint, ``scale_power``,
        # ``sample_time`` -- sees one shape family (profile spec P1).  ``T`` is
        # unknown at construction, so a wrong-width limit cannot be validated
        # here; it surfaces as a cvxpy broadcast error at solve time.
        limit = np.asarray(self.group_limit, dtype=np.float64)
        if limit.ndim == 1:
            limit = limit[:, None]
        elif limit.ndim != 2:
            raise ValueError(
                "DirectedLine.group_limit must be (G,), (G, 1) or (G, T), got shape "
                f"{limit.shape}"
            )
        num_groups = limit.shape[0]
        if limit.size == 0:
            raise ValueError("DirectedLine.group_limit is empty")
        if np.any(group < -1) or np.any(group >= num_groups):
            raise ValueError(
                f"DirectedLine.group values must lie in {{-1}} u [0, {num_groups}), "
                f"got {sorted(set(group.tolist()))}"
            )
        populated = set(group[group >= 0].tolist())
        missing = sorted(set(range(num_groups)) - populated)
        if missing:
            raise ValueError(
                f"DirectedLine.group_limit has {num_groups} group(s) but group(s) {missing} "
                "contain no rows; an empty group is a silently inert constraint"
            )
        if not np.all(np.isfinite(limit)) or np.any(limit <= 0):
            raise ValueError(
                f"DirectedLine.group_limit must be finite and > 0, got {limit.tolist()}"
            )

        matrix = np.zeros((num_groups, self.num_devices), dtype=np.float64)
        matrix[group[group >= 0], np.flatnonzero(group >= 0)] = 1.0

        # A group whose must-flow already exceeds its cap is infeasible by
        # construction, which surfaces as an unbounded VOLL bill or an
        # `infeasible` solver status hours later. Refuse it here instead.
        # `min_power` may itself be hourly, so reduce both sides to one number
        # per group: the group's **largest** must-flow against its **tightest**
        # hour.  For the flat case (one value each) that is exactly the test
        # this has always been; with hourly data it is conservative, which is
        # the right direction for a "this is infeasible by construction" guard.
        min_power = np.atleast_2d(np.asarray(self.min_power, dtype=np.float64))
        capacity = np.asarray(self.nominal_capacity, dtype=np.float64).reshape(-1, 1)
        must_flow = (matrix @ (min_power * capacity)).max(axis=1)
        tightest = limit.min(axis=1)
        bad = np.flatnonzero(must_flow > tightest)
        if bad.size:
            raise ValueError(
                "DirectedLine group(s) "
                f"{[(int(g), float(must_flow[g]), float(tightest[g])) for g in bad]} have a "
                "minimum flow above their group_limit (group, min flow MW, limit MW): "
                "the constraint is infeasible by construction"
            )

        self.group = group
        self.group_limit = make_dynamic(limit)  # already 2-D; the call is symmetry
        self.group_matrix = matrix
        if self.group_name is not None:
            names = np.asarray(self.group_name).reshape(-1)
            if names.size != num_groups:
                raise ValueError(
                    f"DirectedLine.group_name has {names.size} entries but there are "
                    f"{num_groups} group(s)"
                )
            self.group_name = names

    def sample_time(self, time_periods, original_time_horizon):
        """``Transporter.sample_time`` plus the hourly group limit.

        ``Transporter`` slices exactly ``min_power`` / ``max_power`` /
        ``linear_cost``; a ``(G, T)`` :attr:`group_limit` is on the same grid and
        must be sliced with them, or a block solve would meet a full-window cap
        (import-limit profile spec 3.1).  ``Transporter.time_horizon`` is a
        hard-coded ``0``, so nothing else would catch the omission -- only
        cvxpy's broadcast, at solve time.  A ``(G, 1)`` limit is left alone, the
        same ``shape[1] > 1`` rule the inherited method uses.
        """
        dev = super().sample_time(time_periods, original_time_horizon)
        limit = dev.group_limit
        if limit is not None and len(np.shape(limit)) > 1 and np.shape(limit)[1] > 1:
            if isinstance(limit, cp.Parameter):
                raise ValueError(
                    "DirectedLine.group_limit is a cp.Parameter and cannot be sliced by "
                    "sample_time: parametrise the device *after* sampling the window, "
                    "which is what ch3.ra.dispatch's retained-problem path does."
                )
            dev.group_limit = limit[:, time_periods]
        return dev

    @property
    def is_ac(self):
        return False

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def equality_constraints(
        self, power, angle, _, nominal_capacity=None, la=np, envelope=None, mask=None
    ):
        return [power[1] + la.multiply(self.efficiency, power[0])]

    def inequality_constraints(
        self, power, angle, _, nominal_capacity=None, la=np, envelope=None, mask=None
    ):
        """Per-row bounds on ``power[1]`` plus, optionally, the group limit.

        Families 0 and 1 are the inherited per-row bounds.  Family 2, present
        only when :attr:`group_matrix` is set, is ``(G, T)``:
        ``sum_{i in g} power[1][i, t] <= group_limit[g]``.  There is deliberately
        **no** ``self.slack`` term on it: ``slack`` exists to keep an ADMM iterate
        inside its box, and the group limit has no ADMM prox (spec D5).
        """
        constraints = super().inequality_constraints(
            power, angle, _, nominal_capacity=nominal_capacity, la=la, envelope=envelope, mask=mask
        )
        if self.group_matrix is not None:
            matrix, limit = self._group_arrays_like(power[1])
            constraints.append(la.matmul(matrix, power[1]) - limit)
        return constraints

    def _group_arrays_like(self, power1):
        """``(group_matrix, group_limit)`` aligned with ``power1``'s dtype/device.

        Load-bearing on the KKT path.  ``PowerNetwork.kkt_vjp_parameters``
        torchifies the *devices* at ``AbstractDevice.DEFAULT_DTYPE`` (float32)
        but the *state* through ``zap.util.torchify`` at float64, so a torched
        ``group_matrix`` is float32 against a float64 ``power[1]``.  Families 0
        and 1 survive that because elementwise torch ops promote; ``matmul``
        does not -- it raises "expected m1 and m2 to have the same dtype" on the
        first ``backward()`` of any planning run that carries a cap.  Casting
        here (rather than pinning a dtype at build time) keeps the device usable
        at whatever precision the caller chose.
        """
        matrix, limit = self.group_matrix, self.group_limit
        if isinstance(power1, torch.Tensor):
            matrix = torch.as_tensor(matrix, dtype=power1.dtype, device=power1.device)
            limit = torch.as_tensor(limit, dtype=power1.dtype, device=power1.device)
        elif isinstance(matrix, torch.Tensor):  # numpy state against a torched device
            matrix = matrix.detach().cpu().numpy()
            limit = limit.detach().cpu().numpy()
        return matrix, limit

    def operation_cost(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        cost = la.sum(la.multiply(self.linear_cost, power[1]))
        if self.quadratic_cost is not None:
            cost += la.sum(la.multiply(self.quadratic_cost, la.square(power[1])))

        return cost

    # ====
    # DIFFERENTIATION
    # ====

    def _equality_matrices(self, equalities, nominal_capacity=None, la=np):
        size = equalities[0].power[0].shape[1]
        T = size // self.num_devices

        eff = np.broadcast_to(np.asarray(self.efficiency), (self.num_devices, T)).ravel()

        equalities[0].power[0] += sp.diags(eff)
        equalities[0].power[1] += sp.eye(size)

        return equalities

    def _inequality_matrices(self, inequalities, nominal_capacity=None, la=np):
        inequalities = super()._inequality_matrices(
            inequalities, nominal_capacity=nominal_capacity, la=la
        )
        if self.group_matrix is None:
            return inequalities

        # Family 2 is `group_matrix @ power[1] - group_limit`, whose Jacobian in
        # the raveled power[1] is `kron(group_matrix, I_T)`: both the constraint
        # block (G, T) and the variable block (N, T) ravel row-major, so
        # constraint row `g*T + t` meets variable column `i*T + t` exactly.
        size = inequalities[0].power[1].shape[1]
        T = size // self.num_devices
        dense = self.group_matrix
        if isinstance(dense, torch.Tensor):
            dense = dense.detach().cpu().numpy()
        # float64 regardless of the device's dtype: the entries are exactly 0/1,
        # and a float32 block would silently downcast the KKT Jacobian.
        matrix = sp.csr_matrix(np.asarray(dense, dtype=np.float64))
        inequalities[2].power[1] += sp.kron(matrix, sp.eye(T))

        return inequalities

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
        power_weights=None,
        angle_weights=None,
    ):
        assert angle is None
        if self.group_matrix is not None:
            raise NotImplementedError(
                "DirectedLine.admm_prox_update cannot enforce an aggregate group limit "
                "(`system.import_limit.mw` in ch3): the group couples the rows, so the "
                "closed form below is wrong. Use an LP dispatch layer "
                "(import-limit spec D5)."
            )
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity)

        quadratic_cost = 0.0 if self.quadratic_cost is None else self.quadratic_cost
        pmax = torch.multiply(self.max_power, nominal_capacity) + self.slack
        pmin = torch.multiply(self.min_power, nominal_capacity) - self.slack
        eta = self.efficiency

        # Problem is
        #     min_p    a p1^2 + b p1
        #              + (rho/2) ||p0 - z0||^2 + (rho/2) ||p1 - z1||^2
        #              + {p1 box constraints} + {p1 + eta p0 = 0}
        #
        # Substituting p0 = -p1 / eta removes the equality constraint. The
        # derivative in p1 is
        #     2 a p1 + b + (rho / eta) (p1 / eta + z0) + rho (p1 - z1),
        # which is solved by
        #     p1 = (rho (z1 - z0 / eta) - b) / (2 a + rho (1 + 1 / eta^2)).
        num = rho_power * (power[1] - torch.divide(power[0], eta)) - self.linear_cost
        denom = 2 * quadratic_cost + rho_power * (1.0 + torch.divide(1.0, eta**2))

        p1 = torch.divide(num, denom)
        p1 = torch.clip(p1, pmin, pmax)
        p0 = -torch.divide(p1, eta)

        return [p0, p1], None, None

    # ====
    # SCALING
    # ====

    def scale_power(self, scale):
        super().scale_power(scale)
        # Load-bearing: `wy_store.load_system` calls `scale_power(power_unit)` on
        # every device, so an unscaled MW cap would be `power_unit` times too
        # loose (or tight) on any run with `system.power_unit != 1`.
        if self.group_limit is not None:
            self.group_limit = self.group_limit / scale
