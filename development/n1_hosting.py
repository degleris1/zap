"""Preventive N-1 hosting-capacity LP (LODF-compact).

Computes the maximum firm data-center (DC) load that a candidate set of buses
can host such that a *secure base-case generation dispatch exists*: base-case
and every single-line-outage flow stay within thermal limits, all non-DC load
is served, and generation respects its limits -- enforced jointly across a
panel of stressed snapshots that share one placement.

This is the "feasibility / transfer-capacity" definition (no fuel-cost model).
Generation is preventive: co-optimized in the base case but held fixed across
contingencies; post-contingency flows are the closed-form LODF map of the base
flows, so the whole problem is a single LP in the nodal injections.

    maximize    sum_b d_b
    s.t.  per hour t:
            1^T p^t = 0           (per connected component)
            0 <= g^t <= gbar^t
            |f^t_l| <= F_bar_l                              (base, all l)
            |f^t_l + LODF[l,k] f^t_k| <= F_bar_l            (N-1, l in M, k in K)
          0 <= d_b <= site_cap ;  sum_b d_b <= budget
    with  p^t = A_g g^t - A_load l^t - A_C (lf^t d) - A_dc h^t (+ A_batt s^t)
          f^t = PTDF p^t

Full N-1 across a panel is up to ``L^2 * T`` rows, so by default the N-1 set is
discovered by lazy constraint generation (solve, find violated outages by the
explicit LODF map, add them, resolve) and the final active set -- the binding
"umbrella" contingencies -- is reported.
"""

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from zap.contingency.ptdf import (
    find_ac_line,
    build_signed_incidence,
    branch_susceptance,
    thermal_limits,
    connected_components,
    build_ptdf,
)
from zap.contingency.lodf import build_phi, build_lodf, post_contingency_flows


@dataclass
class ContingencySet:
    """The monitored lines, the (non-radial) outaged lines, and the binding set."""

    monitored: np.ndarray
    outaged: np.ndarray
    radial: np.ndarray
    binding: list = field(default_factory=list)  # (t, l, k) tuples that were active


@dataclass
class HostingResult:
    H: float  # total hosted firm DC capacity (sum of d), in network power units
    placement: np.ndarray  # per-candidate-bus DC nameplate
    candidate_buses: np.ndarray
    status: str
    generation: list = field(default_factory=list)  # g^t per hour
    duals: dict = field(default_factory=dict)  # line -> dH/dF_bar (shadow price)
    active_contingencies: list = field(default_factory=list)  # (t, l, k)
    n_rounds: int = 0
    total_shed: float = 0.0  # non-DC load shed summed over the panel


def _injector_incidence(device):
    """(N, n_dev) node incidence of a single-terminal injector device."""
    return device.incidence_matrix[0].tocsc()


def _col(x):
    a = np.asarray(x, dtype=float)
    return a[:, 0] if a.ndim == 2 else a.ravel()


class HostingCapacityProblem:
    """Builds and solves the preventive N-1 hosting-capacity LP.

    Parameters
    ----------
    net : PowerNetwork (provides ``num_nodes``).
    snapshots : list of device-lists, one per panel hour, each a single-period
        (T=1) network (generators, loads, AC line, optional DC lines/batteries).
        The AC-line topology/susceptance/limits must be identical across hours
        (only generator availability and loads vary).
    candidate_buses : array of node indices where DC may be placed.
    load_factors : optional list of per-hour DC load factors ``lf^t`` (scalar or
        per-candidate vector). Defaults to 1.0 (nameplate = draw).
    must_serve_load : if True (default) non-DC load is firm (fixed at full load);
        if False it may be shed in [-load, 0].
    include_batteries : treat batteries as free injections within +/- power
        capacity at each snapshot (optimistic single-period flexibility).
    include_dc_lines : treat DC lines as controllable injections within their
        capacity, shared across contingencies (preventive).
    eps_radial : threshold on ``1 - Phi[k,k]`` for flagging radial/islanding
        outages (excluded from the outaged set).
    """

    def __init__(
        self,
        net,
        snapshots,
        candidate_buses,
        load_factors=None,
        must_serve_load=True,
        include_batteries=True,
        include_dc_lines=True,
        eps_radial=1e-6,
        eps_lodf=1e-4,
    ):
        from zap.devices.injector import Generator, Load
        from zap.devices.store import Battery
        from zap.devices.transporter import DCLine

        self.net = net
        self.N = net.num_nodes
        self.snapshots = list(snapshots)
        self.T = len(self.snapshots)
        self.candidate_buses = np.asarray(candidate_buses, dtype=int)
        self.nC = len(self.candidate_buses)
        self.must_serve_load = must_serve_load
        self.include_batteries = include_batteries
        self.include_dc_lines = include_dc_lines
        self.eps_lodf = eps_lodf

        if load_factors is None:
            load_factors = [1.0] * self.T
        self.load_factors = load_factors

        # --- Static AC topology / distribution factors (from the first hour) ---
        ac_idx, ac = find_ac_line(self.snapshots[0])
        self.ac_index = ac_idx
        self.A = build_signed_incidence(ac, self.N)  # (N, L)
        self.b = branch_susceptance(ac)  # (L,)
        self.F_bar = thermal_limits(ac)  # (L,)
        self.L = self.A.shape[1]

        self.PTDF = build_ptdf(self.A, self.b)  # (L, N)
        self.Phi = build_phi(self.PTDF, self.A)  # (L, L)
        self.lodf, self.is_radial = build_lodf(self.Phi, eps_radial=eps_radial)
        self.outaged = np.where(~self.is_radial)[0]
        self.monitored = np.arange(self.L)

        n_comp, labels = connected_components(self.A, self.b)
        self.components = [np.where(labels == c)[0] for c in range(n_comp)]

        # Candidate incidence (N, nC)
        rows = self.candidate_buses
        cols = np.arange(self.nC)
        self.A_C = sp.csc_matrix(
            (np.ones(self.nC), (rows, cols)), shape=(self.N, self.nC)
        )

        # Device classes for type dispatch
        self._Generator, self._Load = Generator, Load
        self._Battery, self._DCLine = Battery, DCLine

        self._built = False

    # ------------------------------------------------------------------ build
    def _build_injection(self, devices, lf):
        """Return (p_expr, var_list, const_bounds) for one snapshot.

        p_expr : cvxpy expression (N,) net nodal injection (excluding the d term,
            which is added by the caller so it is shared across hours).
        """
        p = cp.Constant(np.zeros(self.N))
        gen_var = None
        extra = []
        constraints = []

        for i, d in enumerate(devices):
            if i == self.ac_index:
                continue  # AC lines enter via PTDF, not as injections
            if isinstance(d, self._Generator):
                Ag = _injector_incidence(d)
                gmax = _col(d.max_power) * _col(d.nominal_capacity)
                g = cp.Variable(d.num_devices, nonneg=True)
                constraints += [g <= gmax]
                p = p + Ag @ g
                gen_var = g
            elif isinstance(d, self._Load):
                Al = _injector_incidence(d)
                load = _col(d.load) * _col(d.nominal_capacity)
                if self.must_serve_load:
                    p = p - Al @ load  # firm: full load withdrawn
                else:
                    shed = cp.Variable(d.num_devices, nonneg=True)
                    constraints += [shed <= load]
                    p = p - Al @ (load - shed)
                    extra.append(("shed", shed))
            elif isinstance(d, self._DCLine) and self.include_dc_lines:
                Adc = (d.incidence_matrix[0] - d.incidence_matrix[1]).tocsc()
                hmax = _col(d.max_power) * _col(d.nominal_capacity)
                h = cp.Variable(d.num_devices)
                constraints += [h <= hmax, h >= -hmax]
                p = p - Adc @ h  # net injection of a transporter is -A_signed @ flow
                extra.append(("dcline", h))
            elif isinstance(d, self._Battery) and self.include_batteries:
                Ab = _injector_incidence(d)
                pmax = _col(d.power_capacity)
                s = cp.Variable(d.num_devices)
                constraints += [s <= pmax, s >= -pmax]
                p = p + Ab @ s
                extra.append(("battery", s))
            # other device types: ignored (no nodal contribution modeled)

        return p, gen_var, extra, constraints

    def build(self, site_cap=None, budget=None):
        """Construct the base LP (no N-1 rows yet; those are added on solve)."""
        self.d = cp.Variable(self.nC, nonneg=True)
        site_constraints = []
        if site_cap is not None:
            site_constraints.append(self.d <= site_cap)
        if budget is not None:
            site_constraints.append(cp.sum(self.d) <= budget)

        self.flow_expr = []  # f^t cvxpy expression per hour
        self.theta_vars = []
        self.gen_vars = []
        self.extra_vars = []
        self.base_thermal_cons = []  # (con_up, con_lo) per hour, for shadow prices
        base_constraints = list(site_constraints)

        # Sparse DC formulation (zap's native B-theta), NOT the dense PTDF map:
        # flow = b * (A^T theta), nodal balance A @ flow == p. PTDF is only used to
        # derive the LODF rows added on top for N-1. One angle reference per island.
        AT = self.A.T.tocsr()
        refs = [int(comp[0]) for comp in self.components]
        self.refs = refs

        for t, devices in enumerate(self.snapshots):
            lf = self.load_factors[t]
            p_base, gen_var, extra, cons = self._build_injection(devices, lf)
            # subtract the shared DC withdrawal (lf^t * d) at candidate buses
            p = p_base - self.A_C @ (lf * self.d)

            theta = cp.Variable(self.N)
            flow = cp.Variable(self.L)              # explicit flow var: cheap to index in N-1 rows
            cons.append(flow == cp.multiply(self.b, AT @ theta))  # DC flow definition
            cons.append(self.A @ flow == p)         # nodal balance => per-island balance
            cons.append(theta[refs] == 0)           # one angle reference per island

            self.flow_expr.append(flow)
            self.theta_vars.append(theta)
            self.gen_vars.append(gen_var)
            self.extra_vars.append(extra)

            # base-case thermal limits (all lines)
            c_up = flow <= self.F_bar
            c_lo = -flow <= self.F_bar
            self.base_thermal_cons.append((c_up, c_lo))
            cons += [c_up, c_lo]

            base_constraints += cons

        # non-DC load-shed variables (present only when must_serve_load=False)
        self.shed_vars = [v for extra in self.extra_vars for (kind, v) in extra if kind == "shed"]
        self.base_constraints = base_constraints
        self._built = True
        return self

    # ------------------------------------------------------------------ solve
    def _violations(self, tol):
        """Find post-contingency overloads at the current solution.

        Returns a list of (t, l, k, viol) with viol = |f_l^(k)| - F_bar_l > tol,
        using the dense LODF over all non-radial outages (certifies the full
        N-1 set, not just the sparsified subset).
        """
        viols = []
        for t in range(self.T):
            fval = np.asarray(self.flow_expr[t].value).ravel()
            # post-contingency flows for every monitored line x non-radial outage
            fpc = post_contingency_flows(fval, self.lodf, outage_set=self.outaged)  # (L, K)
            over = np.abs(fpc) - self.F_bar[:, None]
            ls, ks = np.where(over > tol)
            for li, ki in zip(ls, ks):
                viols.append((t, int(li), int(self.outaged[ki]), float(over[li, ki])))
        return viols

    def _all_n1_constraints(self):
        """Build every significant N-1 row upfront (sparsified by |LODF| > eps)."""
        from zap.contingency.lodf import sparsify_lodf

        sp_lodf = sparsify_lodf(self.lodf, self.eps_lodf).tocoo()
        outaged = set(int(k) for k in self.outaged)
        cons, reg, active = [], [], []
        for mon, k, v in zip(sp_lodf.row, sp_lodf.col, sp_lodf.data):
            mon, k = int(mon), int(k)
            if k not in outaged or mon == k:
                continue  # diagonal: outaged line carries no flow
            for t in range(self.T):
                f = self.flow_expr[t]
                expr = f[mon] + v * f[k]
                cons += [expr <= self.F_bar[mon], -expr <= self.F_bar[mon]]
                reg += [(t, mon, k, "+"), (t, mon, k, "-")]
                active.append((t, mon, k))
        return cons, reg, active

    def solve(
        self,
        solver=None,
        enforce_n1=True,
        lazy=True,
        tol=1e-5,
        max_rounds=40,
        max_add_per_round=2000,
        shed_penalty=1e3,
        verbose=False,
        return_shadow_prices=True,
    ):
        """Solve the hosting-capacity LP.

        Parameters
        ----------
        enforce_n1 : if True, enforce single-line-outage security (the N-1-aware
            problem); if False, only base-case limits (the N-1-*blind* problem,
            used for the cost-of-ignoring-N-1 contrast).
        lazy : when ``enforce_n1`` and True, discover the binding contingencies by
            constraint generation (solve, find overloads, add, repeat); when
            False, add every sparsified N-1 row upfront in one shot.
        """
        if not self._built:
            raise RuntimeError("call build(...) before solve(...)")
        if solver is None:
            installed = cp.installed_solvers()
            # Prefer a solver with clean LP duals for shadow prices; fall back to
            # CLARABEL (always present, used by zap's dispatch).
            for cand in ("HIGHS", "CLARABEL", "GLOP"):
                if cand in installed:
                    solver = getattr(cp, cand)
                    break

        n1_constraints = []  # cvxpy constraints
        n1_registry = []  # parallel (t, l, k, side) for each row (2 per added (t,l,k))
        active = []  # (t, l, k)
        rounds = 0
        prob = None

        if enforce_n1 and not lazy:
            n1_constraints, n1_registry, active = self._all_n1_constraints()

        # firm DC is maximized; any non-DC load shed is penalized so DC never
        # cannibalizes load -- the LP sheds only the unavoidable baseline.
        objective = cp.sum(self.d)
        if self.shed_vars:
            objective = objective - shed_penalty * cp.sum([cp.sum(s) for s in self.shed_vars])

        while True:
            rounds += 1
            constraints = self.base_constraints + n1_constraints
            prob = cp.Problem(cp.Maximize(objective), constraints)
            prob.solve(solver=solver, verbose=verbose)

            if prob.status not in ("optimal", "optimal_inaccurate"):
                return HostingResult(
                    H=float("nan"),
                    placement=np.full(self.nC, np.nan),
                    candidate_buses=self.candidate_buses,
                    status=prob.status,
                    n_rounds=rounds,
                )

            if not (enforce_n1 and lazy):
                break

            viols = self._violations(tol)
            if not viols:
                break
            # add the most-violated first
            viols.sort(key=lambda r: -r[3])
            added = 0
            seen = set(active)
            for (t, mon, k, _) in viols:
                if (t, mon, k) in seen:
                    continue
                f = self.flow_expr[t]
                expr = f[mon] + self.lodf[mon, k] * f[k]
                c_up = expr <= self.F_bar[mon]
                c_lo = -expr <= self.F_bar[mon]
                n1_constraints += [c_up, c_lo]
                n1_registry += [(t, mon, k, "+"), (t, mon, k, "-")]
                active.append((t, mon, k))
                seen.add((t, mon, k))
                added += 1
                if added >= max_add_per_round:
                    break
            if verbose:
                print(f"[hosting] round {rounds}: +{added} N-1 rows, |active|={len(active)}")
            if rounds >= max_rounds:
                break

        # ---- shadow prices: dH/dF_bar_l = sum of duals on rows monitoring l ----
        duals = {}
        if return_shadow_prices:
            duals = self._shadow_prices(constraints, n1_constraints, n1_registry, active)

        gen = [None if g is None else np.asarray(g.value).ravel() for g in self.gen_vars]
        total_shed = float(sum(float(np.asarray(s.value).sum()) for s in self.shed_vars)) \
            if self.shed_vars else 0.0
        return HostingResult(
            H=float(cp.sum(self.d).value),
            placement=np.asarray(self.d.value).ravel(),
            candidate_buses=self.candidate_buses,
            status=prob.status,
            generation=gen,
            duals=duals,
            active_contingencies=active,
            n_rounds=rounds,
            total_shed=total_shed,
        )

    def _shadow_prices(self, all_constraints, n1_constraints, n1_registry, active):
        """Aggregate constraint duals into per-line dH/dF_bar (>= 0).

        F_bar_l appears as the RHS of every constraint that *monitors* line l
        (base ``|f_l| <= F_bar_l`` and N-1 ``|f_l + LODF f_k| <= F_bar_l``); the
        sum of those rows' duals is the marginal hosting capacity per unit of
        added thermal limit on l.
        """
        dH = np.zeros(self.L)

        # N-1 rows: one scalar dual each, attributed to the monitored line.
        for (t, mon, k, side), con in zip(n1_registry, n1_constraints):
            dv = con.dual_value
            if dv is not None:
                dH[mon] += abs(float(np.asarray(dv).ravel()[0]))

        # Base thermal rows: length-L vector duals.
        for (c_up, c_lo) in self.base_thermal_cons:
            for con in (c_up, c_lo):
                dv = con.dual_value
                if dv is not None:
                    dH += np.abs(np.asarray(dv).ravel())

        return {int(mon): float(dH[mon]) for mon in range(self.L) if dH[mon] > 0}

    def binding_set(self, result: HostingResult) -> ContingencySet:
        return ContingencySet(
            monitored=self.monitored,
            outaged=self.outaged,
            radial=np.where(self.is_radial)[0],
            binding=result.active_contingencies,
        )
