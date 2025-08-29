import math
from enum import Enum
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from attrs import Factory, define, field
from numpy.typing import NDArray

from .abstract import AbstractDevice, get_time_horizon, make_dynamic

# Load generation archetypes (from the new synthetic generator)
ARCHETYPES = {
    "interactive": dict(L=0.65, A=0.20, sigma_dev=0.05, sigma_cmn=0.03, rho=0.50),
    "batch": dict(L=0.45, A=0.15, sigma_dev=0.04, sigma_cmn=0.05, rho=0.55),
    "ai-train": dict(L=0.90, A=0.05, sigma_dev=0.08, sigma_cmn=0.07, rho=0.60),
    "ai-infer": dict(L=0.35, A=0.10, sigma_dev=0.07, sigma_cmn=0.04, rho=0.40),
    "hpc": dict(L=0.88, A=0.08, sigma_dev=0.03, sigma_cmn=0.02, rho=0.30),
    "colocation": dict(L=0.75, A=0.05, sigma_dev=0.02, sigma_cmn=0.02, rho=0.20),
    # Legacy compatibility mappings
    "diurnal": dict(
        L=0.65, A=0.20, sigma_dev=0.05, sigma_cmn=0.03, rho=0.50
    ),  # maps to interactive
    "constant": dict(
        L=0.75, A=0.00, sigma_dev=0.01, sigma_cmn=0.01, rho=0.10
    ),  # minimal variation
}

SEED = 42  # reproducibility


def _generate_synthetic_profile(
    workload_type: str,
    time_horizon: int,
    time_resolution_hours: float,
    *,
    L: Optional[float] = None,
    A: Optional[float] = None,
    sigma_dev: Optional[float] = None,
    sigma_cmn: Optional[float] = None,
    rho: Optional[float] = None,
    weekly_amp: float = 0.0,
    headroom: float = 0.02,
    min_lf: float = 0.02,
    seed: int = SEED,
) -> NDArray:
    """
    Returns a per-unit (0..1) load factor time series shaped by:
      - diurnal sinusoid with amplitude A around baseline L
      - AR(1) state with rho memory
      - common & idiosyncratic Gaussian noise (sigma_cmn, sigma_dev)
      - optional weekly modulation (lower weekends / patterning)
    """
    if workload_type not in ARCHETYPES:
        raise ValueError(f"Unknown workload_type {workload_type!r}")

    knobs = ARCHETYPES[workload_type].copy()
    if L is not None:
        knobs["L"] = L
    if A is not None:
        knobs["A"] = A
    if sigma_dev is not None:
        knobs["sigma_dev"] = sigma_dev
    if sigma_cmn is not None:
        knobs["sigma_cmn"] = sigma_cmn
    if rho is not None:
        knobs["rho"] = rho

    rng = np.random.default_rng(seed)
    out = np.empty(time_horizon, dtype=float)
    ar_state = 0.0
    two_pi = 2.0 * math.pi

    for t in range(time_horizon):
        hour = t * time_resolution_hours

        diurnal = knobs["L"] * (1.0 + knobs["A"] * math.sin(two_pi * hour / 24.0))
        weekly = 1.0 + weekly_amp * math.sin(two_pi * hour / (24.0 * 7.0))

        eps_c = rng.normal(0.0, knobs["sigma_cmn"])
        eps_d = rng.normal(0.0, knobs["sigma_dev"])
        ar_state = (
            knobs["rho"] * ar_state
            + math.sqrt(max(1.0 - knobs["rho"] ** 2, 0.0)) * eps_c
        )

        lf = diurnal * weekly * (1.0 + eps_c + eps_d + ar_state)

        lf = max(min_lf, min(1.0 - headroom, lf))
        out[t] = lf

    return out


@define(kw_only=True, slots=False)
class AbstractInjector(AbstractDevice):
    """A single-node device that may deposit or withdraw power from the network. Abstract type that
    should not be instantiated but contains shared behavior among all subclasses."""

    num_nodes: int
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
                0.0 * linear_cost
                if self.quadratic_cost is None
                else self.quadratic_cost
            )
            pmax = torch.multiply(max_power, nominal_capacity)
            pmin = torch.multiply(min_power, nominal_capacity)
            self.admm_data = (quadratic_cost, pmax, pmin)
            self.has_changed = False

        quadratic_cost, pmax, pmin = self.admm_data

        return _admm_prox_update(
            power, rho_power, linear_cost, quadratic_cost, pmin, pmax
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
    """An Injector that can only deposit power."""

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

    # TODO - Add dimension checks

    @property
    def min_power(self):
        return 0.0 * self.dynamic_capacity

    @property
    def max_power(self):
        return self.dynamic_capacity

    def scale_power(self, scale):
        if self.min_nominal_capacity is not None:
            self.min_nominal_capacity /= scale
        if self.max_nominal_capacity is not None:
            self.max_nominal_capacity /= scale
        return super().scale_power(scale)

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
        dev = super().sample_time(time_periods, original_time_horizon)

        if dev.dynamic_capacity.shape[1] > 1:
            dev.dynamic_capacity = dev.dynamic_capacity[:, time_periods]

        return dev


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


@define(kw_only=True, slots=False)
class DataCenterLoad(AbstractInjector):
    """
    Data center device with synthetic *site-level* load profiles driven by nominal_capacity only.
    Produces per-unit load factors (0..1); actual power bounds come from multiplying by nominal_capacity.
    """

    class ProfileType(str, Enum):
        # Keep legacy names for drop-in compatibility
        INTERACTIVE = "interactive"
        BATCH = "batch"
        AI_TRAIN = "ai-train"
        AI_INFER = "ai-infer"
        HPC = "hpc"
        COLOCATION = "colocation"
        DIURNAL = "diurnal"
        CONSTANT = "constant"

    # --- Profile controls
    profile_types: list[ProfileType] = field(
        default=Factory(lambda: [DataCenterLoad.ProfileType.INTERACTIVE])
    )
    time_resolution_hours: float = field(default=0.25)
    settime_horizon: float = field(default=24.0)

    # Optional global knobs for all DCs (can be overridden per-DC via per_dc_overrides)
    weekly_amp: float = field(default=0.0)
    headroom: float = field(default=0.02)
    min_lf: float = field(default=0.02)
    base_seed: int = field(default=SEED)

    # Per-DC overrides: dict(index -> dict of L/A/sigma_dev/sigma_cmn/rho)
    per_dc_overrides: Optional[dict] = field(default=None)

    # Standard device parameters
    profiles: Optional[NDArray] = field(default=None)  # user-supplied per-unit arrays
    min_power: NDArray = field(init=False)
    max_power: NDArray = field(init=False)
    capital_cost: Optional[NDArray] = field(default=None)
    emission_rates: Optional[NDArray] = field(default=None)
    nominal_capacity: NDArray
    linear_cost: NDArray = field(converter=make_dynamic)
    quadratic_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    min_nominal_capacity: Optional[NDArray] = field(
        default=None, converter=make_dynamic
    )
    max_nominal_capacity: Optional[NDArray] = field(
        default=None, converter=make_dynamic
    )

    def __attrs_post_init__(self):
        num_dcs = len(self.nominal_capacity)

        # Length checks / broadcast
        if len(self.profile_types) != num_dcs:
            self.profile_types = [self.profile_types[0]] * num_dcs
        if self.capital_cost is not None and len(self.capital_cost) != num_dcs:
            self.capital_cost = [self.capital_cost[0]] * num_dcs
        if self.emission_rates is not None and len(self.emission_rates) != num_dcs:
            self.emission_rates = [self.emission_rates[0]] * num_dcs
        if len(self.linear_cost) != num_dcs:
            self.linear_cost = [self.linear_cost[0]] * num_dcs
        if self.quadratic_cost is not None and len(self.quadratic_cost) != num_dcs:
            self.quadratic_cost = [self.quadratic_cost[0]] * num_dcs

        T = int(round(self.settime_horizon / self.time_resolution_hours))
        all_profiles = []

        for i in range(num_dcs):
            if self.profiles is not None and self.profiles[i] is not None:
                profile = self._process_custom_profile(self.profiles[i])
            else:
                pt = self.profile_types[i].value
                overrides = (self.per_dc_overrides or {}).get(i, {})
                profile = _generate_synthetic_profile(
                    workload_type=pt,
                    time_horizon=T,
                    time_resolution_hours=self.time_resolution_hours,
                    L=overrides.get("L"),
                    A=overrides.get("A"),
                    sigma_dev=overrides.get("sigma_dev"),
                    sigma_cmn=overrides.get("sigma_cmn"),
                    rho=overrides.get("rho"),
                    weekly_amp=self.weekly_amp,
                    headroom=self.headroom,
                    min_lf=self.min_lf,
                    seed=self.base_seed + 9973 * i,
                )
            all_profiles.append(profile)

        self.profile = np.vstack(all_profiles)  # shape (n_dc, T), per-unit
        self.min_power = -self.profile  # withdraw
        self.max_power = np.zeros_like(self.profile)

        super_class = super()
        if hasattr(super_class, "__attrs_post_init__"):
            super_class.__attrs_post_init__()

    @staticmethod
    def _process_custom_profile(profile: NDArray) -> NDArray:
        if profile.ndim == 1:
            return profile.reshape(-1)
        if profile.shape[0] == 1:
            return profile[0]
        raise ValueError("Custom profile must be 1D or shape (1, T).")

    def get_emissions(self, power, nominal_capacity=None, la=np):
        if self.emission_rates is None:
            return 0.0
        total_emissions = 0.0
        for i in range(len(self.emission_rates)):
            total_emissions += la.sum(la.multiply(self.emission_rates[i], power[0][i]))
        return total_emissions

    def get_investment_cost(self, nominal_capacity=None, la=np):
        if self.capital_cost is None or nominal_capacity is None:
            return 0.0
        return la.sum(
            la.multiply(self.capital_cost, (nominal_capacity - self.nominal_capacity))
        )

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)
        # No additional sampling needed for synthetic profiles as they're generated on demand
        return dev


# @torch.jit.script
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
