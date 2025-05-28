from enum import Enum, auto
from typing import Optional, TypedDict

import numpy as np
import scipy.sparse as sp
import torch
from attrs import Factory, define, field
from numpy.typing import NDArray

from .abstract import AbstractDevice, get_time_horizon, make_dynamic


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


class DataCenterConstants:
    """Constants related to data center infrastructure and costs."""

    class RackDensity:
        """Power density per rack type in kW."""

        GPU = 100.0  # H100 / B100 racks
        CPU = 20.0  # enterprise / cloud general purpose
        STORAGE = 10.0  # disk / flash

    class ITCosts:
        """Cost per rack type in USD."""

        GPU = 1_000_000.0  # H100 / B100 racks
        CPU = 200_000.0  # enterprise / cloud general purpose
        STORAGE = 100_000.0  # disk / flash

    class SpaceRequirements:
        """Space planning constants."""

        SQFT_PER_ACRE = 43_560
        FLOOR_SQFT_PER_RACK = 75
        SITE_GROSS_MULTIPLIER = 2.5  # Total site area vs white space ratio

    class CostFactors:
        """Cost breakdown factors for infrastructure."""

        ELECTRICAL = 0.53  # UPS, switchgear, gensets, PDUs
        COOLING = 0.20  # chillers, CRAH/CRAC, pumps, piping
        SHELL = 0.27  # walls, roof, steel, white-space fit-out


class Location(Enum):
    """Data center location types affecting costs."""

    RURAL = auto()
    SUBURBAN = auto()
    URBAN = auto()

    @property
    def base_capex_per_mw(self) -> float:
        """Baseline 2024-25 construction cost (millions $/MW)."""
        return {
            Location.RURAL: 9.5,
            Location.SUBURBAN: 11.7,
            Location.URBAN: 14.0,
        }[self]

    @property
    def land_cost_per_acre(self) -> float:
        """Land acquisition cost per acre."""
        base_price = 244_000
        multiplier = {
            Location.RURAL: 0.4,
            Location.SUBURBAN: 1.0,
            Location.URBAN: 3.0,
        }[self]
        return multiplier * base_price


class CostBreakdown(TypedDict):
    """Detailed cost breakdown for a data center build."""

    total_capex: float
    electrical_infra: float  # UPS, switchgear, gensets, PDUs
    cooling_infra: float  # chillers, CRAH/CRAC, pumps, piping
    building_shell: float  # walls, roof, steel, white-space fit-out, back-of-house
    land: float  # site acquisition
    it_cost: float  # IT hardware (racks, servers, storage)
    capex_per_mw: float  # effective $/MW after all modifiers
    nominal_it_mw: float  # MW of critical IT load (UPS power)


def _scale_factor(mw: float) -> float:
    """Returns cost scaling factor based on data center size."""
    if mw < 5:
        return 1.15  # edge / micro DC premium
    elif mw <= 20:
        return 1.00  # mid-scale "standard"
    else:
        return 0.85


@define(kw_only=True, slots=False)
class DataCenterLoad(AbstractInjector):
    """
    A data center device that can have different load profiles.

    By default, it has a base load that is 20% of its nominal capacity at all times,
    and follows a diurnal pattern that peaks around 1:30pm.

    This device represents a data center load with dynamic capacity planning capabilities,
    allowing for optimization of both capacity and energy consumption.
    """

    class ProfileType(str, Enum):
        DIURNAL = "diurnal"
        CONSTANT = "constant"
        CUSTOM = "custom"

    profile_types: list[ProfileType] = field(
        default=Factory(lambda: [DataCenterLoad.ProfileType.DIURNAL])
    )
    peak_hours: NDArray = field(default=Factory(lambda: [13.5]))
    base_load_fractions: NDArray = field(default=Factory(lambda: [0.2]))
    time_resolution_hours: float = field(default=0.25)
    profiles: Optional[NDArray] = field(default=None)
    min_power: NDArray = field(init=False)
    max_power: NDArray = field(init=False)
    capital_cost: Optional[NDArray] = field(default=None)
    emission_rates: Optional[NDArray] = field(default=None)
    gpu_racks: NDArray = field(converter=make_dynamic)
    cpu_racks: NDArray = field(converter=make_dynamic)
    storage_racks: NDArray = field(converter=make_dynamic)
    linear_cost: NDArray = field(converter=make_dynamic)
    quadratic_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    settime_horizon: float = field(default=3.0)
    nominal_capacity: NDArray = field(init=False)
    locations: Optional[list[Location]] = field(
        default=Factory(lambda: [Location.SUBURBAN]),
        converter=lambda locs: [locs] if isinstance(locs, Location) else locs,
    )

    @property
    def _nominal_capacity(self) -> NDArray:
        """MW of critical IT load derived from rack counts."""
        kw = (
            self.gpu_racks * DataCenterConstants.RackDensity.GPU
            + self.cpu_racks * DataCenterConstants.RackDensity.CPU
            + self.storage_racks * DataCenterConstants.RackDensity.STORAGE
        )
        return kw / 1000.0

    def __attrs_post_init__(self):
        num_dcs = len(self.cpu_racks)
        if len(self.profile_types) != num_dcs:
            self.profile_types = [self.profile_types[0]] * num_dcs
        if len(self.peak_hours) != num_dcs:
            self.peak_hours = [self.peak_hours[0]] * num_dcs
        if len(self.base_load_fractions) != num_dcs:
            self.base_load_fractions = [self.base_load_fractions[0]] * num_dcs
        if self.profiles is not None and len(self.profiles) != num_dcs:
            self.profiles = [self.profiles[0]] * num_dcs
        if self.capital_cost is not None and len(self.capital_cost) != num_dcs:
            self.capital_cost = [self.capital_cost[0]] * num_dcs
        if self.emission_rates is not None and len(self.emission_rates) != num_dcs:
            self.emission_rates = [self.emission_rates[0]] * num_dcs
        if len(self.linear_cost) != num_dcs:
            self.linear_cost = [self.linear_cost[0]] * num_dcs
        if self.quadratic_cost is not None and len(self.quadratic_cost) != num_dcs:
            self.quadratic_cost = [self.quadratic_cost[0]] * num_dcs
        if self.locations is not None and len(self.locations) != num_dcs:
            self.locations = [self.locations[0]] * num_dcs
        self.nominal_capacity = self._nominal_capacity

        all_profiles = []
        for i in range(num_dcs):
            if self.profiles is not None and self.profiles[i] is not None:
                time_horizon = len(self.profiles[i])
                profile = self._process_custom_profile(
                    self.profiles[i], time_horizon, i
                )
            else:
                time_horizon = self.settime_horizon
                profile = self._create_load_profile(time_horizon, i)
            all_profiles.append(profile)

        self.profile = np.vstack(all_profiles)
        self.min_power = -self.profile
        self.max_power = np.zeros_like(self.profile)

        super_class = super()
        if hasattr(super_class, "__attrs_post_init__"):
            super_class.__attrs_post_init__()

    def _create_diurnal_profile(self, time_horizon: int, dc_idx: int) -> NDArray:
        """
        Generic-dt hardware-aware diurnal profile (storage base, CPU cosine,
        GPU cosine+half-day+AR(1) noise).  Works for any self.time_resolution_hours.
        """
        # ------------------------------------------------------------------
        # 1. Time axis
        # ------------------------------------------------------------------
        dt = float(self.time_resolution_hours)  # hours / sample
        t = np.arange(time_horizon) * dt
        hour = t % 24.0
        peak = float(self.peak_hours[dc_idx])  # desired daily peak hour

        # ------------------------------------------------------------------
        # 2. Rack mix
        # ------------------------------------------------------------------
        P_sto = self.storage_racks[dc_idx] * DataCenterConstants.RackDensity.STORAGE
        P_cpu = self.cpu_racks[dc_idx] * DataCenterConstants.RackDensity.CPU
        P_gpu = self.gpu_racks[dc_idx] * DataCenterConstants.RackDensity.GPU
        P_tot = P_sto + P_cpu + P_gpu
        if P_tot == 0:
            return np.zeros(time_horizon)

        f_sto, f_cpu, f_gpu = P_sto / P_tot, P_cpu / P_tot, P_gpu / P_tot

        # ------------------------------------------------------------------
        # 3. Deterministic components (periods in physical hours)
        # ------------------------------------------------------------------
        phase24 = 2 * np.pi * (hour - peak) / 24.0
        phase12 = 2 * phase24  # 12-h harmonic

        load_storage = np.full(time_horizon, f_sto)
        load_cpu = f_cpu * (0.7 + 0.3 * np.cos(phase24))

        gpu_shape = (
            0.6 * (1 + np.cos(phase24)) / 2  # 24-h
            + 0.4 * (1 + np.cos(phase12)) / 2  # 12-h
        )

        # ------------------------------------------------------------------
        # 4. Correlated noise – time-resolution aware
        # ------------------------------------------------------------------
        rng = np.random.default_rng()
        # variance ∝ dt so *power* stays constant
        noise = rng.normal(scale=0.02 * np.sqrt(dt / 1.0), size=time_horizon)

        # AR(1) coefficient chosen so correlation time ≈ 1 h
        tau = 1.0  # correlation time in hours
        rho = np.exp(-dt / tau)
        for i in range(1, time_horizon):
            noise[i] += rho * noise[i - 1]

        # Smooth over one physical hour (or ≥1 point)
        window_pts = max(1, int(round(1.0 / dt)))
        kernel = np.ones(window_pts) / window_pts
        noise = np.convolve(noise, kernel, mode="same")

        load_gpu = f_gpu * np.clip(gpu_shape + noise, 0.0, 1.2)

        # ------------------------------------------------------------------
        # 5. Aggregate
        # ------------------------------------------------------------------
        profile = load_storage + load_cpu + load_gpu
        profile = np.clip(profile, f_sto, 1.0)
        return profile

    def _create_constant_profile(self, time_horizon: int, dc_idx: int) -> NDArray:
        """Create a constant load profile for a specific data center."""
        return np.ones(time_horizon) * 0.6

    def _process_custom_profile(
        self, profile: NDArray, time_horizon: int, dc_idx: int
    ) -> NDArray:
        """Process and validate a custom profile for a specific data center."""
        if len(profile.shape) == 1:
            profile = profile.reshape(1, -1)
        return profile

    def _create_load_profile(self, time_horizon: int, dc_idx: int) -> NDArray:
        """Create a load profile for a specific data center."""
        if self.profile_types[dc_idx] == self.ProfileType.CONSTANT:
            return self._create_constant_profile(time_horizon, dc_idx)
        else:
            return self._create_diurnal_profile(time_horizon, dc_idx)

    def get_emissions(self, power, nominal_capacity=None, la=np):
        if self.emission_rates is None:
            return 0.0

        total_emissions = 0.0
        for i in range(len(self.emission_rates)):
            total_emissions += la.sum(la.multiply(self.emission_rates[i], power[0][i]))

        return total_emissions

    def get_investment_cost(self, nominal_capacity=None, la=np):
        """Return total investment cost for data center devices.

        Numbers are 2024-Q4 US dollars.
        """
        if self.capital_cost is None or nominal_capacity is None:
            return 0.0

        total_cost = 0.0

        for i in range(self.num_devices):
            gpu_racks = self.gpu_racks[i]
            cpu_racks = self.cpu_racks[i]
            storage_racks = self.storage_racks[i]
            location = self.locations[i]

            it_kw = (
                gpu_racks * DataCenterConstants.RackDensity.GPU
                + cpu_racks * DataCenterConstants.RackDensity.CPU
                + storage_racks * DataCenterConstants.RackDensity.STORAGE
            )
            it_mw = it_kw / 1_000.0
            it_cost = (
                gpu_racks * DataCenterConstants.ITCosts.GPU
                + cpu_racks * DataCenterConstants.ITCosts.CPU
                + storage_racks * DataCenterConstants.ITCosts.STORAGE
            )
            capex_mw_nominal = location.base_capex_per_mw * _scale_factor(it_mw)
            core_capex = capex_mw_nominal * it_mw
            racks_total = gpu_racks + cpu_racks + storage_racks
            white_space_sqft = (
                racks_total * DataCenterConstants.SpaceRequirements.FLOOR_SQFT_PER_RACK
            )
            site_sqft = (
                white_space_sqft
                * DataCenterConstants.SpaceRequirements.SITE_GROSS_MULTIPLIER
            )
            site_acres = site_sqft / DataCenterConstants.SpaceRequirements.SQFT_PER_ACRE
            land_cost = site_acres * location.land_cost_per_acre

            device_total = core_capex + land_cost + it_cost
            total_cost += la.sum(la.multiply(self.capital_cost[i], device_total))

        return total_cost


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
