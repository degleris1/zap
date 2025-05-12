from typing import Optional

import numpy as np
from attrs import Factory, define, field
from numpy.typing import NDArray

from .abstract import (
    AbstractDevice,
    make_dynamic,
)


@define(kw_only=True, slots=False)
class DataCenter(AbstractDevice):
    """
    A data center device that has a base load that is 20% of its nominal capacity at all times,
    and follows a diurnal pattern that peaks around 1:30pm.

    This device represents a data center load with dynamic capacity planning capabilities,
    allowing for optimization of both capacity and energy consumption.
    """

    peak_hour: float = field(default=13.5)
    base_load_fraction: float = field(default=0.2)
    time_resolution_hours: float = field(default=0.25)
    diurnal_profile: NDArray = field(
        init=False, factory=Factory(lambda: np.zeros((0, 0)))
    )
    min_power: NDArray = field(init=False)
    max_power: NDArray = field(init=False)
    linear_cost: NDArray = field(converter=make_dynamic)
    quadratic_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    capital_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    emission_rates: Optional[NDArray] = field(default=None, converter=make_dynamic)
    nominal_capacity: NDArray = field(
        init=False, factory=Factory(lambda: np.zeros((0, 0)))
    )

    def __attrs_post_init__(self):
        time_horizon = (
            self.nominal_capacity.shape[1]
            if len(self.nominal_capacity.shape) > 1
            else 1
        )

        self.diurnal_profile = self._create_diurnal_profile(time_horizon)

        self.min_power = -min(self.diurnal_profile, 0.0)
        self.max_power = np.zeros_like(self.diurnal_profile)

        super_class = super()
        if hasattr(super_class, "__attrs_post_init__"):
            super_class.__attrs_post_init__()

    def _create_diurnal_profile(self, time_horizon: int) -> NDArray:
        """
        Create a diurnal load profile with specified base load, peaking at the given hour.

        Returns:
            NDArray with the profile, with shape (num_devices, time_horizon)
        """
        time_steps = np.arange(time_horizon) * self.time_resolution_hours
        hours_of_day = time_steps % 24
        hour_distance = ((hours_of_day - self.peak_hour) % 24) * (2 * np.pi / 24)
        amplitude = 1.0 - self.base_load_fraction
        profile = self.base_load_fraction + amplitude * np.cos(hour_distance) ** 2
        num_devices = self.nominal_capacity.shape[0]
        profile_expanded = np.tile(profile, (num_devices, 1))

        if profile_expanded.shape[1] != time_horizon:
            if time_horizon == 1:
                profile_expanded = np.mean(profile_expanded, axis=1, keepdims=True)
            else:
                indices = np.linspace(0, profile.size - 1, time_horizon).astype(int)
                profile_expanded = profile_expanded[:, indices]

        return profile_expanded

    def sample_time(self, time_periods, original_time_horizon):
        """
        Override sample_time to ensure the diurnal pattern is properly sampled
        when time periods are selected.
        """
        dev = super().sample_time(time_periods, original_time_horizon)
        time_horizon = (
            dev.nominal_capacity.shape[1] if len(dev.nominal_capacity.shape) > 1 else 1
        )
        dev.diurnal_profile = dev._create_diurnal_profile(time_horizon)

        # Update min_power and max_power based on the new diurnal profile
        dev.min_power = -dev.diurnal_profile
        dev.max_power = np.zeros_like(dev.diurnal_profile)

        return dev

    def get_investment_cost(self, nominal_capacity=None, la=np):
        return 0

    def get_emissions(self, power, nominal_capacity=None, la=np):
        return 0
