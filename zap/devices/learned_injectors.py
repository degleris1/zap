"""Neural‑prox variants of Generator and Load."""

from __future__ import annotations

import torch
import itertools
from torch import nn

from .injector import Generator, Load


def _build_prox_mlp(
    *,
    input_dim: int,
    output_dim: int = 1,
    hidden_width: int = 32,
    hidden_depth: int = 1,
    use_batch_norm: bool = True,
    use_output_layer: bool = True,
) -> nn.Module:
    """Return a fully‑connected network for computing prox operators."""
    layers: list[nn.Module] = []

    # Input layer
    layers.append(nn.Linear(input_dim, hidden_width))
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(hidden_width))
    layers.append(nn.ReLU())

    # Hidden stacks
    hidden_stacks = [
        [
            nn.Linear(hidden_width, hidden_width),
            nn.BatchNorm1d(hidden_width) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
        ]
        for _ in range(max(hidden_depth - 1, 0))
    ]
    layers += list(itertools.chain(*hidden_stacks))

    if use_output_layer:
        layers.append(nn.Linear(hidden_width, output_dim))

    return nn.Sequential(*layers)


class LearnedProxGenerator(Generator):
    """Generator device with a learnable proximal operator."""

    def __init__(
        self,
        *,
        prox_hidden_width: int = 32,
        prox_hidden_depth: int = 1,
        use_batch_norm: bool = True,
        use_output_layer: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prox_net = _build_prox_mlp(
            input_dim=2,
            output_dim=1,
            hidden_width=prox_hidden_width,
            hidden_depth=prox_hidden_depth,
            use_batch_norm=use_batch_norm,
            use_output_layer=use_output_layer,
        )
        self.has_changed = True

    def admm_prox_update(
        self,
        rho_power: float,
        rho_angle: float,
        power,  # list[torch.Tensor]
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

        if self.has_changed:
            self._pmax = torch.multiply(max_power, nominal_capacity)
            self._pmin = torch.multiply(min_power, nominal_capacity)
            self.has_changed = False

        set_p = power[0]
        rho = torch.full_like(set_p, rho_power)
        inp = torch.stack((set_p, rho), dim=-1)
        p = self.prox_net(inp).squeeze(-1)

        p = torch.clip(p, self._pmin, self._pmax)
        return [p], None


class LearnedProxLoad(Load):
    """Load device with a learnable proximal operator."""

    def __init__(
        self,
        *,
        prox_hidden_width: int = 32,
        prox_hidden_depth: int = 1,
        use_batch_norm: bool = True,
        use_output_layer: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prox_net = _build_prox_mlp(
            input_dim=2,
            output_dim=1,
            hidden_width=prox_hidden_width,
            hidden_depth=prox_hidden_depth,
            use_batch_norm=use_batch_norm,
            use_output_layer=use_output_layer,
        )
        self.has_changed = True

    def admm_prox_update(
        self,
        rho_power: float,
        rho_angle: float,
        power,  # list[torch.Tensor]
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

        if self.has_changed:
            self._pmax = torch.multiply(max_power, nominal_capacity)
            self._pmin = torch.multiply(min_power, nominal_capacity)
            self.has_changed = False

        set_p = power[0]
        rho = torch.full_like(set_p, rho_power)
        inp = torch.stack((set_p, rho), dim=-1)
        p = self.prox_net(inp).squeeze(-1)

        # Clamp to bounds (note: for Load pmin is negative, pmax = 0)
        p = torch.clip(p, self._pmin, self._pmax)
        return [p], None
