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
        use_residual: bool = False,
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
        self.use_residual = use_residual
        self.has_changed = True

    def admm_prox_update(
        self,
        rho_power: float,
        rho_angle: float,
        power,
        angle,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        power_weights=None,
        angle_weights=None,
    ):
        p_base, _ = super().admm_prox_update(
            rho_power=rho_power,
            rho_angle=rho_angle,
            power=power,
            angle=angle,
            nominal_capacity=nominal_capacity,
            max_power=max_power,
            min_power=min_power,
            linear_cost=linear_cost,
            power_weights=power_weights,
            angle_weights=angle_weights,
        )
        p_base = p_base[0]

        return _admm_prox_update(
            rho_power, power, p_base=p_base, prox_net=self.prox_net, use_residual=self.use_residual
        )


class LearnedProxLoad(Load):
    """Load device with a learnable proximal operator."""

    def __init__(
        self,
        *,
        prox_hidden_width: int = 32,
        prox_hidden_depth: int = 1,
        use_batch_norm: bool = True,
        use_output_layer: bool = True,
        use_residual: bool = False,
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
        self.use_residual = use_residual
        self.has_changed = True

    def admm_prox_update(
        self,
        rho_power: float,
        rho_angle: float,
        power,
        angle,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        power_weights=None,
        angle_weights=None,
    ):
        p_base, _ = super().admm_prox_update(
            rho_power=rho_power,
            rho_angle=rho_angle,
            power=power,
            angle=angle,
            nominal_capacity=nominal_capacity,
            max_power=max_power,
            min_power=min_power,
            linear_cost=linear_cost,
            power_weights=power_weights,
            angle_weights=angle_weights,
        )

        p_base = p_base[0]

        return _admm_prox_update(
            rho_power, power, p_base=p_base, prox_net=self.prox_net, use_residual=self.use_residual
        )


def _admm_prox_update(
    rho_power: float,
    power: list[torch.Tensor],
    p_base: torch.Tensor = None,
    prox_net: nn.Module = None,
    use_residual: bool = False,
) -> tuple[list[torch.Tensor], None]:
    set_p = power[0]
    orig_shape = set_p.shape
    prox_net = prox_net.to(set_p.device)

    rho_feat = rho_power.expand_as(set_p)
    inp = torch.stack((set_p, rho_feat), dim=-1)
    inp2d = inp.view(-1, 2)

    # Apply prox net
    p_flat = prox_net(inp2d).squeeze(-1)
    if use_residual:
        p = p_flat.view(orig_shape) + p_base
    else:
        p = p_flat.view(orig_shape)

    return [p], None
