"""Concrete planning methods; importing this module registers them.

WP5 spec section 3.1.  Each class carries ``@base.register_method``, so
``planning.METHODS`` is filled as a side effect of this import (``base``'s
``make_method`` triggers it lazily).  ``monolithic`` / ``stochastic`` /
``relaxed`` are config presets over one class (D-W2); the preset -> class map
lives in ``base.METHOD_PRESETS``.
"""

from __future__ import annotations

from .admm import AdmmGradientMethod
from .gradient import GradientMethod
from .single_level import SingleLevelMethod

__all__ = ["AdmmGradientMethod", "GradientMethod", "SingleLevelMethod"]
