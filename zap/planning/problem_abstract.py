import time
from typing import Union
import torch
import numpy as np
from copy import deepcopy

from zap.layer import DispatchLayer
from zap.planning.operation_objectives import AbstractOperationObjective
from zap.planning.investment_objectives import AbstractInvestmentObjective
from zap.planning.constraints import BudgetConstraintSet, ProjectionQP

from .trackers import DEFAULT_TRACKERS, TRACKER_MAPS, LOSS
from .solvers import GradientDescent

from concurrent.futures import ThreadPoolExecutor


class AbstractPlanningProblem:
    """Models long-term multi-value expansion planning."""

    def __init__(
        self,
        operation_objective: AbstractOperationObjective,
        investment_objective: AbstractInvestmentObjective,
        layer: DispatchLayer,
        lower_bounds: dict = None,
        upper_bounds: dict = None,
        budget_constraints: Union[str, BudgetConstraintSet, None] = None,
    ):
        self.operation_objective = operation_objective
        self.investment_objective = investment_objective
        self.layer = layer
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        if self.lower_bounds is None:
            self.lower_bounds = {
                p: getattr(layer.devices[ind], "min_" + pname, None)
                for p, (ind, pname) in self.parameter_names.items()
            }

            # Fallback: use existing device parameter value
            for p, (ind, pname) in self.parameter_names.items():
                if self.lower_bounds[p] is None:
                    self.lower_bounds[p] = getattr(layer.devices[ind], pname)

        if self.upper_bounds is None:
            self.upper_bounds = {
                p: getattr(layer.devices[ind], "max_" + pname, None)
                for p, (ind, pname) in self.parameter_names.items()
            }

            # Fallback: set to infinity
            for p, (ind, pname) in self.parameter_names.items():
                if self.upper_bounds[p] is None:
                    self.upper_bounds[p] = np.inf * self.la.ones_like(self.lower_bounds[p])

        # Initialize budget constraints and projection QP
        self._init_budget_constraints(budget_constraints, layer)

    def _init_budget_constraints(
        self,
        budget_constraints: Union[str, BudgetConstraintSet, None],
        layer: DispatchLayer,
    ):
        """Initialize budget constraints and projection QP.

        Args:
            budget_constraints: Either a CSV path, BudgetConstraintSet, or None
            layer: The dispatch layer (for device info)
        """
        if budget_constraints is None:
            self.budget_constraints = None
            self._projection_qp = None
            return

        # Parse CSV if string path provided
        if isinstance(budget_constraints, str):
            budget_constraints = BudgetConstraintSet.from_csv(
                budget_constraints, self.parameter_names, layer.devices
            )

        self.budget_constraints = budget_constraints

        # Create projection QP
        self._projection_qp = ProjectionQP(
            self.parameter_names,
            self.lower_bounds,
            self.upper_bounds,
            self.budget_constraints,
        )

    @property
    def parameter_names(self):
        return self.layer.parameter_names

    @property
    def time_horizon(self):
        return self.layer.time_horizon

    @property
    def la(self):
        """Return the array module (numpy or torch) based on stored name."""
        if self._la_name == "numpy":
            return np
        elif self._la_name == "torch":
            return torch
        else:
            raise ValueError(f"Unknown la module: {self._la_name}")

    @la.setter
    def la(self, value):
        """Set the array module, storing as string for deepcopy compatibility.

        Accepts either a module (np/torch) or a string name ("numpy"/"torch").
        """
        if value is np or value == "numpy":
            self._la_name = "numpy"
        elif value is torch or value == "torch":
            self._la_name = "torch"
        else:
            raise ValueError(f"Unknown la module: {value}")

    @property
    def num_subproblems(self):
        return 1

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, requires_grad: bool = False, batch=None, **kwargs):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def forward_and_back(self, batch=None, **kwargs):
        J = self.forward(requires_grad=True, batch=batch, **kwargs)
        grad = self.backward()
        return J, grad

    def solve(
        self,
        algorithm=None,
        initial_state=None,
        num_iterations=100,
        trackers=None,
        wandb=None,
        log_wandb_every=1,
        lower_bound=None,
        extra_wandb_trackers=None,
        checkpoint_every=100_000,
        checkpoint_func=lambda x: None,
        batch_size=None,
        batch_strategy="sequential",
        verbosity=10,
        init_full_loss=True,
        peak_net_load_k=None,
        peak_net_load_rerank_every=1,
    ):
        if algorithm is None:
            algorithm = GradientDescent()

        if trackers is None:
            trackers = DEFAULT_TRACKERS

        if batch_size is None or batch_size > self.time_horizon or batch_size <= 0:
            batch_size = self.num_subproblems

        assert all([t in TRACKER_MAPS for t in trackers])

        # If peak_net_load_k is set, batch_strategy controls fill behavior
        if peak_net_load_k is not None:
            _peak_fill_strategy = batch_strategy  # "sequential", "random", "none"
            batch_strategy = "peak_net_load"
        else:
            _peak_fill_strategy = None

        assert batch_strategy in ["sequential", "fixed", "random", "peak_net_load"]

        self.start_time = time.time()
        self.verbosity = verbosity
        self.lower_bound = lower_bound
        self.extra_wandb_trackers = extra_wandb_trackers

        # Setup initial state and history
        state = self.initialize_parameters(deepcopy(initial_state))
        history = self.initialize_history(trackers)

        # RNG for random batch strategy (standalone or as peak_net_load fill)
        _batch_rng = np.random.default_rng(42)

        # Peak net load initialization
        if batch_strategy == "peak_net_load":
            assert peak_net_load_k is not None, (
                "peak_net_load_k must be set when batch_strategy='peak_net_load'"
            )
            assert peak_net_load_k <= self.num_subproblems, (
                f"peak_net_load_k ({peak_net_load_k}) exceeds "
                f"num_subproblems ({self.num_subproblems})"
            )
            _renewable_mask = None
            _last_rerank_iter = -peak_net_load_rerank_every  # force initial ranking
            if peak_net_load_rerank_every == 0:
                # 0 means once per epoch
                peak_net_load_rerank_every = max(
                    1, self.num_subproblems // batch_size
                )

            # Initial ranking
            scores, _renewable_mask = compute_peak_net_loads(
                self.subproblems, state, _renewable_mask
            )
            _cached_top_k = np.argsort(scores)[-peak_net_load_k:]
            _last_rerank_iter = 0

            batch = build_peak_net_load_batch(
                _cached_top_k, batch_size, self.num_subproblems,
                _peak_fill_strategy, list(range(batch_size)), _batch_rng,
            )
        elif batch_strategy == "random":
            batch = sorted(_batch_rng.choice(
                self.num_subproblems, size=batch_size, replace=False
            ).tolist())
        else:
            batch = list(range(batch_size))

        # Run full forward pass to initialize everything
        # TODO - We evaluate the full loss twice :/
        if init_full_loss:
            self.forward(**state)

        # Initialize loop
        self.iteration = 0

        print(batch) if verbosity >= 2 else None
        J, grad = self.forward_and_back(**state, batch=batch)
        if self.la == torch:
            torch.cuda.empty_cache()

        history = self.update_history(
            history, trackers, J, grad, state, None, wandb, log_wandb_every
        )

        # Gradient descent loop
        for iteration in range(num_iterations):
            if self.la == torch:
                last_state = {k: v.detach().clone() for k, v in state.items()}
            else:
                last_state = deepcopy(state)

            self.iteration = iteration + 1
            print("Starting iteration", self.iteration) if verbosity >= 1 else None

            # Checkpoint
            if (self.iteration) % checkpoint_every == 0:
                checkpoint_func(state, history)

            # Gradient step and project
            state = algorithm.step(state, grad)
            state = self.project(state)

            if self.la == torch:
                state = {k: v.detach().clone() for k, v in state.items()}
                torch.cuda.empty_cache()

            # Update batch and loss
            if batch_strategy == "sequential":
                batch = get_next_batch(batch, batch_size, self.num_subproblems)
            elif batch_strategy == "random":
                batch = sorted(_batch_rng.choice(
                    self.num_subproblems, size=batch_size, replace=False
                ).tolist())
            elif batch_strategy == "peak_net_load":
                # Check if rerank is due
                if (self.iteration - _last_rerank_iter) >= peak_net_load_rerank_every:
                    scores, _renewable_mask = compute_peak_net_loads(
                        self.subproblems, state, _renewable_mask
                    )
                    _cached_top_k = np.argsort(scores)[-peak_net_load_k:]
                    _last_rerank_iter = self.iteration
                batch = build_peak_net_load_batch(
                    _cached_top_k, batch_size, self.num_subproblems,
                    _peak_fill_strategy, batch, _batch_rng,
                )
            else:  # fixed
                batch = batch

            print(batch) if verbosity >= 2 else None

            J, grad = self.forward_and_back(**state, batch=batch)

            # Record stuff
            history = self.update_history(
                history, trackers, J, grad, state, last_state, wandb, log_wandb_every
            )

        return state, history

    def initialize_parameters(self, initial_state):
        if initial_state is None:
            return self.layer.initialize_parameters()
        else:
            return initial_state

    def initialize_history(self, trackers):
        return {k: [] for k in trackers}

    def update_history(
        self, history: dict, trackers: dict, J, grad, state, last_state, wandb, log_wandb_every
    ):
        for tracker in trackers:
            f = TRACKER_MAPS[tracker]
            f_val = f(J, grad, state, last_state, self)
            history[tracker] += [f_val]

        if "rolling_loss" not in history:
            history["rolling_loss"] = []

        if isinstance(self, StochasticPlanningProblem):
            if len(history[LOSS]) > 0:
                history["rolling_loss"] += [np.mean(history[LOSS][-self.num_subproblems :])]
            else:
                history["rolling_loss"] += [np.mean(history[LOSS])]
        else:
            history["rolling_loss"] += [history[LOSS][-1]]

        if wandb is not None:
            iteration = len(history[trackers[0]]) - 1

            if (iteration % log_wandb_every == 0) or (iteration == 1):
                print(f"Logging to wandb on iteration {iteration}.\n")

                wand_data = {k: history[k][-1] for k in history.keys()}
                wand_data["iteration"] = iteration

                for k, v in wand_data.items():
                    if k in ["grad", "param"]:
                        # Convert to histogram
                        wand_data[k] = {
                            kk: wandb.Histogram(vv)
                            if isinstance(vv, np.ndarray)
                            else wandb.Histogram(vv.cpu())
                            for kk, vv in v.items()
                        }

                # Add extra trackers
                if self.extra_wandb_trackers is not None:
                    for tracker, f in self.extra_wandb_trackers.items():
                        wand_data[tracker] = f(J, grad, state, last_state, self)

                wandb.log(wand_data)

        return history

    def project(self, state: dict):
        """Project state onto the feasible region.

        If budget constraints are specified, solves a QP to project onto the
        intersection of box constraints and budget constraints. Otherwise,
        falls back to simple box clipping.

        Args:
            state: Dict mapping param_name -> array

        Returns:
            Projected state dict
        """
        if self._projection_qp is not None:
            return self._projection_qp.project(state, la=self.la)
        else:
            # Fallback to simple box projection
            for param in state.keys():
                state[param] = self.la.clip(
                    state[param], self.lower_bounds[param], self.upper_bounds[param]
                )
            return state

    def get_state(self):
        return self.state

    def get_inv_cost(self):
        return self.inv_cost

    def get_op_cost(self):
        return self.op_cost

    def __add__(self, other_problem):
        return StochasticPlanningProblem([self, other_problem])

    def __mul__(self, weight):
        return StochasticPlanningProblem([self], [weight])

    def __rmul__(self, weight):
        return self.__mul__(weight)


class StochasticPlanningProblem(AbstractPlanningProblem):
    """Weighted mixture of planning problems."""

    def __init__(
        self,
        subproblems: list[AbstractPlanningProblem],
        weights: list[float] = None,
        budget_constraints: Union[str, BudgetConstraintSet, None] = None,
    ):
        # Use property setter for deepcopy compatibility
        self.la = subproblems[0].la

        if weights is None:
            weights = [1.0 for _ in subproblems]

        # Merge stochastic subproblems
        new_subproblems = []
        new_weights = []
        for sub, w in zip(subproblems, weights):
            if isinstance(sub, StochasticPlanningProblem):
                new_subproblems.extend(sub.subproblems)
                new_weights.extend([w * w_ for w_ in sub.weights])
            else:
                new_subproblems.append(sub)
                new_weights.append(w)

        # Drop zero weights
        subproblems = [sub for sub, w in zip(subproblems, weights) if w > 0]
        weights = [w for w in weights if w > 0]

        self.subproblems = new_subproblems
        self.weights = new_weights
        #: Per-subproblem forward/backward timing prints are gated on this;
        #: ``AbstractPlanningProblem.solve`` sets it from its ``verbosity`` kwarg.
        self.verbosity = 0
        self.layer = subproblems[0].layer
        self.num_workers = 1

        # Maximum of all sub problem lower bounds
        if self.la is np:
            self.lower_bounds = {
                k: np.max([sub.lower_bounds[k] for sub in self.subproblems], axis=0)
                for k in subproblems[0].lower_bounds.keys()
            }
            self.upper_bounds = {
                k: np.min([sub.upper_bounds[k] for sub in self.subproblems], axis=0)
                for k in subproblems[0].upper_bounds.keys()
            }
        else:  # torch
            self.lower_bounds = {
                k: torch.max(
                    torch.stack([sub.lower_bounds[k] for sub in self.subproblems], dim=0), dim=0
                )[0]
                for k in subproblems[0].lower_bounds.keys()
            }
            self.upper_bounds = {
                k: torch.min(
                    torch.stack([sub.upper_bounds[k] for sub in self.subproblems], dim=0), dim=0
                )[0]
                for k in subproblems[0].upper_bounds.keys()
            }

        assert len(self.subproblems) == len(self.weights)

        # Initialize budget constraints (inherit from first subproblem if not specified)
        if budget_constraints is None:
            # Check if any subproblem has budget constraints
            for sub in self.subproblems:
                if hasattr(sub, "budget_constraints") and sub.budget_constraints is not None:
                    budget_constraints = sub.budget_constraints
                    break

        self._init_budget_constraints(budget_constraints, self.layer)

    @property
    def inv_cost(self):
        return sum([w * sub.get_inv_cost() for w, sub in zip(self.weights, self.subproblems)])

    @property
    def op_cost(self):
        return sum([w * sub.get_op_cost() for w, sub in zip(self.weights, self.subproblems)])

    @property
    def num_subproblems(self):
        return len(self.subproblems)

    def initialize_workers(self, num_workers):
        self.num_workers = num_workers
        self.pool = ThreadPoolExecutor(max_workers=num_workers)
        return None

    def shutdown_workers(self):
        if self.num_workers > 1:
            self.pool.shutdown()

        self.num_workers = 1
        return None

    def forward(self, requires_grad: bool = False, batch=None, **kwargs):
        if batch is None:
            batch = range(self.num_subproblems)

        self.batch = batch

        verbose = getattr(self, "verbosity", 0) >= 2

        if self.num_workers == 1:
            sub_costs = []
            for _idx, b in enumerate(batch):
                _t0 = time.time()
                sub_costs.append(self.subproblems[b].forward(requires_grad, **kwargs))
                _dt = time.time() - _t0
                if verbose and (_dt > 1.0 or _idx == 0 or _idx == len(batch) - 1):
                    print(f"  [fwd] sub {b} ({_idx+1}/{len(batch)}): {_dt:.2f}s")
        else:
            # Developer Note
            # Normally, multi-threading doesn't gain any performance in Python because of the GIL.
            # However, GIL is released when we call the Mosek solver.
            # This is why we can gain performance by multi-threading the forward pass.
            # The same is true for the backward pass, since the linear solver also releases the GIL.
            sub_costs = self.pool.map(
                lambda b: self.subproblems[b].forward(requires_grad, **kwargs), batch
            )
            sub_costs = list(sub_costs)

        return sum([w * c for w, c in zip(self._get_batch_weights(batch), sub_costs)])

    def backward(self):
        batch = self.batch

        verbose = getattr(self, "verbosity", 0) >= 2

        if self.num_workers == 1:
            grads = []
            for _idx, b in enumerate(batch):
                _t0 = time.time()
                grads.append(self.subproblems[b].backward())
                _dt = time.time() - _t0
                if verbose and (_dt > 1.0 or _idx == 0 or _idx == len(batch) - 1):
                    print(f"  [bwd] sub {b} ({_idx+1}/{len(batch)}): {_dt:.2f}s")
        else:
            grads = self.pool.map(lambda b: self.subproblems[b].backward(), batch)
            grads = list(grads)

        return {
            k: sum([w * g[k] for w, g in zip(self._get_batch_weights(batch), grads)])
            for k in grads[0].keys()
        }

    def _get_batch_weights(self, batch):
        total_batch_weight = sum([self.weights[b] for b in batch])
        total_weight = sum(self.weights)
        return (total_weight / total_batch_weight) * np.array(self.weights)[batch]


def weighted_subproblems(problem):
    """``(subproblems, weights)`` of a planning problem, stochastic or not.

    A :class:`StochasticPlanningProblem` evaluates
    ``sum_i w_i * (snapshot_weight_i * op_i + inv_i)`` in :meth:`forward`, where
    ``inv_i`` is subproblem ``i``'s own investment objective -- built from the
    ``sample_time``-sliced devices, so its capital cost is already pro-rated by
    ``block_hours / total_hours``.  Any single-level reformulation of the same
    problem (``MonolithicPlanningProblem``, ``RelaxedPlanningProblem``) has to
    use the same weights and the same per-subproblem investment objectives, or
    its capital cost is off by a factor of the number of blocks.

    A non-stochastic problem is the one-subproblem, unit-weight case.
    """
    if isinstance(problem, StochasticPlanningProblem):
        return list(problem.subproblems), [float(w) for w in problem.weights]
    return [problem], [1.0]


def get_next_batch(batch, batch_size, num_subproblems):
    last_index = batch[-1]
    return [(last_index + 1 + i) % num_subproblems for i in range(batch_size)]


def compute_peak_net_loads(subproblems, state, renewable_mask=None):
    """Compute peak net load score for each subproblem.

    For each subproblem (block), computes max_t(total_load[t] - renewable_available[t])
    where renewable_available uses the current investment capacities from state.

    Parameters
    ----------
    subproblems : list[AbstractPlanningProblem]
        The list of subproblems (one per block).
    state : dict
        Current investment parameters, e.g. {"generator_capacity": array}.
    renewable_mask : np.ndarray or None
        Boolean mask over generators identifying renewables. If None, computed
        from fuel_type and cached for reuse.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (scores, renewable_mask) — scores has shape (num_subproblems,),
        renewable_mask is a boolean array of shape (num_generators,).
    """
    RENEWABLE_FUELS = {"solar", "onwind", "offwind", "offwind_floating"}

    # Build renewable mask from fuel_type (same across all subproblems)
    if renewable_mask is None:
        gen_device = subproblems[0].layer.devices[0]
        fuel_types = np.asarray(gen_device.fuel_type).reshape(-1)
        renewable_mask = np.isin(fuel_types, list(RENEWABLE_FUELS))

    # Get current generator capacities from state
    gen_cap = state.get("generator_capacity", None)
    if gen_cap is None:
        # Fallback to device nominal_capacity
        gen_cap = subproblems[0].layer.devices[0].nominal_capacity

    # Convert to numpy if torch tensor
    if hasattr(gen_cap, "detach"):
        gen_cap = gen_cap.detach().cpu().numpy()
    gen_cap = np.asarray(gen_cap).reshape(-1)

    scores = np.empty(len(subproblems))
    for i, sub in enumerate(subproblems):
        # Load: shape (num_loads, block_hours)
        load_data = sub.layer.devices[1].load
        if hasattr(load_data, "detach"):
            load_data = load_data.detach().cpu().numpy()
        load_data = np.asarray(load_data)
        total_load = load_data.sum(axis=0)  # (block_hours,)

        # Generator capacity factors: shape (num_gens, block_hours)
        dyn_cap = sub.layer.devices[0].dynamic_capacity
        if hasattr(dyn_cap, "detach"):
            dyn_cap = dyn_cap.detach().cpu().numpy()
        dyn_cap = np.asarray(dyn_cap)

        # Renewable available = cf * capacity for renewable generators
        renewable_available = (
            dyn_cap[renewable_mask, :] * gen_cap[renewable_mask, np.newaxis]
        ).sum(axis=0)  # (block_hours,)

        net_load = total_load - renewable_available
        scores[i] = net_load.max()

    return scores, renewable_mask


def build_peak_net_load_batch(
    top_k_indices, batch_size, num_subproblems, fill_strategy, current_batch, rng
):
    """Assemble a batch with top-K peak net load blocks plus fill slots.

    Parameters
    ----------
    top_k_indices : np.ndarray
        Indices of the top-K highest net-load subproblems.
    batch_size : int
        Total batch size.
    num_subproblems : int
        Total number of subproblems.
    fill_strategy : str
        "sequential", "random", or "none".
    current_batch : list[int]
        The current batch (used for sequential fill to track position).
    rng : np.random.Generator
        Random number generator for "random" fill.

    Returns
    -------
    list[int]
        Batch indices of length min(batch_size, num_subproblems).
    """
    top_k_set = set(top_k_indices.tolist())
    k = len(top_k_set)

    if fill_strategy in ("none", "fixed") or batch_size <= k:
        return sorted(top_k_set)[:batch_size]

    fill_count = batch_size - k
    remaining = [i for i in range(num_subproblems) if i not in top_k_set]

    if len(remaining) == 0:
        return sorted(top_k_set)

    if fill_strategy == "random":
        fill_count = min(fill_count, len(remaining))
        fill_indices = rng.choice(remaining, size=fill_count, replace=False).tolist()

    elif fill_strategy == "sequential":
        # Continue from the last position in the current batch
        last_pos = current_batch[-1] if current_batch else -1
        fill_indices = []
        cursor = last_pos + 1
        while len(fill_indices) < fill_count:
            idx = cursor % num_subproblems
            if idx not in top_k_set:
                fill_indices.append(idx)
            cursor += 1
            # Safety: if we've wrapped around fully, stop
            if cursor - (last_pos + 1) >= num_subproblems:
                break
    else:
        raise ValueError(f"Unknown peak_net_load fill strategy: {fill_strategy}")

    return sorted(list(top_k_set) + fill_indices)
