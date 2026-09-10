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
        time_limit_s=None,
        batch_seed=42,
        iteration_hook=None,
        tol_rel_objective=None,
        tol_window=20,
        tol_stationarity=None,
        stationarity_scale=None,
        grad_history_every=0,
    ):
        """Run the descent loop.

        ``time_limit_s`` is a *soft* wall-clock cap: the loop finishes the
        iteration it is in, records it, and then breaks, so the caller gets a
        complete design and a complete history.  ``self.stop_reason`` says which
        of ``"num_iterations"`` / ``"wall_clock"`` ended the loop.  (A hard cap
        -- killing the process -- returns neither.)  Under an outer loop, e.g.
        ``emissions.mode: dual_ascent``, the budget applies **per call**, i.e.
        per outer iteration, not to the whole ascent.

        ``batch_seed`` seeds the minibatch RNG (it was a hardcoded 42), so
        replicates of a stochastic run are reproducible and distinguishable.

        ``iteration_hook(index, state, history, final)`` is called after the
        history of iteration ``index`` is recorded -- ``index`` indexes the
        history lists exactly -- and once more after the loop with
        ``final=True`` (the last index is therefore passed twice; hooks dedupe
        on ``index``).  It is the seam for evaluating a full-horizon objective
        at a checkpoint while a minibatch loop is running.  A hook that returns
        a **truthy** value stops the loop with
        ``stop_reason == "checkpoint_tolerance"``: that is how a stochastic run
        applies the objective-plateau test, whose series has to be the
        checkpoints (a 20-sample window of a sigma ~ 1 B$ per-iteration
        estimator still carries ~3.5 % noise).

        Convergence (spec ``2026-09-10-step-rule-spec.md`` section 3).  Both
        tests are off by default, which is the historical behaviour --
        ``num_iterations`` and ``time_limit_s`` are then the only stops:

        ``tol_rel_objective`` / ``tol_window``
            objective *plateau*.  Over the last ``W = tol_window`` recorded
            losses, the relative decrease from the first half's mean to the
            second half's, ``(F1 - F2) / |Fbar|``, below ``tol_rel_objective``
            stops the loop with ``stop_reason == "objective_tolerance"``.  The
            comparison is signed, so "the objective got worse" also counts as
            "no longer making progress".  This reads ``history[LOSS]``, which is
            the full-horizon objective only when the batch is the whole block
            set; a minibatch run must leave this ``None`` and use the hook.
        ``tol_stationarity`` / ``stationarity_scale``
            first-order optimality: ``max_j |mhat_j| / gamma_j`` over the rows
            strictly inside their bounds, where ``gamma_j`` is the annualised
            capital cost of row ``j`` from ``stationarity_scale``, i.e. every
            interior row's marginal value is within ``tol_stationarity`` of its
            own capex.  ``mhat`` is the step rule's smoothed gradient where it
            has one (Adam's bias-corrected first moment) and the raw gradient
            otherwise.  Recorded every iteration as
            ``history["stationarity_max"]`` whenever ``stationarity_scale`` is
            given, whether or not the tolerance is set.

        ``grad_history_every`` (0 = off) is the sampling period of the
        ``grad_sampled`` / ``opt_moments`` trackers.
        """
        if algorithm is None:
            algorithm = GradientDescent()

        # The trackers read the rule's own diagnostics off the problem, and a
        # stateful rule (Adam's moments, the trust radius) must not carry state
        # across `solve` calls -- under `emissions.mode: dual_ascent` the same
        # algorithm object is reused for every outer iteration.
        self.algorithm = algorithm
        if hasattr(algorithm, "reset"):
            algorithm.reset()
        self.grad_history_every = int(grad_history_every or 0)

        if trackers is None:
            trackers = DEFAULT_TRACKERS

        # A minibatch is a subset of the *subproblems* (blocks), so the guard is
        # against `num_subproblems`.  It used to test `self.time_horizon`, which
        # is the block length in hours: with 24 h blocks any `batch_size > 24`
        # silently collapsed to full-batch gradient descent, and with a 2 h test
        # horizon so did `batch_size = 3`.
        if batch_size is None or batch_size > self.num_subproblems or batch_size <= 0:
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
        _batch_rng = np.random.default_rng(batch_seed)

        # Why the loop stopped; overwritten below if the soft cap fires.
        self.stop_reason = "num_iterations"

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
        # Stamp the minibatch on the problem so `trackers.track_batch` can record
        # which subproblems this iteration's gradient actually saw; `batch` is
        # otherwise a local variable and the trajectory is unattributable.
        self.batch = list(batch)
        J, grad = self.forward_and_back(**state, batch=batch)
        if self.la == torch:
            torch.cuda.empty_cache()

        history = self.update_history(
            history, trackers, J, grad, state, None, wandb, log_wandb_every
        )
        _record_stationarity(history, self, algorithm, grad, state, stationarity_scale)

        if iteration_hook is not None and iteration_hook(0, state, history, False):
            self.stop_reason = "checkpoint_tolerance"
            num_iterations = 0

        # First-order model bookkeeping for `TrustRegionDescent`; NaN says "no
        # model to score yet", which leaves the radius at its initial value.
        _actual_decrease = float("nan")
        _predicted_decrease = float("nan")

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

            # Gradient step and project.  Every rule takes the same keyword
            # arguments (`GradientDescent.step` sinks them in `**kwargs`), so
            # the loop never branches on which rule it is running.
            state = algorithm.step(
                state,
                grad,
                iteration=iteration,
                num_iterations=num_iterations,
                actual_decrease=_actual_decrease,
                predicted_decrease=_predicted_decrease,
                # The box, so a rule that normalises a step can normalise over
                # the *projected* gradient rather than over coordinates that
                # cannot move (`TrustRegionDescent`).
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
            )
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

            # The realised step (post-projection) and the loss it started from,
            # so the trust region can score its own model on the next pass.
            _prev_J = float(J)
            _prev_grad = {k: v for k, v in grad.items()}
            _prev_step = {k: _numpy(state[k]) - _numpy(last_state[k]) for k in state}

            self.batch = list(batch)
            J, grad = self.forward_and_back(**state, batch=batch)

            _actual_decrease = _prev_J - float(J)
            _predicted_decrease = -float(
                sum(
                    float(np.sum(_numpy(_prev_grad[k]) * _prev_step[k]))
                    for k in _prev_step
                )
            )

            # Record stuff
            history = self.update_history(
                history, trackers, J, grad, state, last_state, wandb, log_wandb_every
            )
            stationarity = _record_stationarity(
                history, self, algorithm, grad, state, stationarity_scale
            )

            stop = None
            if iteration_hook is not None and iteration_hook(
                self.iteration, state, history, False
            ):
                stop = "checkpoint_tolerance"

            if (
                stop is None
                and tol_rel_objective is not None
                and _plateau(history.get(LOSS) or [], tol_window, tol_rel_objective)
            ):
                stop = "objective_tolerance"

            if (
                stop is None
                and tol_stationarity is not None
                and stationarity is not None
                and stationarity < float(tol_stationarity)
            ):
                stop = "stationarity"

            # Soft wall-clock cap: the iteration just finished is fully
            # recorded, so the caller keeps a design and a history.
            if stop is None and time_limit_s is not None and (
                time.time() - self.start_time >= float(time_limit_s)
            ):
                stop = "wall_clock"

            if stop is not None:
                self.stop_reason = stop
                break

        if iteration_hook is not None:
            iteration_hook(self.iteration, state, history, True)

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


def _numpy(x) -> np.ndarray:
    """A flat float64 numpy view of a numpy array or a torch tensor."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float).reshape(-1)


def _plateau(series, window, tol) -> bool:
    """Has the objective stopped decreasing over the last ``window`` entries?

    Compares the mean of the first half of the window to the mean of the second
    half, relative to the mean over the window: ``(F1 - F2) / |Fbar| < tol``.

    The window does **not** shrink to fit: with fewer than ``window`` recorded
    values the answer is False.  An earlier version compared whatever history
    existed, which for the first ``W`` iterations degenerates to a
    single-iteration test and is roughly ``W/2`` times stricter than the
    quantity ``tol_rel_objective`` is calibrated against.  On c4's recorded
    series the per-iteration relative decrease is 4.7e-5 and the 20-iteration
    one is ~9.4e-4, so the shrinking form stopped the run after **one**
    iteration at the campaign's ``tol = 1e-4`` while the windowed form
    correctly does not stop at all.

    Signed, not absolute: an objective that *increased* has a negative relative
    decrease and stops the loop, because "no longer making progress" is the
    condition being tested, not "moved by less than tol".  That is also why the
    window has to be full first -- a single non-improving iterate must not end
    a run.
    """
    values = [float(v) for v in series if v is not None and np.isfinite(float(v))]
    n = len(values)
    w = int(window or 0)
    if w < 2 or n < w:
        return False
    segment = np.asarray(values[n - w :], dtype=float)
    half = w // 2
    mean = float(np.mean(segment))
    if mean == 0.0:
        return False
    first = float(np.mean(segment[:half]))
    second = float(np.mean(segment[half:]))
    return ((first - second) / abs(mean)) < float(tol)


def stationarity_max(algorithm, grad, state, problem, scale) -> float:
    """``max_j |mhat_j| / gamma_j`` over the rows strictly inside their bounds.

    ``gamma`` (``scale``) is the annualised capital cost of each row, so the
    quantity is "how far this row's marginal value is from its own capex", in
    units of that capex, and ``tol_stationarity = 0.02`` reads as "every
    interior row is within 2 % of break-even".  Rows at a bound are excluded:
    the KKT condition there is an inequality, and 127 of the 166 rows on
    ``ca2040_z4`` are frozen at ``lower == upper``.
    """
    from .trackers import active_set

    smoothed = {}
    fn = getattr(algorithm, "smoothed_gradient", None)
    if fn is not None:
        try:
            smoothed = fn() or {}
        except Exception:  # noqa: BLE001  # pragma: no cover - never fail a solve
            smoothed = {}

    masks = active_set(state, problem)
    worst = 0.0
    seen = False
    for param, mask in masks.items():
        free = mask["free"]
        if not np.any(free):
            continue
        if param not in scale:
            # No annualised capex for this parameter at all: there is nothing to
            # measure its gradient against, so it is skipped.  Scaling it by 1.0
            # instead would compare a $/MW-yr gradient to a dimensionless
            # tolerance and make `tol_stationarity` unreachable.
            continue
        g = _numpy(smoothed[param]) if param in smoothed else _numpy(grad[param])
        gamma = np.abs(_numpy(scale[param]))
        # A row with no capital cost has no scale to be measured against; it is
        # left out rather than divided by zero.
        usable = free & np.isfinite(gamma) & (gamma > 0.0)
        if not np.any(usable):
            continue
        seen = True
        worst = max(worst, float(np.max(np.abs(g[usable]) / gamma[usable])))
    return worst if seen else float("nan")


def _record_stationarity(history, problem, algorithm, grad, state, scale):
    """Append this iteration's stationarity measure to ``history`` and return it."""
    if scale is None:
        return None
    value = stationarity_max(algorithm, grad, state, problem, scale)
    history.setdefault("stationarity_max", []).append(value)
    return None if not np.isfinite(value) else value


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
