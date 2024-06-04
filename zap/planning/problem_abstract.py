import time
import numpy as np
from copy import deepcopy

from zap.layer import DispatchLayer
from zap.planning.operation_objectives import AbstractOperationObjective
from zap.planning.investment_objectives import AbstractInvestmentObjective

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
                    self.upper_bounds[p] = np.inf

    @property
    def parameter_names(self):
        return self.layer.parameter_names

    @property
    def time_horizon(self):
        return self.layer.time_horizon

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
    ):
        if algorithm is None:
            algorithm = GradientDescent()

        if trackers is None:
            trackers = DEFAULT_TRACKERS

        if batch_size is None or batch_size > self.time_horizon or batch_size <= 0:
            batch_size = self.num_subproblems

        assert all([t in TRACKER_MAPS for t in trackers])
        assert batch_strategy in ["sequential", "fixed"]

        self.start_time = time.time()
        self.lower_bound = lower_bound
        self.extra_wandb_trackers = extra_wandb_trackers

        # Setup initial state and history
        state = self.initialize_parameters(deepcopy(initial_state))
        history = self.initialize_history(trackers)
        batch = list(range(batch_size))
        print(batch)

        # Run full forward pass to initialize everything
        self.forward_and_back(**state)

        # Initialize loop
        self.iteration = 0
        J, grad = self.forward_and_back(**state, batch=batch)
        history = self.update_history(
            history, trackers, J, grad, state, None, wandb, log_wandb_every
        )

        # Gradient descent loop
        for iteration in range(num_iterations):
            last_state = deepcopy(state)
            self.iteration = iteration + 1

            # Checkpoint
            if (iteration + 1) % checkpoint_every == 0:
                checkpoint_func(state, history)

            # Gradient step and project
            state = algorithm.step(state, grad)
            state = self.project(state)

            # Update batch and loss
            if batch_strategy == "sequential":
                batch = get_next_batch(batch, batch_size, self.num_subproblems)
            else:  # fixed
                batch = batch
            print(batch)

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
            iteration = len(history[trackers[0]])

            if (iteration % log_wandb_every == 0) or (iteration == 1):
                print(f"Logging to wandb on iteration {iteration}.")

                wand_data = {k: history[k][-1] for k in history.keys()}
                wand_data["iteration"] = iteration

                # Add extra trackers
                if self.extra_wandb_trackers is not None:
                    for tracker, f in self.extra_wandb_trackers.items():
                        wand_data[tracker] = f(J, grad, state, last_state, self)

                wandb.log(wand_data)

        return history

    def project(self, state: dict):
        for param in state.keys():
            state[param] = np.clip(state[param], self.lower_bounds[param], self.upper_bounds[param])
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

    def __init__(self, subproblems: list[AbstractPlanningProblem], weights: list[float] = None):
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
        self.layer = subproblems[0].layer
        self.num_workers = 1

        # Maximum of all sub problem lower bounds
        self.lower_bounds = {
            k: np.max([sub.lower_bounds[k] for sub in self.subproblems], axis=0)
            for k in subproblems[0].lower_bounds.keys()
        }
        self.upper_bounds = {
            k: np.min([sub.upper_bounds[k] for sub in self.subproblems], axis=0)
            for k in subproblems[0].upper_bounds.keys()
        }

        assert len(self.subproblems) == len(self.weights)

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

        if self.num_workers == 1:
            sub_costs = [self.subproblems[b].forward(requires_grad, **kwargs) for b in batch]
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

        if self.num_workers == 1:
            grads = [self.subproblems[b].backward() for b in batch]
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


def get_next_batch(batch, batch_size, num_subproblems):
    last_index = batch[-1]
    return [(last_index + 1 + i) % num_subproblems for i in range(batch_size)]
