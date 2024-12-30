import pyomo.environ as pyo

from zap.planning.operation_objectives import (
    MultiObjective,
    DispatchCostObjective,
    EmissionsObjective,
)


class PyoAbstractPlannerObjective:
    def get_objective(self, dispatch: pyo.Block):
        raise NotImplementedError


class PyoMultiObjective(PyoAbstractPlannerObjective):
    def __init__(self, obj: MultiObjective):
        self.weights = obj.weights
        self.objectives = [convert_to_pyo_objective(obj) for obj in obj.objectives]

    def get_objective(self, dispatch: pyo.Block):
        return sum(w * obj.get_objective(dispatch) for w, obj in zip(self.weights, self.objectives))


class PyoDispatchObjective(PyoAbstractPlannerObjective):
    def __init__(self):
        pass

    def get_objective(self, dispatch: pyo.Block):
        return dispatch.objective


class PyoEmissionsObjective(PyoAbstractPlannerObjective):
    def __init__(self):
        pass

    def get_objective(self, dispatch: pyo.Block):
        return sum(dispatch.device[d].emissions for d in dispatch.device)


def convert_to_pyo_objective(obj):
    if isinstance(obj, MultiObjective):
        return PyoMultiObjective(obj)

    elif isinstance(obj, DispatchCostObjective):
        return PyoDispatchObjective()

    elif isinstance(obj, EmissionsObjective):
        return PyoEmissionsObjective()
    else:
        raise NotImplementedError(f"Objective {obj} not supported.")
