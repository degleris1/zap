import pao
import pyomo.environ as pyo

from zap.planning.operation_objectives import AbstractOperationObjective
from zap.devices import Generator, DCLine, Battery
from .dispatch import convert_to_pyo, setup_pyomo_model
from .objectives import convert_to_pyo_objective

PAO_SOLVERS = ["pao.pyomo.FA", "pao.pyomo.MIBS", "pao.pyomo.PCCG", "pao.pyomo.REG"]


def solve_bilevel_model(
    net,
    devices,
    time_horizon,
    planner_objective: AbstractOperationObjective,
    param_device_types=[Generator, DCLine, Battery],
    pao_solver="pao.pyomo.FA",
    mip_solver="gurobi",
    verbose=True,
):
    # Settings
    if isinstance(param_device_types[0], int):
        param_devices = param_device_types
    else:
        param_devices = [i for i in range(len(devices)) if type(devices[i]) in param_device_types]

    # Build model
    pyo_devices = [convert_to_pyo(d) for d in devices]

    M = pyo.ConcreteModel()
    M.time_horizon = time_horizon
    M.time_index = pyo.RangeSet(0, time_horizon - 1)
    M.node_index = pyo.RangeSet(0, net.num_nodes - 1)

    # Create parameters
    params = []
    M.param_blocks = pyo.Block(param_devices)
    for p in param_devices:
        par = pyo_devices[p].make_parameteric(M.param_blocks[p])
        pyo_devices[p].add_investment_cost(M.param_blocks[p])

        params += [par]

    # Build dispatch problem
    M.dispatch = pao.pyomo.SubModel(fixed=params)
    setup_pyomo_model(net, devices, time_horizon, model=M.dispatch, pyo_devices=pyo_devices)

    # Create top level objective
    planner_objective = convert_to_pyo_objective(planner_objective)
    M.planner_objective = pyo.Expression(
        expr=planner_objective.get_objective(M.dispatch),
    )
    M.objective = pyo.Objective(
        expr=(M.planner_objective + sum(M.param_blocks[p].investment_cost for p in param_devices)),
        sense=pyo.minimize,
    )

    # Solve bilevel problem
    mip = pao.Solver(mip_solver)
    solver = pao.Solver(pao_solver, mip_solver=mip)  # , linearize_bigm=100.0)
    result = solver.solve(M, tee=verbose)

    return M, {"result": result, "solver": solver}
