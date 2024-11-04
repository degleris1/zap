import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # 1 - ⚡ Zap Basics ⚡


        In this notebook, we introduce the basic types and functionality in Zap.
        We will manually construct a small electricity network, solve a dispatch problem using both CVXPY and ADMM, and analyze the results.

        1. Creating a Network
        2. Solving Dispatch Problems
        3. Analyzing Results
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Creating a Network

        Networks in Zap consist of a `Network` object, which defines the underlying network, and a list of `Device` objects, which are electrical components attached to the network. Devices include generators, loads, transmission lines, batteries, and all other components that connect to the network. Devices can also be used to model financial contracts, such as bids and offers for energy or transmission congestion contracts.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Constructing Devices""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""First, let's initialize a simple network with 3 nodes.""")
    return


@app.cell
def __():
    import zap
    import numpy as np
    return np, zap


@app.cell
def __(zap):
    net = zap.PowerNetwork(num_nodes=3)
    return (net,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Now, let's add a load to the network.""")
    return


@app.cell
def __(net, np, zap):
    load = zap.Load(
        num_nodes=net.num_nodes,
        # The terminal of the device is the node to which it connects
        terminal=np.array([0]),
        # The load argument is a time series of the power consumed by the device
        # This is 2-d array, see more below!
        load=np.array([[10.0, 15.0, 20.0]]),
        # This is the curtailment cost of the device, which we will assume is $500 / MWh
        linear_cost=np.array([500.0]),
    )
    return (load,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        All data passed to the device must be `numpy` arrays or of a similar type (e.g., a PyTorch `Tensor`).
        We can also create groups of devices at the same time.
        Let's add two generators to the network.
        """
    )
    return


@app.cell
def __(net, np, zap):
    generators = zap.Generator(
        num_nodes=net.num_nodes,
        # Since we have two generators, we specify two terminals
        terminal=np.array([1, 2]),
        # Nominal capacity refers to the generator nameplate capacity
        nominal_capacity=np.array([50.0, 25.0]),
        # This is the marginal cost of generation
        linear_cost=np.array([0.1, 30.0]),
        # Emissions rates are optional, but useful for many applications
        emission_rates=np.array([0.0, 500.0]),
        # Dynamic capacity refers to the time-varying capacity factor of each generator
        dynamic_capacity=np.array(
            [
                [0.1, 0.5, 0.1],
                [1.0, 1.0, 1.0],
            ]
        ),
    )
    return (generators,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Since we have two generators, we specify both their parameters at the same time.
        Some of the parameters, such as costs or dynamic capacities, can also time varying, in which case the user must pass a 2d-array of data.
        The first dimension of the data specifies which device is being referred to, and the second dimension of the data specifies the time period. 
        If you pass a 1d-array, Zap will automatically assume this is a static quantity that does not change over time.

        Finally, let's initialize a few transmission lines to connect our network.
        """
    )
    return


@app.cell
def __(net, np, zap):
    lines = zap.ACLine(
        num_nodes=net.num_nodes,
        # AC lines are two-terminal devices, so we specify both a source and sink terminal for each line
        # We will build the classic 3-bus "triangle" network
        source_terminal=np.array([0, 1, 2]),
        sink_terminal=np.array([1, 2, 0]),
        nominal_capacity=np.array([10.0, 20.0, 30.0]),
        # This is the per-MW susceptance, so the susceptance of each line is its nominal capacity
        # times its susceptance
        # We will give the lines uniform total susceptance
        susceptance=np.array([1 / 10.0, 1 / 20.0, 1 / 30.0]),
        # Lines can also have time-vary capacities to simulate outages or de-ratings
        # In this example, we will just make them static
        capacity=np.ones(3),
    )
    return (lines,)


@app.cell
def __(mo):
    mo.md(r"""We will also add an electrical ground device to specify which node is the reference bus.""")
    return


@app.cell
def __(net, np, zap):
    ground = zap.Ground(num_nodes=net.num_nodes, terminal=np.array([0]))
    return (ground,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Putting the Network Together

        To finish up, we will create a list of device (groups). 
        Together with the network and the time horizon, this list fully specifies an electrical network!
        """
    )
    return


@app.cell
def __(generators, ground, lines, load, net):
    devices = [load, generators, lines, ground]
    time_horizon = 3

    net, devices, time_horizon  # Fully network specification
    return devices, time_horizon


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        Before we move on, here are a couple of notes about constructing devices.

        - Many arguments (capital cost, emissions rates, line slacks, etc.) are optional and introduce additional functionality, e.g., for planning studies.
        - For performance reasons, it's best to specify a few device groups (ideally, one for each type of device) with many devices per group, instead of many groups with one or just a few devices per group. This is so that computations can be efficiently vectorized, both on CPU and GPU machines.
        - Remember that the first dimension (rows) of a parameter specifies which device in the group the parameter is for, and the second dimension (columns) specifies the time period. **Dynamic data is always 2-dimensional!**
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Solving Dispatch Problems

        Now that we've defined an electricity dispatch problem, let's solve using two different methods.

        1. `cvxpy` - This method will build a model in `cvxpy` and send it to an off-the-shelf solver, such as ECOS or Clarabel. You can also use commerical solvers like Mosek. This approach is best for small to medium problems and finds highly accurate solutions.

        2. `admm` - This method will use a PyTorch implementation of the alternating direction method of multipliers (ADMM). This can be run on either a CPU or GPU and scales well to larger problems. In general, this method is only capable of finding medium accuracy solutions within a reasonable amount of time.

        ### Solving with CVXPY

        To solve with `cvxpy`, we simply call `net.dispatch` on our devices and (optionally) specify a solver.
        """
    )
    return


@app.cell
def __():
    import cvxpy as cp
    return (cp,)


@app.cell
def __(cp, devices, net, time_horizon):
    outcome = net.dispatch(devices, time_horizon, solver=cp.CLARABEL, add_ground=False)

    outcome
    return (outcome,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""As you can see, the result of a grid dispatch is a complicated object. We will address that in a second.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Solving with ADMM

        Solving with ADMM is a little more complicated and requires two steps:

        1. Transfering device data to PyTorch
        2. Initializing an ADMM solver object.
        """
    )
    return


@app.cell
def __():
    import torch
    from zap.admm import ADMMSolver
    return ADMMSolver, torch


@app.cell
def __(torch):
    machine = "cpu"  # Pick "cuda" for a machine with an Nvidia GPU
    dtype = torch.float32
    return dtype, machine


@app.cell
def __(devices, dtype, machine):
    admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in devices]
    return (admm_devices,)


@app.cell
def __(ADMMSolver, admm_devices, dtype, machine, net, time_horizon):
    admm = ADMMSolver(
        machine=machine,
        dtype=dtype,
        atol=1e-6,
        rtol=1e-6,
    )

    solution_admm, history_admm = admm.solve(net, admm_devices, time_horizon)
    return admm, history_admm, solution_admm


@app.cell
def __(solution_admm):
    solution_admm.power
    return


@app.cell
def __(solution_admm):
    # ADMM solutions need to be cast to a standard DispatchOutcome
    outcome_admm = solution_admm.as_outcome()

    outcome_admm
    return (outcome_admm,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Analyzing Results

        Results are packaged into a hierachically structured `DispatchOutcome` object. 

        1. At the top level, a `DispatchOutcome` has several fields: `power`, `angle`, `prices`, and a few others. You can access these fields like any other Python field, e.g., `outcome.power`.
        2. Each field contains either *device-specific information* or *global information*. Device-specific fields will contain a list of length `len(devices)`, where the `i`th entry in the list contains information specific to the `i`th device. Global fields contain a 2d array of size `(num_nodes, time_horizon)`. You can access the information for device `i` using normal indexing, e.g., `outcome.power[i]`.
        3. For device-specific information, each block of information is further broken down by the *terminal* of the device. Many devices, such as generator, loads, and batteries, have just a single terminal and will always be indexed as `outcome.power[i][0]`. Some devices, like transmission lines, have two or more terminals. The data for terminal `j` is stored in `outcome.power[i][j]`.
        4. Finally, the data for terminal `j` of device `i` is just a 2d array of size `(num_devices, time_horizon)`, where `num_devices = devices[i].num_devices` is the number of devices in device group `i`.
        """
    )
    return


@app.cell
def __(outcome):
    outcome.prices  # Global
    return


@app.cell
def __(outcome):
    outcome.power  # Device-specific
    return


@app.cell
def __(outcome):
    outcome.power[2]  # Power flows for device 2, which we made the transmission lines
    return


@app.cell
def __(outcome):
    # Indexing the second terminal of the lines---the result is an object of size (num_ac_lines, time_horizon)
    outcome.power[2][1]
    return


@app.cell
def __(outcome):
    # Indexing the first (and only) terminal of the generators
    outcome.power[1][0]
    return


@app.cell
def __(outcome):
    # Power output of generator 0 at timestep 2
    outcome.power[1][0][0, 2]
    return


if __name__ == "__main__":
    app.run()
