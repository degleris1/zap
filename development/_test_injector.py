import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import zap
    return mo, np, zap


@app.cell
def __(zap):
    net = zap.PowerNetwork(num_nodes=5)
    return (net,)


@app.cell
def __(net, np, zap):
    injector = zap.Injector(
        num_nodes=net.num_nodes,
        terminal=np.array([0]),
        nominal_capacity=np.array([1.0]),
        min_power=np.array([[10.0, 15.0, 20.0]]),
        max_power=np.array([[10.0, 15.0, 20.0]]),
        linear_cost=np.array([500.0]),
    )
    injector
    return (injector,)


@app.cell
def __(net, np, zap):
    load = zap.Load(
        num_nodes=net.num_nodes,
        terminal=np.array([0]),
        load=np.array([[10.0, 15.0, 20.0]]),
        linear_cost=np.array([500.0]),
    )
    load
    return (load,)


@app.cell
def __(net, np, zap):
    generators = zap.Generator(
        num_nodes=net.num_nodes,
        terminal=np.array([1, 2]),
        nominal_capacity=np.array([50.0, 25.0]),
        linear_cost=np.array([0.1, 30.0]),
        emission_rates=np.array([0.0, 500.0]),
        dynamic_capacity=np.array(
            [
                [0.1, 0.5, 0.1],
                [1.0, 1.0, 1.0],
            ]
        ),
    )
    generators
    return (generators,)


@app.cell
def __(generators):
    generators.min_power
    return


if __name__ == "__main__":
    app.run()
