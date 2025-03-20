import marimo

__generated_with = "0.10.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import cvxpy as cp
    import torch

    import zap
    return cp, mo, np, torch, zap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Import a Network""")
    return


@app.cell
def _(np, zap):
    net, devices = zap.importers.load_test_network()
    devices = devices[:3]
    devices += [zap.Ground(num_nodes=net.num_nodes, terminal=np.array([0]))]
    return devices, net


@app.cell
def _(net):
    net
    return


@app.cell
def _(devices):
    devices
    return


@app.cell
def _(devices):
    devices[1].min_power
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Construct an Optimization Layer""")
    return


@app.cell
def _():
    from zap import DispatchLayer
    from zap.network import DispatchOutcome
    return DispatchLayer, DispatchOutcome


@app.cell
def _():
    parameter_names = {"load": (1, "min_power")}
    return (parameter_names,)


@app.cell
def _(DispatchLayer, cp, devices, net, parameter_names):
    x_star = DispatchLayer(
        net,
        devices,
        parameter_names,
        time_horizon=4,
        solver=cp.CLARABEL
    )
    return (x_star,)


@app.cell
def _(devices, np):
    demand_0 = devices[1].min_power
    demand_0 -= 10*np.random.randn(*demand_0.shape)
    return (demand_0,)


@app.cell
def _(demand_0, x_star):
    y = x_star(load=demand_0)
    return (y,)


@app.cell
def _(np, y):
    dy = y.package(np.zeros_like(y.vectorize()))
    return (dy,)


@app.cell
def _(dy):
    dy.prices
    return


@app.cell
def _(DispatchOutcome, torch):
    def compute_lmb(y: DispatchOutcome):
        return torch.linalg.norm(y.prices)
    return (compute_lmb,)


@app.cell
def _(y):
    y_torch = y.torchify(requires_grad=True)
    return (y_torch,)


@app.cell
def _(compute_lmb, y_torch):
    alpha = compute_lmb(y_torch)
    alpha.backward()
    return (alpha,)


@app.cell
def _(DispatchOutcome, y_torch, zap):
    dalpha_dy = DispatchOutcome(*[zap.util.grad_or_zero(x) for x in y_torch])
    return (dalpha_dy,)


@app.cell
def _(dalpha_dy):
    dalpha_dy.prices
    return


if __name__ == "__main__":
    app.run()
