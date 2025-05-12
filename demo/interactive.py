import marimo

__generated_with = "0.8.3"
app = marimo.App(width="full")


@app.cell
def __():
    import importlib
    import json
    from copy import deepcopy
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch

    return Path, deepcopy, importlib, json, mo, np, torch


@app.cell
def __():
    import zap

    return (zap,)


@app.cell
def __(importlib):
    from experiments.plan import runner

    _ = importlib.reload(runner)
    return (runner,)


@app.cell
def __(importlib):
    from experiments.plan import plotter

    _ = importlib.reload(plotter)
    return (plotter,)


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_theme(style="white")
    return plt, seaborn


@app.cell
def __(runner):
    # Load config data

    config = runner.load_config("experiments/plan/config/demo_small_v01.yaml")
    return (config,)


@app.cell
def __():
    # devices[1]
    return


@app.cell
def __(
    battery_cost_input,
    config,
    deepcopy,
    load_input,
    runner,
    solar_cost_input,
    wind_cost_input,
):
    # Load dataset
    _data_config = deepcopy(config["data"])
    _data_config["case"] = load_input.value

    _data_config["battery_cost_scale"] = battery_cost_input.value
    _data_config["generator_cost_scale"]["solar"] = solar_cost_input.value
    _data_config["generator_cost_scale"]["wind"] = wind_cost_input.value

    _dataset = runner.load_dataset(**_data_config)
    net, devices = _dataset["net"], _dataset["devices"]
    return devices, net


@app.cell
def __(carbon_input, config, deepcopy, devices, net, runner):
    _problem_config = deepcopy(config["problem"])
    _problem_config["emissions_weight"] = carbon_input.value

    problem_data = runner.setup_problem(net, devices, **_problem_config)

    J_full = problem_data["problem"]
    J_stoch = problem_data["stochastic_problem"]
    layer = problem_data["layer"]
    return J_full, J_stoch, layer, problem_data


@app.cell
def __(layer):
    baseline_params = layer.initialize_parameters()
    return (baseline_params,)


@app.cell
def __():
    init_state_file = "demo_small_v01/000/model_00300"
    return (init_state_file,)


@app.cell
def __(init_state_file, json, np, runner):
    if init_state_file is not None:
        initial_path = runner.datadir("results", f"{init_state_file}.json")
        with open(initial_path, "r") as f:
            initial_state = json.load(f)
            initial_state = {
                k: np.array(v).reshape(-1, 1) for k, v in initial_state.items()
            }
    else:
        initial_state = None
    return f, initial_path, initial_state


@app.cell
def __(mo):
    mo.md(
        """
        # ⚡️ zap: interactive differentiable planning ⚡️

        $~$

        ## Settings
        """
    )
    return


@app.cell
def __(
    battery_cost_input,
    carbon_input,
    load_input,
    mo,
    radiogroup,
    run_button,
    solar_cost_input,
    wind_cost_input,
):
    _col1 = mo.vstack(
        [
            mo.md(
                f"Carbon Penalty: &nbsp;&nbsp; {carbon_input} {carbon_input.value} \$ / mTCO$_2$"
            ),
            mo.md(
                f"Solar Cost: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {solar_cost_input} {100.0 * solar_cost_input.value} % of projected"
            ),
            mo.md(
                f"Wind Cost: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {wind_cost_input} {100.0 * wind_cost_input.value} % of projected"
            ),
            mo.md(
                f"Battery Cost: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {battery_cost_input} {100.0 * battery_cost_input.value} % of projected"
            ),
        ]
    )
    _col2 = mo.vstack(
        [
            mo.md(f"Load Intensity: {load_input}"),
            radiogroup,
        ],
        align="center",
        gap=1.0,
    )

    mo.hstack(
        [_col1, _col2, mo.md(f"{run_button}")],
        justify="center",
        gap=10.0,
    )
    return


@app.cell
def __(mo):
    radiogroup = mo.ui.radio(
        options={"Reference Solution": False, "Previous Solution": True},
        value="Reference Solution",
        label="Initial State",
    )
    return (radiogroup,)


@app.cell
def __(mo):
    battery_cost_input = mo.ui.slider(start=0.1, stop=2.0, step=0.1, value=1.0)
    solar_cost_input = mo.ui.slider(start=0.1, stop=2.0, step=0.1, value=1.0)
    wind_cost_input = mo.ui.slider(start=0.1, stop=2.0, step=0.1, value=1.0)
    return battery_cost_input, solar_cost_input, wind_cost_input


@app.cell
def __(mo):
    carbon_input = mo.ui.slider(1.0, 500.0, value=100.0)
    return (carbon_input,)


@app.cell
def __(mo):
    load_input = mo.ui.dropdown(
        options={"Medium": "load_medium", "High": "load_high"},
        value="Medium",
        label="<Load Intensity>",
    )
    return (load_input,)


@app.cell
def __(mo):
    run_button = mo.ui.run_button(
        label="Update Plan!", tooltip="Press to start computation."
    )
    return (run_button,)


@app.cell
def __():
    refresh_interval = 2
    return (refresh_interval,)


@app.cell
def __():
    total_update_iterations = 4
    return (total_update_iterations,)


@app.cell
def __(initial_state, mo):
    get_solution, set_solution = mo.state((0, initial_state), allow_self_loops=True)
    return get_solution, set_solution


@app.cell(hide_code=True)
def __(
    initial_state,
    json,
    mo,
    np,
    radiogroup,
    run_button,
    runner,
    set_solution,
):
    mo.stop(not run_button.value)

    # Load last optimized state
    with open(runner.DATA_PATH / "demo_model.json", "r") as _f:
        last_opt_state = json.load(_f)
        last_opt_state = {
            k: np.array(v).reshape(-1, 1) for k, v in last_opt_state.items()
        }

    # Reset iteration count and state
    if radiogroup.value:
        set_solution((0, last_opt_state))
    else:
        set_solution((0, initial_state))
    return (last_opt_state,)


@app.cell
def __(
    J_stoch,
    json,
    mo,
    refresh_interval,
    runner,
    set_solution,
    theta_opt,
    total_iter,
    total_update_iterations,
    zap,
):
    mo.stop(total_iter >= total_update_iterations)

    # Run solver
    _new_theta, _history = J_stoch.solve(
        num_iterations=refresh_interval,
        algorithm=zap.planning.GradientDescent(step_size=1e-2, clip=5000.0),
        batch_size=1,
        init_full_loss=False,
        initial_state=theta_opt,
        checkpoint_func=None,
    )

    # Update state
    set_solution((total_iter + refresh_interval, _new_theta))

    # Save model
    with open(runner.DATA_PATH / "demo_model.json", "w") as _f:
        json.dump({k: v.ravel().tolist() for k, v in _new_theta.items()}, _f)
    return


@app.cell
def __(mo):
    mo.md("""## Results""")
    return


@app.cell
def __():
    mosek_tol = 1e-8
    return (mosek_tol,)


@app.cell
def __(
    J_stoch,
    baseline_params,
    devices,
    gc_names,
    gc_vals,
    get_solution,
    mo,
    mosek_tol,
    plotter,
    plt,
    pn,
    pypsa,
    total_update_iterations,
):
    total_iter, theta_opt = get_solution()

    # Capacity plot
    _fig1, _axes1 = plt.subplots(1, 4, figsize=(6.5, 2), width_ratios=[1, 1, 1, 10])
    plotter.capacity_plot(baseline_params, theta_opt, devices, fig=_fig1, axes=_axes1)
    _axes1[2].set_ylim(0, 100)
    _fig1.suptitle("Capacity Investments", fontsize=10)
    _fig1.tight_layout()

    # Stack plot
    _layer = J_stoch.subproblems[0].layer
    _layer.solver_kwargs["mosek_params"]["MSK_DPAR_INTPNT_TOL_REL_GAP"] = mosek_tol
    _layer.solver_kwargs["mosek_params"]["MSK_DPAR_INTPNT_TOL_DFEAS"] = mosek_tol
    _layer.solver_kwargs["mosek_params"]["MSK_DPAR_INTPNT_TOL_PFEAS"] = mosek_tol
    # _y0 = getattr(J_stoch.subproblems[0], "state", None)

    _fig2, _ax2 = plt.subplots(figsize=(6.5, 2))
    plotter.stackplot(theta_opt, _layer, fig=_fig2, ax=_ax2)
    _ax2.set_xlabel("Hour")
    _ax2.set_title("Peak Day Dispatch", fontsize=10)
    _fig2.tight_layout()

    # Map
    # Replace map data
    pn.generators["p_nom_opt"] = theta_opt["generator"].ravel()
    pn.lines["p_nom_opt"] = theta_opt["ac_line"].ravel()
    _gen = (
        pn.generators.assign(pnom=pn.generators.p_nom_opt)
        .groupby(["bus", "carrier"])
        .p_nom_opt.sum()
    )

    plt.figure()
    pn.plot(
        bus_sizes=_gen / 30.0,
        bus_colors=pn.carriers.color.to_dict(),
        # margin=0.5,
        # flow="mean",
        line_widths=0.25 * pn.lines.p_nom_opt,
        line_colors="black",
        # link_widths=0,
    )
    pypsa.plot.add_legend_patches(
        plt.gca(),
        gc_vals,
        gc_names,
        legend_kw={"fontsize": 10, "loc": "upper right", "bbox_to_anchor": (1.5, 1.0)},
    )
    plt.tight_layout()
    _fig3 = plt.gca()

    # Solver info
    output_string = f"Finished {total_iter}/{total_update_iterations} iterations."
    if total_iter < total_update_iterations:
        output_string += ".. still solving."
    else:
        output_string += " Algorithm complete!"

    mo.hstack(
        [mo.vstack([output_string, _fig2, _fig1], gap=2.0), _fig3], align="center"
    )
    return output_string, theta_opt, total_iter


@app.cell
def __():
    return


@app.cell
def __():
    import pypsa

    return (pypsa,)


@app.cell
def __(np, pn):
    gen_colors = pn.carriers.color.to_dict()
    gc_names = np.array(list(gen_colors.keys()))[[0, 1, 2, 3, 4, 5, 6, 7, 8, 11]]
    gc_vals = np.array(list(gen_colors.values()))[[0, 1, 2, 3, 4, 5, 6, 7, 8, 11]]
    return gc_names, gc_vals, gen_colors


@app.cell
def __(config, np, pypsa, runner):
    _csv_dir = f"load_medium/elec_s_100"
    if config["data"]["use_extra_components"]:
        _csv_dir += "_ec"

    pn = pypsa.Network()
    pn.import_from_csv_folder(runner.DATA_PATH / "pypsa/western/" / _csv_dir)

    pn.lines = pn.lines[~np.isinf(pn.lines.x)]
    return (pn,)


@app.cell
def __():
    # p2 = plotter.stackplot(get_solution()[1], J_stoch.subproblems[0].layer, J_stoch.subproblems[0].state)[0]
    return


@app.cell
def __():
    # mo.hstack([p1, p2])
    return


@app.cell
def __():
    # run_button.p
    return


if __name__ == "__main__":
    app.run()
