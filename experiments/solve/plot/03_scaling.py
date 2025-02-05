import marimo

__generated_with = "0.10.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch
    import importlib
    import pickle

    from pathlib import Path
    return Path, importlib, mo, np, pd, pickle, torch


@app.cell
def _():
    import zap
    return (zap,)


@app.cell
def _(importlib):
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_theme(
        style="whitegrid",
        palette="bright",
        rc={
            "axes.edgecolor": "0.15",
            "axes.linewidth": 1.25,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        },
    )

    from experiments.solve import plotter

    _ = importlib.reload(plotter)
    return plotter, plt, seaborn


@app.cell
def _(importlib):
    from experiments.solve import runner

    _ = importlib.reload(runner)
    return (runner,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Helper Functions""")
    return


@app.cell
def _(runner):
    def open_configs(path):
        return runner.expand_config(runner.load_config(path))
    return (open_configs,)


@app.cell
def _(extract_runtime, np, pd):
    def build_runtime_table(configs, skip_missing=False, iter=-1):
        df = {}

        # Add config info
        # Index, solver, load_scale, hps full config reference
        index = [cfg["index"] for cfg in configs]
        df["solver"] = [cfg["solver"] for cfg in configs]
        df["scale_load"] = [cfg["data"]["args"]["scale_load"] for cfg in configs]
        df["hours_per_scenario"] = [
            cfg["parameters"]["hours_per_scenario"] for cfg in configs
        ]
        df["num_nodes"] = [cfg["data"]["num_nodes"] for cfg in configs]
        df["num_contingencies"] = [cfg.get("num_contingencies", 0) for cfg in configs]

        # Extract runtime
        runtimes, primal_residuals, dual_residuals, total_resids, objective_vals, data = zip(
            *[extract_runtime(cfg, skip_missing=skip_missing, iter=iter) for cfg in configs]
        )

        df["median_runtime"] = [np.median(rt) for rt in runtimes]
        df["mean_runtime"] = [np.mean(rt) for rt in runtimes]
        df["runtimes"] = runtimes

        df["primal_residuals"] = primal_residuals
        df["dual_residuals"] = dual_residuals
        df["total_residuals"] = total_resids

        df["median_resid"] = [None if r is None else np.median(r) for r in total_resids]
        df["upper_resid"] = [None if r is None else np.max(r) for r in total_resids]

        df["obj_vals"] = objective_vals
        df["median_obj_vals"] = [None if f is None else np.median(f) for f in objective_vals]

        df = pd.DataFrame(df, index=index)

        if skip_missing:
            df = df[df.mean_runtime >= 0.0]

        df["_obj_cvx"] = np.where(df["solver"] == "cvxpy", df["median_obj_vals"], -1.0)
        df["best_obj"] = df.groupby(
            ["num_contingencies", "num_nodes", "hours_per_scenario", "scale_load"]
        )["_obj_cvx"].transform("max")
        df["subopt"] = np.abs(df["median_obj_vals"] - df["best_obj"]) / df["best_obj"]

        return df, data
    return (build_runtime_table,)


@app.cell
def _(
    Path,
    get_dual_resid_scaled,
    get_primal_resid_scaled,
    np,
    pickle,
    runner,
):
    def extract_runtime(config, skip_missing=False, iter=-1):
        path = Path(runner.get_results_path(config["id"], config["index"])) / "solver_data.pkl"

        if skip_missing and (not path.exists()):
            return [-1.0], None, None, None, None, None

        with open(path, "rb") as f:
            solver_data = pickle.load(f)

        runtimes = [[d["time"] for d in data] for data in solver_data]

        if config["solver"] == "admm":
            primal_residuals = [[get_primal_resid_scaled(d, iter) for d in data] for data in solver_data]
            dual_residuals = [[get_dual_resid_scaled(d, iter) for d in data] for data in solver_data]
            total_resids = [[np.maximum(rp, rd) for rp, rd in zip(primals, duals)] for primals, duals in zip(primal_residuals, dual_residuals)]
            objective_vals = [[d["history"].objective[iter] for d in data] for data in solver_data]

        else:
            primal_residuals = None
            dual_residuals = None
            total_resids = None
            objective_vals = [[d["problem_data"][0]["value"] for d in data] for data in solver_data]

        data = solver_data

        return runtimes, primal_residuals, dual_residuals, total_resids, objective_vals, data
    return (extract_runtime,)


@app.cell
def _(np):
    def get_primal_resid_scaled(data, iter):
        h = data["history"]
        root_n = np.sqrt(data["num_ac_terminals"]) + np.sqrt(data["num_dc_terminals"])

        return np.sqrt(h.power[iter] ** 2 + h.phase[iter] ** 2) / root_n

    def get_dual_resid_scaled(data, iter):
        h = data["history"]
        root_n = np.sqrt(data["num_ac_terminals"]) + np.sqrt(data["num_dc_terminals"])

        return np.sqrt(h.dual_power[iter] ** 2 + h.dual_phase[iter] ** 2) / root_n
    return get_dual_resid_scaled, get_primal_resid_scaled


@app.cell
def _(solver_data):
    solver_data[-1][0][0]["problem_data"][0]["value"]
    return


@app.cell
def _(plt):
    def plot_runtimes(
        df,
        fig=None,
        ax=None,
        compare="solver",
        x_index="scale_load",
        value="mean_runtime",
        labels=None,
    ):
        if fig is None:
            fig, ax = plt.subplots(figsize=(3, 2))

        keys = labels.keys()

        for k in keys:
            data = df[df[compare] == k]

            data = data[[x_index, value]]
            data = data.groupby(x_index).mean()
            data = data.sort_values(by=x_index)

            if labels is None:
                label = k
            else:
                label = labels[k]

            ax.plot(data.index, data[value], label=label, marker=".", ms=4)
            ax.set_xlabel(x_index)
            ax.set_ylabel(value)

        ax.legend()

        return fig, ax
    return (plot_runtimes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Scaling""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Time Horizon""")
    return


@app.cell
def _(df_hours):
    print(df_hours.subopt.max())
    print(df_hours.subopt.median())
    return


@app.cell
def _(build_runtime_table, open_configs, pd):
    _configs = open_configs("./experiments/solve/config/scaling_hours_v03.yaml")
    df_hours, solver_data = build_runtime_table(_configs)
    df_hours.loc[df_hours.solver == "admm", "solver"] = "admm_gpu"

    df_hours_cpu, _ = build_runtime_table(open_configs("./experiments/solve/config/scaling_hours_v04cpu.yaml"))
    df_hours_cpu["solver"] = "admm_cpu"

    df_hours = pd.concat([df_hours, df_hours_cpu])

    df_hours["num_days"] = df_hours.hours_per_scenario / 24
    df_hours.sort_values(by=["median_resid", "hours_per_scenario", "solver"])

    df_hours.sort_values(by=["hours_per_scenario", "scale_load", "solver"], ascending=False)
    return df_hours, df_hours_cpu, solver_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Number of Nodes""")
    return


@app.cell
def _(df_nodes):
    print(df_nodes.subopt.max())
    print(df_nodes.subopt.median())
    return


@app.cell
def _():
    # plt.plot(solver_data_nodes[28][0][0]["history"].objective)
    return


@app.cell
def _(build_runtime_table, open_configs, pd):
    _configs = open_configs("./experiments/solve/config/scaling_devices_v03.yaml")
    df_nodes, solver_data_nodes = build_runtime_table(_configs)
    df_nodes.loc[df_nodes.solver == "admm", "solver"] = "admm_gpu"

    df_nodes_cpu, _ = build_runtime_table(open_configs("./experiments/solve/config/scaling_devices_v04cpu.yaml"))
    df_nodes_cpu["solver"] = "admm_cpu"

    df_nodes = pd.concat([df_nodes, df_nodes_cpu])
    df_nodes = df_nodes[df_nodes.num_nodes >= 500]

    df_nodes.sort_values(by=["upper_resid"], ascending=False)
    df_nodes.sort_values(by=["num_nodes", "scale_load", "solver"], ascending=False)
    return df_nodes, df_nodes_cpu, solver_data_nodes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Contingencies""")
    return


@app.cell
def _(df_cont):
    print(df_cont.subopt.max())
    print(df_cont.subopt.median())
    return


@app.cell
def _():
    # plt.plot(solver_data_cont[54][0][0]["history"].dual_power)
    # plt.yscale("log")
    # plt.show()
    return


@app.cell
def _(build_runtime_table, np, open_configs, pd):
    _configs = open_configs("./experiments/solve/config/scaling_cont_v04.yaml")
    df_cont, solver_data_cont = build_runtime_table(_configs, skip_missing=False)
    df_cont.loc[df_cont.solver == "admm", "solver"] = "admm_gpu"

    df_cont_cpu, _ = build_runtime_table(open_configs("./experiments/solve/config/scaling_cont_v04cpu.yaml"))
    df_cont_cpu["solver"] = "admm_cpu"

    # df_cont_big, _solver_data = build_runtime_table(
    #     open_configs("./experiments/solve/config/scaling_cont_big_v04.yaml")
    # )
    # df_cont_big["solver"] = "admm_gpu"
    # df_cont_base, _solver_data = build_runtime_table(
    #     open_configs("./experiments/solve/config/scaling_cont_big_base_v01.yaml"),
    #     skip_missing=True,
    # )

    df_cont = pd.concat([df_cont, df_cont_cpu])  #, df_cont_big, df_cont_base])


    df_cont["num_contingencies"] = np.minimum(df_cont["num_contingencies"], 1158) + 1

    df_cont.sort_values(by=["upper_resid"], ascending=False)
    df_cont.sort_values(by=["num_contingencies", "scale_load"], ascending=False)
    return df_cont, df_cont_cpu, solver_data_cont


@app.cell
def _(df_cont, df_hours, df_nodes, np):
    _median = np.median([df_hours["median_resid"].median(), df_nodes["median_resid"].median(), df_cont["median_resid"].median()])
    _max = np.max([df_hours["upper_resid"].max(), df_nodes["upper_resid"].max(), df_cont["upper_resid"].max()])

    print(f"Median Residual: {_median}")
    print(f"Max Residual: {_max}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Combine Plots""")
    return


@app.cell
def _(df_cont, df_hours, df_nodes, plot_runtimes, plt):
    def plot_scaling(figsize):
        _fig, _axes = plt.subplots(1, 3, figsize=figsize)
        labels = {"cvxpy": "Mosek", "admm_cpu": "ADMM CPU", "admm_gpu": "ADMM GPU"}

        plot_runtimes(
            df_nodes,
            fig=_fig,
            ax=_axes[0],
            x_index="num_nodes",
            labels=labels,
        )
        plot_runtimes(
            df_hours,
            fig=_fig,
            ax=_axes[1],
            x_index="hours_per_scenario",
            labels=labels,
        )
        plot_runtimes(
            df_cont,
            fig=_fig,
            ax=_axes[2],
            x_index="num_contingencies",
            labels=labels,
        )

        _axes[0].set_ylim(1.0, 1000.0)
        _axes[0].set_yscale("log")
        _axes[0].set_xlabel("Network Size")
        _axes[0].set_ylabel("Mean Runtime (s)")
        _axes[0].get_legend().remove()
        _axes[0].set_xticks([500, 2000, 4000])

        _axes[1].set_ylim(1.0, 1000.0)
        _axes[1].get_legend().remove()
        _axes[1].set_xlabel("Time Horizon")
        _axes[1].set_ylabel("")
        _axes[1].set_yscale("log")
        _axes[1].set_xticks([24, 384, 768])
        _axes[1].tick_params(labelleft=False)

        _axes[2].set_ylim(1.0, 1000.0)
        _axes[2].set_ylabel("")
        _axes[2].set_yscale("log")
        _axes[2].set_xscale("log")
        _axes[2].get_legend().remove()
        _axes[2].set_xlabel("Contingencies")
        _axes[2].tick_params(labelleft=False)


        for ax in _axes:
            ax.grid(True, which='minor', axis="y", color="lightgray", linestyle="dashed", linewidth=0.3)

        _fig.align_xlabels()
        _fig.tight_layout()

        _axes[1].legend(loc="upper center", framealpha=1, ncol=3, bbox_to_anchor=(0.5, 1.3))
        _fig.subplots_adjust(top=0.8)

        # _fig.tight_layout()

        return _fig, _axes
    return (plot_scaling,)


@app.cell
def _(Path, plot_scaling, plotter):
    plotter.set_full_style()
    _fig, _axes = plot_scaling(figsize=(plotter.FIGWIDTH_FULL, 2.5))
    _fig.savefig(Path().home() / "figures/gpu/scaling_devices_hours_fig_full.eps")
    _fig.savefig(Path().home() / "figures/gpu/scaling_devices_hours_fig_full.pdf")
    _fig
    return


@app.cell
def _(Path, plot_scaling, plotter):
    plotter.set_small_style()
    _fig, _axes = plot_scaling(figsize=(plotter.FIGWIDTH_SMALL, 2))
    _fig.savefig(Path().home() / "figures/gpu/scaling_devices_hours_fig_small.eps")
    _fig.savefig(Path().home() / "figures/gpu/Fig4.eps")

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Debug""")
    return


@app.cell
def _():
    # _index = 1
    # _path = Path(runner.get_results_path(configs[_index]["id"], _index))

    # with open(_path / "solver_data.pkl", "rb") as f:
    #     data = pickle.load(f)
    return


@app.cell
def _():
    # _c = data[1][0]
    # _c["time"]
    return


if __name__ == "__main__":
    app.run()
