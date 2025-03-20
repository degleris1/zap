import zap
import numpy as np
import datetime as dt
import pandas as pd
import pypsa

from pathlib import Path


ZAP_PATH = Path(zap.__file__).parent.parent
DATA_PATH = ZAP_PATH / "data"


def load_pypsa24hour():
    return load_pypsa_network(num_hours=24)


def load_pypsa1hour():
    return load_pypsa_network(num_hours=1)


def load_pypsa_network(num_hours=1, num_nodes=100):
    dates = pd.date_range(
        dt.datetime(2019, 1, 2, 0),
        dt.datetime(2019, 1, 2, 0) + dt.timedelta(hours=num_hours),
        freq="1h",
        inclusive="left",
    )
    pn = pypsa.Network()
    pn.import_from_csv_folder(DATA_PATH / f"pypsa/western/elec_s_{num_nodes}")

    net, devices = zap.importers.load_pypsa_network(
        pn, dates, power_unit=1e3, cost_unit=10.0
    )
    parameters = None
    return net, devices, parameters


def load_example_network(name=None):
    """
    Load an example pypsa network from the data directory.

    Parameters
    ----------
    name : str, optional
        Name of the example network to load. If None, returns a list of available networks.

    Returns
    -------
    pypsa.Network or list
        The requested network or a list of available networks.
    """

    data_dir = Path(__file__).parent.parent / "resources" / "networks"

    if not data_dir.exists():
        return []

    available_networks = [f.stem for f in data_dir.glob("*.nc")]

    if name is None:
        return available_networks

    if name not in available_networks:
        raise ValueError(
            f"Network '{name}' not found. Available networks: {available_networks}"
        )

    network_path = data_dir / f"{name}.nc"
    return pypsa.Network(network_path)


def load_test_network():
    net, devices = zap.importers.load_test_network()
    parameters = None
    return net, devices, parameters


def load_simple_network():
    num_nodes = 7

    net = zap.PowerNetwork(num_nodes=num_nodes)

    parameters = [
        {"nominal_capacity": np.array([1.0, 2.0, 1.0])},
        {},
        {"nominal_capacity": np.array([1.0, 1.0, 1.0])},
        {"power_capacity": np.array([5.0])},
    ]

    generators = zap.Generator(
        num_nodes=num_nodes,
        terminal=np.array([0, 1, 3]),
        dynamic_capacity=np.array(
            [
                [100.0, 100.0, 100.0, 100.0],  # Peaker
                [10.0, 50.0, 50.0, 15.0],  # Solar panel
                [15.0, 15.0, 15.0, 15.0],  # CC Gas
            ]
        ),
        linear_cost=np.array([100.0, 0.5, 40.0]),
        nominal_capacity=parameters[0]["nominal_capacity"],
    )

    loads = zap.Load(
        num_nodes=num_nodes,
        terminal=np.array([0]),
        load=np.array([[30.0, 40.0, 45.0, 80.0]]),
        linear_cost=np.array([200.0]),
    )

    lines = zap.ACLine(
        num_nodes=num_nodes,
        source_terminal=np.array([0, 1, 3]),
        sink_terminal=np.array([1, 3, 0]),
        capacity=np.array([45.0, 50.0, 11.0]),
        susceptance=np.array([0.1, 0.05, 1.0]),
        linear_cost=0.025 * np.ones(3),
        nominal_capacity=parameters[2]["nominal_capacity"],
    )

    batteries = zap.Battery(
        num_nodes=num_nodes,
        terminal=np.array([1]),
        duration=np.array([4.0]),
        linear_cost=np.array([0.01]),
        power_capacity=parameters[3]["power_capacity"],
    )

    devices = [generators, loads, lines, batteries]
    return net, devices, parameters
