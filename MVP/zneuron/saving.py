"""Save and load a whole brain (every engine array) plus the robot's place in the world."""

import json
import pickle

import numpy as np

FORMAT = 1


def save(net, path, world=None):
    data = {
        "format": FORMAT,
        "layout": net.layout, "size": net.size,
        "arrays": {k: np.asarray(v) for k, v in net.S._asdict().items()},
        "world": json.dumps(world.state()) if world is not None else None,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(net, path, world=None):
    """Load into an existing network of the same shape (same neuron capacity and settings)."""
    with open(path, "rb") as f:
        data = pickle.load(f)   # only load files you saved yourself: pickle can run code
    if data.get("format") != FORMAT:
        raise ValueError("saved with a different version")
    arrays = data["arrays"]
    for k, v in net.S._asdict().items():
        if k not in arrays or arrays[k].shape != v.shape:
            raise ValueError(f"brain shape differs ({k}); start run_live.py with the same settings")
    for k, v in net.S._asdict().items():
        v[...] = arrays[k]
    net._lines_key = net._causes_key = None
    if world is not None and data.get("world"):
        world.load_state(json.loads(data["world"]))
