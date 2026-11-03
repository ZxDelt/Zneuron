import json
import warnings

import matplotlib
import numpy

from zneuron import SPEC_PATH


def test_libraries_import():
    assert numpy.__version__.startswith("2.")
    assert matplotlib.__version__.startswith("3.")


def test_spec_is_readable():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    items = [i for g in spec["groups"] for i in g["items"]]
    assert items
    open_items = [i["id"] for i in items if i.get("todo") and not i.get("confirmed")]
    if open_items:   # answers written but not yet confirmed in neuron_spec.html
        warnings.warn(f"spec items waiting for your Submit: {open_items}")
