"""Zneuron v2: leak, no line limit, mini-buckets with the choke rule (single-neuron XOR), slow memory,
pruning only weak lines. v1 behaviour is covered by the older tests (all switches off by default)."""

import numpy as np
import pytest

from zneuron.network import Network
from zneuron.params import Params, v2


def fired(net, nid, since):
    return any(k == nid and t >= since for t, k in net.fire_log)


def test_leak_keeps_charge_that_the_dump_threw_away():
    for leak, expect in ((50.0, True), (0.0, False)):      # 0 = v1's 20 ms dump
        net = Network(n=1, sensors=0, start_lines=0, seed=0, params=v2(leak_tau_ms=leak, branches=0))
        net.stimulate(0, 0.2)
        net.run(25)
        net.stimulate(0, 0.2)                               # 0.2 x e^(-25/50) + 0.2 = 0.32 >= 0.3
        net.run(3)
        assert fired(net, 0, 0) == expect


def test_no_8_line_limit():
    net = Network(n=30, sensors=0, start_lines=0, seed=0, params=v2())
    lines = [net.connect(k, 0, 0.5) for k in range(1, 25)]
    assert all(m is not None for m in lines) and net.S.nlines[0] == 24


def xor_neuron(params):
    """2 input neurons on the same mini-bucket of neuron 2."""
    net = Network(n=3, sensors=2, start_lines=0, seed=0, params=params)
    for k in range(3):
        net.neurons[k].sign = 1
    net.connect(0, 2, 0.5, branch=0)
    net.connect(1, 2, 0.5, branch=0)
    return net


def respond(net, a, b):
    t0 = net.t
    for k, on in ((0, a), (1, b)):
        if on:
            net.stimulate(k, 0.3)                           # exactly the threshold: one clean input spike
    net.run(15)
    out = fired(net, 2, t0)
    net.run(300)                                            # let everything settle before the next pattern
    return out


def test_one_neuron_solves_xor_with_the_choke_rule():
    net = xor_neuron(v2())
    table = {(a, b): respond(net, a, b) for a in (0, 1) for b in (0, 1)}
    assert table == {(0, 0): False, (0, 1): True, (1, 0): True, (1, 1): False}


def test_without_the_choke_rule_it_is_just_or():
    net = xor_neuron(v2(branch_hi=9.0))
    table = {(a, b): respond(net, a, b) for a in (0, 1) for b in (0, 1)}
    assert table == {(0, 0): False, (0, 1): True, (1, 0): True, (1, 1): True}


@pytest.mark.parametrize("model", ["lif", "alif"])
def test_point_neurons_cannot_do_xor(model):
    """Any weights: if A alone and B alone fire a LIF/ALIF neuron, A+B fires it too (excitatory inputs add up)."""
    from bees.alif import ALIFBrain
    from bees.flybrain import FlyBrain
    from bees.lif import LIFBrain
    cls = LIFBrain if model == "lif" else ALIFBrain
    for w in np.linspace(0.05, 1.0, 20):
        res = {}
        for a, b in ((1, 0), (0, 1), (1, 1)):
            brain = cls(FlyBrain(3, 2, 0, [1, 1, 1], [(0, 2, w), (1, 2, w)]), seed=0)
            brain.set_drive([0, 1], [a * 1.0, b * 1.0])        # one spike each at t = 0
            brain.run(1)
            brain.set_drive([0, 1], [0.0, 0.0])
            brain.run(10)
            res[(a, b)] = bool(brain.fires_n()[2] > 0)
        assert not (res[(1, 0)] and res[(0, 1)] and not res[(1, 1)])


def test_slow_memory_lets_a_late_reward_land():
    out = {}
    for name, p in (("v1", Params()), ("v2", v2(elig_slow_tau_ms=3000.0, elig_slow_w=1.0, branches=0))):
        net = Network(n=2, sensors=0, start_lines=0, seed=0, params=p)
        for nrn in net.neurons:
            nrn.sign = 1
        net.lfuel = 0.0
        m = net.connect(0, 1, 0.5)
        net.stimulate(0, 1.0)
        net.run(2)
        net.stimulate(1, 1.0)
        net.run(1)
        net.run(1500)                                       # reward arrives 1.5 s later
        net.reward(1.0)
        out[name] = m.size - 0.5
    assert out["v1"] < 1e-4 < out["v2"]


def test_quiet_lines_are_not_pruned_in_v2():
    for prune_quiet, expect in ((1, False), (0, True)):
        p = v2(prune_quiet=prune_quiet, unused_ms=1000, prune_slices=1, branches=0, grow_any=0, starve_ms=10**9)
        net = Network(n=4, sensors=0, start_lines=0, seed=0, params=p)
        for nrn in net.neurons:
            nrn.sign = 1
        net.connect(0, 3, 0.5)                              # neuron 0 keeps 3 lines: above its 2-line floor
        net.connect(0, 1, 0.2)
        net.connect(0, 2, 0.5)
        net.connect(2, 1, 0.5)                              # neuron 1 keeps 3 lines too
        net.connect(3, 1, 0.5)
        net.run(3000)                                       # nothing fires: every line is quiet
        assert (net.line_between(0, 1) is not None) == expect
