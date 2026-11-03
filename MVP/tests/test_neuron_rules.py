"""One test per spec rule, on tiny hand-wired networks."""

import pytest

from superneuron.network import Network


def make(n=3):
    net = Network(n=n, sensors=0, start_lines=0, seed=0)
    for nrn in net.neurons:
        nrn.sign = 1
    return net


def run(net, ticks):
    log = []
    for _ in range(ticks):
        t = net.t
        log += [(t, f.id) for f in net.step()]
    return log


def test_fire_and_carry_over():          # #6, D4
    net = make()
    net.stimulate(0, 0.8)
    assert [f.id for f in net.step()] == [0]
    assert net.neurons[0].hub == pytest.approx(0.3)


def test_rest_blocks_firing_for_2_ms():   # D3, #27 bursts
    net = make()
    net.stimulate(0, 1.0)
    assert run(net, 5) == [(0, 0), (3, 0)]  # hub 1 -> fire, keep 0.5 -> fire again after rest


def test_conduction_delay_2_ms():         # #11
    net = make()
    net.connect(0, 1, 1.0)
    net.stimulate(0, 0.5)
    log = run(net, 4)
    assert (0, 0) in log and (2, 1) in log


def test_pool_holds_charge_until_full():  # F1
    net = make()
    net.connect(0, 2, 0.3)
    net.connect(1, 2, 0.3)                # neuron 2 has 2 lines -> pool capacity 0.5
    net.stimulate(0, 0.5)
    run(net, 3)
    n2 = net.neurons[2]
    assert n2.hub == 0.0
    assert sum(n2.pools.values()) == pytest.approx(0.3)


def test_window_dumps_charge_after_20_ms():  # #5
    net = make()
    net.stimulate(0, 0.3)                 # below threshold, never fires
    run(net, 19)
    assert net.neurons[0].hub == pytest.approx(0.3)
    run(net, 2)
    assert net.neurons[0].hub == 0.0
    assert 0 not in net.awake             # goes to sleep (#26)


def test_inhibition_goes_to_hub_with_floor():  # #15
    net = make()
    net.neurons[0].sign = -1
    net.connect(0, 1, 0.4)
    net.stimulate(1, 0.3)
    net.stimulate(0, 0.5)
    run(net, 3)                            # inhibitory spike lands at t=2
    assert net.neurons[1].hub == 0.0       # 0.3 − 0.4 floors at 0
    assert net.lines[0].ab.thickness == pytest.approx(-0.4)


def test_ltp_when_input_arrives_before_fire():  # #17 #19
    net = make()
    m = net.connect(0, 1, 0.6)
    net.stimulate(0, 0.5)
    net.neurons[1].hub = 0.0
    run(net, 2)
    net.stimulate(1, 0.5)                  # arrival at t=2, receiver fires at t=2 (gap 0)
    run(net, 1)
    assert m.size > 0.6


def test_ltd_when_input_arrives_after_fire():  # #18 #19
    net = make()
    m = net.connect(0, 1, 0.6)
    net.stimulate(1, 0.5)                  # receiver fires at t=0
    net.stimulate(0, 0.5)                  # sender fires at t=0, arrives at t=2
    run(net, 3)
    assert m.size < 0.6


def test_fatigue_weakens_repeated_spikes():  # #20
    net = make()
    m = net.connect(0, 1, 0.5)
    net.stimulate(0, 0.5)
    run(net, 1)
    assert m.energy == pytest.approx(0.7)


def test_efuel_blocks_firing_when_empty():  # #36
    net = make()
    net.neurons[0].efuel = 0.0
    net.neurons[0].efuel_t = 0
    net.stimulate(0, 1.0)
    assert net.step() == []


def test_threshold_clamped_below_hub_cap():  # F5
    net = make()
    nrn = net.neurons[0]
    nrn.thr_h, nrn.rfuel_local = 0.8, 2.0
    assert net.final_threshold(nrn) == 0.95


def test_sign_ignored_for_limits():       # #15 limits use size
    net = make()
    net.neurons[0].sign = -1
    m = net.connect(0, 1, 0.5)
    assert m.thickness == -0.5 and m.size == 0.5
