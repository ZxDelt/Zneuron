"""Reward timing, sensor drive, motor counting, neurogenesis and neuron death."""

import pytest

from superneuron.network import Network
from superneuron.params import Params


def pair():
    net = Network(n=2, sensors=0, start_lines=0, seed=0)
    for nrn in net.neurons:
        nrn.sign = 1
    return net


def learn_once(net, m):
    """0 fires, its spike reaches 1 at t=2 and 1 fires then -> LTP on 0->1."""
    net.stimulate(0, 0.5)
    net.run(2)
    net.stimulate(1, 0.5)
    net.run(1)
    return m.size


def test_reward_plus_repeats_recent_learning():      # #30
    net = pair()
    m = net.connect(0, 1, 0.5)
    after_ltp = learn_once(net, m)
    gained = after_ltp - 0.5
    assert gained > 0
    assert net.reward(1.0) >= 1
    assert m.size == pytest.approx(after_ltp + gained, rel=0.05)


def test_reward_minus_undoes_recent_learning():      # #30
    net = pair()
    m = net.connect(0, 1, 0.5)
    learn_once(net, m)
    net.reward(-1.0)
    assert m.size == pytest.approx(0.5, abs=0.002)


def test_old_learning_is_not_rewarded():             # eligibility fades over ~1 s
    net = pair()
    m = net.connect(0, 1, 0.5)
    after_ltp = learn_once(net, m)
    net.run(3000)
    net.reward(1.0)
    assert m.size == pytest.approx(after_ltp, abs=1e-4)


def test_sensor_drive_is_rate_coded():                # #25: stronger sense = more spikes
    counts = []
    for rate in (0.005, 0.02, 0.1):                   # spikes per ms asked for
        net = Network(n=3, sensors=1, start_lines=0, seed=0)
        net.set_drive([0], [rate])
        net.run(1000)
        counts.append(sum(1 for t, k in net.fire_log if k == 0))
    assert counts[0] == 5 and counts[1] == 20
    assert 40 <= counts[2] < 100                      # Efuel (#36) caps a constantly driven neuron
    net.set_drive([0], [0.0])
    before = len(net.fire_log)
    net.run(500)
    assert len(net.fire_log) == before


def test_motor_spikes_are_counted_and_reset():
    net = Network(n=4, sensors=1, motors=1, start_lines=0, seed=0)
    motor = net.motor_ids()[0]
    net.stimulate(motor, 1.0)
    net.run(10)
    assert net.read_motors([motor])[0] >= 1
    assert net.read_motors([motor])[0] == 0


def test_idle_neuron_dies_and_spot_is_free():         # #37
    net = Network(n=3, sensors=0, start_lines=0, seed=0, params=Params(death_ms=2000, starve_ms=10**9))
    net.run(3001)
    assert net.counts["death"] == 3 and net.alive_count() == 0


def test_region_grows_a_new_neuron_when_starving():   # #37
    p = Params(max_lines=2, birth_starving=3, birth_gap_ms=1000, starve_ms=10**9)
    net = Network(n=8, capacity=9, sensors=0, seed=0, start_lines=2, layout="grid", params=p)
    assert net.alive_count() == 8
    net.run(2001)
    assert net.counts["birth"] >= 1 and net.alive_count() == 9
    born = [n for n in net.neurons if n.alive and n.id == 8][0]
    assert len(born.lines) >= 1                        # wired in from busy neighbours


def test_reward_only_reaches_lines_into_neurons_that_just_fired():   # targeted reward
    net = pair()
    m = net.connect(0, 1, 0.5)
    after_ltp = learn_once(net, m)
    net.run(400)                                       # neuron 1 fired > 300 ms ago
    net.reward(1.0)
    assert m.size == pytest.approx(after_ltp, abs=1e-6)


def test_resting_lfuel_zero_learns_only_with_reward():                # three-factor rule (#30 proposal)
    net = pair()
    net.lfuel = 0.0
    m = net.connect(0, 1, 0.5)
    assert learn_once(net, m) == pytest.approx(0.5)   # timing alone changes nothing
    net.reward(1.0)
    assert m.size > 0.5                                # the reward makes it stick
