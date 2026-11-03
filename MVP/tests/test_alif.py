"""Adaptive LIF bee neuron (bees/alif.py): adaptive threshold, same interface and learning rule as the LIF."""

from types import SimpleNamespace

import numpy as np

from bees import flybrain as F
from bees.alif import ALIFBrain, BETA, TAU_A
from bees.lif import THR, LIFBrain

SENSOR, MOTOR, CELL = 0, 1, 2


def tiny(w=0.3):                                          # one sensor -> one cell, plus an unused motor
    return SimpleNamespace(n=3, sensors=1, motors=1, sign=[1.0, 1.0, 1.0], edges=[(SENSOR, CELL, w)])


def fly_brains(seed=3, scale=2.0):
    fly = F.build(seed)
    fly.edges = [(a, b, min(1.0, w * scale)) for a, b, w in fly.edges]
    return LIFBrain(fly, seed), ALIFBrain(fly, seed), fly


def test_threshold_rises_with_firing_and_decays_back():
    b = ALIFBrain(tiny())
    assert b.thresholds()[CELL] == THR
    b.set_drive([SENSOR], [0.2])
    b.run(500)
    assert b.fires_n()[CELL] > 0 and b.thresholds()[CELL] > THR + 0.1
    high = b.thresholds()[CELL] - THR
    b.set_drive([SENSOR], [0.0])
    b.run(int(5 * TAU_A))
    assert b.fires_n()[CELL] > 0 and b.thresholds()[CELL] - THR < 0.01 * high + 1e-9


def test_alif_fires_less_than_lif_under_the_same_drive():
    lif, alif = LIFBrain(tiny()), ALIFBrain(tiny())
    for b in (lif, alif):
        b.set_drive([SENSOR], [0.2])
        b.run(2000)
    assert alif.fires_n()[SENSOR] == lif.fires_n()[SENSOR]       # same sensor coding: input is identical
    assert 0 < alif.fires_n()[CELL] < 0.8 * lif.fires_n()[CELL]


def test_interface_matches_the_lif_bee():
    lif, alif, fly = fly_brains()
    assert alif.kind == "alif" and alif.t == 0 and alif.line_count() == lif.line_count() == len(fly.edges)
    sensors = list(range(F.N_SENSORS))
    motors = list(range(F.N_SENSORS, F.N_SENSORS + F.N_MOTORS))
    for b in (lif, alif):
        b.set_drive(motors, [0.01] * len(motors))
        b.set_drive(sensors[:8], [0.1] * 8)
        b.run(1000)
    assert alif.t == 1000
    assert np.array_equal(alif.fires_n()[:8], lif.fires_n()[:8])    # sensor spike trains are identical
    m = alif.read_motors(motors)
    assert m.shape == (F.N_MOTORS,) and m.sum() > 0 and alif.read_motors(motors).sum() == 0
    ts, ids = alif.recent_fires(900)
    assert len(ts) > 0 and ts.min() >= 900 and ids.max() < fly.n
    src, dst, w = alif.lines()
    assert len(src) == alif.line_count() and (w > 0).all()
    a, b, _ = fly.edges[0]
    alif.cut([(a, b)])
    assert alif.weight(a, b) == 0.0 and alif.line_count() == len(fly.edges) - 1


def test_reward_changes_weights_only_with_eligibility():
    b = ALIFBrain(tiny())
    assert b.reward(1.0) == 0 and b.weight(SENSOR, CELL) == 0.3      # nothing fired yet: no eligibility
    b.set_drive([SENSOR], [0.2])
    b.run(200)                                            # sensor/cell spike pairs leave an STDP trace
    assert b.reward(1.0) == 1 and b.weight(SENSOR, CELL) != 0.3
    w = b.weight(SENSOR, CELL)
    assert b.reward(1.0) == 0 and b.weight(SENSOR, CELL) == w        # trace used up by the reward


def test_alif_is_bare_bones():
    assert TAU_A == 200.0 and BETA == 0.05
    b = ALIFBrain(tiny())
    assert set(b.S._fields) - set(LIFBrain(tiny()).S._fields) == {"a"}     # only the adaptation variable is added
