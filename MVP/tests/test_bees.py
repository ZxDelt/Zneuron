"""Bee lab: fly wiring limits, shared arena rules, both neuron models, fairness, the comparison runner."""

import math

import numpy as np

from bees import arena as A
from bees import flybrain as F
from bees.brains import Match, make_brain
from bees.compare import paired_test, run_match

GRAB = np.zeros(F.N_MOTORS)
GRAB[[m - F.N_SENSORS for m in F.motor_ids("grab")]] = 1             # 4 grab spikes = a grab


def test_fly_wiring_respects_the_8_line_limit():
    for seed in range(5):
        fly = F.build(seed)
        deg = np.zeros(fly.n, int)
        for a, b, _ in fly.edges:
            deg[a] += 1
            deg[b] += 1
        assert deg.max() <= 8 and fly.n == F.N
        assert all(deg[F.KC0:F.MBON0] >= 2)                 # every Kenyon cell is wired in and out


def test_both_bees_start_identical():                       # fairness: same wiring, same start weights
    flies = [make_brain(m, 3, 2.0)[1] for m in ("hub2", "hub1", "lif", "alif")]
    assert all(f.edges == flies[0].edges and f.sign == flies[0].sign for f in flies)


def test_hub_bees_have_no_add_ons():                        # no targeted reward, three-factor rule like LIF
    m = Match("basic", 1)
    for k in ("hub2", "hub1"):
        net = m.bees[k].brain.net
        assert net.p.reward_window_ms == 0 and net.lfuel == 0


def test_eyes_see_a_square_ahead():
    w = A.World("basic", 1)
    b = w.bodies["hub"]
    w.squares = [[b.x + 3 * math.cos(b.heading), b.y + 3 * math.sin(b.heading)]]
    d = w.sense(b)
    assert d[F.eye_id("square", F.SECTORS // 2, 0)] > 0
    assert d[F.eye_id("square", 0, 0)] == 0


def test_starts_are_fair():
    """Every bee starts at the same distance from the spot, seeing it at the same angle (views equal up to rotation)."""
    names = ("hub2", "hub1", "lif", "alif")
    for seed in (1, 2, 3):
        w = A.World("basic", seed, names=names)
        h = w.size / 2
        dists, views = set(), set()
        for b in w.bodies.values():
            dists.add(round(math.hypot(b.x - h, b.y - h), 6))
            views.add(round(A.wrap(math.atan2(h - b.y, h - b.x) - b.heading), 6))
        assert len(dists) == 1 and len(views) == 1


def test_pick_up_then_place_on_the_spot():
    w = A.World("basic", 1)
    b = w.bodies["hub"]
    w.squares = [[b.x, b.y]]
    assert w.act(b, GRAB) > 0 and b.carrying
    w.act(b, GRAB)                                         # grabbing off the spot does nothing
    assert b.carrying and b.stats["drops"] == 1
    b.x, b.y = w.spot[0], w.spot[1]
    assert w.act(b, GRAB) >= A.R_DELIVER - 0.2 and not b.carrying
    assert b.stats["deliveries"] == 1 and len(w.squares) == 1


def test_squares_are_shared():
    w = A.World("basic", 1)
    hub, lif = w.bodies["hub"], w.bodies["lif"]
    lif.x, lif.y = hub.x, hub.y
    w.squares = [[hub.x, hub.y]]
    w.act(hub, GRAB)
    w.act(lif, GRAB)
    assert hub.carrying and not lif.carrying               # the first bee took the only square


def test_match_runs_all_bees_and_damage_hits_all():
    m = Match("damaged", 2, minutes=1)
    m.run(29_980)                                          # damage lands at 30 s
    before = {k: b.brain.line_count() for k, b in m.bees.items()}
    m.run(40)
    for k, b in m.bees.items():
        assert b.spikes() > 0 and b.body.stats["distance"] > 0
        assert b.damaged and b.brain.line_count() < before[k]


def test_compare_match_and_paired_test():
    r = run_match("basic", 1, 1, 2.0)
    assert set(r["bees"]) == {"hub2", "hub1", "lif", "alif"} and len(r["bees"]["hub2"]["curve"]) == 1
    assert paired_test([5, 6, 4, 7, 5, 6, 8, 5]) < 0.01
    assert paired_test([1, -1, 2, -2]) == 1.0
