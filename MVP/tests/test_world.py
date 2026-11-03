"""World, robot body, senses, hearing chat, and saving/loading a brain."""

import numpy as np

from superneuron import saving, world as W
from superneuron.network import Network


def brain(seed=1):
    net = Network(n=300, capacity=343, sensors=W.N_SENSORS, motors=W.N_MOTORS, seed=seed, layout="cube")
    return net, W.World(seed)


def test_senses_have_the_right_shape_and_range():
    net, w = brain()
    rates, rewards, events = w.sense(0, 0.02)
    assert rates.shape == (W.N_SENSORS,)
    assert (rates >= 0).all() and (rates <= 0.5).all()
    assert w.senses["eyes"].shape == (2, W.RETINA, W.RETINA)
    assert w.senses["skin"][2] == 1.0 and w.senses["skin"][3] == 1.0       # standing: both feet on the ground


def test_eyes_see_sky_above_and_ground_below_in_daylight():
    _, w = brain()
    w.sense(30_000, 0.02)                         # a quarter into the day: sun is up
    eyes = w.senses["eyes"][0]
    assert eyes[0].mean() > eyes[-1].mean()


def test_night_is_darker_than_day():
    _, w = brain()
    w.sense(30_000, 0.02)
    day = w.senses["eyes"].mean()
    w.sense(90_000, 0.02)
    assert w.senses["eyes"].mean() < day


def test_chat_text_is_heard_as_sound():
    _, w = brain()
    w.hear("left", 0)
    heard = 0.0
    for t in range(0, 400, 20):
        w.sense(t, 0.02)
        heard += w.senses["ears"][:, 1:3].sum()   # speech uses the bands, wind is band 0
    assert heard > 0


def test_motors_move_the_body():
    net, w = brain()
    walk = [k for k, (name, _, _) in enumerate(W.JOINTS) if name == "walk"][0]
    counts = np.zeros(W.N_MOTORS)
    counts[2 * walk] = 1                          # the "walk forward" motor neuron fires every body tick
    for _ in range(100):
        w.apply_motors(counts, 0.02)
    assert w.z > 1.0                              # walked forward along +z


def test_agent_runs_and_babbles():
    net, w = brain()
    a = W.Agent(net, w)
    a.run(5000)
    assert net.t == 5000
    assert any(abs(v) > 0 for v in w.cmd.values())


def test_save_and_load_brain(tmp_path):
    net, w = brain()
    a = W.Agent(net, w)
    a.run(1000)
    path = tmp_path / "b.brain"
    saving.save(net, path, w)
    t, lines, x = net.t, len(net.lines), w.x
    a.run(3000)
    saving.load(net, path, w)
    assert net.t == t and len(net.lines) == lines and w.x == x


def test_reflex_arcs_let_senses_reach_motors():
    net = W.make_brain(seed=3, neurons=300, capacity=343)
    a = W.Agent(net, W.World(3))
    net.set_drive(net.motor_ids(), [0.0] * W.N_MOTORS)       # babbling off
    a.run(5000)
    fired = {k for t, k in net.fire_log if t > net.t - 3000}
    assert any(net.neurons[k].motor for k in fired)


def test_teacher_runs_lessons():
    from superneuron.teacher import Teacher
    net = W.make_brain(seed=4, neurons=300, capacity=343)
    teacher = Teacher(W.Agent(net, W.World(4)), seed=4)
    for _ in range(5):
        assert teacher.trial() in ("right", "wrong", "none")
    assert len(teacher.results) == 5
