import json
import threading
import urllib.error
import urllib.request

from run_live import sensor_pattern
from superneuron import insight
from superneuron.network import Network
from superneuron.server import Sim, serve


def chain_net():
    """0 -> 1 -> 2, strong lines, so firing 0 makes 1 then 2 fire."""
    net = Network(n=3, sensors=1, start_lines=0, seed=0)
    for nrn in net.neurons:
        nrn.sign = 1
    net.connect(0, 1, 1.0)
    net.connect(1, 2, 1.0)
    return net


def run(net, ticks):
    for _ in range(ticks):
        net.step()


def test_why_fired_traces_back_to_outside_input():
    net = chain_net()
    net.stimulate(0, 1.0)
    run(net, 6)
    tree = insight.why_fired(net, 2)
    assert tree["id"] == 2
    assert tree["causes"][0]["id"] == 1
    assert tree["causes"][0]["causes"][0]["id"] == 0
    assert tree["causes"][0]["causes"][0]["causes"][0]["outside"]


def test_chains_show_cause_and_effect():
    net = chain_net()
    net.stimulate(0, 1.0)
    run(net, 6)
    assert [0, 1, 2] in insight.chains(net)


def test_groups_keep_their_name():
    net = Network(n=40, sensors=4, seed=1, start_lines=4)
    stim = sensor_pattern(50, 5)
    tracker = insight.GroupTracker()
    for _ in range(3000):
        stim(net)
        net.step()
    first = {g["name"] for g in tracker.update(net)}
    for _ in range(100):
        stim(net)
        net.step()
    second = {g["name"] for g in tracker.update(net)}
    assert first and first & second            # same thoughts keep the same name


def test_grid_layout_wires_direct_neighbours_and_grows():
    net = Network(n=100, sensors=5, seed=1, layout="grid")
    step = net.size / 10
    assert all(n.x == 0.5 * step for n in net.neurons[:5])          # sensors on the left edge
    for line in net.lines.values():
        a, b = net.neurons[line.a], net.neurons[line.b]
        assert round(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5) == round(step)  # direct neighbours
    stim = sensor_pattern(50, 5)
    for _ in range(3000):
        stim(net)
        net.step()
    assert net.counts["grow"] > 0


def test_alarm_on_silence():
    net = Network(n=10, sensors=0, seed=0, start_lines=2)
    run(net, 2500)
    assert any("Silent" in msg for _, msg in insight.alarms(net))


def test_server_streams_frames_and_takes_commands():
    net = Network(n=12, sensors=2, seed=3)
    sim = Sim(net, sensor_pattern(50, 5), speed=5)
    server = serve(sim, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
        assert b"dockview-core.min.js" in page
        assert b"SuperNeuron Live" in urllib.request.urlopen(f"http://127.0.0.1:{port}/classic", timeout=5).read()
        js = urllib.request.urlopen(f"http://127.0.0.1:{port}/js/brain3d.js", timeout=5)
        assert js.headers["Content-Type"].startswith("text/javascript")
        try:                                   # nothing outside web/ is served
            urllib.request.urlopen(f"http://127.0.0.1:{port}/../run_live.py", timeout=5)
            raise AssertionError("served a file outside web/")
        except urllib.error.HTTPError as err:
            assert err.code == 404
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/stream", timeout=5) as stream:
            line = stream.readline()
            while not line.startswith(b"data:"):
                line = stream.readline()
        frame = json.loads(line[5:])
        assert len(frame["neurons"]) == 12 and "detail" in frame
        req = urllib.request.Request(f"http://127.0.0.1:{port}/control",
                                     data=json.dumps({"cmd": "select", "id": 5}).encode(), method="POST")
        urllib.request.urlopen(req, timeout=5)
        assert sim.selected == 5
    finally:
        sim.running = False
        server.shutdown()
