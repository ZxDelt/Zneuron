"""Watch the Hub Neuron network live in Chrome.

    .venv/Scripts/python.exe run_live.py                       (robot + world, 1000-spot cube brain)
    .venv/Scripts/python.exe run_live.py --no-body             (no robot: fixed sensor test pattern)
    .venv/Scripts/python.exe run_live.py --warmup 30000        (skip the first 30 s)
    .venv/Scripts/python.exe run_live.py --no-body --layout grid --neurons 100 --capacity 100 --sensors 5 --motors 0 --gap 5
    Old 2D view: http://127.0.0.1:8765/classic

Opens http://127.0.0.1:8765 in your browser. Press Ctrl+C in the terminal to stop.
"""

import argparse
import webbrowser
from pathlib import Path

from zneuron import world
from zneuron.network import Network
from zneuron.server import Sim, serve


def sensor_pattern(every_ms, gap_ms):
    """Stand-in for sensors: sensors fire one after another (0, 1, 2, ...) every `every_ms`.

    Each sensor gets exactly its threshold of charge, so it fires one clean spike
    (a bigger push would make it fire twice through carry-over).
    """
    def stimulus(net):
        sensors = [n.id for n in net.neurons if n.sensor]
        phase = net.t % every_ms
        if phase % gap_ms == 0 and phase // gap_ms < len(sensors):
            sid = sensors[phase // gap_ms]
            net.stimulate(sid, net.final_threshold(net.neurons[sid]))

    def install(net):
        """Same pattern as a schedule inside the compiled engine (no Python call per ms)."""
        sensors = [n.id for n in net.neurons if n.sensor]
        net.set_schedule(every_ms, [(k * gap_ms, sid, -1.0) for k, sid in enumerate(sensors) if k * gap_ms < every_ms])

    stimulus.install = install
    return stimulus


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-body", action="store_true",
                    help="no robot/world: sensors get a fixed test pattern instead (the older demo)")
    ap.add_argument("--layout", choices=["cube", "grid", "random"], default="cube",
                    help="cube = 3D (sensors on one face, motors on the opposite face); grid = 2D square grid")
    ap.add_argument("--neurons", type=int, default=900, help="neurons alive at start")
    ap.add_argument("--capacity", type=int, default=1000, help="spots in the brain (free spots let new neurons grow)")
    ap.add_argument("--sensors", type=int, default=None, help="default: what the body needs (86) or 25 without body")
    ap.add_argument("--motors", type=int, default=None, help="default: what the body needs (20) or 25 without body")
    ap.add_argument("--start-lines", type=int, default=2, help="input lines per neuron at start (R14: 2)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--every", type=int, default=50, help="ms between sensor sweeps")
    ap.add_argument("--gap", type=int, default=2, help="ms between one sensor and the next")
    ap.add_argument("--warmup", type=int, default=0, help="ms to simulate before showing")
    ap.add_argument("--speed", type=float, default=1, help="× real time at start (0.03, 0.1, 0.3, 1, 3, 10, ... 10000)")
    ap.add_argument("--reward-only", action="store_true",
                    help="resting Lfuel 0: learning only sticks when rewarded (thumbs up/down, flowers, rocks)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    args = ap.parse_args()

    body = not args.no_body
    sensors = args.sensors if args.sensors is not None else (world.N_SENSORS if body else 25)
    motors = args.motors if args.motors is not None else (world.N_MOTORS if body else 25)
    if body and args.layout == "cube" and args.sensors is None and args.motors is None:
        # the standard robot brain: senses on one face, motors on the other, sensor->motor reflex arcs
        net = world.make_brain(seed=args.seed, neurons=args.neurons, capacity=args.capacity)
    else:
        net = Network(n=args.neurons, capacity=args.capacity, sensors=sensors, seed=args.seed,
                      start_lines=args.start_lines, layout=args.layout, motors=motors)
    if args.reward_only:
        net.lfuel = 0.0
    if body:
        agent, stimulus = world.Agent(net, world.World(args.seed)), None
        agent.run(args.warmup)
    else:
        agent, stimulus = None, sensor_pattern(args.every, args.gap)
        stimulus.install(net)
        net.run(args.warmup)

    sim = Sim(net, stimulus, speed=args.speed, agent=agent, save_dir=Path(__file__).parent / "brains")
    server = serve(sim, port=args.port)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Zneuron live at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
