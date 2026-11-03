"""Behaviour probe: why does a bee deliver more or less? One match, then per bee:
motor drive per action, how much it turns vs walks, time at walls, time near squares, time carrying,
how steering follows what the eyes see (does it turn toward squares / the spot?), firing per layer.

    .venv/Scripts/python.exe -m bees.probe --condition basic --scale 1 --seed 3 --minutes 5
"""

import argparse
import json
import math

import numpy as np

from . import flybrain as F
from .brains import Match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="basic")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--over", default="{}")
    a = ap.parse_args()
    over = json.loads(a.over)
    if over:
        from superneuron import params as PP
        from . import brains as B
        base = PP.v2
        B.v2 = lambda **kw: base(**{**over, **kw})
    m = Match(a.condition, a.seed, 20, a.scale, models=["hub2", "lif", "alif"])
    stats = {k: dict(act=np.zeros(4), turn=0.0, speed=0.0, wall=0, near=0, carry=0, steer_ok=0, steer_n=0)
             for k in m.bees}
    steps = int(a.minutes * 60_000 // 20)
    for _ in range(steps):
        # read the senses the brain is about to get, to check steering toward targets
        sees = {}
        for k, b in m.bees.items():
            d = m.world.sense(b.body)
            per = F.SECTORS * F.EYE_COPIES
            kind = 1 if b.body.carrying else 0                       # carrying: look for the spot, else squares
            sec = np.array([d[kind * per + s * F.EYE_COPIES] for s in range(F.SECTORS)])
            sees[k] = sec
        m.step()
        for k, b in m.bees.items():
            s, body = stats[k], b.body
            s["turn"] += abs(body.turn)
            s["speed"] += body.speed
            w = m.world
            if min(body.x, body.y, w.size - body.x, w.size - body.y) < 1.0:
                s["wall"] += 1
            if any(math.hypot(x - body.x, y - body.y) < 3 for x, y in w.squares):
                s["near"] += 1
            s["carry"] += body.carrying
            sec = sees[k]
            if sec.max() > 0:
                side = int(np.argmax(sec)) - F.SECTORS // 2         # >0: target on the left (sector 6 = far left)
                if side != 0 and abs(body.turn) > 0.2:
                    s["steer_n"] += 1
                    s["steer_ok"] += (side > 0) == (body.turn > 0)
    print(f"{a.condition} x{a.scale:g} seed {a.seed}, {a.minutes} min")
    for k, b in m.bees.items():
        s = stats[k]
        f = b.brain.fires_n()
        rates = {g: round(float(f[x:y].sum()) / (y - x) / (a.minutes * 60), 1) for g, (x, y) in F.GROUPS.items()}
        print(f"  {k:5s} delivered {b.body.stats['deliveries']:3d} picked {b.body.stats['pickups']:3d} "
              f"| turn {s['turn']/steps:4.2f} speed {s['speed']/steps:4.2f} wall {100*s['wall']/steps:3.0f}% "
              f"near-square {100*s['near']/steps:3.0f}% carrying {100*s['carry']/steps:3.0f}% "
              f"| steers toward target {100*s['steer_ok']/max(1,s['steer_n']):3.0f}% of {s['steer_n']} turns | Hz {rates}")


if __name__ == "__main__":
    main()
