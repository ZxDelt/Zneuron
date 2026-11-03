"""2D arena shared by both bees: pick up cyan squares and place them on the magenta spot. Red squares are decoys.

One World (squares, decoys, spot) holds several Bodies. Squares are shared: a square one bee takes is gone for the
other. Bees do not see or collide with each other. Each bee senses with compound-eye sectors and body sense
(flybrain.py) and moves with 4 actions: turn left, turn right, forward, grab. One body step = 20 ms of brain time.
"""

import math
import random

import numpy as np

from . import flybrain as F

BODY_MS = 20
DT = BODY_MS / 1000.0
FOV = math.radians(210)
SECTOR = FOV / F.SECTORS
EYE_RATE = 0.08          # spikes per ms for an object right in front (80 Hz)
CTX_RATE = 0.03
TOUCH_RATE = 0.08
BEE_R, TOUCH_D = 0.5, 1.1
GRAB_SPIKES = 2          # grab motor spikes in one body step that make a grab
SHAPE_MS = 200

# a carried square stays held until it is placed on the spot; closer and farther cancel out
R_PICK, R_DELIVER, R_DECOY, R_BUMP, R_CLOSER = 0.4, 1.0, -0.6, -0.3, 0.1

# conditions to compare the two bees in (squares scale with two bees sharing the arena)
CONDITIONS = {
    "basic":       dict(label="Basic", size=26.0, squares=6, reds=0, eye_range=12.0,
                        about="6 squares, spot in the middle, clear senses"),
    "far":         dict(label="Far & sparse", size=42.0, squares=4, reds=0, eye_range=14.0,
                        about="bigger arena, fewer squares: long trips"),
    "decoys":      dict(label="Decoys", size=26.0, squares=5, reds=4, eye_range=12.0,
                        about="4 red decoys: grabbing one costs reward"),
    "moving_spot": dict(label="Moving spot", size=26.0, squares=6, reds=0, eye_range=12.0, move_spot_ms=60_000,
                        about="the drop spot jumps every minute: must re-adapt"),
    "noisy":       dict(label="Noisy senses", size=26.0, squares=6, reds=0, eye_range=12.0, noise=0.25,
                        about="25% of sense readings blanked, plus random flicker"),
    "damaged":     dict(label="Brain damage", size=26.0, squares=6, reds=0, eye_range=12.0, damage=0.2,
                        about="20% of connections cut halfway through (same ones in both brains)"),
}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Body:
    def __init__(self, name, x, y, heading, noise_seed):
        self.name = name
        self.x, self.y, self.heading = x, y, heading
        self.speed = self.turn = 0.0
        self.carrying = False
        self.bump_until = 0
        self.last_bump = -10_000
        self.target_d = None
        self.rng = random.Random(noise_seed)          # sense noise: same stream for both bees
        self.stats = dict(deliveries=0, pickups=0, decoys=0, drops=0, bumps=0, reward=0.0, first_delivery_ms=None,
                          distance=0.0)
        self.events = []
        self.senses = np.zeros(F.N_SENSORS)

    def snapshot(self, world):
        return dict(x=round(self.x, 3), y=round(self.y, 3), heading=round(self.heading, 3), carrying=self.carrying,
                    bump=world.t < self.bump_until)


class World:
    def __init__(self, condition="basic", seed=1, names=("hub", "lif")):
        self.cond_name = condition
        self.c = dict(CONDITIONS[condition])
        self.rng = random.Random(seed)
        self.size = self.c["size"]
        self.eye_range = self.c["eye_range"]
        self.noise = self.c.get("noise", 0.0)
        h = self.size / 2
        self.spot = [h, h, 2.5]
        self.squares = [self._free_xy() for _ in range(self.c["squares"])]
        self.reds = [self._free_xy() for _ in range(self.c["reds"])]
        self.t = 0
        # starts evenly spaced on a circle around the spot, each facing the spot at the same offset (identical views
        # up to rotation); which bee gets which start rotates with the seed
        k = len(names)
        phi0 = self.rng.uniform(-math.pi, math.pi)
        off = self.rng.uniform(-math.pi, math.pi)
        starts = []
        for j in range(k):
            ang = phi0 + 2 * math.pi * j / k
            starts.append((h + 6 * math.cos(ang), h + 6 * math.sin(ang), wrap(ang + math.pi + off)))
        self.bodies = {n: Body(n, *starts[(j + seed) % k], noise_seed=seed * 31 + 7) for j, n in enumerate(names)}

    # ---------- layout ----------

    def _free_xy(self):
        for _ in range(100):
            x, y = self.rng.uniform(1.5, self.size - 1.5), self.rng.uniform(1.5, self.size - 1.5)
            if math.hypot(x - self.spot[0], y - self.spot[1]) > self.spot[2] + 2.0:
                return [x, y]
        return [1.5, 1.5]

    def _move_spot(self):
        m = self.spot[2] + 2
        self.spot[0], self.spot[1] = self.rng.uniform(m, self.size - m), self.rng.uniform(m, self.size - m)
        for b in self.bodies.values():
            b.target_d = None

    # ---------- senses ----------

    @staticmethod
    def _touching(b, objs):
        for k, (ox, oy) in enumerate(objs):
            if math.hypot(ox - b.x, oy - b.y) <= TOUCH_D:
                return k
        return None

    def on_spot(self, b):
        return math.hypot(self.spot[0] - b.x, self.spot[1] - b.y) <= self.spot[2]

    def sense(self, b):
        """Drive for every sensor neuron of body b, spikes per ms."""
        d = np.zeros(F.N_SENSORS)
        seen = {k: np.zeros(F.SECTORS) for k in F.KINDS}
        objs = [("square", s[0], s[1]) for s in self.squares] + [("red", r[0], r[1]) for r in self.reds]
        objs.append(("spot", self.spot[0], self.spot[1]))
        for kind, ox, oy in objs:
            dx, dy = ox - b.x, oy - b.y
            dist = math.hypot(dx, dy)
            if dist > self.eye_range:
                continue
            rel = wrap(math.atan2(dy, dx) - b.heading)
            if kind == "spot" and dist <= self.spot[2]:
                rel = 0.0                                   # standing on it: it fills the front view
            if abs(rel) > FOV / 2:
                continue
            sec = min(F.SECTORS - 1, int((rel + FOV / 2) / SECTOR))
            seen[kind][sec] = max(seen[kind][sec], 1.0 - dist / self.eye_range)
        for kind in F.KINDS:
            for sec in range(F.SECTORS):
                for c in range(F.EYE_COPIES):
                    d[F.eye_id(kind, sec, c)] = EYE_RATE * seen[kind][sec]
        touching = self._touching(b, self.squares) is not None or self._touching(b, self.reds) is not None
        ctx = {"carrying": CTX_RATE if b.carrying else 0.0,
               "empty": 0.0 if b.carrying else CTX_RATE,
               "touch": TOUCH_RATE if not b.carrying and touching else 0.0,
               "on_spot": TOUCH_RATE if self.on_spot(b) else 0.0,
               "bump": 0.1 if self.t < b.bump_until else 0.0}
        for name, rate in ctx.items():
            d[F.context_ids(name)] = rate
        if self.noise:
            blank = np.array([b.rng.random() < self.noise for _ in range(F.N_SENSORS)])
            d[blank] = 0.0
            d += np.array([b.rng.random() * 0.01 for _ in range(F.N_SENSORS)])
        b.senses = d
        return d

    # ---------- bodies ----------

    def act(self, b, counts):
        """counts: spikes per motor neuron in the last body step. Moves body b, returns its reward."""
        a = {name: int(counts[[m - F.N_SENSORS for m in F.motor_ids(name)]].sum()) for name in F.ACTIONS}
        r = 0.0
        events = []
        turn_target = max(-4.0, min(4.0, 2.5 * (a["left"] - a["right"])))
        speed_target = min(4.0, 1.5 * a["forward"])
        b.turn += (turn_target - b.turn) * 0.25
        b.speed += (speed_target - b.speed) * 0.2
        b.heading = wrap(b.heading + b.turn * DT)
        nx = b.x + math.cos(b.heading) * b.speed * DT
        ny = b.y + math.sin(b.heading) * b.speed * DT
        lo, hi = BEE_R, self.size - BEE_R
        if not (lo <= nx <= hi and lo <= ny <= hi):
            nx, ny = min(hi, max(lo, nx)), min(hi, max(lo, ny))
            b.speed *= 0.3
            b.bump_until = self.t + 200
            if self.t - b.last_bump > 500:
                r += R_BUMP
                b.stats["bumps"] += 1
                events.append("bump")
            b.last_bump = self.t
        b.stats["distance"] += math.hypot(nx - b.x, ny - b.y)
        b.x, b.y = nx, ny

        if a["grab"] >= GRAB_SPIKES:
            if b.carrying:
                if self.on_spot(b):                                 # place it: grab while standing on the spot
                    b.carrying = False
                    r += R_DELIVER
                    b.stats["deliveries"] += 1
                    if b.stats["first_delivery_ms"] is None:
                        b.stats["first_delivery_ms"] = self.t
                    self.squares.append(self._free_xy())           # a new square appears somewhere
                    events.append("deliver")
                    b.target_d = None
                else:
                    b.stats["drops"] += 1                           # grab off the spot: nothing happens
            else:
                k = self._touching(b, self.squares)
                if k is not None:
                    self.squares.pop(k)
                    b.carrying = True
                    r += R_PICK
                    b.stats["pickups"] += 1
                    events.append("pickup")
                    b.target_d = None
                else:
                    k = self._touching(b, self.reds)
                    if k is not None:
                        self.reds[k] = self._free_xy()
                        r += R_DECOY
                        b.stats["decoys"] += 1
                        events.append("decoy")

        # shaping: getting closer to the current goal (nearest square, or the spot when carrying)
        if (self.t + BODY_MS) % SHAPE_MS == 0:
            d = self._goal_distance(b)
            if d is not None and b.target_d is not None:
                if d < b.target_d - 0.15:
                    r += R_CLOSER
                elif d > b.target_d + 0.15:
                    r -= R_CLOSER
            b.target_d = d
        b.stats["reward"] += r
        b.events = events
        return max(-1.0, min(1.0, r))

    def tick(self):
        """After all bodies acted: the world clock moves on one body step."""
        self.t += BODY_MS
        mv = self.c.get("move_spot_ms")
        if mv and self.t % mv == 0:
            self._move_spot()
            return ["spot moved"]
        return []

    def _goal_distance(self, b):
        if b.carrying:
            return math.hypot(self.spot[0] - b.x, self.spot[1] - b.y)
        if not self.squares:
            return None
        return min(math.hypot(sx - b.x, sy - b.y) for sx, sy in self.squares)

    def snapshot(self):
        return dict(size=self.size, spot=[round(v, 2) for v in self.spot],
                    squares=[[round(a, 2), round(b, 2)] for a, b in self.squares],
                    reds=[[round(a, 2), round(b, 2)] for a, b in self.reds], eye_range=self.eye_range,
                    bees={n: b.snapshot(self) for n, b in self.bodies.items()})
