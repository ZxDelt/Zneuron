"""The world and the robot body, simulated in Python on the same clock as the brain.

- World: endless flat ground made of 16 m chunks (generated on demand from the seed), flowers, rocks,
  a day/night cycle, and sounds (birds near flowers, wind, footsteps, your chat as speech).
- Body: a ZX-style robot. Motor neurons drive head, torso, shoulders, elbows, walk and turn.
- Senses (the only inputs): eyes (6x6 retina per eye), ears (left/right x 4 pitch bands), skin (6 touch points).
Coordinates: metres, y up, a heading of 0 faces +z.
"""

import math
import random

import numpy as np
from numba import njit

CHUNK = 16.0
VIEW = 34.0                    # objects within this distance are simulated and drawn
DAY_MS = 120_000               # one day-night cycle (brain time)
BODY_MS = 20                   # body + senses update every 20 ms of brain time
RETINA = 6                     # retina is RETINA x RETINA per eye
FOV = math.radians(70)
BANDS = 8                      # pitch bands per ear: 0 low (wind, footsteps), 1-6 speech, 7 high (birds)
SKIN = ("hand L", "hand R", "foot L", "foot R", "chest", "back")

# motor pairs: (name, + neuron role, - neuron role); each pair is 2 motor neurons pulling opposite ways
JOINTS = (
    ("head turn", "left", "right"), ("head nod", "up", "down"), ("torso twist", "left", "right"),
    ("torso bend", "forward", "back"), ("left shoulder", "raise", "lower"), ("left elbow", "bend", "straighten"),
    ("right shoulder", "raise", "lower"), ("right elbow", "bend", "straighten"),
    ("walk", "forward", "back"), ("turn", "left", "right"),
)
LIMITS = {  # radians (walk/turn are speeds, no limit)
    "head turn": (-1.2, 1.2), "head nod": (-0.7, 0.6), "torso twist": (-0.8, 0.8), "torso bend": (-0.3, 1.2),
    "left shoulder": (-0.5, 3.0), "left elbow": (0.0, 2.4), "right shoulder": (-0.5, 3.0), "right elbow": (0.0, 2.4),
}
N_SENSORS = 2 * RETINA * RETINA + 2 * BANDS + len(SKIN)
BABBLE, BABBLE_MIN = 0.01, 0.002     # motor babbling, spikes per ms: starts at 10/s, fades toward 2/s
N_MOTORS = 2 * len(JOINTS)

# letters -> (band, ear loudness) for heard speech: each letter is 60 ms of sound in one band
SPEECH_MS = 60


@njit(cache=True)
def _cast(origin, dirs, sph, sun, light):
    """Brightness seen along each ray: sky, sun, ground, or the nearest flower/rock sphere."""
    out = np.empty(dirs.shape[0])
    for r in range(dirs.shape[0]):
        dx, dy, dz = dirs[r, 0], dirs[r, 1], dirs[r, 2]
        if dy > 0:
            val = 0.15 + 0.75 * light
            if dx * sun[0] + dy * sun[1] + dz * sun[2] > 0.985:
                val = max(val, light)
            near = np.inf
        else:
            near = -origin[1] / min(dy, -1e-6)
            val = (0.45 * light + 0.03) * math.exp(-near / 40)
        for k in range(sph.shape[0]):
            ox, oy, oz = origin[0] - sph[k, 0], origin[1] - sph[k, 1], origin[2] - sph[k, 2]
            b = dx * ox + dy * oy + dz * oz
            c = ox * ox + oy * oy + oz * oz - sph[k, 3] ** 2
            disc = b * b - c
            if disc > 0:
                t = -b - math.sqrt(disc)
                if 0 < t < near:
                    near = t
                    val = (sph[k, 4] * light + 0.03) * math.exp(-t / 40)
        out[r] = min(1.0, max(0.0, val))
    return out


def _rot_y(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c])


class World:
    def __init__(self, seed=1):
        self.seed = seed
        self.chunks = {}
        self.taken = {}             # flower id -> brain time it was picked (respawns after 60 s)
        self.speech = []            # pending (start ms, band) of heard chat letters
        # robot
        self.x, self.z, self.heading = 0.0, 0.0, 0.0
        self.joints = {name: 0.0 for name in LIMITS}
        self.cmd = {name: 0.0 for name, _, _ in JOINTS}   # smoothed motor commands, -1..1
        self.speed = self.turn = 0.0
        self.step_phase = 0.0
        self.prev_retina = None
        self.touching = set()
        self.events = []            # (t, text) for the brain-state teller / chat
        self.last_reward_dist = None
        self.senses = {"eyes": np.zeros((2, RETINA, RETINA)), "ears": np.zeros((2, BANDS)), "skin": np.zeros(len(SKIN))}
        self._near_key = None           # cache of nearby objects (rebuilt when the robot moves 1 m or every 0.5 s)
        self._near = ([], [])

    # ---------- world content ----------

    def _chunk(self, cx, cz):
        key = (cx, cz)
        if key not in self.chunks:
            rng = random.Random(hash((self.seed, cx, cz)) & 0xFFFFFFFF)
            flowers = []
            for k in range(rng.randint(5, 11)):
                flowers.append((f"f{cx},{cz},{k}", cx * CHUNK + rng.uniform(0, CHUNK), cz * CHUNK + rng.uniform(0, CHUNK),
                                rng.choice(("#ff5c8a", "#ffd23f", "#b388ff", "#ff8a3d", "#ffffff")), rng.uniform(1.5, 4.0)))
            rocks = []
            for k in range(rng.randint(1, 4)):
                rocks.append((f"r{cx},{cz},{k}", cx * CHUNK + rng.uniform(0, CHUNK), cz * CHUNK + rng.uniform(0, CHUNK),
                              rng.uniform(0.4, 1.3)))
            if (cx, cz) == (0, 0):                      # keep the start spot clear
                rocks = [r for r in rocks if math.hypot(r[1], r[2]) > 3]
            self.chunks[key] = (flowers, rocks)
        return self.chunks[key]

    def nearby(self, t):
        """Flowers and rocks within VIEW metres of the robot (cached; picked flowers are dropped at once)."""
        key = (round(self.x), round(self.z), t // 500)
        if key != self._near_key:
            self._near_key = key
            self._near = self._find_nearby(t)
        flowers, rocks = self._near
        return [f for f in flowers if t - self.taken.get(f[0], -10**9) > 60_000], rocks

    def _find_nearby(self, t):
        flowers, rocks = [], []
        r = int(VIEW // CHUNK) + 1
        cx0, cz0 = int(math.floor(self.x / CHUNK)), int(math.floor(self.z / CHUNK))
        for cx in range(cx0 - r, cx0 + r + 1):
            for cz in range(cz0 - r, cz0 + r + 1):
                fl, rk = self._chunk(cx, cz)
                for f in fl:
                    if math.hypot(f[1] - self.x, f[2] - self.z) < VIEW and t - self.taken.get(f[0], -10**9) > 60_000:
                        flowers.append(f)
                for k in rk:
                    if math.hypot(k[1] - self.x, k[2] - self.z) < VIEW:
                        rocks.append(k)
        return flowers, rocks

    def daylight(self, t):
        """0 at midnight .. 1 at noon. The day starts at sunrise."""
        return max(0.0, math.sin(2 * math.pi * (t % DAY_MS) / DAY_MS))

    def sun_dir(self, t):
        a = 2 * math.pi * (t % DAY_MS) / DAY_MS
        return np.array([math.cos(a), math.sin(a), 0.35])

    def hear(self, text, t):
        """Chat text arrives as sound: each letter is 60 ms in one pitch band (same word = same pattern)."""
        start = max([s for s, _ in self.speech] + [t]) + SPEECH_MS
        for ch in text.lower():
            if ch.isalnum():
                self.speech.append((start, 1 + ord(ch) % (BANDS - 2)))     # speech uses bands 1..6 only
            start += SPEECH_MS

    # ---------- body ----------

    def _facing(self):
        return self.heading + self.joints["torso twist"] + self.joints["head turn"]

    def head_pos(self):
        bend = self.joints["torso bend"]
        fwd = _rot_y(np.array([0.0, 0.0, 1.0]), self.heading + self.joints["torso twist"])
        return np.array([self.x, 0.9 + 0.65 * math.cos(bend), self.z]) + fwd * 0.65 * math.sin(bend)

    def hand_pos(self, side):
        """Forward kinematics: shoulder -> upper arm -> forearm (0.3 m each)."""
        s = self.joints[f"{side} shoulder"]
        e = self.joints[f"{side} elbow"]
        bend = self.joints["torso bend"]
        yaw = self.heading + self.joints["torso twist"]
        lateral = 0.24 if side == "left" else -0.24
        shoulder = np.array([lateral, 0.9 + 0.45 * math.cos(bend), 0.45 * math.sin(bend)])
        arm = lambda a: np.array([0.0, -math.cos(a + bend), math.sin(a + bend)])  # noqa: E731
        local = shoulder + 0.3 * arm(s) + 0.3 * arm(s + e)
        return np.array([self.x, 0.0, self.z]) + _rot_y(local, yaw)

    def apply_motors(self, counts, dt):
        """counts: spikes per motor neuron in the last body tick, ordered as JOINTS pairs (+, -)."""
        for k, (name, _, _) in enumerate(JOINTS):
            push = max(-1.0, min(1.0, float(counts[2 * k]) - float(counts[2 * k + 1])))
            if name in ("walk", "turn"):
                # momentum: each spike pushes, the push fades over ~0.6 s, so walking builds up
                self.cmd[name] = max(-1.0, min(1.0, self.cmd[name] * math.exp(-dt / 0.6) + 0.35 * push))
            else:
                # joints: smooth over ~150 ms so single spikes nudge and steady firing drives
                self.cmd[name] += (push - self.cmd[name]) * min(1.0, dt / 0.15)
        for name, (lo, hi) in LIMITS.items():
            self.joints[name] = min(hi, max(lo, self.joints[name] + self.cmd[name] * 2.5 * dt))
        self.speed = 1.2 * self.cmd["walk"]
        self.turn = 1.5 * self.cmd["turn"]
        self.heading += self.turn * dt
        fwd = _rot_y(np.array([0.0, 0.0, 1.0]), self.heading)
        self.x += fwd[0] * self.speed * dt
        self.z += fwd[2] * self.speed * dt
        self.step_phase += abs(self.speed) * dt * 3.0

    # ---------- senses ----------

    def sense(self, t, dt):
        """Update eyes, ears, skin; returns (sensor drive rates in spikes/ms, rewards, events)."""
        flowers, rocks = self.nearby(t)
        light = self.daylight(t)
        rewards, events = [], []

        # body vs rocks: push out, touch chest/back, small negative reward
        touch = np.zeros(len(SKIN))
        for rid, rx, rz, rr in rocks:
            d = math.hypot(self.x - rx, self.z - rz)
            if d < rr + 0.35:
                nx, nz = (self.x - rx) / max(d, 1e-6), (self.z - rz) / max(d, 1e-6)
                self.x, self.z = rx + nx * (rr + 0.35), rz + nz * (rr + 0.35)
                fwd = _rot_y(np.array([0.0, 0.0, 1.0]), self.heading)
                front = fwd[0] * -nx + fwd[2] * -nz > 0
                touch[4 if front else 5] = 1.0
                if "rock" not in self.touching:
                    rewards.append(-0.5)
                    events.append("bumped into a rock")
                    self.touching.add("rock")
        if touch[4] == 0 and touch[5] == 0:
            self.touching.discard("rock")

        # feet: both on the ground when standing, alternating while walking
        if abs(self.speed) < 0.05:
            touch[2] = touch[3] = 1.0
        else:
            left_down = math.sin(self.step_phase) > 0
            touch[2], touch[3] = (1.0, 0.3) if left_down else (0.3, 1.0)

        # hands touching flowers: picking one is a reward
        for k, side in ((0, "left"), (1, "right")):
            hand = self.hand_pos(side)
            for fid, fx, fz, color, _ in flowers:
                if (hand[0] - fx) ** 2 + (hand[1] - 0.35) ** 2 + (hand[2] - fz) ** 2 < 0.3 ** 2:
                    touch[k] = 1.0
                    self.taken[fid] = t
                    rewards.append(0.6)
                    events.append(f"{side} hand picked a flower")
                    break

        # shaping: getting closer to the nearest flower is a small reward (checked once a second)
        if flowers and t % 1000 < BODY_MS:
            dist = min(math.hypot(f[1] - self.x, f[2] - self.z) for f in flowers)
            if self.last_reward_dist is not None and dist < self.last_reward_dist - 0.5:
                rewards.append(0.15)
            self.last_reward_dist = dist

        # eyes: rays from each eye through a 6x6 grid; brightness of what each ray hits
        eyes = self._look(t, flowers, rocks, light)
        motion = np.abs(eyes - self.prev_retina) if self.prev_retina is not None else np.zeros_like(eyes)
        self.prev_retina = eyes

        # ears: left/right x 4 bands
        ears = self._listen(t, flowers)

        self.senses = {"eyes": eyes, "ears": ears, "skin": touch}
        # drive in spikes per ms: motion matters most, steady brightness a little (fatigue fades it anyway)
        eye_rate = 0.045 * np.minimum(1.0, motion * 4) + 0.006 * eyes
        ear_rate = 0.06 * np.minimum(1.0, ears)
        skin_rate = 0.08 * touch
        rates = np.concatenate([eye_rate.ravel(), ear_rate.ravel(), skin_rate])
        return rates, rewards, events

    def _look(self, t, flowers, rocks, light):
        facing = self._facing()
        pitch = self.joints["head nod"] - self.joints["torso bend"] * 0.6
        head = self.head_pos()
        sun = self.sun_dir(t)
        sun /= np.linalg.norm(sun)
        spheres = [(fx, 0.35, fz, 0.2, 1.0) for _, fx, fz, _, _ in flowers] + \
                  [(rx, rr * 0.6, rz, rr, 0.55) for _, rx, rz, rr in rocks]
        sph = np.array(spheres, dtype=np.float64).reshape(-1, 5)
        out = np.zeros((2, RETINA, RETINA))
        g = (np.arange(RETINA) + 0.5) / RETINA - 0.5
        for e, side in enumerate((1, -1)):          # left eye, right eye
            origin = head + _rot_y(np.array([0.06 * side, 0.0, 0.1]), facing)
            yaw = facing + g[None, :] * -FOV         # columns: left .. right
            pit = pitch + g[::-1, None] * FOV        # rows: top .. bottom
            d = np.stack([np.sin(yaw) * np.cos(pit), np.sin(pit) * np.ones_like(yaw), np.cos(yaw) * np.cos(pit)], -1)
            d = np.ascontiguousarray(d.reshape(-1, 3))
            out[e] = _cast(origin, d, sph, sun, light).reshape(RETINA, RETINA)
        return np.clip(out, 0, 1)

    def _listen(self, t, flowers):
        facing = self._facing()
        head = self.head_pos()
        ears = np.zeros((2, BANDS))
        ear_dirs = [_rot_y(np.array([1.0, 0, 0]), facing), _rot_y(np.array([-1.0, 0, 0]), facing)]
        sources = [(None, 0, 0.12 + 0.08 * math.sin(t / 7000))]              # wind, everywhere
        for fid, fx, fz, _, period in flowers[:24]:                            # birds near flowers
            if (t + int(hash(fid) % 5000)) % int(period * 1000) < 150:
                sources.append((np.array([fx, 1.5, fz]), BANDS - 1, 1.0))
        if abs(self.speed) > 0.05 and math.sin(self.step_phase * 2) > 0.95:    # footsteps
            sources.append((None, 0, 0.5))
        self.speech = [(s, b) for s, b in self.speech if s + SPEECH_MS > t]
        for s, b in self.speech:                                               # your chat, as speech
            if s <= t:
                sources.append((None, b, 1.0))
                break
        for pos, band, loud in sources:
            for e in range(2):
                if pos is None:
                    ears[e, band] += loud
                else:
                    v = pos - head
                    dist = np.linalg.norm(v)
                    toward = max(0.0, float(ear_dirs[e] @ (v / dist)))
                    ears[e, band] += loud * (0.4 + 0.6 * toward) / (1 + 0.04 * dist ** 2)
        return np.clip(ears, 0, 1)

    # ---------- for the page ----------

    def snapshot(self, t):
        flowers, rocks = self.nearby(t)
        return {
            "robot": {"x": round(self.x, 3), "z": round(self.z, 3), "heading": round(self.heading, 4),
                      "joints": {k: round(v, 3) for k, v in self.joints.items()},
                      "speed": round(self.speed, 3), "step": round(self.step_phase, 3)},
            "time": t, "day_ms": DAY_MS, "daylight": round(self.daylight(t), 3),
            "flowers": [[round(fx, 2), round(fz, 2), c] for _, fx, fz, c, _ in flowers],
            "rocks": [[round(rx, 2), round(rz, 2), round(rr, 2)] for _, rx, rz, rr in rocks],
            "senses": {k: np.round(v, 2).tolist() for k, v in self.senses.items()},
            "motors": {name: round(self.cmd[name], 2) for name, _, _ in JOINTS},
        }

    def state(self):
        return {"x": self.x, "z": self.z, "heading": self.heading, "joints": self.joints, "taken": self.taken}

    def load_state(self, st):
        self.x, self.z, self.heading = st["x"], st["z"], st["heading"]
        self.joints.update(st["joints"])
        self.taken = dict(st["taken"])


def ear_sensor_slots():
    """Positions of the ear sensors within the sensor list (eyes come first, then ears, then skin)."""
    start = 2 * RETINA * RETINA
    return range(start, start + 2 * BANDS)


def speech_sensor_slots():
    """Ear sensors for the speech bands (1..6), both ears."""
    start = 2 * RETINA * RETINA
    return [start + ear * BANDS + b for ear in range(2) for b in range(1, BANDS - 1)]


def make_brain(seed=1, neurons=900, capacity=1000, reflexes=3, orienting=4, params=None):
    """The standard body brain: 1000-spot cube, senses on one face, motors on the other, reflex arcs.
    Orienting reflex (wiring choice): the head-turn motor neurons get `orienting` extra lines from the speech bands,
    like the reflex that turns your head toward a sound."""
    from .network import Network
    net = Network(n=neurons, capacity=capacity, sensors=N_SENSORS, motors=N_MOTORS, seed=seed, layout="cube",
                  reflexes=reflexes, reflex_from=ear_sensor_slots(), params=params)
    head_turn = net.motor_ids()[:2]                       # JOINTS[0] = head turn: left, right
    for m in head_turn:
        for src in net.rng.sample(speech_sensor_slots(), orienting):
            net.connect(src, m, 0.5)
    return net


class Agent:
    """Couples a brain and the world: every 20 ms of brain time, senses go in and motor spikes come out."""

    def __init__(self, net, world):
        self.net, self.world = net, world
        sensors, motors = net.sensor_ids(), net.motor_ids()
        assert len(sensors) >= N_SENSORS, f"brain needs {N_SENSORS} sensors (has {len(sensors)})"
        assert len(motors) >= N_MOTORS, f"brain needs {N_MOTORS} motors (has {len(motors)})"
        self.sensors = sensors[:N_SENSORS]
        self.motors = motors[:N_MOTORS]
        self.carry = 0
        self.babble = BABBLE
        net.babble(self.babble)        # motor babbling: the body moves a little on its own (#32)
        self._msyn = self._mall = 0
        self.rewards = []              # (t, value, reason)
        self.events = []               # (t, text)

    def _fade_babbling(self):
        """Babbling fades as motor neurons are driven more by their input lines than by noise."""
        S = self.net.S
        syn, all_ = int(S.msyn[self.motors].sum()), int(S.mall[self.motors].sum())
        d_syn, d_all = syn - self._msyn, all_ - self._mall
        self._msyn, self._mall = syn, all_
        if d_all > 0:
            share = d_syn / d_all
            self.babble = max(BABBLE_MIN, min(BABBLE, BABBLE * (1.0 - share)))
            self.net.babble(self.babble)

    def roles(self):
        """Which sensor/motor neuron does what (for the page)."""
        names = [f"eye L {r},{c}" for r in range(RETINA) for c in range(RETINA)] + \
                [f"eye R {r},{c}" for r in range(RETINA) for c in range(RETINA)] + \
                [f"ear {side} band {b}" for side in ("L", "R") for b in range(BANDS)] + [f"skin {s}" for s in SKIN]
        m = [f"{name}: {role}" for name, plus, minus in JOINTS for role in (plus, minus)]
        return {"sensors": dict(zip(self.sensors, names)), "motors": dict(zip(self.motors, m))}

    def run(self, ticks):
        """Advance brain + body by `ticks` ms, in 20 ms body steps."""
        net, w = self.net, self.world
        dt = BODY_MS / 1000.0
        self.carry += ticks
        while self.carry >= BODY_MS:
            self.carry -= BODY_MS
            rates, rewards, events = w.sense(net.t, dt)
            net.set_drive(self.sensors, rates)
            net.run(BODY_MS)
            w.apply_motors(net.read_motors(self.motors), dt)
            for r in rewards:
                net.reward(r)
                self.rewards.append((net.t, r, events[0] if events else "getting closer to a flower"))
            for e in events:
                self.events.append((net.t, e))
            if net.t % 1000 < BODY_MS:
                self._fade_babbling()
        self.rewards = self.rewards[-50:]
        self.events = self.events[-50:]
