"""The bees: same fly wiring, same body, same learning rule; only the neuron model differs.

- hub2: Zneuron v2 (params.v2: leak, threshold 0.3, no line limit, mini-buckets with the choke rule,
        gentle homeostasis, pruning only weak lines, lock on reward, grow by teamwork, free spots to grow neurons)
- hub1: Zneuron v1 (spec defaults), for reference
- lif:  textbook leaky integrate-and-fire (bees/lif.py)
- alif: adaptive LIF, the strongest standard spiking neuron (bees/alif.py)

Fair test rules:
- no per-model tuning: all start from the same weights (one shared scale, compared at several levels)
- one shared learning rule outside the neuron: reward-gated STDP (same sizes, 20 ms window) with two memories
  (250 ms + 2 s), driven by reward surprise (reward minus its running average) x a shared gain
- targeted reward is off for every bee
"""

import random
import time

import numpy as np

from zneuron import engine as E
from zneuron.network import Network
from zneuron.params import Params, v2

from . import arena as A
from . import flybrain as F
from .alif import ALIFBrain
from .lif import LIFBrain

# babbling (spontaneous motor firing, #32), spikes per ms per motor neuron; grab is rarer so a grab means something
BABBLE = {"left": 0.004, "right": 0.004, "forward": 0.006, "grab": 0.0015}
MODELS = {
    "hub2": dict(label="Zneuron v2", color="yellow"),
    "hub1": dict(label="Zneuron v1", color="violet"),
    "lif": dict(label="LIF (textbook)", color="pink"),
    "alif": dict(label="ALIF (adaptive)", color="orange"),
}
SCALES = (1.0, 2.0, 3.0)          # shared start-weight levels the comparison runs at
DEFAULT_SCALE = 2.0

# the shared learning rule (same for every bee)
SLOW_TAU, SLOW_W = 2000.0, 1.0    # second, slow eligibility memory
REWARD_GAIN = 4.0                 # reward surprise x this
BASELINE_RATE = 0.05              # how fast the expected reward follows the real one
FREE_SPOTS = 40                   # v2 brains get room to grow new neurons (#37)


class HubBrain:
    kind = "hub"

    def __init__(self, fly, seed=1, version=2):
        if version == 2:
            p = v2(elig_slow_tau_ms=SLOW_TAU, elig_slow_w=SLOW_W)
            cap = fly.n + FREE_SPOTS
        else:
            p = Params(reward_window_ms=0.0, elig_slow_tau_ms=SLOW_TAU, elig_slow_w=SLOW_W)
            cap = fly.n
        net = Network(n=fly.n, capacity=cap, sensors=fly.sensors, motors=fly.motors, seed=seed,
                      start_lines=0, layout="grid", params=p)
        net.S.sign[:fly.n] = fly.sign
        for a, b, w in fly.edges:
            net.connect(a, b, w)
        if version == 2:                      # each neuron its own target rate: Kenyon cells sparse, MBONs busier
            a, b = F.GROUPS["kenyon"]
            net.S.target[a:b] = 1.0
            a, b = F.GROUPS["mbon"]
            net.S.target[a:b] = 5.0
        net.lfuel = 0.0                       # three-factor rule: timing learning only sticks with a reward
        net.S.drive_acc[:fly.n] = np.random.default_rng(seed).random(fly.n)
        self.net = net
        self.n = fly.n

    @property
    def t(self):
        return self.net.t

    def set_drive(self, ids, rates):
        self.net.set_drive(ids, rates)

    def run(self, ticks):
        self.net.run(ticks)

    def read_motors(self, ids):
        return self.net.read_motors(ids)

    def reward(self, r):
        return self.net.reward(r)

    def _mini(self, a, b):
        L = E.line_between(self.net.S, a, b)
        return E.NONE if L == E.NONE else E.mini_from(self.net.S, L, a)

    def cut(self, pairs):
        for a, b in pairs:
            mi = self._mini(a, b)
            if mi != E.NONE:
                self.net.S.thick[mi] = 0.0

    def fires_n(self):
        return self.net.S.fires_n

    def lines(self):
        """Active connections now: (src, dst, strength 0..1). The Hub brain can grow and prune its own."""
        S = self.net.S
        L = np.flatnonzero(S.line_alive)
        src = np.concatenate([S.line_a[L], S.line_b[L]])
        dst = np.concatenate([S.line_b[L], S.line_a[L]])
        w = np.abs(np.concatenate([S.thick[2 * L], S.thick[2 * L + 1]]))
        m = w > 0
        return src[m], dst[m], w[m]

    def line_count(self):
        return int(len(self.lines()[0]))

    def alive_count(self):
        return self.net.alive_count()

    def recent_fires(self, since):
        S = self.net.S
        k = min(int(S.sc[E.FL_N]), E.FLC)
        h = int(S.sc[E.FL_HEAD])
        idx = (np.arange(h - k, h)) % E.FLC
        t, i = S.fl_t[idx], S.fl_id[idx]
        m = t >= since
        return t[m], i[m]


def make_brain(model, seed=1, scale=DEFAULT_SCALE):
    fly = F.build(seed)
    fly.edges = [(a, b, min(1.0, w * scale)) for a, b, w in fly.edges]
    if model == "hub2":
        return HubBrain(fly, seed, version=2), fly
    if model == "hub1":
        return HubBrain(fly, seed, version=1), fly
    if model == "alif_g":                     # fairness control: ALIF + the Hub v2's reward-guided growth rule
        p = v2()
        grow = (p.reward_grow, p.reward_grow_min, p.reward_grow_ms, p.reward_grow_size, p.trial_ms, p.trial_permanent)
        return ALIFBrain(fly, seed, SLOW_TAU, SLOW_W, grow), fly
    cls = LIFBrain if model == "lif" else ALIFBrain
    return cls(fly, seed, SLOW_TAU, SLOW_W), fly


class Bee:
    """One brain driving one body in a shared world."""

    def __init__(self, model, body, seed=1, scale=DEFAULT_SCALE, learn=True):
        self.model = model
        self.body = body
        self.learn = learn                               # False = control: rewards are not passed to the brain
        self.brain, self.fly = make_brain(model, seed, scale)
        self.sensors = list(range(F.N_SENSORS))
        self.motors = list(range(F.N_SENSORS, F.N_SENSORS + F.N_MOTORS))
        rates = []
        for name in F.ACTIONS:
            rates += [BABBLE[name]] * F.MOTOR_COPIES
        self.brain.set_drive(self.motors, rates)
        self.start_lines = self.brain.line_count()
        self.compute_s = 0.0
        self.damaged = False
        self.expected = 0.0                              # running average reward (for reward surprise)

    def give_reward(self, r):
        """Shared rule: learn from reward surprise (reward minus the expected reward), times a shared gain."""
        surprise = r - self.expected
        self.expected += BASELINE_RATE * (r - self.expected)
        t0 = time.perf_counter()
        self.brain.reward(max(-1.0, min(1.0, REWARD_GAIN * surprise)))
        self.compute_s += time.perf_counter() - t0

    def spikes(self):
        return int(self.brain.fires_n().sum())

    def neurons(self):
        return self.brain.alive_count() if hasattr(self.brain, "alive_count") else self.brain.n


class Match:
    """All bees in one shared world, stepping in lockstep (same brain time)."""

    def __init__(self, condition="basic", seed=1, minutes=20, scale=DEFAULT_SCALE, learn=True, models=None):
        self.condition, self.seed, self.scale = condition, seed, scale
        self.models = list(models or MODELS)       # may also include "alif_g" (control: ALIF + growth)
        self.world = A.World(condition, seed, names=tuple(self.models))
        self.bees = {m: Bee(m, self.world.bodies[m], seed, scale, learn) for m in self.models}
        self.damage_at = None
        dmg = self.world.c.get("damage")
        if dmg:
            fly = self.bees[self.models[0]].fly
            rng = random.Random(seed * 7919)
            self.damage_pairs = [(a, b) for a, b, _ in rng.sample(fly.edges, round(len(fly.edges) * dmg))]
            self.damage_at = minutes * 30_000            # halfway, in ms
        self.events = []                                  # (t ms, bee or "", event) of the last step
        self.n_steps = 0

    @property
    def t(self):
        return self.world.t

    def step(self):
        """One 20 ms body step for all bees. Who acts first rotates, so nobody always wins a shared square."""
        self.events = []
        counts = {}
        for m, bee in self.bees.items():
            t0 = time.perf_counter()
            bee.brain.set_drive(bee.sensors, self.world.sense(bee.body))
            bee.brain.run(A.BODY_MS)
            counts[m] = bee.brain.read_motors(bee.motors)
            bee.compute_s += time.perf_counter() - t0
        k = self.n_steps % len(self.models)
        for m in self.models[k:] + self.models[:k]:
            bee = self.bees[m]
            r = self.world.act(bee.body, counts[m])
            if r != 0.0 and bee.learn:
                bee.give_reward(r)
            self.events += [(self.world.t, m, e) for e in bee.body.events]
        self.events += [(self.world.t, "", e) for e in self.world.tick()]
        self.n_steps += 1
        if self.damage_at is not None and self.world.t >= self.damage_at and not self.bees[self.models[0]].damaged:
            for bee in self.bees.values():
                bee.brain.cut(self.damage_pairs)
                bee.damaged = True
            self.events.append((self.world.t, "", "brain damage"))

    def run(self, ms):
        for _ in range(int(ms) // A.BODY_MS):
            self.step()
