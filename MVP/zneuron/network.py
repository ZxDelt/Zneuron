"""Hub Neuron network, following neuron_spec.json.

The rules run in `engine.py` (compiled with Numba, all state in numpy arrays). This module builds the
brain (positions, starting wiring, glia regions) and gives friendly views on top of the arrays:
`net.neurons[i].hub`, `net.lines[id].ab.thickness`, `net.fire_log`, ... so the server, the insight tools
and the tests don't need to know about arrays.
"""

import math
import random
from collections import defaultdict

import numpy as np

from . import engine as E
from .params import Params


# ---------- views onto the arrays ----------

def _opt(v):
    return None if v == E.NONE else int(v)


class MiniLine:
    """One direction of a line (D8). thickness 0 = not connected (R15)."""

    __slots__ = ("net", "i")

    def __init__(self, net, i):
        self.net, self.i = net, i

    S = property(lambda self: self.net.S)
    src = property(lambda self: int(self.S.line_a[self.i // 2] if self.i % 2 == 0 else self.S.line_b[self.i // 2]))
    dst = property(lambda self: int(self.S.line_b[self.i // 2] if self.i % 2 == 0 else self.S.line_a[self.i // 2]))
    line = property(lambda self: self.i // 2)
    energy = property(lambda self: float(self.S.energy[self.i]))
    pending = property(lambda self: float(self.S.pending[self.i]))
    trial = property(lambda self: bool(self.S.trial[self.i]))
    locked = property(lambda self: bool(self.S.locked[self.i]))
    created_t = property(lambda self: int(self.S.created[self.i]))
    last_spike_t = property(lambda self: _opt(self.S.last_spike[self.i]))
    last_arrival_t = property(lambda self: _opt(self.S.last_arrival[self.i]))
    thick_since = property(lambda self: _opt(self.S.thick_since[self.i]))

    @property
    def thickness(self):
        return float(self.S.thick[self.i])

    @thickness.setter
    def thickness(self, v):
        self.S.thick[self.i] = v

    @property
    def active(self):
        return self.S.thick[self.i] != 0.0

    @property
    def size(self):
        # limits use the size of the thickness, ignoring the sign (#15)
        return abs(float(self.S.thick[self.i]))


class Line:
    """A line between two neurons holding two one-way mini-lines (#33, D8)."""

    __slots__ = ("net", "id")

    def __init__(self, net, lid):
        self.net, self.id = net, lid

    a = property(lambda self: int(self.net.S.line_a[self.id]))
    b = property(lambda self: int(self.net.S.line_b[self.id]))
    delay = property(lambda self: int(self.net.S.line_delay[self.id]))
    ab = property(lambda self: MiniLine(self.net, 2 * self.id))
    ba = property(lambda self: MiniLine(self.net, 2 * self.id + 1))

    def out_of(self, nid):
        return self.ab if nid == self.a else self.ba

    def into(self, nid):
        return self.ba if nid == self.a else self.ab

    def other(self, nid):
        return self.b if nid == self.a else self.a


def _field(name, cast=float):
    def get(self):
        return cast(getattr(self.net.S, name)[self.id])

    def set_(self, v):
        getattr(self.net.S, name)[self.id] = v
    return property(get, set_)


class Neuron:
    __slots__ = ("net", "id")

    def __init__(self, net, nid):
        self.net, self.id = net, nid

    x = _field("px")                           # position in the brain space (F7)
    y = _field("py")
    z = _field("pz")
    glia = _field("region", int)               # glia region (#34)
    alive = _field("alive", bool)              # False = free spot for a new neuron (#37)
    motor = _field("motor", bool)              # its firing drives the body

    sign = _field("sign", int)                 # +1 excitatory, -1 inhibitory (#15)
    sensor = _field("sensor", bool)            # driven from outside (eyes, ears, skin)
    hub = _field("hub")
    thr_h = _field("thr_h")                    # homeostasis threshold (#31)
    rfuel_local = _field("rfuel_local")        # set by its glia (#34)
    rest_until = _field("rest_until", int)     # D3
    efuel = _field("efuel")                    # #36
    efuel_t = _field("efuel_t", int)
    last_growth = _field("last_growth", int)

    @property
    def cfuel(self):                           # #29, per neuron (fades lazily, same formula as engine.cfuel_now)
        net, P = self.net, self.net.P
        steps = (net.t + P.cfuel_every_ms - 1) // P.cfuel_every_ms - net.S.cfuel_b[self.id]
        c = float(net.S.cfuel[self.id])
        return c * math.exp(-steps * P.cfuel_every_ms / P.cfuel_tau_ms) if steps > 0 else c

    @property
    def window_start(self):
        return _opt(self.net.S.window_start[self.id])

    @property
    def last_fire(self):
        return _opt(self.net.S.last_fire[self.id])

    @property
    def lines(self):
        S = self.net.S
        return [int(S.slot_line[self.id, s]) for s in range(S.nlines[self.id])]

    @property
    def pools(self):
        S = self.net.S
        return {int(S.slot_line[self.id, s]): float(S.pool[self.id, s]) for s in range(S.nlines[self.id])}

    @property
    def nearest(self):
        S = self.net.S
        return [int(k) for k in S.nearest[self.id, :S.n_nearest[self.id]]]

    @property
    def fires(self):
        S = self.net.S
        return sorted(int(v) for v in S.fires[self.id, :min(S.fires_n[self.id], E.FR)])

    @property
    def empty(self):
        return bool(E.is_empty(self.net.S, self.id))


class _Ring:
    """Read-only list view of a ring buffer log, oldest first."""

    def __init__(self, net, size, head, count, make):
        self.net, self.size, self.head, self.count, self.make = net, size, head, count, make

    def _idx(self):
        S = self.net.S
        n = min(int(S.sc[self.count]), self.size)
        start = (int(S.sc[self.head]) - n) % self.size
        return np.arange(start, start + n) % self.size

    def __iter__(self):
        S = self.net.S
        idx = self._idx()
        return iter(self.make(S, idx)) if len(idx) else iter(())

    def __len__(self):
        return min(int(self.net.S.sc[self.count]), self.size)

    def __getitem__(self, i):
        return list(self)[i]

    def __reversed__(self):
        return reversed(list(self))


class _Awake:
    def __init__(self, net):
        self.net = net

    def __contains__(self, i):
        return 0 <= i < len(self.net.neurons) and bool(self.net.S.awake[i])

    def __len__(self):
        return int(self.net.S.sc[E.AWAKE_N])

    def __iter__(self):
        S = self.net.S
        return iter([int(v) for v in S.awake_list[:S.sc[E.AWAKE_N]]])


# ---------- the network ----------

class Network:
    def __init__(self, n=40, sensors=4, seed=1, size_um=1000.0, params=None, start_lines=2,
                 layout="random", motors=0, capacity=None, reflexes=0, reflex_from=None):
        """n neurons alive at start; capacity = spots in the brain space (free spots let new neurons grow, #37)."""
        self.p = params or Params()
        self.P = E.make_params(self.p)
        self.rng = random.Random(seed)
        self.size = size_um
        self.layout = layout
        cap = max(n, capacity or n)
        self.capacity = cap

        motors = min(motors, n - sensors)
        if layout in ("grid", "cube"):
            positions = self._lattice_positions(cap, sensors, motors, dims=3 if layout == "cube" else 2)
        else:
            positions = [(self.rng.uniform(0, size_um), self.rng.uniform(0, size_um), 0.0) for _ in range(cap)]
        if cap > n:
            # free spots are scattered at random among the non-sensor, non-motor positions
            rest = positions[sensors + motors:]
            self.rng.shuffle(rest)
            positions = positions[:sensors + motors] + rest
        regions = self._regions(positions)
        self.S = E.make_state(cap, cap * self.p.max_lines // 2 + 16, self.p.nearest, max(regions) + 1,
                              maxl=self.p.max_lines, branches=self.p.branches)
        self.S.target[:] = self.p.target_hz
        E.seed_rng(self.S, seed)
        S = self.S
        n_inh = round((n - sensors - motors) * self.p.inhibitory_frac)
        inh = set(self.rng.sample(range(sensors + motors, n), n_inh))
        for i, (x, y, z) in enumerate(positions):
            # F7: positions in the brain space (2D square: random or grid; 3D: cube)
            S.px[i], S.py[i], S.pz[i], S.region[i] = x, y, z, regions[i]
            S.sign[i] = -1 if i in inh else 1
            S.sensor[i] = i < sensors
            S.motor[i] = sensors <= i < sensors + motors
            S.alive[i] = i < n
            S.thr_h[i] = self.p.thr_start
        # neurons that can be driven from outside: sensors (senses) and motors (babbling, #32)
        S.sensor_list[:sensors + motors] = np.arange(sensors + motors)
        S.sc[E.SENSOR_N] = sensors + motors
        self.neurons = [Neuron(self, i) for i in range(cap)]
        E.compute_nearest(S)
        self.glia = defaultdict(list)
        for i in range(n):
            self.glia[regions[i]].append(i)
        for nrn in self.neurons:            # R14: every neuron starts with 2 input lines
            if nrn.sensor or not nrn.alive:
                continue
            pool = self.direct_neighbours(nrn) if layout in ("grid", "cube") else nrn.nearest
            for src in self.rng.sample(pool, min(start_lines, len(pool))):
                self.connect(src, nrn.id, self.rng.uniform(0.3, 0.7))
        # reflex arcs (wiring choice, not neuron spec): each motor neuron gets `reflexes` weak input lines straight
        # from sensor neurons, like spinal reflexes; the first comes from `reflex_from` (e.g. the ears) if given
        sensor_ids = list(range(sensors))
        for m in range(sensors, sensors + motors):
            picks = []
            if reflexes and reflex_from:
                picks.append(self.rng.choice(list(reflex_from)))
            picks += self.rng.sample([i for i in sensor_ids if i not in picks], max(0, reflexes - len(picks)))
            for src in picks[:reflexes]:
                self.connect(src, m, 0.3)

        self.awake = _Awake(self)
        self.fire_log = _Ring(self, E.FLC, E.FL_HEAD, E.FL_N, lambda S, k: list(zip(
            S.fl_t[k].tolist(), S.fl_id[k].tolist())))
        self.learn_log = _Ring(self, E.LLC, E.LL_HEAD, E.LL_N, lambda S, k: list(zip(
            S.ll_t[k].tolist(), S.ll_src[k].tolist(), S.ll_dst[k].tolist(), S.ll_d[k].tolist())))
        self.events = _Ring(self, E.EVC, E.EV_HEAD, E.EV_N, lambda S, k: [
            (t, E.EVENT_NAMES[kind], a, b) for t, kind, a, b in zip(
                S.ev_t[k].tolist(), S.ev_kind[k].tolist(), S.ev_src[k].tolist(), S.ev_dst[k].tolist())])
        self._causes_key, self._causes = None, {}
        self._lines_key, self._lines = None, {}

    # ---------- setup ----------

    def direct_neighbours(self, nrn):
        d0 = self._dist(nrn, self.neurons[nrn.nearest[0]])
        return [k for k in nrn.nearest if self._dist(nrn, self.neurons[k]) <= d0 * 1.01]

    def _lattice_positions(self, n, sensors, motors, dims):
        """Regular grid (2D square) or cube (3D). Sensors on the x = min face (left edge in 2D),
        motors on the x = max face, so signals flow across the brain."""
        side = math.ceil(n ** (1 / dims) - 1e-9)
        step = self.size / side
        coords = [(c + 0.5) * step for c in range(side)]
        if dims == 3:
            cells = [(x, y, z) for z in coords for y in reversed(coords) for x in coords]
        else:
            cells = [(x, y, 0.0) for y in reversed(coords) for x in coords][:n]

        def spread(face, k):
            return [face[round(i * (len(face) - 1) / max(1, k - 1))] for i in range(k)] if k else []

        pick_s = spread([c for c in cells if c[0] == coords[0]], sensors)
        pick_m = spread([c for c in cells if c[0] == coords[-1] and c not in pick_s], motors)
        rest = [c for c in cells if c not in pick_s and c not in pick_m]
        return (pick_s + pick_m + rest)[:n]

    @staticmethod
    def _dist(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    def _regions(self, positions):
        # #34 / F7: split the brain space into regions of ~50 neurons, 1 glia per region
        dims = 3 if self.layout == "cube" else 2
        side = max(1, math.ceil((len(positions) / self.p.glia_group) ** (1 / dims)))
        self.glia_side = side
        cell = lambda v: min(side - 1, int(v / self.size * side))  # noqa: E731
        return [((cell(z) if dims == 3 else 0) * side + cell(y)) * side + cell(x) for x, y, z in positions]

    # ---------- state ----------

    t = property(lambda self: int(self.S.sc[E.T]))

    @property
    def lfuel(self):                  # #30, fixed at 1 in v1
        return float(self.S.fs[E.LFUEL])

    @lfuel.setter
    def lfuel(self, v):
        self.S.fs[E.LFUEL] = v

    @property
    def rfuel(self):                  # D5, global, 1 in v1
        return float(self.S.fs[E.RFUEL])

    @rfuel.setter
    def rfuel(self, v):
        self.S.fs[E.RFUEL] = v

    def alive_count(self):
        return int(self.S.alive.sum())

    @property
    def counts(self):
        return {name: int(self.S.counts[k]) for k, name in enumerate(E.EVENT_NAMES)}

    @property
    def lines(self):
        # rebuilt only when lines were added or removed (or time moved on)
        S = self.S
        key = (self.t, int(S.sc[E.FREE_N]), int(S.counts[E.EV_GROW]), int(S.counts[E.EV_PRUNE]))
        if key != self._lines_key:
            self._lines_key = key
            self._lines = {int(L): Line(self, int(L)) for L in np.flatnonzero(S.line_alive)}
        return self._lines

    @property
    def fire_causes(self):
        """(t, id) -> [(src or -1 for outside input, arrive t, charge)] for recent fires."""
        S = self.S
        key = (int(S.sc[E.FL_N]), self.t)
        if key != self._causes_key:
            out = {}
            n = min(int(S.sc[E.FL_N]), E.FLC)
            start = (int(S.sc[E.FL_HEAD]) - n) % E.FLC
            for j in range(n):
                k = (start + j) % E.FLC
                c = int(S.cause_n[k])
                out[(int(S.fl_t[k]), int(S.fl_id[k]))] = list(zip(
                    S.cause_src[k, :c].tolist(), S.cause_t[k, :c].tolist(), S.cause_ch[k, :c].tolist()))
            self._causes_key, self._causes = key, out
        return self._causes

    def line_between(self, a, b):
        S = self.S
        for L in S.slot_line[a, :S.nlines[a]].tolist():
            if (S.line_a[L] == b and S.line_b[L] == a) or (S.line_a[L] == a and S.line_b[L] == b):
                return Line(self, L)
        return None

    def connect(self, src, dst, size, trial=False, branch=None):
        """Activate the src->dst mini-line, creating the line if needed. Sign comes from src (#15).
        branch: put the line on this mini-bucket of dst (v2), instead of the automatic choice."""
        mi = E.connect(self.S, self.P, src, dst, size, trial, self.t, self.p.delay_ms)
        if mi == E.NONE:
            return None
        if branch is not None:
            self.S.sbr[dst, E.slot_of(self.S, dst, int(mi) // 2)] = branch
        return MiniLine(self, int(mi))

    # ---------- running ----------

    def stimulate(self, nid, charge=1.0):
        """Outside input this tick (sensors)."""
        k = self.S.sc[E.STIM_N]
        self.S.stim_id[k], self.S.stim_ch[k] = nid, charge
        self.S.sc[E.STIM_N] = k + 1

    def set_schedule(self, period_ms, entries):
        """Repeating outside input: [(phase ms, neuron id, charge or -1 = exactly its threshold)]."""
        S = self.S
        entries = entries[:E.SCH]
        for k, (phase, nid, ch) in enumerate(entries):
            S.sch_phase[k], S.sch_id[k], S.sch_ch[k] = phase, nid, ch
        S.sc[E.SCH_N], S.sc[E.SCH_PERIOD] = len(entries), period_ms

    def step(self):
        """One 1 ms tick. Returns the neurons that fired."""
        E.run(self.S, self.P, 1, self.p.delay_ms)
        return [self.neurons[int(i)] for i in self.S.last_fired[:self.S.sc[E.LAST_FIRED_N]]]

    def run(self, ticks):
        """Many ticks in one compiled call (fast)."""
        if ticks > 0:
            E.run(self.S, self.P, int(ticks), self.p.delay_ms)

    def select(self, nid):
        """Neuron whose hub and threshold are traced every tick (for the viewer)."""
        self.S.sc[E.SEL] = nid
        self.S.sc[E.TR_N] = 0

    def reward(self, value):
        """Reward r in -1..1 (thumbs up/down, task rewards). Learning from the last ~1 s is applied again x r:
        + strengthens what just happened, - undoes it (#30). Returns how many lines changed."""
        return int(E.apply_reward(self.S, self.P, float(value), self.t))

    # ---------- body link ----------

    def sensor_ids(self):
        return [int(i) for i in self.S.sensor_list[:self.S.sc[E.SENSOR_N]] if self.S.sensor[i]]

    def babble(self, rate=0.01):
        """Spontaneous firing for motor neurons (#32, optional per neuron): random movement to learn from."""
        self.set_drive(self.motor_ids(), [rate] * len(self.motor_ids()))

    def motor_ids(self):
        return [int(i) for i in np.flatnonzero(self.S.motor)]

    def set_drive(self, ids, rates):
        """Sense strength for sensor neurons, as spikes per ms (0..0.5). Rate coding (#25)."""
        self.S.drive[np.asarray(ids, dtype=np.int64)] = np.clip(rates, 0.0, 0.5)

    def read_motors(self, ids):
        """Spikes per motor neuron since the last read (then reset)."""
        ids = np.asarray(ids, dtype=np.int64)
        out = self.S.mcount[ids].copy()
        self.S.mcount[ids] = 0
        return out

    def drives(self):
        """Drive samples every 100 ms: (t, Lfuel, Rfuel incl. local, mean Cfuel), oldest first."""
        S = self.S
        n = min(int(S.sc[E.DR_N]), E.DRC)
        start = (int(S.sc[E.DR_HEAD]) - n) % E.DRC
        return [(int(S.dr_t[k]), float(S.dr_l[k]), round(float(S.dr_r[k]), 3), round(float(S.dr_c[k]), 3))
                for k in ((start + j) % E.DRC for j in range(n))]

    def trace(self):
        S = self.S
        n = min(int(S.sc[E.TR_N]), E.TRC)
        start = (int(S.sc[E.TR_HEAD]) - n) % E.TRC
        idx = [(start + k) % E.TRC for k in range(n)]
        return [(int(S.tr_t[k]), float(S.tr_hub[k]), float(S.tr_thr[k])) for k in idx]

    # ---------- helpers ----------

    def final_threshold(self, nrn):
        # F5: homeostasis threshold × Rfuel (global × local), kept between 0.15 and 0.95
        S, P = self.S, self.P
        return min(P.final_thr_max, max(P.final_thr_min, float(S.thr_h[nrn.id] * S.fs[E.RFUEL] * S.rfuel_local[nrn.id])))

    def thresholds(self):
        """Final threshold of every neuron at once (F5)."""
        S, P = self.S, self.P
        return np.clip(S.thr_h * S.fs[E.RFUEL] * S.rfuel_local, P.final_thr_min, P.final_thr_max)

    def efuel(self, nrn):
        return float(E.efuel_now(self.S, self.P, nrn.id, self.t))

    def energy(self, m):
        return float(E.energy_now(self.S, self.P, m.i, self.t))

    def rate_hz(self, nrn, window_ms=None):
        S, w = self.S, float(window_ms or self.p.rate_window_ms)
        f = S.fires[nrn.id, :min(int(S.fires_n[nrn.id]), E.FR)]
        return int((f > self.t - w).sum()) * 1000.0 / w

    def resting(self, nrn):
        return self.t < self.S.rest_until[nrn.id]

    def inflight(self):
        """Spikes travelling right now: (src, dst, fraction of the way, charge)."""
        S, t, out = self.S, self.t, []
        for b in range(E.QD):
            for j in range(S.q_cnt[b]):
                mi = int(S.q_mini[b, j])
                L = mi // 2
                if not S.line_alive[L] or S.q_gen[b, j] != S.line_gen[L]:
                    continue
                send = int(S.q_send[b, j])
                span = max(1, int(S.line_delay[L]))
                out.append((int(E.mini_src(S, mi)), int(E.mini_dst(S, mi)), (t - send) / span, float(S.q_ch[b, j])))
        return out
