"""The pink bee's neuron: the textbook leaky integrate-and-fire (LIF) neuron, bare bones.

dv/dt = -v / 20 ms + input; fire at v >= 0.5, reset to 0, 2 ms refractory. Nothing else: no adaptation,
no homeostasis, no growth. Learning is the same shared rule as the Hub bee (bees/brains.py): reward-modulated
STDP (Izhikevich 2007 / Florian 2007 style) with the same sizes, window and eligibility. Same wiring, delays,
start weights, sensor rate coding and motor babbling as the Hub bee.
"""

import math
from collections import namedtuple

import numpy as np
from numba import njit

from zneuron.engine import rand01

NONE = -1
QD = 8
FLC = 4096

LIFState = namedtuple("LIFState", [
    "sc", "v", "sign", "is_sensor", "is_motor", "refr", "last_fire", "fires_n", "mcount", "drive", "drive_acc",
    "inbuf", "src", "dst", "w", "elig", "elig_t", "last_arr", "out_start", "out_e", "in_start", "in_e",
    "fl_t", "fl_id", "elig2", "elig2_t", "cfg", "rng",
    # optional reward-guided growth (fairness control for the Hub v2 rule): a separate list of grown lines
    "gsrc", "gdst", "gw", "gelig", "gelig_t", "gelig2", "gelig2_t", "glast", "gborn", "gtrial", "gcfg",
])
GMAX = 256                     # grown lines a brain can hold
# gcfg: lines per reward (0 = off), min reward, window ms, start size, trial ms, permanent size
T, FL_HEAD, FL_N = range(3)

THR, TAU_M, REFR_MS, DELAY = 0.5, 20.0, 2, 2
LTP, LTD, WIN, ELIG_TAU = 0.01, 0.012, 20.0, 250.0


PAD_EDGES = 1400      # GPU batches need equal sizes: edges are padded with zero-strength dummies (always skipped)


def make(n, sensors, motors, sign, edges, seed=1, slow_tau=0.0, slow_w=0.0, grow=None):
    edges = list(edges) + [(0, 0, 0.0)] * max(0, PAD_EDGES - len(edges))
    E = len(edges)
    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)
    w = np.array([e[2] for e in edges], dtype=np.float64)
    order_out = np.argsort(src, kind="stable")
    order_in = np.argsort(dst, kind="stable")
    out_start = np.searchsorted(src[order_out], np.arange(n + 1)).astype(np.int64)
    in_start = np.searchsorted(dst[order_in], np.arange(n + 1)).astype(np.int64)
    is_sensor = np.zeros(n, dtype=np.bool_)
    is_sensor[:sensors] = True
    is_motor = np.zeros(n, dtype=np.bool_)
    is_motor[sensors:sensors + motors] = True
    S = LIFState(
        sc=np.zeros(3, dtype=np.int64), v=np.zeros(n), sign=np.array(sign, dtype=np.float64),
        is_sensor=is_sensor, is_motor=is_motor, refr=np.zeros(n, dtype=np.int64),
        last_fire=np.full(n, NONE, dtype=np.int64), fires_n=np.zeros(n, dtype=np.int64),
        mcount=np.zeros(n, dtype=np.int64), drive=np.zeros(n), drive_acc=np.random.default_rng(seed).random(n),
        inbuf=np.zeros((QD, n)), src=src, dst=dst, w=w, elig=np.zeros(E), elig_t=np.zeros(E, dtype=np.int64),
        last_arr=np.full(E, NONE, dtype=np.int64), out_start=out_start, out_e=order_out.astype(np.int64),
        in_start=in_start, in_e=order_in.astype(np.int64),
        fl_t=np.zeros(FLC, dtype=np.int64), fl_id=np.zeros(FLC, dtype=np.int64),
        elig2=np.zeros(E), elig2_t=np.zeros(E, dtype=np.int64), cfg=np.array([slow_tau, slow_w], dtype=np.float64),
        rng=np.zeros(1, dtype=np.uint64),
        gsrc=np.zeros(GMAX, np.int64), gdst=np.zeros(GMAX, np.int64), gw=np.zeros(GMAX), gelig=np.zeros(GMAX),
        gelig_t=np.zeros(GMAX, np.int64), gelig2=np.zeros(GMAX), gelig2_t=np.zeros(GMAX, np.int64),
        glast=np.full(GMAX, NONE, np.int64), gborn=np.zeros(GMAX, np.int64), gtrial=np.zeros(GMAX, np.bool_),
        gcfg=np.array(grow or (0, 0.2, 300.0, 0.2, 3000.0, 0.3), dtype=np.float64))
    seed_rng(S, seed)
    return S


def seed_rng(S, seed):
    from zneuron.engine import seed_rng as engine_seed
    engine_seed(S, seed)                                 # same own random generator as the Hub engine (GPU-ready)


@njit(cache=True, inline='always')
def elig_now(S, e, t):
    return S.elig[e] * math.exp(-(t - S.elig_t[e]) / ELIG_TAU)


@njit(cache=True, inline='always')
def elig2_now(S, e, t):
    if S.cfg[0] <= 0:
        return 0.0
    return S.elig2[e] * math.exp(-(t - S.elig2_t[e]) / S.cfg[0])


@njit(cache=True, inline='always')
def add_elig(S, e, d, t):
    S.elig[e] = elig_now(S, e, t) + d
    S.elig_t[e] = t
    if S.cfg[0] > 0:                                     # shared rule upgrade: slow memory (seconds) for late rewards
        S.elig2[e] = elig2_now(S, e, t) + d
        S.elig2_t[e] = t


@njit(cache=True, inline='always')
def gelig_add(S, g, d, t):
    S.gelig[g] = S.gelig[g] * math.exp(-(t - S.gelig_t[g]) / ELIG_TAU) + d
    S.gelig_t[g] = t
    if S.cfg[0] > 0:
        S.gelig2[g] = S.gelig2[g] * math.exp(-(t - S.gelig2_t[g]) / S.cfg[0]) + d
        S.gelig2_t[g] = t


@njit(cache=True, inline='always')
def grown_fire(S, i, t):
    """Grown lines (growth control only): send along lines from i, LTP on lines into i."""
    for g in range(GMAX):
        if S.gw[g] == 0.0:
            continue
        if S.gsrc[g] == i:
            d = S.gdst[g]
            S.inbuf[(t + DELAY) % QD, d] += S.sign[i] * S.gw[g]
            S.glast[g] = t + DELAY
            lf = S.last_fire[d]
            if lf != NONE and 0 < t + DELAY - lf <= WIN:
                gelig_add(S, g, -LTD * (1.0 - (t + DELAY - lf) / WIN), t)
        elif S.gdst[g] == i:
            la = S.glast[g]
            if la != NONE and 0 <= t - la <= WIN:
                gelig_add(S, g, LTP * (1.0 - (t - la) / WIN), t)


@njit(cache=True)
def grown_prune(S, t):
    """Trial rule, as the Hub's: a grown line must reach the permanent size within trial ms, else it is removed."""
    for g in range(GMAX):
        if S.gw[g] != 0.0 and S.gtrial[g]:
            if S.gw[g] >= S.gcfg[5]:
                S.gtrial[g] = False
            elif t - S.gborn[g] >= S.gcfg[4]:
                S.gw[g] = 0.0


@njit(cache=True)
def grown_reward(S, r, t):
    """Reward on grown lines, then reward-guided growth: same rule as engine.reward_growth."""
    for g in range(GMAX):
        if S.gw[g] == 0.0:
            continue
        el = (S.gelig[g] * math.exp(-(t - S.gelig_t[g]) / ELIG_TAU)
              + (S.cfg[1] * S.gelig2[g] * math.exp(-(t - S.gelig2_t[g]) / S.cfg[0]) if S.cfg[0] > 0 else 0.0))
        if abs(el) < 1e-6:
            continue
        S.gw[g] = min(1.0, max(1e-6, S.gw[g] + el * r))
        S.gelig[g] = 0.0
        S.gelig2[g] = 0.0
    if S.gcfg[0] <= 0 or r < S.gcfg[1]:
        return
    k = min(S.sc[FL_N], FLC)
    h = S.sc[FL_HEAD]
    grown = 0
    for back in range(k):
        idx = (h - 1 - back) % FLC
        tp = S.fl_t[idx]
        if t - tp > S.gcfg[2]:
            break
        post = S.fl_id[idx]
        if S.is_sensor[post] or rand01(S) > 0.25:
            continue
        for b2 in range(back + 1, k):
            j = (h - 1 - b2) % FLC
            dt = tp - S.fl_t[j]
            if dt < 2:
                continue
            if dt > WIN:
                break
            pre = S.fl_id[j]
            if pre == post or S.is_motor[pre] or rand01(S) >= 0.3 or connected(S, pre, post):
                continue
            free = -1
            for g in range(GMAX):
                if S.gw[g] == 0.0:
                    free = g
                    break
            if free >= 0:
                S.gsrc[free], S.gdst[free], S.gw[free] = pre, post, S.gcfg[3]
                S.gelig[free], S.gelig2[free], S.glast[free] = 0.0, 0.0, NONE
                S.gborn[free], S.gtrial[free] = t, True
                grown += 1
            break
        if grown >= S.gcfg[0]:
            break


@njit(cache=True)
def connected(S, a, b):
    for k in range(S.out_start[a], S.out_start[a + 1]):
        e = S.out_e[k]
        if S.dst[e] == b and S.w[e] != 0.0:
            return True
    for g in range(GMAX):
        if S.gw[g] != 0.0 and S.gsrc[g] == a and S.gdst[g] == b:
            return True
    return False


@njit(cache=True, inline='always')
def fire(S, i, t):
    S.v[i] = 0.0
    S.refr[i] = t + REFR_MS + 1
    S.last_fire[i] = t
    S.fires_n[i] += 1
    if S.is_motor[i]:
        S.mcount[i] += 1
    h = S.sc[FL_HEAD]
    S.fl_t[h], S.fl_id[h] = t, i
    S.sc[FL_HEAD] = (h + 1) % FLC
    S.sc[FL_N] += 1
    b = (t + DELAY) % QD
    for k in range(S.out_start[i], S.out_start[i + 1]):
        e = S.out_e[k]
        if S.w[e] == 0.0:
            continue
        d = S.dst[e]
        S.inbuf[b, d] += S.sign[i] * S.w[e]
        S.last_arr[e] = t + DELAY
        lf = S.last_fire[d]                              # LTD: arrives after the receiver fired
        if lf != NONE and 0 < t + DELAY - lf <= WIN:
            add_elig(S, e, -LTD * (1.0 - (t + DELAY - lf) / WIN), t)
    for k in range(S.in_start[i], S.in_start[i + 1]):  # LTP: input arrived before this spike
        e = S.in_e[k]
        la = S.last_arr[e]
        if la != NONE and S.w[e] != 0.0 and 0 <= t - la <= WIN:
            add_elig(S, e, LTP * (1.0 - (t - la) / WIN), t)
    if S.gcfg[0] > 0:
        grown_fire(S, i, t)


@njit(cache=True)
def run(S, ticks):
    decay = math.exp(-1.0 / TAU_M)
    n = S.v.shape[0]
    for _ in range(ticks):
        t = S.sc[T]
        b = t % QD
        for i in range(n):
            S.v[i] = S.v[i] * decay + S.inbuf[b, i]
            S.inbuf[b, i] = 0.0
        for i in range(n):
            if S.drive[i] > 0.0:
                if S.is_motor[i]:
                    if rand01(S) < S.drive[i]:
                        S.v[i] = max(S.v[i], THR)
                elif S.is_sensor[i]:
                    S.drive_acc[i] += S.drive[i]
                    if S.drive_acc[i] >= 1.0:
                        S.drive_acc[i] -= 1.0
                        S.v[i] = max(S.v[i], THR)
        for i in range(n):
            if t >= S.refr[i] and S.v[i] >= THR:
                fire(S, i, t)
        if S.gcfg[0] > 0 and t % 100 == 0:
            grown_prune(S, t)
        S.sc[T] = t + 1


@njit(cache=True)
def apply_reward(S, r):
    t = S.sc[T]
    changed = 0
    for e in range(S.w.shape[0]):
        if S.w[e] == 0.0:
            continue
        el = elig_now(S, e, t) + S.cfg[1] * elig2_now(S, e, t)
        if abs(el) < 1e-6:
            continue
        S.w[e] = min(1.0, max(1e-6, S.w[e] + el * r))   # stays a (very thin) line, like Hub thick_min floor
        S.elig[e] = 0.0
        S.elig2[e] = 0.0
        changed += 1
    if S.gcfg[0] > 0:
        grown_reward(S, r, t)
    return changed


class LIFBrain:
    kind = "lif"

    def __init__(self, fly, seed=1, slow_tau=0.0, slow_w=0.0, grow=None):
        self.S = make(fly.n, fly.sensors, fly.motors, fly.sign, fly.edges, seed, slow_tau, slow_w, grow)
        self.n = fly.n
        self.index = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(self.S.src, self.S.dst))}

    @property
    def t(self):
        return int(self.S.sc[T])

    def set_drive(self, ids, rates):
        self.S.drive[np.asarray(ids)] = np.clip(rates, 0.0, 0.5)

    def run(self, ticks):
        run(self.S, int(ticks))

    def read_motors(self, ids):
        ids = np.asarray(ids)
        out = self.S.mcount[ids].copy()
        self.S.mcount[ids] = 0
        return out

    def reward(self, r):
        return int(apply_reward(self.S, float(r)))

    def cut(self, pairs):
        for a, b in pairs:
            e = self.index.get((a, b))
            if e is not None:
                self.S.w[e] = 0.0

    def fires_n(self):
        return self.S.fires_n

    def weight(self, a, b):
        e = self.index.get((a, b))
        return 0.0 if e is None else float(self.S.w[e])

    def lines(self):
        m = self.S.w > 0
        g = self.S.gw > 0
        return (np.concatenate([self.S.src[m], self.S.gsrc[g]]), np.concatenate([self.S.dst[m], self.S.gdst[g]]),
                np.concatenate([self.S.w[m], self.S.gw[g]]))

    def line_count(self):
        return int((self.S.w > 0).sum() + (self.S.gw > 0).sum())

    def recent_fires(self, since):
        k = min(int(self.S.sc[FL_N]), FLC)
        h = int(self.S.sc[FL_HEAD])
        idx = (np.arange(h - k, h)) % FLC
        t, i = self.S.fl_t[idx], self.S.fl_id[idx]
        m = t >= since
        return t[m], i[m]
