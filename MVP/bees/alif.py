"""A third bee neuron: the adaptive LIF (ALIF), the strongest widely used standard spiking neuron, bare bones.

The textbook LIF (bees/lif.py) plus one thing: an adaptive threshold (Bellec et al. 2018, "Long short-term memory
and learning-to-learn in networks of spiking neurons", NeurIPS (LSNN); Bellec et al. 2020, "A solution to the
learning dilemma for recurrent networks of spiking neurons", Nat. Commun. (e-prop)):

    threshold = 0.5 + BETA * a       a decays with TAU_A = 200 ms, a += 1 after every spike

- TAU_A 200 ms: the textbook "hundreds of ms" adaptation (Bellec used 200 ms to seconds, matched to the task).
- BETA 0.05: a settles near rate * TAU_A, so 50 Hz sustained firing (a ~ 10) doubles the threshold to ~1.0.

Everything else is the LIF's, unchanged: 20 ms leak, reset to 0, 2 ms refractory, 2 ms delay, same wiring, same
sensor rate coding and motor babbling, same reward-gated STDP (LTP 0.01, LTD 0.012, 20 ms window, 250 ms
eligibility, no targeted reward). A forced sensor/babble spike lifts v to the neuron's current threshold, so sensor
spike trains and babbling rates stay identical to the LIF bee. No homeostasis, no growth, no other add-ons.
"""

import math
from collections import namedtuple

import numpy as np
from numba import njit

from . import lif as L
from .lif import DELAY, FL_HEAD, FL_N, FLC, LTD, LTP, NONE, QD, REFR_MS, T, TAU_M, THR, WIN, add_elig

TAU_A, BETA = 200.0, 0.05

ALIFState = namedtuple("ALIFState", L.LIFState._fields + ("a",))


def make(n, sensors, motors, sign, edges, seed=1, slow_tau=0.0, slow_w=0.0, grow=None):
    return ALIFState(*L.make(n, sensors, motors, sign, edges, seed, slow_tau, slow_w, grow), a=np.zeros(n))


@njit(cache=True, inline='always')
def fire(S, i, t):
    S.v[i] = 0.0
    S.a[i] += 1.0                                         # the only change from lif.fire: the threshold steps up
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
        L.grown_fire(S, i, t)


@njit(cache=True)
def run(S, ticks):
    decay = math.exp(-1.0 / TAU_M)
    adecay = math.exp(-1.0 / TAU_A)
    n = S.v.shape[0]
    for _ in range(ticks):
        t = S.sc[T]
        b = t % QD
        for i in range(n):
            S.v[i] = S.v[i] * decay + S.inbuf[b, i]
            S.inbuf[b, i] = 0.0
            S.a[i] *= adecay
        for i in range(n):
            if S.drive[i] > 0.0:
                thr = THR + BETA * S.a[i]
                if S.is_motor[i]:
                    if L.rand01(S) < S.drive[i]:
                        S.v[i] = max(S.v[i], thr)
                elif S.is_sensor[i]:
                    S.drive_acc[i] += S.drive[i]
                    if S.drive_acc[i] >= 1.0:
                        S.drive_acc[i] -= 1.0
                        S.v[i] = max(S.v[i], thr)
        for i in range(n):
            if t >= S.refr[i] and S.v[i] >= THR + BETA * S.a[i]:
                fire(S, i, t)
        if S.gcfg[0] > 0 and t % 100 == 0:
            L.grown_prune(S, t)
        S.sc[T] = t + 1


class ALIFBrain(L.LIFBrain):
    """Same interface as LIFBrain; only the state (adds `a`) and the tick loop differ."""
    kind = "alif"

    def __init__(self, fly, seed=1, slow_tau=0.0, slow_w=0.0, grow=None):
        self.S = make(fly.n, fly.sensors, fly.motors, fly.sign, fly.edges, seed, slow_tau, slow_w, grow)
        self.n = fly.n
        self.index = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(self.S.src, self.S.dst))}

    def run(self, ticks):
        run(self.S, int(ticks))

    def thresholds(self):
        return THR + BETA * self.S.a
