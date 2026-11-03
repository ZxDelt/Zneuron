"""Zneuron on the GPU, one thread per neuron (one thread block per brain).

The CPU engine (engine.py) is event-driven: only awake neurons are touched and spikes travel through a queue. That
is fast on a CPU and slow on a GPU (gpu.py runs it one brain per thread). Here every neuron ticks every ms in its own
thread, and the same rules are reorganised so threads never write the same memory:

- pull delivery: a neuron reads its own input lines ("did my sender fire one delay ago?"); the sender only writes
  its own outgoing lines (time, charge sent). No spike queue, no atomics on charges.
- learning touches only lines into the learning neuron (LTP at its spike, LTD when a spike reaches it).
- growth, pruning, locking and new neurons change shared wiring, so they are queued and done by one thread in a
  short serial step (growth each tick if asked for, pruning/locks with the glia check, neurogenesis each second).
- the fire log (used by growth toward recent firers) is written with an atomic counter.

Same rules and parameters as engine.py, helper functions reused from it (compiled for the GPU by gpu.py), but the
order of events inside a tick differs, so results match the CPU on average, not spike for spike. Logs used only by
the viewer (causes, recent inputs, learning log) are not kept here.
"""

import numpy as np
from numba import cuda

from . import engine as E
from .gpu import device_engine

BLOCK = 512          # threads per brain: neurons per brain must be <= BLOCK
MAXREG = int(__import__("os").environ.get("SN_GPU_MAXREG", "40"))   # registers per thread (fewer = more brains/SM)
MAXREQ = 64          # growth requests a brain can queue per tick
# fields the neuron-parallel engine never touches (spike queue, viewer's cause log): kept 1 wide on the GPU to save
# memory (~9 MB per brain), so hundreds of brains fit
SLIM = ("q_mini", "q_ch", "q_en", "q_send", "q_gen", "cause_src", "cause_t", "cause_ch")

_KERNELS = '''
@cuda.jit(device=True)
def nrand(R, i):
    x = R[i]
    x ^= x >> uint64(12)
    x ^= x << uint64(25)
    x ^= x >> uint64(27)
    R[i] = x
    return ((x * uint64(2685821657736338717)) >> uint64(11)) * (1.0 / 9007199254740992.0)


@cuda.jit(device=True)
def in_mini(S, L, i):
    """The mini-line of line L that delivers INTO neuron i."""
    return 2 * L + 1 if S.line_a[L] == i else 2 * L


@cuda.jit(device=True)
def learn_p(S, P, i, mi, delta, t):
    """engine.learn without the shared learning log / eligibility list (only neuron i's own input line)."""
    if delta > 0:
        delta *= 1.0 + cfuel_now(S, P, i, t)
    elif S.locked[mi]:
        delta *= P.lock_ltd_scale
    S.elig[mi] = elig_now(S, P, mi, t) + delta
    S.elig_t[mi] = t
    if P.elig_slow_tau_ms > 0:
        S.elig2[mi] = elig2_now(S, P, mi, t) + delta
        S.elig2_t[mi] = t
    old = abs(S.thick[mi])
    size = _min(P.thick_max, _max(0.0, old + delta * S.fs[LFUEL]))
    S.thick[mi] = S.sign[mini_src(S, mi)] * size


@cuda.jit(device=True)
def request(REQ, NREQ, b, pre, post, size, trial):
    k = cuda.atomic.add(NREQ, b, 1)
    if k < REQ.shape[1]:
        REQ[b, k, 0] = pre
        REQ[b, k, 1] = post
        REQ[b, k, 2] = 1 if trial else 0
        REQ[b, k, 3] = int(size * 1e6)


@cuda.jit(device=True)
def fire_p(S, P, i, t, SENT, REQ, NREQ, b):
    S.hub[i] -= final_threshold(S, P, i)            # carry-over (D4)
    S.efuel[i] -= P.efuel_cost
    S.rest_until[i] = t + int(P.rest_ms) + 1
    S.window_start[i] = NONE if is_empty(S, i) else t
    S.last_fire[i] = t
    if S.motor[i]:
        S.mcount[i] += 1
    k = S.fires_head[i]
    S.fires[i, k] = t
    S.fires_head[i] = (k + 1) % FR
    S.fires_n[i] += 1
    h = cuda.atomic.add(S.sc, FL_HEAD, 1) % FLC     # fire log (FL_HEAD counts up, readers use it modulo FLC)
    S.fl_t[h] = t
    S.fl_id[h] = i
    cuda.atomic.add(S.sc, FL_N, 1)
    for s in range(S.nlines[i]):
        L = S.slot_line[i, s]
        out = mini_from(S, L, i)
        if S.thick[out] != 0.0:                      # all output lines fire (D1): record time and charge sent
            en = energy_now(S, P, out, t)
            SENT[out] = S.thick[out] * en
            S.energy[out] = en * P.fatigue_mult      # #20
            S.last_spike[out] = t
        inp = out ^ 1                                # LTP on this neuron's own input line
        if S.thick[inp] != 0.0 and S.last_arrival[inp] != NONE:
            gap = t - S.last_arrival[inp]
            credit = True
            if P.branches > 0 and S.thick[inp] > 0:
                bs = S.bspk[i, S.sbr[i, s]]
                credit = bs != NONE and t - bs <= P.stdp_window_ms
            if credit and 0 <= gap <= P.stdp_window_ms:
                learn_p(S, P, i, inp, P.ltp * (1.0 - gap / P.stdp_window_ms), t)
    # Hebbian growth (#21): queued, done by the serial step
    if P.grow_need > 0 and rate_hz(S, i, t, 1000.0) >= P.grow_need * S.target[i]:
        return
    for kk in range(S.n_nearest[i]):
        other = S.nearest[i, kk]
        if not S.alive[other]:
            continue
        lf = S.last_fire[other]
        if lf == NONE or not (0 <= t - lf <= P.hebb_gap_ms):
            continue
        if line_between(S, other, i) != NONE:
            continue
        S.cof_t[i, kk, S.cof_n[i, kk] % HEB] = t
        S.cof_n[i, kk] += 1
        recent = 0
        for r in range(_min(S.cof_n[i, kk], HEB)):
            if S.cof_t[i, kk, r] > t - P.hebb_window_ms:
                recent += 1
        if recent >= P.hebb_count:
            request(REQ, NREQ, b, other, i, P.trial_start, True)
            S.cof_n[i, kk] = 0
            for r in range(HEB):
                S.cof_t[i, kk, r] = NONE


@cuda.jit(device=True)
def glia_neuron(S, P, i, t, REQ, NREQ, b, R):
    """engine.glia_check for one neuron: astrocyte, homeostasis, starvation growth (queued)."""
    if S.sensor[i] or S.motor[i] or not S.alive[i]:
        return
    if rate_hz(S, i, t, 1000.0) > (P.glia_hz if P.glia_hz > 0 else S.target[i]):
        S.rfuel_local[i] = _min(P.rfuel_local_max, S.rfuel_local[i] + P.astro_step)
    else:
        S.rfuel_local[i] = _max(1.0, S.rfuel_local[i] - P.astro_step)
    rate = rate_hz(S, i, t, P.rate_window_ms)
    if rate < S.target[i]:
        S.thr_h[i] = _max(P.thr_min, S.thr_h[i] - P.homeo_step)
    elif rate > S.target[i]:
        S.thr_h[i] = _min(P.thr_max, S.thr_h[i] + P.homeo_step)
        if P.homeo_thin < 1.0:
            for s in range(S.nlines[i]):
                inp = in_mini(S, S.slot_line[i, s], i)
                if (S.thick[inp] != 0.0 and not S.locked[inp]
                        and not (P.protect_thin > 0 and S.prot_t[inp] != NONE and t - S.prot_t[inp] < P.protect_ms)):
                    S.thick[inp] *= P.homeo_thin
    quiet_since = S.last_fire[i] if S.last_fire[i] != NONE else 0
    if (t - quiet_since >= P.starve_ms and rate < S.target[i]
            and t - S.last_growth[i] >= P.starve_ms and S.nlines[i] < P.max_lines):
        chosen = NONE
        if P.grow_any > 0:
            k = _min(S.sc[FL_N], FLC)
            h = S.sc[FL_HEAD]
            for back in range(_min(k, 64)):
                j = (h - 1 - back) % FLC
                cand = S.fl_id[j]
                if t - S.fl_t[j] > 50:
                    break
                if (cand != i and S.alive[cand] and not S.motor[cand] and line_between(S, cand, i) == NONE
                        and S.nlines[cand] < P.max_lines and nrand(R, i) < 0.5):
                    chosen = cand
                    break
        else:
            nn = S.n_nearest[i]
            for _ in range(8):
                if nn == 0:
                    break
                k = S.nearest[i, _min(nn - 1, int(nrand(R, i) * nn))]
                if S.alive[k] and line_between(S, k, i) == NONE and S.nlines[k] < P.max_lines:
                    chosen = k
                    break
        if chosen != NONE:
            request(REQ, NREQ, b, chosen, i, P.trial_start, True)
            S.last_growth[i] = t


@cuda.jit(device=True)
def prune_slice(S, P, t):
    """engine.glia_check's pruning and locking, on this check's slice of lines (serial)."""
    nL = S.line_alive.shape[0]
    chunk = (nL + int(P.prune_slices) - 1) // int(P.prune_slices)
    start = S.sc[PRUNE_AT]
    S.sc[PRUNE_AT] = (start + chunk) % nL
    for j in range(chunk):
        L = (start + j) % nL
        if not S.line_alive[L]:
            continue
        for mi in (2 * L, 2 * L + 1):
            if not S.line_alive[L] or S.thick[mi] == 0.0:
                continue
            if abs(S.thick[mi]) >= P.lock_thick or S.thick_since[mi] != NONE or S.locked[mi]:
                check_lock(S, P, mi, t)
            if S.locked[mi]:
                continue
            size = abs(S.thick[mi])
            if S.trial[mi] and size >= P.trial_permanent:
                S.trial[mi] = False
            if S.prot_t[mi] != NONE and t - S.prot_t[mi] < P.protect_ms:
                continue
            last_used = S.last_spike[mi] if S.last_spike[mi] != NONE else S.created[mi]
            remove = (size < P.thick_min or (S.trial[mi] and t - S.created[mi] >= P.trial_ms)
                      or (P.prune_quiet > 0 and t - last_used >= P.unused_ms and size < P.unused_thin))
            if remove:
                last_two = S.thick[mi ^ 1] == 0.0 and (S.nlines[S.line_a[L]] <= P.min_lines
                                                     or S.nlines[S.line_b[L]] <= P.min_lines)
                if last_two:
                    S.thick[mi] = S.sign[mini_src(S, mi)] * _max(size, P.thick_min)
                    S.trial[mi] = False
                else:
                    remove_mini(S, mi, t)


@cuda.jit(device=True)
def tick_p(S, P, i, n, delay, SENT, REQ, NREQ, b, R):
    t = S.sc[T]
    live = i < n and S.alive[i]
    # 0. small leak (v2)
    if live and P.leak_tau_ms > 0:
        decay = math.exp(-1.0 / P.leak_tau_ms)
        S.hub[i] = S.hub[i] * decay if S.hub[i] >= 1e-4 else 0.0
        for s in range(S.nlines[i]):
            S.pool[i, s] = S.pool[i, s] * decay if S.pool[i, s] >= 1e-4 else 0.0
        for bb in range(S.bch.shape[1]):
            S.bch[i, bb] = S.bch[i, bb] * decay if S.bch[i, bb] >= 1e-4 else 0.0
    # 1. pull delivery: spikes my senders fired one delay ago
    if live:
        for s in range(S.nlines[i]):
            L = S.slot_line[i, s]
            mi = in_mini(S, L, i)
            if S.thick[mi] == 0.0 or S.last_spike[mi] != t - S.line_delay[L]:
                continue
            ch = SENT[mi]
            S.last_arrival[mi] = t
            S.nov_sum[i] += S.energy[mi] / P.fatigue_mult if P.fatigue_mult > 0 else 1.0
            S.nov_cnt[i] += 1
            if ch < 0:
                S.hub[i] = _max(0.0, S.hub[i] + ch)
            else:
                if S.window_start[i] == NONE and is_empty(S, i):
                    S.window_start[i] = t
                if P.close_elig > 0:
                    v = S.hub[i] / final_threshold(S, P, i)
                    psi = 1.0 - abs(v - 1.0) / 0.5
                    if psi > 0:
                        S.elig[mi] = elig_now(S, P, mi, t) + P.close_elig * psi
                        S.elig_t[mi] = t
                if P.branches > 0:
                    S.bch[i, S.sbr[i, s]] += ch
                else:
                    S.pool[i, s] += ch
            lf = S.last_fire[i]
            if lf != NONE and 0 < t - lf <= P.stdp_window_ms:
                learn_p(S, P, i, mi, -P.ltd * (1.0 - (t - lf) / P.stdp_window_ms), t)
    # 2. outside input: senses (rate coding) and motor babbling
    if live and S.drive[i] > 0.0 and (S.sensor[i] or S.motor[i]):
        if S.motor[i]:
            if nrand(R, i) < S.drive[i]:
                apply_input(S, P, i, final_threshold(S, P, i), t)
        else:
            S.drive_acc[i] += S.drive[i]
            if S.drive_acc[i] >= 1.0:
                S.drive_acc[i] -= 1.0
                apply_input(S, P, i, final_threshold(S, P, i), t)
    # 3. integrate: mini-buckets (v2) or pools (v1), 20 ms window (v1), fire check
    fires = False
    if live:
        nl = S.nlines[i]
        nb = int(P.branches)
        if nb > 0:
            for bb in range(nb):
                c = S.bch[i, bb]
                if c >= P.branch_hi:
                    c = 0.0
                elif c >= P.branch_lo:
                    S.hub[i] += c * P.dend_gain
                    S.bspk[i, bb] = t
                    c = 0.0
                S.bch[i, bb] = c
        elif nl > 0:
            cap = 1.0 / nl
            for s in range(nl):
                pool = S.pool[i, s]
                while pool >= cap - 1e-9:
                    pool -= cap
                    S.hub[i] += cap
                S.pool[i, s] = _max(0.0, pool)
        if P.leak_tau_ms > 0 and is_empty(S, i):
            S.window_start[i] = NONE
        S.hub[i] = _min(P.hub_cap, S.hub[i])
        ws = S.window_start[i]
        if P.leak_tau_ms <= 0 and ws != NONE and t - ws >= P.window_ms:
            S.hub[i] = 0.0
            for s in range(nl):
                S.pool[i, s] = 0.0
            for bb in range(S.bch.shape[1]):
                S.bch[i, bb] = 0.0
            S.window_start[i] = NONE
        fires = t >= S.rest_until[i] and efuel_now(S, P, i, t) >= P.efuel_cost and S.hub[i] >= final_threshold(S, P, i)
    cuda.syncthreads()
    if fires:
        fire_p(S, P, i, t, SENT, REQ, NREQ, b)
    # 6. Cfuel every 10 ms
    if live and t % int(P.cfuel_every_ms) == 0 and S.nov_cnt[i] > 0:
        c = cfuel_now(S, P, i, t)
        c += P.cfuel_rise * (S.nov_sum[i] / S.nov_cnt[i] - c)
        S.cfuel[i] = c
        S.cfuel_b[i] = t // int(P.cfuel_every_ms) + 1
        S.nov_sum[i] = 0.0
        S.nov_cnt[i] = 0
    # glia, per neuron
    if live and t % int(P.check_ms) == 0 and t > 0:
        glia_neuron(S, P, i, t, REQ, NREQ, b, R)
    cuda.syncthreads()
    # serial step: queued growth, pruning slice, neurogenesis, clock
    if i == 0:
        nreq = _min(NREQ[b], REQ.shape[1])
        for k in range(nreq):
            pre, post = REQ[b, k, 0], REQ[b, k, 1]
            if (line_between(S, pre, post) == NONE and S.nlines[pre] < P.max_lines
                    and S.nlines[post] < P.max_lines):
                connect(S, P, pre, post, REQ[b, k, 3] / 1e6, REQ[b, k, 2] == 1, t, delay)
        NREQ[b] = 0
        if t % int(P.check_ms) == 0 and t > 0:
            prune_slice(S, P, t)
            if t % 1000 == 0:
                neurogenesis(S, P, t, delay)
        if S.sc[REWARD_UNTIL] >= 0 and t + 1 >= S.sc[REWARD_UNTIL]:
            S.fs[LFUEL] = 1.0
            S.sc[REWARD_UNTIL] = -1
        S.sc[T] = t + 1
    cuda.syncthreads()


@cuda.jit(device=True)
def reward_p(S, P, i, n, r, t):
    """engine.apply_reward for neuron i's own input lines."""
    if i >= n or not S.alive[i]:
        return
    if P.reward_window_ms > 0:
        lf = S.last_fire[i]
        if lf == NONE or t - lf > P.reward_window_ms:
            return
    for s in range(S.nlines[i]):
        L = S.slot_line[i, s]
        mi = in_mini(S, L, i)
        if S.thick[mi] == 0.0:
            continue
        e = elig_now(S, P, mi, t) + P.elig_slow_w * elig2_now(S, P, mi, t)
        if abs(e) < 1e-6 or (r < 0 and S.locked[mi]):
            continue
        old = abs(S.thick[mi])
        size = _min(P.thick_max, _max(0.0, old + e * r))
        S.thick[mi] = S.sign[mini_src(S, mi)] * size
        S.elig[mi] = 0.0
        S.elig2[mi] = 0.0
        if r > 0 and e > 0:
            S.prot_t[mi] = t
'''


def _kernel_src():
    fields = E.State._fields
    args = ", ".join(f"F{k}" for k in range(len(fields)))
    views = ", ".join(f"F{k}[b]" for k in range(len(fields)))
    return _KERNELS + f'''

@cuda.jit(max_registers=MAXREG)
def par_run_kernel({args}, P, ticks, delay, n, SENT, REQ, NREQ, NR):
    b = cuda.blockIdx.x
    i = cuda.threadIdx.x
    S = State({views})
    for _ in range(ticks):
        tick_p(S, P, i, n, delay, SENT[b], REQ, NREQ, b, NR[b])


@cuda.jit(max_registers=MAXREG)
def par_reward_kernel({args}, P, r, n):
    b = cuda.blockIdx.x
    i = cuda.threadIdx.x
    if r[b] == 0.0:
        return
    S = State({views})
    t = S.sc[T]
    reward_p(S, P, i, n, r[b], t)
    cuda.syncthreads()
    if i == 0 and P.reward_grow > 0 and r[b] >= P.reward_grow_min:
        reward_growth(S, P, t)
'''


_NS = None


def device_par():
    global _NS
    if _NS is None:
        ns = device_engine()
        ns["MAXREG"] = MAXREG
        exec(compile(_kernel_src(), "engine_gpu_par", "exec"), ns)
        _NS = ns
    return _NS


class GPUParBatch:
    """Many Hub brains on the GPU, one thread per neuron. Same interface as gpu.GPUBatch."""

    def __init__(self, states, P, delay):
        self.dev = device_par()
        self.states, self.P, self.delay = states, P, int(delay)
        self.nb = len(states)
        self.n = states[0].hub.shape[0]
        assert self.n <= BLOCK, f"{self.n} neurons > {BLOCK} threads per block"
        self.arrays = [cuda.to_device(np.stack([getattr(S, f)[..., :1] if f in SLIM else getattr(S, f) for S in states]))
                       for f in E.State._fields]
        self.idx = {f: k for k, f in enumerate(E.State._fields)}
        two_l = states[0].thick.shape[0]
        self.sent = cuda.to_device(np.zeros((self.nb, two_l)))
        self.req = cuda.to_device(np.zeros((self.nb, MAXREQ, 4), np.int64))
        self.nreq = cuda.to_device(np.zeros(self.nb, np.int64))
        seeds = np.array([[(int(S.rng[0]) * 6364136223846793005 + 1442695040888963407 * (k + 1)) % 2 ** 64 or 1
                           for k in range(self.n)] for S in states], dtype=np.uint64)
        self.nrng = cuda.to_device(seeds)
        # pull delivery needs the charge each queued spike carries: seed from the CPU queue is not needed (fresh brains)
        self.threads = ((self.n + 31) // 32) * 32

    def field(self, name):
        return self.arrays[self.idx[name]]

    def run(self, ticks):
        self.dev["par_run_kernel"][self.nb, self.threads](*self.arrays, self.P, int(ticks), self.delay, self.n,
                                                          self.sent, self.req, self.nreq, self.nrng)

    def reward(self, r):
        self.dev["par_reward_kernel"][self.nb, self.threads](*self.arrays, self.P, r, self.n)

    def pull(self, names=None):
        cuda.synchronize()
        for f in names or E.State._fields:
            if f in SLIM:
                continue
            host = self.field(f).copy_to_host()
            for b, S in enumerate(self.states):
                getattr(S, f)[...] = host[b]

    def push(self, names):
        for f in names:
            self.arrays[self.idx[f]] = cuda.to_device(np.stack([getattr(S, f) for S in self.states]))
