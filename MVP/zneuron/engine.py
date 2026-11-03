"""Compiled core of the Hub Neuron network (Numba). Same rules as neuron_spec.json, spec IDs in comments.

All state lives in numpy arrays inside one `State` namedtuple so a single compiled call can run
thousands of 1 ms ticks without returning to Python. `network.py` wraps this with friendly views.

Tick order (F8): deliver spikes -> outside input -> pools pass to hub -> 20 ms window -> fire ->
learn -> Cfuel (every 10 ms) -> sleep -> glia checks (every 100 ms).
"""

import math
from collections import namedtuple

import numpy as np
from numba import njit, uint64

NONE = -1
MAXL = 8          # slots per neuron (max lines, #21)
FR = 256          # fire-time ring per neuron (rate over 5 s, #31)
RI = 32           # recent inputs kept per neuron (for "why it fired")
QD = 32           # spike queue buckets; delays must be < QD ms (#11 max 20)
FLC = 5000        # fire log size
LLC = 20000       # learning log size
EVC = 500         # event log size
HEB = 8           # co-fire times kept per neighbour pair (#21 Hebbian, needs 5)
TRC = 512         # trace of the selected neuron
SCH = 256         # outside-input schedule entries
DRC = 300         # drive samples kept (one per 100 ms)

# integer scalars in S.sc
(T, FL_HEAD, FL_N, LL_HEAD, LL_N, EV_HEAD, EV_N, FREE_N, TR_HEAD, TR_N, SEL, STIM_N,
 SCH_N, SCH_PERIOD, AWAKE_N, DIRTY_N, LAST_FIRED_N, DR_HEAD, DR_N, REWARD_UNTIL, SENSOR_N, ELIST_N, PRUNE_AT,
 NSC) = range(24)
# float scalars in S.fs
LFUEL, RFUEL, NFS = range(3)
# event kinds
EV_GROW, EV_PRUNE, EV_LOCK, EV_UNLOCK, EV_BIRTH, EV_DEATH = range(6)
EVENT_NAMES = ("grow", "prune", "lock", "unlock", "birth", "death")

State = namedtuple("State", [
    "sc", "fs", "counts",
    # neurons
    "sign", "sensor", "hub", "thr_h", "rfuel_local", "rest_until", "window_start", "last_fire",
    "efuel", "efuel_t", "cfuel", "cfuel_b", "nov_sum", "nov_cnt", "last_growth",
    "nlines", "slot_line", "pool", "awake", "awake_list", "awake_pos", "dirty", "dirty_list",
    "fires", "fires_head", "fires_n", "ri_src", "ri_t", "ri_ch", "ri_n",
    "nearest", "n_nearest", "cof_t", "cof_n",
    # lines and mini-lines (mini 2L = a->b, 2L+1 = b->a)
    "line_a", "line_b", "line_delay", "line_alive", "line_gen", "free_lines",
    "thick", "energy", "energy_t", "pending", "last_spike", "last_arrival", "created", "trial",
    "locked", "thick_since",
    # spikes in flight
    "q_cnt", "q_mini", "q_ch", "q_en", "q_send", "q_gen",
    # outside input
    "stim_id", "stim_ch", "sch_phase", "sch_id", "sch_ch",
    # logs
    "fl_t", "fl_id", "cause_src", "cause_t", "cause_ch", "cause_n",
    "ll_t", "ll_src", "ll_dst", "ll_d", "ev_t", "ev_kind", "ev_src", "ev_dst",
    "tr_t", "tr_hub", "tr_thr", "last_fired", "fired_buf",
    "dr_t", "dr_l", "dr_r", "dr_c",
    # body link, reward timing, neurogenesis
    "alive", "motor", "px", "py", "pz", "region", "region_birth", "born",
    "drive", "drive_acc", "sensor_list", "mcount", "elig", "elig_t", "msyn", "mall",
    # v2: mini-buckets (branches), slow eligibility, eligible-line list, reward protection, per-neuron target rate
    "sbr", "bch", "bspk", "elig2", "elig2_t", "elist", "in_elist", "prot_t", "target",
    # own random generator (xorshift64*, same on CPU and GPU) and scratch space for the nearest search
    "rng", "nd", "nsel",
])

P_FIELDS = ["hub_cap", "thr_min", "thr_max", "final_thr_min", "final_thr_max", "window_ms", "rest_ms",
            "min_lines", "max_lines", "fatigue_mult", "fatigue_tau_ms", "ltp", "ltd", "stdp_window_ms",
            "thick_min", "thick_max", "trial_start", "trial_permanent", "trial_ms", "unused_ms", "unused_thin",
            "starve_ms", "hebb_gap_ms", "hebb_count", "hebb_window_ms", "target_hz", "rate_window_ms",
            "homeo_step", "homeo_thin", "check_ms", "efuel_cost", "efuel_tau_ms", "cfuel_every_ms",
            "cfuel_rise", "cfuel_tau_ms", "lock_thick", "lock_ms", "lock_ltd_scale", "astro_step",
            "rfuel_local_max", "elig_tau_ms", "birth_starving", "birth_gap_ms", "death_ms", "inhibitory_frac",
            "reward_window_ms", "thr_start", "leak_tau_ms", "branches", "branch_lo", "branch_hi", "dend_gain",
            "glia_hz", "prune_quiet", "prune_slices", "elig_slow_tau_ms", "elig_slow_w", "protect_ms",
            "grow_any", "birth_full_lines", "grow_need", "reward_grow", "reward_grow_min", "reward_grow_ms",
            "reward_grow_size", "delay_ms", "protect_thin", "close_elig"]
P = namedtuple("P", P_FIELDS)


def make_params(params):
    return P(*[float(getattr(params, f)) for f in P_FIELDS])


def make_state(n, n_lines_cap, k_near, n_regions=1, maxl=MAXL, branches=0):
    z = lambda shape, dt: np.zeros(shape, dtype=dt)  # noqa: E731
    f = lambda shape, v=NONE: np.full(shape, v, dtype=np.int64)  # noqa: E731
    q_cap = max(64, min(n * maxl, 4096))           # spikes in flight per ms; capped so many brains fit on the GPU
    nb = max(1, branches)
    L = n_lines_cap
    S = State(
        sc=z(NSC, np.int64), fs=np.array([1.0, 1.0]), counts=z(6, np.int64),
        sign=np.ones(n, np.int64), sensor=z(n, np.bool_), hub=z(n, np.float64), thr_h=np.full(n, 0.5),
        rfuel_local=np.ones(n), rest_until=z(n, np.int64), window_start=f(n), last_fire=f(n),
        efuel=np.ones(n), efuel_t=z(n, np.int64), cfuel=z(n, np.float64), cfuel_b=z(n, np.int64),
        nov_sum=z(n, np.float64), nov_cnt=z(n, np.int64), last_growth=np.full(n, -10**9, np.int64),
        nlines=z(n, np.int64), slot_line=f((n, maxl)), pool=z((n, maxl), np.float64),
        awake=z(n, np.bool_), awake_list=z(n, np.int64), awake_pos=f(n), dirty=z(n, np.bool_),
        dirty_list=z(n, np.int64),
        fires=f((n, FR)), fires_head=z(n, np.int64), fires_n=z(n, np.int64),
        ri_src=z((n, RI), np.int64), ri_t=z((n, RI), np.int64), ri_ch=z((n, RI), np.float64), ri_n=z(n, np.int64),
        nearest=f((n, k_near)), n_nearest=z(n, np.int64), cof_t=f((n, k_near, HEB)), cof_n=z((n, k_near), np.int64),
        line_a=f(L), line_b=f(L), line_delay=z(L, np.int64), line_alive=z(L, np.bool_), line_gen=z(L, np.int64),
        free_lines=np.arange(L - 1, -1, -1, dtype=np.int64),
        thick=z(2 * L, np.float64), energy=np.ones(2 * L), energy_t=z(2 * L, np.int64), pending=z(2 * L, np.float64),
        last_spike=f(2 * L), last_arrival=f(2 * L), created=z(2 * L, np.int64), trial=z(2 * L, np.bool_),
        locked=z(2 * L, np.bool_), thick_since=f(2 * L),
        q_cnt=z(QD, np.int64), q_mini=z((QD, q_cap), np.int64), q_ch=z((QD, q_cap), np.float64),
        q_en=z((QD, q_cap), np.float64), q_send=z((QD, q_cap), np.int64), q_gen=z((QD, q_cap), np.int64),
        stim_id=z(n * 4 + 16, np.int64), stim_ch=z(n * 4 + 16, np.float64),
        sch_phase=z(SCH, np.int64), sch_id=z(SCH, np.int64), sch_ch=z(SCH, np.float64),
        fl_t=z(FLC, np.int64), fl_id=z(FLC, np.int64), cause_src=z((FLC, RI), np.int64),
        cause_t=z((FLC, RI), np.int64), cause_ch=z((FLC, RI), np.float64), cause_n=z(FLC, np.int64),
        ll_t=z(LLC, np.int64), ll_src=z(LLC, np.int64), ll_dst=z(LLC, np.int64), ll_d=z(LLC, np.float64),
        ev_t=z(EVC, np.int64), ev_kind=z(EVC, np.int64), ev_src=z(EVC, np.int64), ev_dst=z(EVC, np.int64),
        tr_t=z(TRC, np.int64), tr_hub=z(TRC, np.float64), tr_thr=z(TRC, np.float64),
        last_fired=z(n, np.int64), fired_buf=z(n, np.int64),
        dr_t=z(DRC, np.int64), dr_l=z(DRC, np.float64), dr_r=z(DRC, np.float64), dr_c=z(DRC, np.float64),
        alive=np.ones(n, np.bool_), motor=z(n, np.bool_), px=z(n, np.float64), py=z(n, np.float64),
        pz=z(n, np.float64), region=z(n, np.int64), region_birth=np.full(n_regions, -10**9, np.int64),
        born=z(n, np.int64), drive=z(n, np.float64), drive_acc=z(n, np.float64), sensor_list=z(n, np.int64),
        mcount=z(n, np.int64), elig=z(2 * L, np.float64), elig_t=z(2 * L, np.int64),
        msyn=z(n, np.int64), mall=z(n, np.int64),
        sbr=z((n, maxl), np.int64), bch=z((n, nb), np.float64), bspk=f((n, nb)),
        elig2=z(2 * L, np.float64), elig2_t=z(2 * L, np.int64), elist=z(2 * L, np.int64),
        in_elist=z(2 * L, np.bool_), prot_t=f(2 * L), target=np.full(n, 2.0),
        rng=np.array([0x9E3779B97F4A7C15], dtype=np.uint64), nd=z(n, np.float64), nsel=z(n, np.bool_),
    )
    S.sc[FREE_N] = L
    S.sc[SEL] = -1
    S.sc[REWARD_UNTIL] = -1
    return S


def seed_rng(S, seed):
    """Seed the state's own random generator (never 0 for xorshift)."""
    S.rng[0] = np.uint64((seed * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019) % (2 ** 64)) or np.uint64(1)


@njit(cache=True, inline='always', error_model='numpy')
def rand01(S):
    """Uniform 0..1 from xorshift64* kept in S.rng: works the same compiled for CPU or GPU."""
    x = S.rng[0]
    x ^= x >> uint64(12)
    x ^= x << uint64(25)
    x ^= x >> uint64(27)
    S.rng[0] = x
    return ((x * uint64(2685821657736338717)) >> uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, inline='always', error_model='numpy')
def randint(S, n):
    return min(n - 1, int(rand01(S) * n))


# ---------- small helpers ----------

@njit(cache=True, inline='always', error_model='numpy')
def final_threshold(S, P, i):
    # F5: homeostasis threshold × Rfuel (global × local), kept between 0.15 and 0.95
    thr = S.thr_h[i] * S.fs[RFUEL] * S.rfuel_local[i]
    return min(P.final_thr_max, max(P.final_thr_min, thr))


@njit(cache=True, inline='always', error_model='numpy')
def efuel_now(S, P, i, t):
    e = 1.0 - (1.0 - S.efuel[i]) * math.exp(-(t - S.efuel_t[i]) / P.efuel_tau_ms)
    S.efuel[i] = e
    S.efuel_t[i] = t
    return e


@njit(cache=True, inline='always', error_model='numpy')
def energy_now(S, P, mi, t):
    e = 1.0 - (1.0 - S.energy[mi]) * math.exp(-(t - S.energy_t[mi]) / P.fatigue_tau_ms)
    S.energy[mi] = e
    S.energy_t[mi] = t
    return e


@njit(cache=True, inline='always', error_model='numpy')
def cfuel_now(S, P, i, t):
    # Cfuel fades ×exp(-10/300) at every 10 ms boundary without new input (#29); computed lazily
    boundaries = (t + P.cfuel_every_ms - 1) // P.cfuel_every_ms
    steps = boundaries - S.cfuel_b[i]
    if steps > 0:
        return S.cfuel[i] * math.exp(-steps * P.cfuel_every_ms / P.cfuel_tau_ms)
    return S.cfuel[i]


@njit(cache=True, inline='always', error_model='numpy')
def rate_hz(S, i, t, window):
    # walk back from the newest fire and stop at the first one outside the window
    c = 0
    k = S.fires_head[i]
    for _ in range(min(S.fires_n[i], FR)):
        k = (k - 1) % FR
        if S.fires[i, k] <= t - window:
            break
        c += 1
    return c * 1000.0 / window


@njit(cache=True, error_model='numpy')
def drive_means(S, P, t):
    """Mean local Rfuel and mean Cfuel over non-sensor neurons (for the Drives panel)."""
    r = 0.0
    c = 0.0
    n = 0
    for i in range(S.hub.shape[0]):
        if not S.sensor[i] and S.alive[i]:
            r += S.rfuel_local[i]
            c += cfuel_now(S, P, i, t)
            n += 1
    if n == 0:
        return 1.0, 0.0
    return r / n, c / n


@njit(cache=True, inline='always', error_model='numpy')
def is_empty(S, i):
    if S.hub[i] != 0.0:
        return False
    for s in range(S.nlines[i]):
        if S.pool[i, s] != 0.0:
            return False
    for b in range(S.bch.shape[1]):
        if S.bch[i, b] != 0.0:
            return False
    return True


@njit(cache=True, inline='always', error_model='numpy')
def wake(S, i):
    if not S.awake[i]:
        S.awake[i] = True
        n = S.sc[AWAKE_N]
        S.awake_list[n] = i
        S.awake_pos[i] = n
        S.sc[AWAKE_N] = n + 1


@njit(cache=True, inline='always', error_model='numpy')
def sleep(S, i):
    if S.awake[i]:
        S.awake[i] = False
        pos = S.awake_pos[i]
        last = S.awake_list[S.sc[AWAKE_N] - 1]
        S.awake_list[pos] = last
        S.awake_pos[last] = pos
        S.awake_pos[i] = NONE
        S.sc[AWAKE_N] -= 1


@njit(cache=True, inline='always', error_model='numpy')
def line_between(S, a, b):
    for s in range(S.nlines[a]):
        L = S.slot_line[a, s]
        if (S.line_a[L] == a and S.line_b[L] == b) or (S.line_a[L] == b and S.line_b[L] == a):
            return L
    return NONE


@njit(cache=True, inline='always', error_model='numpy')
def mini_from(S, L, src):
    return 2 * L if S.line_a[L] == src else 2 * L + 1


@njit(cache=True, inline='always', error_model='numpy')
def slot_of(S, i, L):
    for s in range(S.nlines[i]):
        if S.slot_line[i, s] == L:
            return s
    return NONE


@njit(cache=True, inline='always', error_model='numpy')
def log_event(S, t, kind, src, dst):
    h = S.sc[EV_HEAD]
    S.ev_t[h], S.ev_kind[h], S.ev_src[h], S.ev_dst[h] = t, kind, src, dst
    S.sc[EV_HEAD] = (h + 1) % EVC
    S.sc[EV_N] += 1
    S.counts[kind] += 1


@njit(cache=True, inline='always', error_model='numpy')
def push_input(S, i, src, t, ch):
    k = S.ri_n[i] % RI
    S.ri_src[i, k], S.ri_t[i, k], S.ri_ch[i, k] = src, t, ch
    S.ri_n[i] += 1


@njit(cache=True, inline='always', error_model='numpy')
def mini_src(S, mi):
    L = mi // 2
    return S.line_a[L] if mi % 2 == 0 else S.line_b[L]


@njit(cache=True, inline='always', error_model='numpy')
def mini_dst(S, mi):
    L = mi // 2
    return S.line_b[L] if mi % 2 == 0 else S.line_a[L]


# ---------- structure: connect / remove ----------

@njit(cache=True, inline='always', error_model='numpy')
def pick_branch(S, P, x, s):
    """v2 smart growth: a new line joins the mini-bucket holding the most charge right now (inputs that are active
    together end up together); with nothing active, lines are dealt round-robin."""
    nb = int(P.branches)
    if nb <= 0:
        return 0
    best, best_ch = s % nb, 0.0
    for b in range(nb):
        if S.bch[x, b] > best_ch:
            best, best_ch = b, S.bch[x, b]
    return best


@njit(cache=True, inline='always', error_model='numpy')
def connect(S, P, src, dst, size, trial, t, delay):
    """Activate the src->dst mini-line, creating the line if needed. Sign comes from src (#15)."""
    L = line_between(S, src, dst)
    if L == NONE:
        if S.nlines[src] >= P.max_lines or S.nlines[dst] >= P.max_lines or S.sc[FREE_N] == 0:
            return NONE
        S.sc[FREE_N] -= 1
        L = S.free_lines[S.sc[FREE_N]]
        S.line_a[L], S.line_b[L], S.line_delay[L], S.line_alive[L] = src, dst, delay, True
        for mi in (2 * L, 2 * L + 1):
            S.thick[mi] = 0.0
            S.pending[mi] = 0.0
            S.last_spike[mi] = NONE
            S.last_arrival[mi] = NONE
            S.trial[mi] = False
            S.locked[mi] = False
            S.thick_since[mi] = NONE
        for x in (src, dst):
            s = S.nlines[x]
            S.slot_line[x, s] = L
            S.pool[x, s] = 0.0
            S.sbr[x, s] = pick_branch(S, P, x, s)
            S.nlines[x] = s + 1
    mi = mini_from(S, L, src)
    S.thick[mi] = S.sign[src] * min(P.thick_max, size)
    S.created[mi] = t
    S.trial[mi] = trial
    S.locked[mi] = False
    S.thick_since[mi] = NONE
    S.energy[mi] = 1.0
    S.energy_t[mi] = t
    if trial:
        log_event(S, t, EV_GROW, src, dst)
    return mi


@njit(cache=True, inline='always', error_model='numpy')
def remove_mini(S, mi, t):
    log_event(S, t, EV_PRUNE, mini_src(S, mi), mini_dst(S, mi))
    S.thick[mi] = 0.0
    L = mi // 2
    if S.thick[mi ^ 1] != 0.0:
        return
    for x in (S.line_a[L], S.line_b[L]):
        s = slot_of(S, x, L)
        last = S.nlines[x] - 1
        S.slot_line[x, s] = S.slot_line[x, last]
        S.pool[x, s] = S.pool[x, last]
        S.sbr[x, s] = S.sbr[x, last]
        S.slot_line[x, last] = NONE
        S.pool[x, last] = 0.0
        S.nlines[x] = last
    S.line_alive[L] = False
    S.line_gen[L] += 1                       # spikes still flying on this line are dropped
    S.free_lines[S.sc[FREE_N]] = L
    S.sc[FREE_N] += 1


# ---------- learning (#17 #18 #19 #29 #30 #35) ----------

@njit(cache=True, inline='always', error_model='numpy')
def learn(S, P, receiver, mi, delta, t):
    if delta > 0:
        delta *= 1.0 + cfuel_now(S, P, receiver, t)     # Cfuel speeds up learning
    elif S.locked[mi]:
        delta *= P.lock_ltd_scale                        # locked lines barely weaken
    S.pending[mi] += delta
    # eligibility: the same change, remembered and fading over ~1 s, so a reward can boost or undo it (#30)
    S.elig[mi] = elig_now(S, P, mi, t) + delta
    S.elig_t[mi] = t
    if P.elig_slow_tau_ms > 0:                          # v2: a second, slow memory (seconds) for late rewards
        S.elig2[mi] = elig2_now(S, P, mi, t) + delta
        S.elig2_t[mi] = t
    if not S.in_elist[mi]:                              # remember which lines have eligibility (fast rewards)
        S.in_elist[mi] = True
        S.elist[S.sc[ELIST_N]] = mi
        S.sc[ELIST_N] += 1
    old = abs(S.thick[mi])
    # v1: Lfuel is fixed, so pending is applied at once (#30)
    size = min(P.thick_max, max(0.0, old + S.pending[mi] * S.fs[LFUEL]))
    S.pending[mi] = 0.0
    h = S.sc[LL_HEAD]
    S.ll_t[h], S.ll_src[h], S.ll_dst[h], S.ll_d[h] = t, mini_src(S, mi), mini_dst(S, mi), size - old
    S.sc[LL_HEAD] = (h + 1) % LLC
    S.sc[LL_N] += 1
    S.thick[mi] = S.sign[mini_src(S, mi)] * size


@njit(cache=True, inline='always', error_model='numpy')
def add_eligibility(S, P, mi, delta, t):
    """Eligibility only (no direct weight change): remembered for a reward, like learn() does."""
    S.elig[mi] = elig_now(S, P, mi, t) + delta
    S.elig_t[mi] = t
    if P.elig_slow_tau_ms > 0:
        S.elig2[mi] = elig2_now(S, P, mi, t) + delta
        S.elig2_t[mi] = t
    if not S.in_elist[mi]:
        S.in_elist[mi] = True
        S.elist[S.sc[ELIST_N]] = mi
        S.sc[ELIST_N] += 1


@njit(cache=True, inline='always', error_model='numpy')
def elig_now(S, P, mi, t):
    return S.elig[mi] * math.exp(-(t - S.elig_t[mi]) / P.elig_tau_ms)


@njit(cache=True, inline='always', error_model='numpy')
def elig2_now(S, P, mi, t):
    if P.elig_slow_tau_ms <= 0:
        return 0.0
    return S.elig2[mi] * math.exp(-(t - S.elig2_t[mi]) / P.elig_slow_tau_ms)


@njit(cache=True, error_model='numpy')
def apply_reward(S, P, r, t):
    """Reward r (-1..1): recent learning (eligibility) is applied again x r, so a thumbs-up strengthens
    what just happened and a thumbs-down undoes it (#30). Locked lines are not undone.
    Only lines with eligibility are visited (kept in S.elist), not every line in the brain."""
    changed = 0
    k = 0
    while k < S.sc[ELIST_N]:
        mi = S.elist[k]
        L = mi // 2
        e = elig_now(S, P, mi, t) + P.elig_slow_w * elig2_now(S, P, mi, t)
        if not S.line_alive[L] or S.thick[mi] == 0.0 or abs(e) < 1e-6:
            # nothing left to reward here: drop it from the list (swap with the last entry)
            S.in_elist[mi] = False
            S.sc[ELIST_N] -= 1
            S.elist[k] = S.elist[S.sc[ELIST_N]]
            continue
        k += 1
        # targeted: only lines into neurons that fired in the last reward_window_ms (they took part in the act)
        if P.reward_window_ms > 0:
            lf = S.last_fire[mini_dst(S, mi)]
            if lf == NONE or t - lf > P.reward_window_ms:
                continue
        if r < 0 and S.locked[mi]:
            continue
        old = abs(S.thick[mi])
        size = min(P.thick_max, max(0.0, old + e * r))
        h = S.sc[LL_HEAD]
        S.ll_t[h], S.ll_src[h], S.ll_dst[h], S.ll_d[h] = t, mini_src(S, mi), mini_dst(S, mi), size - old
        S.sc[LL_HEAD] = (h + 1) % LLC
        S.sc[LL_N] += 1
        S.thick[mi] = S.sign[mini_src(S, mi)] * size
        S.elig[mi] = 0.0
        S.elig2[mi] = 0.0
        if r > 0 and e > 0:
            S.prot_t[mi] = t                            # v2 lock on reward: safe from pruning for protect_ms
        changed += 1
    if P.reward_grow > 0 and r >= P.reward_grow_min:
        reward_growth(S, P, t)
    return changed


@njit(cache=True, inline='always', error_model='numpy')
def reward_growth(S, P, t):
    """Research: reward-guided growth. A neuron that fired shortly before a reward grows a trial line from a neuron
    that fired 2..20 ms before it (its likely cause), if they are not connected yet: wire up what just worked."""
    k = min(S.sc[FL_N], FLC)
    h = S.sc[FL_HEAD]
    grown = 0
    for back in range(k):
        idx = (h - 1 - back) % FLC
        tp = S.fl_t[idx]
        if t - tp > P.reward_grow_ms:
            break
        post = S.fl_id[idx]
        if S.sensor[post] or not S.alive[post] or S.nlines[post] >= P.max_lines or rand01(S) > 0.25:
            continue
        for b2 in range(back + 1, k):
            j = (h - 1 - b2) % FLC
            dt = tp - S.fl_t[j]
            if dt < 2:
                continue
            if dt > P.stdp_window_ms:
                break
            pre = S.fl_id[j]
            if (pre != post and S.alive[pre] and not S.motor[pre] and S.nlines[pre] < P.max_lines
                    and rand01(S) < 0.3 and line_between(S, pre, post) == NONE):
                if connect(S, P, pre, post, P.reward_grow_size, True, t, int(P.delay_ms)) != NONE:
                    grown += 1
                break
        if grown >= P.reward_grow:
            break


# ---------- outside input ----------

@njit(cache=True, inline='always', error_model='numpy')
def apply_input(S, P, i, ch, t):
    if S.window_start[i] == NONE and is_empty(S, i):
        S.window_start[i] = t
    S.hub[i] = min(P.hub_cap, S.hub[i] + ch)
    push_input(S, i, NONE, t, ch)
    wake(S, i)


# ---------- firing ----------

@njit(cache=True, inline='always', error_model='numpy')
def fire(S, P, i, t, delay):
    S.hub[i] -= final_threshold(S, P, i)            # carry-over (D4)
    S.efuel[i] -= P.efuel_cost                       # #36 (efuel_now ran in the check)
    S.rest_until[i] = t + int(P.rest_ms) + 1         # can't fire for the next 2 ms (D3)
    S.window_start[i] = NONE if is_empty(S, i) else t
    S.last_fire[i] = t                               # stamp (#24)
    if S.motor[i]:
        S.mcount[i] += 1                             # read and reset by the body every body tick
        S.mall[i] += 1                               # all motor spikes, and those driven by input lines
        for r in range(min(S.ri_n[i], RI)):          # (not babbling): lets babbling fade as control grows
            if S.ri_src[i, r] != NONE and t - S.ri_t[i, r] <= P.window_ms:
                S.msyn[i] += 1
                break
    k = S.fires_head[i]
    S.fires[i, k] = t
    S.fires_head[i] = (k + 1) % FR
    S.fires_n[i] += 1
    # fire log + what built up the charge (for "why it fired")
    h = S.sc[FL_HEAD]
    S.fl_t[h], S.fl_id[h] = t, i
    c = 0
    for r in range(min(S.ri_n[i], RI)):
        if t - S.ri_t[i, r] <= P.window_ms:
            S.cause_src[h, c], S.cause_t[h, c], S.cause_ch[h, c] = S.ri_src[i, r], S.ri_t[i, r], S.ri_ch[i, r]
            c += 1
    S.cause_n[h] = c
    S.ri_n[i] = 0
    S.sc[FL_HEAD] = (h + 1) % FLC
    S.sc[FL_N] += 1

    for s in range(S.nlines[i]):
        L = S.slot_line[i, s]
        out = mini_from(S, L, i)
        if S.thick[out] != 0.0:                      # all output lines fire (D1)
            en = energy_now(S, P, out, t)
            b = (t + S.line_delay[L]) % QD
            q = S.q_cnt[b]
            if q < S.q_mini.shape[1]:
                S.q_mini[b, q], S.q_ch[b, q], S.q_en[b, q] = out, S.thick[out] * en, en
                S.q_send[b, q], S.q_gen[b, q] = t, S.line_gen[L]
                S.q_cnt[b] = q + 1
            S.energy[out] = en * P.fatigue_mult      # #20
            S.last_spike[out] = t
        # LTP: input arrived before this fire, within 20 ms (#17 #19)
        inp = out ^ 1
        if S.thick[inp] != 0.0 and S.last_arrival[inp] != NONE:
            gap = t - S.last_arrival[inp]
            # v2 branch credit: an excitatory input only gets credit if its mini-bucket poured in just now
            credit = True
            if P.branches > 0 and S.thick[inp] > 0:
                bs = S.bspk[i, S.sbr[i, s]]
                credit = bs != NONE and t - bs <= P.stdp_window_ms
            if credit and 0 <= gap <= P.stdp_window_ms:
                learn(S, P, i, inp, P.ltp * (1.0 - gap / P.stdp_window_ms), t)

    # Hebbian growth: a nearby unconnected neuron fired just before, 5+ times in 1 s (#21)
    # v2 grow as needed: only while this neuron fires below grow_need x its target rate (busy neurons don't grow)
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
        for r in range(min(S.cof_n[i, kk], HEB)):
            if S.cof_t[i, kk, r] > t - P.hebb_window_ms:
                recent += 1
        if recent >= P.hebb_count:
            if connect(S, P, other, i, P.trial_start, True, t, delay) != NONE:
                S.cof_n[i, kk] = 0
                for r in range(HEB):
                    S.cof_t[i, kk, r] = NONE


# ---------- one tick ----------

@njit(cache=True, inline='always', error_model='numpy')
def tick(S, P, delay):
    t = S.sc[T]
    # 0. v2 small leak: charge already held fades a little before this tick's new input arrives
    if P.leak_tau_ms > 0:
        decay = math.exp(-1.0 / P.leak_tau_ms)
        for a in range(S.sc[AWAKE_N]):
            i = S.awake_list[a]
            S.hub[i] = S.hub[i] * decay if S.hub[i] >= 1e-4 else 0.0
            for s in range(S.nlines[i]):
                S.pool[i, s] = S.pool[i, s] * decay if S.pool[i, s] >= 1e-4 else 0.0
            for bb in range(S.bch.shape[1]):
                S.bch[i, bb] = S.bch[i, bb] * decay if S.bch[i, bb] >= 1e-4 else 0.0
    # 1. deliver spikes arriving now (+ into pools, − straight to the hub)
    b = t % QD
    for j in range(S.q_cnt[b]):
        mi = S.q_mini[b, j]
        L = mi // 2
        if not S.line_alive[L] or S.q_gen[b, j] != S.line_gen[L]:
            continue                                  # line was pruned while the spike flew
        dst = mini_dst(S, mi)
        ch = S.q_ch[b, j]
        S.last_arrival[mi] = t
        S.nov_sum[dst] += S.q_en[b, j]
        S.nov_cnt[dst] += 1
        if not S.dirty[dst]:
            S.dirty[dst] = True
            S.dirty_list[S.sc[DIRTY_N]] = dst
            S.sc[DIRTY_N] += 1
        push_input(S, dst, mini_src(S, mi), t, ch)
        if ch < 0:
            S.hub[dst] = max(0.0, S.hub[dst] + ch)
        else:
            if S.window_start[dst] == NONE and is_empty(S, dst):
                S.window_start[dst] = t
            s = slot_of(S, dst, L)
            if P.close_elig > 0:
                # research: credit for pushing the neuron close to its threshold, even without a spike
                v = S.hub[dst] / final_threshold(S, P, dst)
                psi = 1.0 - abs(v - 1.0) / 0.5
                if psi > 0:
                    add_eligibility(S, P, mi, P.close_elig * psi, t)
            if s != NONE:
                if P.branches > 0:
                    S.bch[dst, S.sbr[dst, s]] += ch
                else:
                    S.pool[dst, s] += ch
        # LTD: spike arrived after the receiver fired (#18 #19)
        lf = S.last_fire[dst]
        if lf != NONE and 0 < t - lf <= P.stdp_window_ms:
            learn(S, P, dst, mi, -P.ltd * (1.0 - (t - lf) / P.stdp_window_ms), t)
        wake(S, dst)
    S.q_cnt[b] = 0

    # outside input: senses from the body as a rate (#25 rate coding), one-off pushes, the repeating schedule
    for k in range(S.sc[SENSOR_N]):
        i = S.sensor_list[k]
        if S.drive[i] > 0.0:
            if S.motor[i]:
                # babbling (#32 spontaneous firing): random moments, so opposing motors don't cancel out
                if rand01(S) < S.drive[i]:
                    apply_input(S, P, i, final_threshold(S, P, i), t)
            else:
                # senses: steady rate coding (#25)
                S.drive_acc[i] += S.drive[i]
                if S.drive_acc[i] >= 1.0:
                    S.drive_acc[i] -= 1.0
                    apply_input(S, P, i, final_threshold(S, P, i), t)
    for k in range(S.sc[STIM_N]):
        apply_input(S, P, S.stim_id[k], S.stim_ch[k], t)
    S.sc[STIM_N] = 0
    if S.sc[SCH_PERIOD] > 0:
        ph = t % S.sc[SCH_PERIOD]
        for k in range(S.sc[SCH_N]):
            if S.sch_phase[k] == ph:
                i = S.sch_id[k]
                ch = S.sch_ch[k] if S.sch_ch[k] >= 0 else final_threshold(S, P, i)   # <0 = exactly its threshold
                apply_input(S, P, i, ch, t)

    # 2-4 for awake neurons: pools pass when full (F1), 20 ms window (#5), fire check
    nf = 0
    nb = int(P.branches)
    for a in range(S.sc[AWAKE_N]):
        i = S.awake_list[a]
        nl = S.nlines[i]
        if nb > 0:
            # v2 mini-buckets in the bucket: each fills from its lines and pours into the hub (x dend_gain) once it
            # holds branch_lo; overfilled past branch_hi it chokes and goes quiet (lets one neuron solve XOR)
            for b in range(nb):
                c = S.bch[i, b]
                if c >= P.branch_hi:
                    c = 0.0
                elif c >= P.branch_lo:
                    S.hub[i] += c * P.dend_gain
                    S.bspk[i, b] = t
                    c = 0.0
                S.bch[i, b] = c
        elif nl > 0:
            cap = 1.0 / nl
            for s in range(nl):
                pool = S.pool[i, s]
                while pool >= cap - 1e-9:
                    pool -= cap
                    S.hub[i] += cap
                S.pool[i, s] = max(0.0, pool)
        if P.leak_tau_ms > 0 and is_empty(S, i):
            S.window_start[i] = NONE              # v2: no 20 ms dump; the leak (step 0) does the forgetting
        S.hub[i] = min(P.hub_cap, S.hub[i])
        ws = S.window_start[i]
        if P.leak_tau_ms <= 0 and ws != NONE and t - ws >= P.window_ms:
            S.hub[i] = 0.0
            for s in range(nl):
                S.pool[i, s] = 0.0
            for b in range(S.bch.shape[1]):
                S.bch[i, b] = 0.0
            S.window_start[i] = NONE
            S.ri_n[i] = 0
        if t >= S.rest_until[i] and efuel_now(S, P, i, t) >= P.efuel_cost and S.hub[i] >= final_threshold(S, P, i):
            S.fired_buf[nf] = i
            nf += 1
    for k in range(nf):
        fire(S, P, S.fired_buf[k], t, delay)
        S.last_fired[k] = S.fired_buf[k]
    S.sc[LAST_FIRED_N] = nf

    # 6. Cfuel every 10 ms (#29): only neurons that got input need updating, the rest fade lazily
    if t % int(P.cfuel_every_ms) == 0:
        for d in range(S.sc[DIRTY_N]):
            i = S.dirty_list[d]
            c = cfuel_now(S, P, i, t)
            c += P.cfuel_rise * (S.nov_sum[i] / S.nov_cnt[i] - c)
            S.cfuel[i] = c
            S.cfuel_b[i] = t // int(P.cfuel_every_ms) + 1
            S.nov_sum[i] = 0.0
            S.nov_cnt[i] = 0
            S.dirty[i] = False
        S.sc[DIRTY_N] = 0

    # 7. sleep: nothing to compute (#26)
    a = 0
    while a < S.sc[AWAKE_N]:
        i = S.awake_list[a]
        if is_empty(S, i) and t >= S.rest_until[i]:
            S.window_start[i] = NONE
            sleep(S, i)
        else:
            a += 1

    # trace of the selected neuron (for the viewer)
    sel = S.sc[SEL]
    if sel >= 0:
        h = S.sc[TR_HEAD]
        S.tr_t[h], S.tr_hub[h], S.tr_thr[h] = t + 1, S.hub[sel], final_threshold(S, P, sel)
        S.sc[TR_HEAD] = (h + 1) % TRC
        S.sc[TR_N] += 1

    if t % int(P.check_ms) == 0 and t > 0:
        glia_check(S, P, t, delay)
        if t % 1000 == 0:
            neurogenesis(S, P, t, delay)
    # drives for the viewer, every 100 ms
    if (t + 1) % 100 == 0:
        r, c = drive_means(S, P, t + 1)
        h = S.sc[DR_HEAD]
        S.dr_t[h], S.dr_l[h], S.dr_r[h], S.dr_c[h] = t + 1, S.fs[LFUEL], S.fs[RFUEL] * r, c
        S.sc[DR_HEAD] = (h + 1) % DRC
        S.sc[DR_N] += 1
    # a temporary reward (Lfuel) ends: back to the neutral 1 (#30, v1)
    if S.sc[REWARD_UNTIL] >= 0 and t + 1 >= S.sc[REWARD_UNTIL]:
        S.fs[LFUEL] = 1.0
        S.sc[REWARD_UNTIL] = -1
    S.sc[T] = t + 1


@njit(cache=True, error_model='numpy')
def run(S, P, ticks, delay):
    for _ in range(ticks):
        tick(S, P, delay)


# ---------- glia: every 100 ms (#34) ----------

@njit(cache=True, error_model='numpy')
def glia_check(S, P, t, delay):
    n = S.hub.shape[0]
    for i in range(n):
        # sensors and motors are driven from outside (senses, babbling), so glia and homeostasis leave them alone
        if S.sensor[i] or S.motor[i] or not S.alive[i]:
            continue
        # astrocyte: calm only a neuron that is itself firing above target (#34); v2: only real runaway (glia_hz)
        if rate_hz(S, i, t, 1000.0) > (P.glia_hz if P.glia_hz > 0 else S.target[i]):
            S.rfuel_local[i] = min(P.rfuel_local_max, S.rfuel_local[i] + P.astro_step)
        else:
            S.rfuel_local[i] = max(1.0, S.rfuel_local[i] - P.astro_step)
    for i in range(n):
        if S.sensor[i] or S.motor[i] or not S.alive[i]:
            continue
        # homeostasis (#31)
        rate = rate_hz(S, i, t, P.rate_window_ms)
        if rate < S.target[i]:
            S.thr_h[i] = max(P.thr_min, S.thr_h[i] - P.homeo_step)
        elif rate > S.target[i]:
            S.thr_h[i] = min(P.thr_max, S.thr_h[i] + P.homeo_step)
            if P.homeo_thin < 1.0:
                for s in range(S.nlines[i]):
                    inp = mini_from(S, S.slot_line[i, s], i) ^ 1
                    if (S.thick[inp] != 0.0 and not S.locked[inp]
                            and not (P.protect_thin > 0 and S.prot_t[inp] != NONE and t - S.prot_t[inp] < P.protect_ms)):
                        S.thick[inp] *= P.homeo_thin
        # starvation growth (#21)
        quiet_since = S.last_fire[i] if S.last_fire[i] != NONE else 0
        if (t - quiet_since >= P.starve_ms and rate < S.target[i]
                and t - S.last_growth[i] >= P.starve_ms and S.nlines[i] < P.max_lines):
            # pick a random free neighbour among the nearest: try random ones first (cheap),
            # fall back to a full scan; either way every free neighbour is equally likely
            nn = S.n_nearest[i]
            chosen = NONE
            if P.grow_any > 0:
                # v2 grow by teamwork: take a line from a neuron that just fired, from any layer
                k = min(S.sc[FL_N], FLC)
                h = S.sc[FL_HEAD]
                for back in range(min(k, 64)):
                    cand = S.fl_id[(h - 1 - back) % FLC]
                    if t - S.fl_t[(h - 1 - back) % FLC] > 50:
                        break
                    if (cand != i and S.alive[cand] and not S.motor[cand] and line_between(S, cand, i) == NONE
                            and S.nlines[cand] < P.max_lines and rand01(S) < 0.5):
                        chosen = cand
                        break
            for _ in range(8 if chosen == NONE else 0):
                k = S.nearest[i, randint(S, nn)] if nn > 0 else NONE
                if k != NONE and S.alive[k] and line_between(S, k, i) == NONE and S.nlines[k] < P.max_lines:
                    chosen = k
                    break
            if chosen == NONE and P.grow_any <= 0:
                choices = 0
                for kk in range(nn):
                    k = S.nearest[i, kk]
                    if S.alive[k] and line_between(S, k, i) == NONE and S.nlines[k] < P.max_lines:
                        choices += 1
                if choices > 0:
                    pick = randint(S, choices)
                    for kk in range(nn):
                        k = S.nearest[i, kk]
                        if S.alive[k] and line_between(S, k, i) == NONE and S.nlines[k] < P.max_lines:
                            if pick == 0:
                                chosen = k
                                break
                            pick -= 1
            if chosen != NONE:
                connect(S, P, chosen, i, P.trial_start, True, t, delay)
                S.last_growth[i] = t

    # pruning and locking (#22 #35), on a rotating slice of the lines (prune_slices checks to cover all)
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
            # only lines that are thick, counting toward a lock, or locked can change lock state
            if abs(S.thick[mi]) >= P.lock_thick or S.thick_since[mi] != NONE or S.locked[mi]:
                check_lock(S, P, mi, t)
            if S.locked[mi]:
                continue
            size = abs(S.thick[mi])
            if S.trial[mi] and size >= P.trial_permanent:
                S.trial[mi] = False
            last_used = S.last_spike[mi] if S.last_spike[mi] != NONE else S.created[mi]
            if S.prot_t[mi] != NONE and t - S.prot_t[mi] < P.protect_ms:
                continue                                 # v2: used in a rewarded act, kept for now
            remove = (size < P.thick_min or (S.trial[mi] and t - S.created[mi] >= P.trial_ms)
                      or (P.prune_quiet > 0 and t - last_used >= P.unused_ms and size < P.unused_thin))
            if remove:
                last_two = S.thick[mi ^ 1] == 0.0 and (S.nlines[S.line_a[L]] <= P.min_lines
                                                     or S.nlines[S.line_b[L]] <= P.min_lines)
                if last_two:
                    # a neuron's last 2 lines stop at 0.1 instead (#18 #22)
                    S.thick[mi] = S.sign[mini_src(S, mi)] * max(size, P.thick_min)
                    S.trial[mi] = False
                else:
                    remove_mini(S, mi, t)


@njit(cache=True, error_model='numpy')
def compute_nearest(S):
    """Nearest alive neurons for every alive neuron (F7), by 3D distance."""
    # numpy-free (runs on the GPU too): pick the k smallest distances one by one; ties go to the lower index,
    # the same order a stable sort gives
    n = S.hub.shape[0]
    k = S.nearest.shape[1]
    d = S.nd
    taken = S.nsel
    for i in range(n):
        if not S.alive[i]:
            S.n_nearest[i] = 0
            continue
        for j in range(n):
            taken[j] = j == i or not S.alive[j]
            d[j] = (S.px[i] - S.px[j]) ** 2 + (S.py[i] - S.py[j]) ** 2 + (S.pz[i] - S.pz[j]) ** 2
        c = 0
        while c < k:
            best = NONE
            for j in range(n):
                if not taken[j] and (best == NONE or d[j] < d[best]):
                    best = j
            if best == NONE:
                break
            taken[best] = True
            S.nearest[i, c] = best
            c += 1
        S.n_nearest[i] = c


@njit(cache=True, error_model='numpy')
def add_nearest(S, i):
    """A neuron was born at spot i: find its own nearest neighbours, and put it into its neighbours' lists where it is
    closer than their current farthest one. O(n x k) instead of recomputing every list (O(n^2 x k))."""
    n = S.hub.shape[0]
    k = S.nearest.shape[1]
    d = S.nd
    taken = S.nsel
    for j in range(n):
        taken[j] = j == i or not S.alive[j]
        d[j] = (S.px[i] - S.px[j]) ** 2 + (S.py[i] - S.py[j]) ** 2 + (S.pz[i] - S.pz[j]) ** 2
    c = 0
    while c < k:
        best = NONE
        for j in range(n):
            if not taken[j] and (best == NONE or d[j] < d[best]):
                best = j
        if best == NONE:
            break
        taken[best] = True
        S.nearest[i, c] = best
        c += 1
    S.n_nearest[i] = c
    for j in range(n):
        if j == i or not S.alive[j]:
            continue
        m = S.n_nearest[j]
        if m == k:
            far = S.nearest[j, m - 1]
            fd = (S.px[j] - S.px[far]) ** 2 + (S.py[j] - S.py[far]) ** 2 + (S.pz[j] - S.pz[far]) ** 2
            if d[j] >= fd:
                continue
            m -= 1                                   # drop the farthest to make room
        pos = m                                      # insert i, keeping the list sorted by distance
        while pos > 0:
            q = S.nearest[j, pos - 1]
            qd = (S.px[j] - S.px[q]) ** 2 + (S.py[j] - S.py[q]) ** 2 + (S.pz[j] - S.pz[q]) ** 2
            if qd <= d[j]:
                break
            S.nearest[j, pos] = q
            S.cof_n[j, pos] = S.cof_n[j, pos - 1]
            pos -= 1
        S.nearest[j, pos] = i
        S.cof_n[j, pos] = 0
        S.n_nearest[j] = m + 1


@njit(cache=True, error_model='numpy')
def neurogenesis(S, P, t, delay):
    """#37: a region grows 1 new neuron when 3+ of its neurons are starving (below target) while already
    full (8 lines) and there is a free spot; a neuron with no lines and no spike for 60 s dies."""
    n = S.hub.shape[0]
    changed = False
    for i in range(n):
        if (S.alive[i] and not S.sensor[i] and not S.motor[i] and S.nlines[i] == 0
                and t - S.born[i] >= P.death_ms
                and (S.last_fire[i] == NONE or t - S.last_fire[i] >= P.death_ms)):
            S.alive[i] = False
            S.hub[i] = 0.0
            sleep(S, i)
            log_event(S, t, EV_DEATH, i, i)
            changed = True
    for r in range(S.region_birth.shape[0]):
        if t - S.region_birth[r] < P.birth_gap_ms:
            continue
        starving = 0
        free = NONE
        for i in range(n):
            if S.region[i] != r:
                continue
            if not S.alive[i]:
                if free == NONE:
                    free = i
            elif (not S.sensor[i] and S.nlines[i] >= (P.birth_full_lines if P.birth_full_lines > 0 else P.max_lines)
                  and rate_hz(S, i, t, P.rate_window_ms) < S.target[i]):
                starving += 1
        if starving < P.birth_starving or free == NONE:
            continue
        i = free
        S.alive[i] = True
        S.born[i] = t
        S.sign[i] = -1 if rand01(S) < P.inhibitory_frac else 1
        S.hub[i] = 0.0
        S.thr_h[i] = P.thr_start
        S.rfuel_local[i] = 1.0
        for b in range(S.bch.shape[1]):
            S.bch[i, b] = 0.0
            S.bspk[i, b] = NONE
        S.last_fire[i] = NONE
        S.window_start[i] = NONE
        S.fires_n[i] = 0
        S.efuel[i] = 1.0
        S.efuel_t[i] = t
        S.cfuel[i] = 0.0
        S.last_growth[i] = t
        S.region_birth[r] = t
        add_nearest(S, i)
        # 2 input lines from the busiest nearby neurons that still have room (R14)
        for _ in range(2):
            best = NONE
            best_rate = -1.0
            for kk in range(S.n_nearest[i]):
                k = S.nearest[i, kk]
                if S.alive[k] and S.nlines[k] < P.max_lines and line_between(S, k, i) == NONE:
                    rk = rate_hz(S, k, t, P.rate_window_ms)
                    if rk > best_rate:
                        best, best_rate = k, rk
            if best != NONE:
                connect(S, P, best, i, 0.5, False, t, delay)
        log_event(S, t, EV_BIRTH, i, i)
        changed = True
    # (dead neurons stay in their neighbours' lists; every use of a neighbour checks it is alive)


@njit(cache=True, inline='always', error_model='numpy')
def check_lock(S, P, mi, t):
    if abs(S.thick[mi]) >= P.lock_thick:
        if S.thick_since[mi] == NONE:
            S.thick_since[mi] = t
        r = mini_dst(S, mi)
        locked = 0
        for s in range(S.nlines[r]):
            if S.locked[mini_from(S, S.slot_line[r, s], r) ^ 1]:
                locked += 1
        if (not S.locked[mi] and t - S.thick_since[mi] >= P.lock_ms and S.fs[LFUEL] > 0
                and locked < S.nlines[r] // 2):
            S.locked[mi] = True
            log_event(S, t, EV_LOCK, mini_src(S, mi), r)
    else:
        S.thick_since[mi] = NONE
    if S.locked[mi] and S.fs[LFUEL] < -0.5:
        S.locked[mi] = False
        log_event(S, t, EV_UNLOCK, mini_src(S, mi), mini_dst(S, mi))
