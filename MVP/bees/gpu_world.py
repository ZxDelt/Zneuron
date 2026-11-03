"""The shared arena (bees/arena.py) written over plain arrays, so many worlds run at once on the GPU.

Same rules as arena.World: compound-eye sectors, body sense, 4 actions, shared squares, rewards, reward surprise.
Functions are @njit (testable on the CPU against arena.World); gpu_match.py compiles this same source for the GPU.
Starting layouts come from arena.World (pack_world), so a GPU world starts exactly like the CPU one; later respawn
positions use this module's own random generator, so long runs differ in detail from the CPU arena, not in rules.

Per world w and bee k:
  wi[w]: t, squares, decoys, move_spot_ms, steps, learn                 wf[w]: size, eye range, noise, spot x, y, r
  sq[w, i]: square x, y      rd[w, i]: decoy x, y
  bf[w, k]: x, y, heading, speed, turn, goal distance (-1 none), expected reward
  bi[w, k]: carrying, bump_until, last_bump
  st[w, k]: deliveries, pickups, decoys, drops, bumps, reward, first delivery ms (-1), distance
  curve[w, k, minute]: deliveries in that minute      rng[w, 0] world, rng[w, 1 + k] sense noise of bee k
"""

import math

import numpy as np
from numba import njit, uint64

from . import arena as A
from . import flybrain as F

# indices
T, NSQ, NRD, MOVE, STEPS, LEARN = range(6)
SIZE, EYE, NOISE, SPX, SPY, SPR = range(6)
BX, BY, BH, BSPD, BTURN, BGOAL, BEXP = range(7)
CARRY, BUMPU, LASTB = range(3)
DELIV, PICK, DECOY, DROP, BUMP, REW, FIRST, DIST = range(8)

BODY_MS = A.BODY_MS
DT = A.DT
FOV = A.FOV
SECTOR = A.SECTOR
EYE_RATE, CTX_RATE, TOUCH_RATE = A.EYE_RATE, A.CTX_RATE, A.TOUCH_RATE
BEE_R, TOUCH_D, GRAB_SPIKES, SHAPE_MS = A.BEE_R, A.TOUCH_D, A.GRAB_SPIKES, A.SHAPE_MS
R_PICK, R_DELIVER, R_DECOY, R_BUMP, R_CLOSER = A.R_PICK, A.R_DELIVER, A.R_DECOY, A.R_BUMP, A.R_CLOSER
SECTORS, EYE_COPIES, NS = F.SECTORS, F.EYE_COPIES, F.N_SENSORS
C_CARRY, C_EMPTY, C_TOUCH, C_SPOT, C_BUMP = (F.context_ids(n)[0] for n in ("carrying", "empty", "touch", "on_spot", "bump"))
N_CARRY, N_EMPTY, N_TOUCH, N_SPOT, N_BUMP = (len(F.context_ids(n)) for n in ("carrying", "empty", "touch", "on_spot", "bump"))
M0 = F.N_SENSORS
MC = F.MOTOR_COPIES
MAX_SQ, MAX_RD = 12, 8


@njit(cache=True)
def rnd(rng, i):
    """Uniform 0..1 from the xorshift64* generator in rng[i]."""
    x = rng[i]
    x ^= x >> uint64(12)
    x ^= x << uint64(25)
    x ^= x >> uint64(27)
    rng[i] = x
    return ((x * uint64(2685821657736338717)) >> uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@njit(cache=True)
def free_xy(wf, rng):
    for _ in range(100):
        x = 1.5 + rnd(rng, 0) * (wf[SIZE] - 3.0)
        y = 1.5 + rnd(rng, 0) * (wf[SIZE] - 3.0)
        if math.hypot(x - wf[SPX], y - wf[SPY]) > wf[SPR] + 2.0:
            return x, y
    return 1.5, 1.5


@njit(cache=True)
def touching(objs, n, x, y):
    for i in range(n):
        if math.hypot(objs[i, 0] - x, objs[i, 1] - y) <= TOUCH_D:
            return i
    return -1


@njit(cache=True)
def see(seen, kind, ox, oy, b, wf, is_spot):
    dx, dy = ox - b[BX], oy - b[BY]
    dist = math.hypot(dx, dy)
    if dist > wf[EYE]:
        return
    rel = wrap(math.atan2(dy, dx) - b[BH])
    if is_spot and dist <= wf[SPR]:
        rel = 0.0                                          # standing on it: it fills the front view
    if abs(rel) > FOV / 2:
        return
    sec = int((rel + FOV / 2) / SECTOR)
    if sec > SECTORS - 1:
        sec = SECTORS - 1
    v = 1.0 - dist / wf[EYE]
    if v > seen[kind * SECTORS + sec]:
        seen[kind * SECTORS + sec] = v


@njit(cache=True)
def sense(wi, wf, sq, rd, b, bi, rng, k, seen, drive):
    """Writes this bee's sense rates (spikes per ms) into drive[0:NS] (the brain's drive row)."""
    for i in range(3 * SECTORS):
        seen[i] = 0.0
    for i in range(wi[NSQ]):
        see(seen, 0, sq[i, 0], sq[i, 1], b, wf, False)
    for i in range(wi[NRD]):
        see(seen, 2, rd[i, 0], rd[i, 1], b, wf, False)
    see(seen, 1, wf[SPX], wf[SPY], b, wf, True)
    for kind in range(3):
        for sec in range(SECTORS):
            for c in range(EYE_COPIES):
                drive[(kind * SECTORS + sec) * EYE_COPIES + c] = EYE_RATE * seen[kind * SECTORS + sec]
    carrying = bi[CARRY] == 1
    on_spot = math.hypot(wf[SPX] - b[BX], wf[SPY] - b[BY]) <= wf[SPR]
    touch = (not carrying) and (touching(sq, wi[NSQ], b[BX], b[BY]) >= 0 or touching(rd, wi[NRD], b[BX], b[BY]) >= 0)
    for i in range(N_CARRY):
        drive[C_CARRY + i] = CTX_RATE if carrying else 0.0
    for i in range(N_EMPTY):
        drive[C_EMPTY + i] = 0.0 if carrying else CTX_RATE
    for i in range(N_TOUCH):
        drive[C_TOUCH + i] = TOUCH_RATE if touch else 0.0
    for i in range(N_SPOT):
        drive[C_SPOT + i] = TOUCH_RATE if on_spot else 0.0
    for i in range(N_BUMP):
        drive[C_BUMP + i] = 0.1 if wi[T] < bi[BUMPU] else 0.0
    if wf[NOISE] > 0:
        for i in range(NS):
            if rnd(rng, 1 + k) < wf[NOISE]:
                drive[i] = 0.0
        for i in range(NS):
            drive[i] += rnd(rng, 1 + k) * 0.01


@njit(cache=True)
def goal_distance(wi, wf, sq, b, bi):
    if bi[CARRY] == 1:
        return math.hypot(wf[SPX] - b[BX], wf[SPY] - b[BY])
    if wi[NSQ] == 0:
        return -1.0
    best = 1e30
    for i in range(wi[NSQ]):
        d = math.hypot(sq[i, 0] - b[BX], sq[i, 1] - b[BY])
        if d < best:
            best = d
    return best


@njit(cache=True)
def act(wi, wf, sq, rd, b, bi, st, curve, rng, mcount):
    """Moves one bee from its motor spikes (mcount row, read and reset), returns its reward (-1..1)."""
    left = right = fwd = grab = 0
    for c in range(MC):
        left += mcount[M0 + c]
        right += mcount[M0 + MC + c]
        fwd += mcount[M0 + 2 * MC + c]
        grab += mcount[M0 + 3 * MC + c]
    for c in range(4 * MC):
        mcount[M0 + c] = 0
    t = wi[T]
    r = 0.0
    turn_target = 2.5 * (left - right)
    if turn_target > 4.0:
        turn_target = 4.0
    if turn_target < -4.0:
        turn_target = -4.0
    speed_target = 1.5 * fwd
    if speed_target > 4.0:
        speed_target = 4.0
    b[BTURN] += (turn_target - b[BTURN]) * 0.25
    b[BSPD] += (speed_target - b[BSPD]) * 0.2
    b[BH] = wrap(b[BH] + b[BTURN] * DT)
    nx = b[BX] + math.cos(b[BH]) * b[BSPD] * DT
    ny = b[BY] + math.sin(b[BH]) * b[BSPD] * DT
    lo, hi = BEE_R, wf[SIZE] - BEE_R
    if not (lo <= nx <= hi and lo <= ny <= hi):
        nx = min(hi, max(lo, nx))
        ny = min(hi, max(lo, ny))
        b[BSPD] *= 0.3
        bi[BUMPU] = t + 200
        if t - bi[LASTB] > 500:
            r += R_BUMP
            st[BUMP] += 1
        bi[LASTB] = t
    st[DIST] += math.hypot(nx - b[BX], ny - b[BY])
    b[BX], b[BY] = nx, ny

    if grab >= GRAB_SPIKES:
        if bi[CARRY] == 1:
            if math.hypot(wf[SPX] - b[BX], wf[SPY] - b[BY]) <= wf[SPR]:      # place it on the spot
                bi[CARRY] = 0
                r += R_DELIVER
                st[DELIV] += 1
                curve[t // 60000] += 1
                if st[FIRST] < 0:
                    st[FIRST] = t
                x, y = free_xy(wf, rng)                                      # a new square appears somewhere
                sq[wi[NSQ], 0], sq[wi[NSQ], 1] = x, y
                wi[NSQ] += 1
                b[BGOAL] = -1.0
            else:
                st[DROP] += 1
        else:
            i = touching(sq, wi[NSQ], b[BX], b[BY])
            if i >= 0:
                last = wi[NSQ] - 1                                           # same order as list.pop(i)
                for j in range(i, last):
                    sq[j, 0], sq[j, 1] = sq[j + 1, 0], sq[j + 1, 1]
                wi[NSQ] = last
                bi[CARRY] = 1
                r += R_PICK
                st[PICK] += 1
                b[BGOAL] = -1.0
            else:
                i = touching(rd, wi[NRD], b[BX], b[BY])
                if i >= 0:
                    x, y = free_xy(wf, rng)
                    rd[i, 0], rd[i, 1] = x, y
                    r += R_DECOY
                    st[DECOY] += 1

    if (t + BODY_MS) % SHAPE_MS == 0:                     # shaping: closer to the current goal
        d = goal_distance(wi, wf, sq, b, bi)
        if d >= 0 and b[BGOAL] >= 0:
            if d < b[BGOAL] - 0.15:
                r += R_CLOSER
            elif d > b[BGOAL] + 0.15:
                r -= R_CLOSER
        b[BGOAL] = d
    st[REW] += r
    return min(1.0, max(-1.0, r))


@njit(cache=True)
def surprise(b, r, gain, rate):
    """Shared learning rule: reward minus the expected reward, x gain (bees/brains.py Bee.give_reward)."""
    s = r - b[BEXP]
    b[BEXP] += rate * (r - b[BEXP])
    v = gain * s
    return min(1.0, max(-1.0, v))


@njit(cache=True)
def world_tick(wi, wf, bf, rng, K):
    wi[T] += BODY_MS
    wi[STEPS] += 1
    if wi[MOVE] > 0 and wi[T] % wi[MOVE] == 0:
        m = wf[SPR] + 2
        wf[SPX] = m + rnd(rng, 0) * (wf[SIZE] - 2 * m)
        wf[SPY] = m + rnd(rng, 0) * (wf[SIZE] - 2 * m)
        for k in range(K):
            bf[k, BGOAL] = -1.0


def pack_worlds(specs, names, minutes):
    """specs: [(condition, seed, learn)]. Start layouts copied from arena.World so CPU and GPU worlds start equal."""
    W, K = len(specs), len(names)
    wi = np.zeros((W, 6), np.int64)
    wf = np.zeros((W, 6))
    sq = np.zeros((W, MAX_SQ, 2))
    rd = np.zeros((W, MAX_RD, 2))
    bf = np.zeros((W, K, 7))
    bi = np.zeros((W, K, 3), np.int64)
    st = np.zeros((W, K, 8))
    curve = np.zeros((W, K, minutes + 1))
    rng = np.zeros((W, 1 + K), np.uint64)
    for w, (cond, seed, learn) in enumerate(specs):
        cw = A.World(cond, seed, names=tuple(names))
        wi[w] = (0, len(cw.squares), len(cw.reds), cw.c.get("move_spot_ms", 0), 0, int(learn))
        wf[w] = (cw.size, cw.eye_range, cw.noise, cw.spot[0], cw.spot[1], cw.spot[2])
        for i, (x, y) in enumerate(cw.squares):
            sq[w, i] = (x, y)
        for i, (x, y) in enumerate(cw.reds):
            rd[w, i] = (x, y)
        for k, n in enumerate(names):
            body = cw.bodies[n]
            bf[w, k] = (body.x, body.y, body.heading, 0.0, 0.0, -1.0, 0.0)
            bi[w, k] = (0, 0, -10_000)
            st[w, k, FIRST] = -1
        rng[w, 0] = np.uint64((seed * 0x9E3779B97F4A7C15 + 7) % 2 ** 64) or 1
        rng[w, 1:] = np.uint64((seed * 31 + 7) * 0x2545F4914F6CDD1D % 2 ** 64) or 1
    return dict(wi=wi, wf=wf, sq=sq, rd=rd, bf=bf, bi=bi, st=st, curve=curve, rng=rng)
