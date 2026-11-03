"""LIF and ALIF bee brains on the GPU: bees/lif.py and bees/alif.py compiled as CUDA device code (same source),
plus one kernel each to run many brains and to give them rewards. See superneuron/gpu.py for the Hub brains."""

import re
import types
from pathlib import Path

import numpy as np
from numba import cuda

from superneuron.gpu import THREADS, _HELPERS, device_engine

from . import alif as A
from . import lif as L

_NS = {}


def _to_device_src(src):
    src = re.sub(r"@njit\([^)]*\)", "@cuda.jit(device=True)", src)
    src = src.replace("from numba import njit", "from numba import cuda" + _HELPERS)
    src = re.sub(r"(?<![\w.])min\(", "_min(", src)
    src = re.sub(r"(?<![\w.])max\(", "_max(", src)
    return src


def _kernels(state_cls, run_call):
    fields = state_cls._fields
    args = ", ".join(f"F{k}" for k in range(len(fields)))
    views = ", ".join(f"F{k}[b]" for k in range(len(fields)))
    return f'''
@cuda.jit
def run_kernel({args}, ticks, nb):
    b = cuda.grid(1)
    if b >= nb:
        return
    S = {state_cls.__name__}({views})
    {run_call}


@cuda.jit
def reward_kernel({args}, r, nb):
    b = cuda.grid(1)
    if b >= nb or r[b] == 0.0:
        return
    S = {state_cls.__name__}({views})
    apply_reward(S, r[b])
'''


def device_lif():
    if "lif" not in _NS:
        src = _to_device_src(Path(L.__file__).read_text(encoding="utf-8"))
        src = src.replace("from superneuron.engine import rand01", "rand01 = _ENGINE['rand01']")
        ns = {"__name__": "bees.lif_gpu", "_ENGINE": device_engine()}
        exec(compile(src, "lif_gpu", "exec"), ns)
        exec(compile(_kernels(L.LIFState, "run(S, ticks)"), "lif_gpu_kernels", "exec"), ns)
        _NS["lif"] = ns
    return _NS["lif"]


def device_alif():
    if "alif" not in _NS:
        lif = device_lif()
        src = _to_device_src(Path(A.__file__).read_text(encoding="utf-8"))
        src = src.replace("from . import lif as L", "L = _LIF")
        src = re.sub(r"from \.lif import ([\w, ]+)\n",
                     lambda m: "".join(f"{n.strip()} = _LIF.{n.strip()}\n" for n in m.group(1).split(",")), src)
        src = src.replace("L.rand01(", "rand01(")        # GPU code can't look up attributes on a namespace
        ns = {"__name__": "bees.alif_gpu", "_LIF": types.SimpleNamespace(**lif), "rand01": lif["rand01"]}
        exec(compile(src, "alif_gpu", "exec"), ns)
        ns["apply_reward"] = lif["apply_reward"]
        exec(compile(_kernels(A.ALIFState, "run(S, ticks)"), "alif_gpu_kernels", "exec"), ns)
        _NS["alif"] = ns
    return _NS["alif"]


class GPUPointBatch:
    """Many LIF or ALIF brains (same wiring shape) on the GPU. Same interface as superneuron.gpu.GPUBatch."""

    def __init__(self, states, kind):
        self.dev = device_lif() if kind == "lif" else device_alif()
        self.cls = L.LIFState if kind == "lif" else A.ALIFState
        self.states = states
        self.nb = len(states)
        self.blocks = (self.nb + THREADS - 1) // THREADS
        self.arrays = [cuda.to_device(np.stack([getattr(S, f) for S in states])) for f in self.cls._fields]
        self.idx = {f: k for k, f in enumerate(self.cls._fields)}

    def field(self, name):
        return self.arrays[self.idx[name]]

    def run(self, ticks):
        self.dev["run_kernel"][self.blocks, THREADS](*self.arrays, int(ticks), self.nb)

    def reward(self, r):
        self.dev["reward_kernel"][self.blocks, THREADS](*self.arrays, r, self.nb)

    def pull(self, names=None):
        cuda.synchronize()
        for f in names or self.cls._fields:
            host = self.field(f).copy_to_host()
            for b, S in enumerate(self.states):
                getattr(S, f)[...] = host[b]

    def push(self, names):
        for f in names:
            self.arrays[self.idx[f]] = cuda.to_device(np.stack([getattr(S, f) for S in self.states]))


# ---------- one thread per neuron (one block per brain), like superneuron/gpu_par.py ----------

_PAR = '''
@cuda.jit(device=True)
def nrand(R, i):
    x = R[i]
    x ^= x >> uint64(12)
    x ^= x << uint64(25)
    x ^= x >> uint64(27)
    R[i] = x
    return ((x * uint64(2685821657736338717)) >> uint64(11)) * (1.0 / 9007199254740992.0)


@cuda.jit(device=True)
def tick_par(S, i, n, R):
    t = S.sc[T]
    b = t % QD
    thr = THR
    if i < n:
        S.v[i] = S.v[i] * math.exp(-1.0 / TAU_M) + S.inbuf[b, i]
        S.inbuf[b, i] = 0.0
        if ADAPT:  # ADAPT
            S.a[i] *= math.exp(-1.0 / TAU_A)  # ADAPT
            thr = THR + BETA * S.a[i]  # ADAPT
        if S.drive[i] > 0.0:
            if S.is_motor[i]:
                if nrand(R, i) < S.drive[i]:
                    S.v[i] = _max(S.v[i], thr)
            elif S.is_sensor[i]:
                S.drive_acc[i] += S.drive[i]
                if S.drive_acc[i] >= 1.0:
                    S.drive_acc[i] -= 1.0
                    S.v[i] = _max(S.v[i], thr)
    cuda.syncthreads()
    if i < n and t >= S.refr[i] and S.v[i] >= thr:
        S.v[i] = 0.0
        if ADAPT:  # ADAPT
            S.a[i] += 1.0  # ADAPT
        S.refr[i] = t + REFR_MS + 1
        S.last_fire[i] = t
        S.fires_n[i] += 1
        if S.is_motor[i]:
            S.mcount[i] += 1
        h = cuda.atomic.add(S.sc, FL_HEAD, 1) % FLC
        S.fl_t[h] = t
        S.fl_id[h] = i
        bb = (t + DELAY) % QD
        for k in range(S.out_start[i], S.out_start[i + 1]):
            e = S.out_e[k]
            if S.w[e] == 0.0:
                continue
            d = S.dst[e]
            cuda.atomic.add(S.inbuf, (bb, d), S.sign[i] * S.w[e])
            S.last_arr[e] = t + DELAY
            lf = S.last_fire[d]
            if lf != NONE and 0 < t + DELAY - lf <= WIN:
                add_elig(S, e, -LTD * (1.0 - (t + DELAY - lf) / WIN), t)
        for k in range(S.in_start[i], S.in_start[i + 1]):
            e = S.in_e[k]
            la = S.last_arr[e]
            if la != NONE and S.w[e] != 0.0 and 0 <= t - la <= WIN:
                add_elig(S, e, LTP * (1.0 - (t - la) / WIN), t)
    cuda.syncthreads()
    if i == 0:
        S.sc[T] = t + 1
    cuda.syncthreads()


@cuda.jit(device=True)
def reward_par(S, i, nt, r):
    t = S.sc[T]
    for e in range(i, S.w.shape[0], nt):
        if S.w[e] == 0.0:
            continue
        el = elig_now(S, e, t) + S.cfg[1] * elig2_now(S, e, t)
        if abs(el) < 1e-6:
            continue
        S.w[e] = _min(1.0, _max(1e-6, S.w[e] + el * r))
        S.elig[e] = 0.0
        S.elig2[e] = 0.0
'''


def _par_kernels(state_cls, adapt):
    fields = state_cls._fields
    args = ", ".join(f"F{k}" for k in range(len(fields)))
    views = ", ".join(f"F{k}[b]" for k in range(len(fields)))
    keep = [ln for ln in _PAR.splitlines() if adapt or "# ADAPT" not in ln]   # LIF has no adaptation value
    src = "\n".join(keep).replace("if ADAPT:", "if True:")
    return src + f'''

@cuda.jit(max_registers=40)
def par_run_kernel({args}, ticks, n, NR):
    b = cuda.blockIdx.x
    i = cuda.threadIdx.x
    S = {state_cls.__name__}({views})
    for _ in range(ticks):
        tick_par(S, i, n, NR[b])


@cuda.jit(max_registers=40)
def par_reward_kernel({args}, r):
    b = cuda.blockIdx.x
    if r[b] == 0.0:
        return
    S = {state_cls.__name__}({views})
    reward_par(S, cuda.threadIdx.x, cuda.blockDim.x, r[b])
'''


def device_par(kind):
    key = kind + "_par"
    if key not in _NS:
        ns = dict(device_lif())                          # LIF helpers (eligibility, ...), then ALIF on top
        if kind == "alif":
            ns.update(device_alif())
        if kind == "lif":
            ns.update(TAU_A=1.0, BETA=0.0)                 # unused (ADAPT=False)
        from numba import uint64
        ns["uint64"] = uint64
        exec(compile(_par_kernels(L.LIFState if kind == "lif" else A.ALIFState, kind == "alif"),
                     f"{kind}_gpu_par", "exec"), ns)
        _NS[key] = ns
    return _NS[key]


class GPUPointParBatch(GPUPointBatch):
    """LIF or ALIF brains on the GPU, one thread per neuron (fast). Same interface."""

    def __init__(self, states, kind):
        super().__init__(states, kind)
        self.pdev = device_par(kind)
        self.n = states[0].v.shape[0]
        self.threads = ((self.n + 31) // 32) * 32
        seeds = np.array([[(int(S.rng[0]) * 6364136223846793005 + 1442695040888963407 * (k + 1)) % 2 ** 64 or 1
                           for k in range(self.n)] for S in states], dtype=np.uint64)
        self.nrng = cuda.to_device(seeds)

    def run(self, ticks):
        self.pdev["par_run_kernel"][self.nb, self.threads](*self.arrays, int(ticks), self.n, self.nrng)

    def reward(self, r):
        self.pdev["par_reward_kernel"][self.nb, self.threads](*self.arrays, r)
