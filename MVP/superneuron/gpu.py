"""Run many Hub brains at once on the GPU (Numba CUDA), one brain per GPU thread.

The GPU code is not a second copy of the neuron: engine.py's own source is compiled for the GPU here (its
`@njit` helpers become CUDA device functions), so every rule stays in one place. Brains of one batch must have the
same shape (same neuron count, capacity, line limit, mini-buckets) and the same parameters.

    batch = GPUBatch([net1.S, net2.S, ...], net1.P, delay_ms)
    batch.run(20)               # 20 ms for every brain
    batch.reward(r)             # r: one reward per brain (0 = none)
    batch.pull(["fires_n"])     # copy fields back to the host states
"""

import math
import re
from pathlib import Path

import numpy as np
from numba import cuda, uint64

from . import engine as E

THREADS = 64
_DEV = None
_HELPERS = '''


@cuda.jit(device=True)
def _min(a, b):
    return a if a < b else b


@cuda.jit(device=True)
def _max(a, b):
    return a if a > b else b
'''


def device_engine():
    """engine.py compiled for the GPU: a namespace with the same function names, as CUDA device functions."""
    global _DEV
    if _DEV is not None:
        return _DEV
    src = Path(E.__file__).read_text(encoding="utf-8")
    src = re.sub(r"@njit\([^)]*\)", "@cuda.jit(device=True)", src)
    src = src.replace("from numba import njit, uint64", "from numba import cuda, uint64" + _HELPERS)
    # this Numba's CUDA target mis-types the min/max builtins inside device functions: use 2-argument helpers
    src = re.sub(r"(?<![\w.])min\(", "_min(", src)
    src = re.sub(r"(?<![\w.])max\(", "_max(", src)
    ns = {"__name__": "superneuron.engine_gpu"}
    exec(compile(src, "engine_gpu", "exec"), ns)
    fields = E.State._fields
    args = ", ".join(f"F{k}" for k in range(len(fields)))
    views = ", ".join(f"F{k}[b]" for k in range(len(fields)))
    kern = f'''
@cuda.jit
def run_kernel({args}, P, ticks, delay, nb):
    b = cuda.grid(1)
    if b >= nb:
        return
    S = State({views})
    run(S, P, ticks, delay)


@cuda.jit
def reward_kernel({args}, P, r, nb):
    b = cuda.grid(1)
    if b >= nb or r[b] == 0.0:
        return
    S = State({views})
    apply_reward(S, P, r[b], S.sc[T])
'''
    exec(compile(kern, "engine_gpu_kernels", "exec"), ns)
    _DEV = ns
    return ns


class GPUBatch:
    """Many brains (engine States of the same shape) living on the GPU."""

    def __init__(self, states, P, delay):
        self.dev = device_engine()
        self.states = states
        self.P = P
        self.delay = int(delay)
        self.nb = len(states)
        self.blocks = (self.nb + THREADS - 1) // THREADS
        self.arrays = [cuda.to_device(np.stack([getattr(S, f) for S in states])) for f in E.State._fields]
        self.idx = {f: k for k, f in enumerate(E.State._fields)}

    def field(self, name):
        """The device array for one state field, shape (brains, ...)."""
        return self.arrays[self.idx[name]]

    def run(self, ticks):
        self.dev["run_kernel"][self.blocks, THREADS](*self.arrays, self.P, int(ticks), self.delay, self.nb)

    def reward(self, r):
        """r: device or host array, one reward per brain (0 = no reward for that brain)."""
        self.dev["reward_kernel"][self.blocks, THREADS](*self.arrays, self.P, r, self.nb)

    def pull(self, names=None):
        """Copy fields back into the host States (all fields if names is None)."""
        cuda.synchronize()
        for f in names or E.State._fields:
            host = self.field(f).copy_to_host()
            for b, S in enumerate(self.states):
                getattr(S, f)[...] = host[b]

    def push(self, names):
        """Copy host States' fields to the GPU (after editing them on the host)."""
        for f in names:
            self.arrays[self.idx[f]] = cuda.to_device(np.stack([getattr(S, f) for S in self.states]))
