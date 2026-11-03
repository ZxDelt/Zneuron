"""Many bee matches at once, entirely on the GPU: brains (all four neuron models), bodies and shared arenas.

Brains run one GPU thread per neuron (zneuron/gpu_par.py, bees/gpu_bees.py). Each 20 ms body step is a
handful of GPU kernel launches for ALL matches together:
  sense (every bee of every world) -> run 20 ms of each brain batch -> act (every world) -> rewards per brain batch.
Brains follow the CPU engine's rules (same averages, not spike for spike). Arenas follow arena.World's rules.

    python -m bees.compare --gpu        (uses this)
"""

import random
import re
import time
from pathlib import Path

import numpy as np
from numba import cuda

from zneuron.gpu import THREADS, _HELPERS
from zneuron.gpu_par import GPUParBatch

from . import gpu_world as GW
from .brains import BABBLE, BASELINE_RATE, MODELS, REWARD_GAIN, make_brain
from . import flybrain as F
from .gpu_bees import GPUPointParBatch

_WNS = None


def device_world():
    """gpu_world.py compiled for the GPU, plus the sense and act kernels (4 bee types)."""
    global _WNS
    if _WNS is not None:
        return _WNS
    src = Path(GW.__file__).read_text(encoding="utf-8")
    src = re.sub(r"@njit\([^)]*\)", "@cuda.jit(device=True)", src)
    src = src.replace("from numba import njit, uint64", "from numba import cuda, uint64" + _HELPERS)
    src = re.sub(r"(?<![\w.])min\(", "_min(", src)
    src = re.sub(r"(?<![\w.])max\(", "_max(", src)
    src += '''

@cuda.jit
def sense_kernel(wi, wf, sq, rd, bf, bi, rng, seen, D0, D1, D2, D3, W, K):
    j = cuda.grid(1)
    if j >= W * K:
        return
    w, k = j // K, j % K
    if k == 0:
        D = D0[w]
    elif k == 1:
        D = D1[w]
    elif k == 2:
        D = D2[w]
    else:
        D = D3[w]
    sense(wi[w], wf[w], sq[w], rd[w], bf[w, k], bi[w, k], rng[w], k, seen[w, k], D)


@cuda.jit
def act_kernel(wi, wf, sq, rd, bf, bi, st, curve, rng, M0, M1, M2, M3, R0, R1, R2, R3, W, K, gain, rate):
    w = cuda.grid(1)
    if w >= W:
        return
    off = wi[w, STEPS] % K
    for kk in range(K):
        k = (kk + off) % K
        if k == 0:
            M = M0[w]
        elif k == 1:
            M = M1[w]
        elif k == 2:
            M = M2[w]
        else:
            M = M3[w]
        r = act(wi[w], wf[w], sq[w], rd[w], bf[w, k], bi[w, k], st[w, k], curve[w, k], rng[w], M)
        v = 0.0
        if wi[w, LEARN] == 1 and r != 0.0:
            v = surprise(bf[w, k], r, gain, rate)
        if k == 0:
            R0[w] = v
        elif k == 1:
            R1[w] = v
        elif k == 2:
            R2[w] = v
        else:
            R3[w] = v
    world_tick(wi[w], wf[w], bf[w], rng[w], K)
'''
    ns = {"__name__": "bees.gpu_world_dev"}
    exec(compile(src, "gpu_world_dev", "exec"), ns)
    _WNS = ns
    return ns


class GPUMatches:
    """specs: [(condition, seed, learn, scale)], all `minutes` long (worlds may mix conditions and weight levels)."""

    def __init__(self, specs, minutes, models=None):
        self.models = list(models or MODELS)
        assert len(self.models) == 4, "kernels are written for 4 bees per world"
        self.specs, self.minutes = specs, minutes
        self.W, self.K = len(specs), len(self.models)
        self.dev = device_world()
        motors = list(range(F.N_SENSORS, F.N_SENSORS + F.N_MOTORS))
        babble = [BABBLE[a] for a in F.ACTIONS for _ in range(F.MOTOR_COPIES)]
        self.brains = {m: [] for m in self.models}
        flies = []
        for cond, seed, learn, scale in specs:
            for m in self.models:
                b, fly = make_brain(m, seed, scale)
                b.set_drive(motors, babble)
                self.brains[m].append(b)
            flies.append(fly)
        self.start_lines = {m: [b.line_count() for b in self.brains[m]] for m in self.models}
        self.batches = {}
        for m in self.models:
            bs = self.brains[m]
            if m.startswith("hub"):
                self.batches[m] = GPUParBatch([b.net.S for b in bs], bs[0].net.P, bs[0].net.p.delay_ms)
            else:
                self.batches[m] = GPUPointParBatch([b.S for b in bs], m)
        self.world_host = GW.pack_worlds([s[:3] for s in specs], self.models, minutes)
        self.world = {k: cuda.to_device(v) for k, v in self.world_host.items()}
        self.seen = cuda.device_array((self.W, self.K, 3 * F.SECTORS))
        self.R = {m: cuda.to_device(np.zeros(self.W)) for m in self.models}
        # brain damage (same lines cut in every bee of a world), halfway through
        self.damage = {}
        for w, (cond, seed, learn, scale) in enumerate(specs):
            dmg = GW.A.CONDITIONS[cond].get("damage")
            if dmg:
                rng = random.Random(seed * 7919)
                self.damage[w] = [(a, b) for a, b, _ in rng.sample(flies[w].edges, round(len(flies[w].edges) * dmg))]
        self.damage_at = minutes * 30_000 if self.damage else None
        self.t = 0
        self.wall = 0.0

    def _damage_now(self):
        for m in self.models:
            batch = self.batches[m]
            batch.pull()
            for w, pairs in self.damage.items():
                self.brains[m][w].cut(pairs)
            batch.push(["thick"] if m.startswith("hub") else ["w"])

    def step(self):
        d, wd, K, W = self.dev, self.world, self.K, self.W
        D = [self.batches[m].field("drive") for m in self.models]
        Mc = [self.batches[m].field("mcount") for m in self.models]
        d["sense_kernel"][(W * K + THREADS - 1) // THREADS, THREADS](
            wd["wi"], wd["wf"], wd["sq"], wd["rd"], wd["bf"], wd["bi"], wd["rng"], self.seen, *D, W, K)
        for m in self.models:
            self.batches[m].run(GW.BODY_MS)
        d["act_kernel"][(W + THREADS - 1) // THREADS, THREADS](
            wd["wi"], wd["wf"], wd["sq"], wd["rd"], wd["bf"], wd["bi"], wd["st"], wd["curve"], wd["rng"],
            *Mc, *[self.R[m] for m in self.models], W, K, REWARD_GAIN, BASELINE_RATE)
        for m in self.models:
            self.batches[m].reward(self.R[m])
        self.t += GW.BODY_MS
        if self.damage_at is not None and self.t == self.damage_at:
            self._damage_now()

    def run(self, progress=None, duty=1.0):
        """duty < 1 leaves the GPU free part of the time (for other apps): after each brain-second the run waits
        (1 - duty) / duty times as long as that second took."""
        t0 = time.perf_counter()
        total = self.minutes * 60_000
        mark = time.perf_counter()
        while self.t < total:
            self.step()
            if duty < 1.0 and self.t % 1000 == 0:
                cuda.synchronize()
                busy = time.perf_counter() - mark
                time.sleep(busy * (1.0 - duty) / duty)
                mark = time.perf_counter()
            if progress and self.t % 60_000 == 0:
                cuda.synchronize()
                progress(self.t // 60_000, self.minutes, time.perf_counter() - t0)
        cuda.synchronize()
        self.wall = time.perf_counter() - t0

    def results(self):
        """One record per world, in the same format as compare.run_match."""
        for m in self.models:
            self.batches[m].pull()
        host = {k: v.copy_to_host() for k, v in self.world.items()}
        out = []
        third = max(1, self.minutes // 3)
        total_s = self.minutes * 60
        for w, (cond, seed, learn, scale) in enumerate(self.specs):
            bees = {}
            for k, m in enumerate(self.models):
                st, c = host["st"][w, k], [int(x) for x in host["curve"][w, k, :self.minutes]]
                b = self.brains[m][w]
                spikes = int(b.fires_n().sum())
                deliv = int(st[GW.DELIV])
                r = dict(curve=c, deliveries=deliv, pickups=int(st[GW.PICK]), decoys=int(st[GW.DECOY]),
                         drops=int(st[GW.DROP]), bumps=int(st[GW.BUMP]), reward=round(float(st[GW.REW]), 2),
                         carry_success=round(100 * deliv / st[GW.PICK], 1) if st[GW.PICK] else 0.0,
                         gain=round((sum(c[-third:]) - sum(c[:third])) / third, 3),
                         first_delivery_s=round(st[GW.FIRST] / 1000, 1) if st[GW.FIRST] >= 0 else float(total_s),
                         spikes_per_s=round(spikes / total_s, 1),
                         spikes_per_delivery=round(spikes / deliv) if deliv else None,
                         speed_x=None,
                         lines_kept=round(100 * b.line_count() / self.start_lines[m][w], 1),
                         neurons=b.alive_count() if hasattr(b, "alive_count") else b.n)
                if w in self.damage:
                    half = self.minutes // 2
                    r["after_damage"] = round(sum(c[half:]) / max(1, self.minutes - half), 3)
                bees[m] = r
            out.append(dict(condition=cond, seed=seed, minutes=self.minutes, scale=scale, learn=learn,
                            bees=bees, gpu=True))
        return out
