"""GPU engines (skipped without CUDA):
- gpu.py (one brain per GPU thread) gives exactly the CPU spikes,
- gpu_par.py / gpu_bees.py (one GPU thread per neuron) give the CPU's firing rates on average."""

import numpy as np
import pytest

numba_cuda = pytest.importorskip("numba.cuda")
if not numba_cuda.is_available():
    pytest.skip("no CUDA GPU", allow_module_level=True)

from bees import flybrain as F                     # noqa: E402
from bees.brains import make_brain                 # noqa: E402

SIM = 2000


def brains(model, n=2):
    out = []
    for s in range(n):
        b, _ = make_brain(model, seed=1 + s, scale=2.0)
        rates = np.zeros(F.N_SENSORS)
        rates[5 + s] = 0.08
        rates[70] = 0.03
        b.set_drive(list(range(F.N_SENSORS)), rates)
        b.set_drive(list(range(F.N_SENSORS, F.N_SENSORS + F.N_MOTORS)), [0.004] * F.N_MOTORS)
        out.append(b)
    return out


def test_one_brain_per_thread_is_exact():
    from superneuron.gpu import GPUBatch
    cpu, gpu = brains("hub1"), brains("hub1")
    batch = GPUBatch([b.net.S for b in gpu], gpu[0].net.P, 2)
    batch.run(SIM)
    batch.pull(["fires_n"])
    for b in cpu:
        b.run(SIM)
    for c, g in zip(cpu, gpu):
        assert np.array_equal(c.net.S.fires_n, g.net.S.fires_n)


@pytest.mark.parametrize("model", ["hub2", "lif", "alif"])
def test_one_thread_per_neuron_matches_cpu_rates(model):
    from bees.gpu_bees import GPUPointParBatch
    from superneuron.gpu_par import GPUParBatch
    cpu, gpu = brains(model), brains(model)
    for b in cpu:
        b.run(SIM)
    if model == "hub2":
        batch = GPUParBatch([b.net.S for b in gpu], gpu[0].net.P, 2)
    else:
        batch = GPUPointParBatch([b.S for b in gpu], model)
    for _ in range(SIM // 20):
        batch.run(20)
    batch.pull(["fires_n"])
    c = sum(int(b.fires_n().sum()) for b in cpu)
    g = sum(int(b.fires_n().sum()) for b in gpu)
    assert abs(g - c) <= 0.25 * c, (c, g)
