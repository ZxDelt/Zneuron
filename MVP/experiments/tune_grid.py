"""Compare tuning options on the 100-neuron grid. Prints one line per variant (avg of 3 seeds).

    .venv/Scripts/python.exe -m experiments.tune_grid
    .venv/Scripts/python.exe -m experiments.tune_grid A C     (only some variants)

Add a variant by giving Params overrides in VARIANTS.
"""

import sys
from concurrent.futures import ProcessPoolExecutor

from run_live import sensor_pattern
from zneuron.network import Network
from zneuron.params import Params

SECONDS = 90

VARIANTS = {
    "A baseline":         {},
    "B fast homeostasis": {"homeo_step": 0.005},
    "C strong fatigue":   {"fatigue_mult": 0.5},
    "D no lock":          {"lock_ms": 10**12},
}


def run(name, seed):
    net = Network(n=100, sensors=5, seed=seed, layout="grid", params=Params(**VARIANTS[name]))
    stim = sensor_pattern(50, 5)
    step = net.size / 10
    inner = [n for n in net.neurons if not n.sensor]
    cols_seen, fire_frac = [], []
    for sec in range(1, SECONDS + 1):
        for _ in range(1000):
            stim(net)
            net.step()
        if sec > SECONDS - 30:      # judge the last 30 s, after settling
            recent = {k for t, k in net.fire_log if t > net.t - 1000 and not net.neurons[k].sensor}
            cols_seen.append(len({int(net.neurons[k].x // step) for k in recent}))
            fire_frac.append(len(recent) / len(inner))
    # does activity keep going on its own once the sensors stop? (runaway check)
    t0 = net.t
    for _ in range(2000):
        net.step()
    rates = [net.rate_hz(n) for n in inner]
    minis = [m for line in net.lines.values() for m in (line.ab, line.ba) if m.active]
    return {
        "mean_hz": sum(rates) / len(rates), "over_10hz": sum(r > 10 for r in rates),
        "cols": sum(cols_seen) / len(cols_seen), "active": sum(fire_frac) / len(fire_frac),
        "net_lines": net.counts["grow"] - net.counts["prune"],
        "locked": sum(m.locked for m in minis),
        "after_stop": sum(1 for t, _ in net.fire_log if t > t0 + 1000),
    }


def main():
    seeds = [1, 2, 3]
    names = [n for n in VARIANTS if len(sys.argv) < 2 or n.split()[0] in sys.argv[1:]]
    with ProcessPoolExecutor() as pool:
        jobs = {(n, s): pool.submit(run, n, s) for n in names for s in seeds}
        print(f"{'variant':20s} mean Hz  >10Hz  columns  active  +lines  locked  spikes 1-2 s after sensors stop")
        for n in names:
            rs = [jobs[(n, s)].result() for s in seeds]
            a = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}
            print(f"{n:20s} {a['mean_hz']:7.1f} {a['over_10hz']:6.1f} {a['cols']:6.1f}/10 {a['active']:6.0%} "
                  f"{a['net_lines']:7.0f} {a['locked']:7.0f} {a['after_stop']:6.0f}")


if __name__ == "__main__":
    main()
