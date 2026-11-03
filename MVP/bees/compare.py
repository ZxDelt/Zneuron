"""SuperNeuron v2 vs v1 vs textbook LIF vs ALIF, sharing one arena, in every condition, at every shared
start-weight level, many seeds each, in parallel.

    .venv/Scripts/python.exe -m bees.compare                      # 6 conditions x 3 levels x 10 seeds x 60 brain-min
    .venv/Scripts/python.exe -m bees.compare --minutes 10 --seeds 4 --conditions basic decoys --scales 2

All bees of a match get the same wiring, the same start weights and the same arena (they share it). Because each
seed is one match, the test is paired: for every metric, SuperNeuron v2's per-seed difference to each other bee is
tested with an exact sign-flip permutation test. Writes bees/results/compare.json and prints one JSON progress line per match.
"""

import argparse
import itertools
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "results"

METRICS = {   # key: (label, higher is better? None = information only)
    "deliveries": ("Squares delivered", True),
    "pickups": ("Squares picked up", True),
    "carry_success": ("Pick-ups that got delivered (%)", True),
    "decoys": ("Decoys grabbed", False),
    "drops": ("Wasted grabs while carrying", False),
    "bumps": ("Wall bumps", False),
    "reward": ("Total reward", True),
    "gain": ("Learning gain (deliveries/min, last third - first third)", True),
    "first_delivery_s": ("Time to first delivery (s)", False),
    "after_damage": ("Deliveries/min after damage", True),
    "spikes_per_s": ("Spikes per second (energy)", False),
    "spikes_per_delivery": ("Spikes per delivery (efficiency)", False),
    "speed_x": ("Simulation speed (x real time)", True),
    "lines_kept": ("Connections now vs start (%)", None),
    "neurons": ("Neurons at the end", None),
}
FOCUS = "hub2"


def run_match(condition, seed, minutes, scale, learn=True):
    from .brains import Match
    match = Match(condition, seed, minutes, scale, learn)
    total = minutes * 60_000
    curves = {m: [] for m in match.bees}
    for _ in range(minutes):
        before = {m: b.body.stats["deliveries"] for m, b in match.bees.items()}
        match.run(60_000)
        for m, b in match.bees.items():
            curves[m].append(b.body.stats["deliveries"] - before[m])
    out = {}
    third = max(1, minutes // 3)
    for m, bee in match.bees.items():
        s, c = bee.body.stats, curves[m]
        spikes = bee.spikes()
        r = dict(curve=c, deliveries=s["deliveries"], pickups=s["pickups"], decoys=s["decoys"], drops=s["drops"],
                 bumps=s["bumps"], reward=round(s["reward"], 2),
                 carry_success=round(100 * s["deliveries"] / s["pickups"], 1) if s["pickups"] else 0.0,
                 gain=round((sum(c[-third:]) - sum(c[:third])) / third, 3),
                 first_delivery_s=round(s["first_delivery_ms"] / 1000, 1) if s["first_delivery_ms"] else minutes * 60.0,
                 spikes_per_s=round(spikes / (total / 1000), 1),
                 spikes_per_delivery=round(spikes / s["deliveries"]) if s["deliveries"] else None,
                 speed_x=round(total / 1000 / bee.compute_s, 1),
                 lines_kept=round(100 * bee.brain.line_count() / bee.start_lines, 1), neurons=bee.neurons())
        if match.damage_at:
            half = minutes // 2
            r["after_damage"] = round(sum(c[half:]) / max(1, minutes - half), 3)
        out[m] = r
    return dict(condition=condition, seed=seed, minutes=minutes, scale=scale, learn=learn, bees=out)


def paired_test(diffs):
    """Two-sided sign-flip permutation test on paired differences: p for 'mean difference is 0'."""
    d = np.asarray(diffs, float)
    if len(d) < 2 or np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    if len(d) <= 16:
        signs = np.array(list(itertools.product((1, -1), repeat=len(d))))
    else:
        signs = np.random.default_rng(0).choice((1, -1), size=(20000, len(d)))
    return float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).mean())


def perm_test(a, b, n=5000):
    """Two-sided permutation test for two independent groups."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    both = np.concatenate([a, b])
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(n):
        x = rng.permutation(both)
        hits += abs(x[:len(a)].mean() - x[len(a):].mean()) >= obs - 1e-12
    return hits / n


def summarize(runs, conditions, scale, models):
    """Per condition and metric: each bee's mean/sd, and SuperNeuron v2 vs each other bee (paired test, winner)."""
    from .arena import CONDITIONS
    out = []
    for cond in conditions:
        rs = [r for r in runs if r["condition"] == cond and r["scale"] == scale and r["learn"]]
        if not rs:
            continue
        row = dict(condition=cond, label=CONDITIONS[cond]["label"], about=CONDITIONS[cond]["about"], metrics={})
        for key, (label, higher) in METRICS.items():
            vals = {m: [r["bees"][m].get(key) for r in rs] for m in models}
            if all(v is None for v in vals[FOCUS]):
                continue
            sd = lambda x: round(float(np.std(x, ddof=1)), 3) if len(x) > 1 else 0.0
            m_out = dict(label=label, higher=higher, bees={}, vs={})
            for m in models:
                x = np.array([v for v in vals[m] if v is not None], float)
                m_out["bees"][m] = dict(mean=round(float(x.mean()), 3) if len(x) else None, sd=sd(x) if len(x) else 0.0)
            for opp in models:
                if opp == FOCUS:
                    continue
                pairs = [(h, o) for h, o in zip(vals[FOCUS], vals[opp]) if h is not None and o is not None]
                if not pairs:
                    continue
                h = np.array([p[0] for p in pairs], float)
                o = np.array([p[1] for p in pairs], float)
                p = paired_test(h - o)
                winner = "tie"
                if higher is not None and p < 0.05 and h.mean() != o.mean():
                    winner = FOCUS if (h.mean() > o.mean()) == higher else opp
                m_out["vs"][opp] = dict(p=round(p, 4), winner=winner, ahead=int((h > o).sum()), behind=int((o > h).sum()))
            row["metrics"][key] = m_out
        row["curve"] = {m: np.mean([r["bees"][m]["curve"] for r in rs], axis=0).round(3).tolist() for m in models}
        row["vs"] = {opp: v["winner"] for opp, v in row["metrics"]["deliveries"]["vs"].items()}
        out.append(row)
    return out


def control(runs, scale, models):
    """Learning on vs off (basic): did rewards change deliveries?"""
    res = {}
    for m in models:
        on = [r["bees"][m]["deliveries"] for r in runs if r["learn"] and r["condition"] == "basic" and r["scale"] == scale]
        off = [r["bees"][m]["deliveries"] for r in runs if not r["learn"] and r["scale"] == scale]
        if not on or not off:
            return None
        res[m] = dict(learning_on=round(float(np.mean(on)), 2), learning_off=round(float(np.mean(off)), 2),
                      p=round(perm_test(on, off), 4),
                      curve_off=np.mean([r["bees"][m]["curve"] for r in runs if not r["learn"] and r["scale"] == scale],
                                        axis=0).round(3).tolist())
    return res


def main():
    from .arena import CONDITIONS
    from .brains import MODELS, SCALES
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--scales", type=float, nargs="*", default=list(SCALES))
    ap.add_argument("--workers", type=int, default=max(1, min(30, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--out", default=str(OUT / "compare.json"))
    ap.add_argument("--gpu", action="store_true", help="run every match at once on the GPU (bees/gpu_match.py)")
    ap.add_argument("--gpu-duty", type=float, default=1.0, help="share of time the GPU is used (e.g. 0.5 leaves half free)")
    a = ap.parse_args()

    jobs = [(c, s, a.minutes, k) for k in a.scales for c in a.conditions for s in range(1, a.seeds + 1)]
    if "basic" in a.conditions:
        # control: rewards switched off in "basic", to show how much is learned vs built in
        jobs += [("basic", s, a.minutes, k, False) for k in a.scales for s in range(1, a.seeds + 1)]
    random.Random(0).shuffle(jobs)
    t0 = time.time()
    runs = []
    if a.gpu:
        from .gpu_match import GPUMatches
        specs = [(j[0], j[1], j[5] if len(j) > 5 else True, j[3]) for j in jobs]
        g = GPUMatches(specs, a.minutes)
        g.run(lambda m, M, s: print(json.dumps(dict(progress=m, total=M, last=f"GPU minute {m}/{M}, {s:.0f}s")), flush=True),
              duty=a.gpu_duty)
        runs = g.results()
        jobs = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_match, *j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            runs.append(r)
            print(json.dumps(dict(progress=len(runs), total=len(jobs),
                                  last=f"{r['condition']} x{r['scale']:g} seed {r['seed']}"
                                       f"{'' if r['learn'] else ' (rewards off)'}: "
                                       + " · ".join(f"{m} {b['deliveries']}" for m, b in r["bees"].items()))),
                  flush=True)

    OUT.mkdir(exist_ok=True)
    res = dict(made=time.strftime("%Y-%m-%d %H:%M"), minutes=a.minutes, seeds=a.seeds, wall_s=round(time.time() - t0),
               scales=[f"{k:g}" for k in a.scales], models=MODELS, focus=FOCUS, conditions_order=a.conditions,
               by_scale={f"{k:g}": dict(conditions=summarize(runs, a.conditions, k, list(MODELS)),
                                        control=control(runs, k, list(MODELS))) for k in a.scales},
               runs=runs)
    Path(a.out).write_text(json.dumps(res))
    print(json.dumps(dict(done=True, out=a.out, wall_s=res["wall_s"])), flush=True)


if __name__ == "__main__":
    main()
