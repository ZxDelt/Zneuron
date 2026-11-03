"""Fast research fights: Zneuron variants vs LIF and ALIF, all conditions x all start-weight levels, in parallel.

    .venv/Scripts/python.exe -m bees.research                         # current v2
    .venv/Scripts/python.exe -m bees.research --over '{"leak_tau_ms": 30}' --name leak30
    .venv/Scripts/python.exe -m bees.research --minutes 10 --seeds 8

Only Zneuron's own neuron parameters (params.v2 overrides) change between variants; wiring, senses, rewards, the
learning rule and the rivals stay fixed, so a better score means a better neuron. Prints a scoreboard: for every
condition x weight level, v2's mean deliveries vs each rival and the paired-test winner. Appends to
bees/results/research.jsonl so every variant ever tried is kept.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "results" / "research.jsonl"
RIVALS = ("lif", "alif")             # --rivals lif alif alif_g adds the control: ALIF with the same growth rule


def set_rivals(r):
    global RIVALS
    RIVALS = tuple(r)


def one(cond, seed, minutes, scale, over, rivals=None):
    if rivals:
        set_rivals(rivals)
    from zneuron import params as PP
    from . import brains as B
    if over:
        base = PP.v2
        B.v2 = lambda **kw: base(**{**over, **kw})
    import math
    m = B.Match(cond, seed, minutes, scale, models=["hub2", *RIVALS])
    steps = minutes * 3000
    # steering accuracy: when the goal (nearest square, or the spot when carrying) is in view off to one side and the
    # bee turns, does it turn toward it? 50% = chance. Kept for the first and the last third of the run.
    steer = {k: [[0, 0], [0, 0]] for k in m.bees}
    for i in range(steps):
        m.step()
        part = 0 if i < steps // 3 else 1 if i >= 2 * steps // 3 else -1
        if part < 0:
            continue
        w = m.world
        for k, b in m.bees.items():
            body = b.body
            if abs(body.turn) < 0.2:
                continue
            if body.carrying:
                gx, gy = w.spot[0], w.spot[1]
            elif w.squares:
                gx, gy = min(w.squares, key=lambda q: (q[0] - body.x) ** 2 + (q[1] - body.y) ** 2)
            else:
                continue
            rel = (math.atan2(gy - body.y, gx - body.x) - body.heading + math.pi) % (2 * math.pi) - math.pi
            if 0.15 < abs(rel) < 1.8 and math.hypot(gx - body.x, gy - body.y) < w.eye_range:
                steer[k][part][1] += 1
                steer[k][part][0] += (rel > 0) == (body.turn > 0)
    out = {}
    for k, b in m.bees.items():
        f = b.brain.fires_n()
        acc = [round(100 * a / max(1, n), 1) for a, n in steer[k]]
        out[k] = dict(deliveries=b.body.stats["deliveries"], pickups=b.body.stats["pickups"],
                      reward=round(b.body.stats["reward"], 2), spikes=int(f.sum()), steer_early=acc[0], steer_late=acc[1])
    return cond, seed, scale, out


def score(rows):
    from .compare import paired_test
    table = {}
    for cond, seed, scale, out in rows:
        table.setdefault((cond, scale), []).append(out)
    res = {}
    for key, outs in sorted(table.items()):
        h = np.array([o["hub2"]["deliveries"] for o in outs], float)
        cell = dict(hub2=round(float(h.mean()), 2))
        for k in ("hub2", *RIVALS):
            cell[f"steer_{k}"] = (round(float(np.mean([o[k]["steer_early"] for o in outs])), 1),
                                  round(float(np.mean([o[k]["steer_late"] for o in outs])), 1))
        for r in RIVALS:
            x = np.array([o[r]["deliveries"] for o in outs], float)
            p = paired_test(h - x)
            cell[r] = round(float(x.mean()), 2)
            cell[f"p_{r}"] = round(p, 3)
            cell[f"win_{r}"] = "W" if p < 0.05 and h.mean() > x.mean() else "L" if p < 0.05 else ("w" if h.mean() > x.mean() else "l" if h.mean() < x.mean() else "=")
        res[f"{key[0]}@{key[1]:g}"] = cell
    return res


def main():
    from .arena import CONDITIONS
    ap = argparse.ArgumentParser()
    ap.add_argument("--over", default="{}", help="JSON overrides for params.v2")
    ap.add_argument("--name", default="v2")
    ap.add_argument("--minutes", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--scales", type=float, nargs="*", default=[1.0, 2.0, 3.0])
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--workers", type=int, default=max(1, min(30, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--rivals", nargs="*", default=list(RIVALS))
    a = ap.parse_args()
    set_rivals(a.rivals)
    over = json.loads(a.over)
    jobs = [(c, s, a.minutes, k, over, tuple(a.rivals)) for k in a.scales for c in a.conditions
            for s in range(1, a.seeds + 1)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, *zip(*jobs)))
    res = score(rows)
    wins = {r: sum(c[f"win_{r}"] == "W" for c in res.values()) for r in RIVALS}
    losses = {r: sum(c[f"win_{r}"] == "L" for c in res.values()) for r in RIVALS}
    ahead = {r: sum(c[f"win_{r}"] in ("W", "w") for c in res.values()) for r in RIVALS}
    tot = {k: round(float(np.mean([c[k] for c in res.values()])), 2) for k in ("hub2", *RIVALS)}
    print(f"== {a.name}  {json.dumps(over)}  ({a.minutes} min x {a.seeds} seeds, {round(time.time() - t0)}s)")
    print(f"   avg delivered: v2 {tot['hub2']}  lif {tot['lif']}  alif {tot['alif']}")
    for r in RIVALS:
        print(f"   vs {r:4s}: significant W {wins[r]:2d} / L {losses[r]:2d}   ahead in {ahead[r]}/{len(res)} cells")
    print("   cell             v2     lif   alif   (W/L = significant, w/l = not)   steering % early->late: v2 | lif | alif")
    for key, c in res.items():
        st = " | ".join(f"{c['steer_' + k][0]:4.1f}->{c['steer_' + k][1]:4.1f}" for k in ("hub2", *RIVALS))
        print(f"   {key:16s} {c['hub2']:5.1f}  {c['lif']:5.1f}{c['win_lif']} {c['alif']:5.1f}{c['win_alif']}    {st}")
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(name=a.name, over=over, minutes=a.minutes, seeds=a.seeds, made=time.strftime("%Y-%m-%d %H:%M"),
                                avg=tot, wins=wins, losses=losses, ahead=ahead, cells=res)) + "\n")


if __name__ == "__main__":
    main()
