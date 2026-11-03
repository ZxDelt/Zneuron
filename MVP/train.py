"""Train many brains in parallel (one per CPU core), each in its own world, and report total brain time.

    .venv/Scripts/python.exe train.py                          # one brain per CPU core (up to 30), 20 brain-min each
    .venv/Scripts/python.exe train.py --brains 4 --minutes 60
    .venv/Scripts/python.exe train.py --no-teach               # free play only (flowers, rocks)

Each brain is saved to brains/train-<seed>.brain; load one in the app with "Load brain".
A "control" brain with rewards switched off is trained alongside, to show whether learning beats chance.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from zneuron import saving, world as W
from zneuron.params import Params
from zneuron.teacher import Teacher

BRAINS = Path(__file__).parent / "brains"


def train_one(seed, minutes, teach, control, reward_scale=1.0, lock=True, lfuel=1.0):
    params = None if lock else Params(lock_ms=10**12)
    net = W.make_brain(seed=seed, params=params)
    agent = W.Agent(net, W.World(seed))
    net.lfuel = lfuel        # 1 = timing learning always sticks (v1); 0 = it only sticks when rewarded (#30)
    if control:
        net.reward = lambda value: 0           # rewards have no effect: the brain only babbles and does STDP
    elif reward_scale != 1.0:
        base = net.reward
        net.reward = lambda value: base(value * reward_scale)
    teacher = Teacher(agent, seed=seed)
    end = minutes * 60_000
    blocks = []
    start = time.perf_counter()
    while net.t < end:
        if teach:
            teacher.trial()
            if len(teacher.results) % 50 == 0:
                blocks.append(teacher.score(50))
        else:
            agent.run(5000)
    wall = time.perf_counter() - start
    name = f"train-{seed}{'-control' if control else ''}"
    BRAINS.mkdir(exist_ok=True)
    saving.save(net, BRAINS / f"{name}.brain", agent.world)
    return {"name": name, "brain_s": net.t / 1000, "wall_s": wall, "blocks": blocks,
            "trials": len(teacher.results), "rewards": len(agent.rewards), "alive": net.alive_count(),
            "births": net.counts["birth"], "locked": net.counts["lock"], "babble": agent.babble}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brains", type=int, default=max(1, min(30, (os.cpu_count() or 4) - 2)),
                    help="brains trained at once (default: CPU cores - 2, max 30)")
    ap.add_argument("--minutes", type=float, default=20, help="brain-minutes per brain")
    ap.add_argument("--no-teach", action="store_true", help="free play only, no word lessons")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--reward-scale", type=float, default=1.0, help="multiply every reward (stronger learning)")
    ap.add_argument("--no-lock", action="store_true", help="switch off line locking (#35) while training")
    ap.add_argument("--lfuel", type=float, default=0.0,
                    help="resting Lfuel: 0 = learning only sticks when rewarded (works for word lessons), 1 = v1 spec")
    args = ap.parse_args()

    jobs = [(args.seed + k, args.minutes, not args.no_teach, k == 0, args.reward_scale, not args.no_lock, args.lfuel)
            for k in range(args.brains)]
    print(f"training {args.brains} brains x {args.minutes} brain-minutes in parallel "
          f"({'word lessons' if not args.no_teach else 'free play'}; brain {args.seed} is the no-reward control)")
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.brains) as pool:
        results = list(pool.map(train_one, *zip(*jobs)))
    wall = time.perf_counter() - start
    total = sum(r["brain_s"] for r in results)
    for r in results:
        blocks = " ".join("-" if b is None else f"{b:.0%}" for b in r["blocks"])
        print(f"{r['name']:22s} {r['brain_s'] / 60:5.1f} brain-min in {r['wall_s']:5.0f} s "
              f"({r['brain_s'] / r['wall_s']:5.1f}x) | trials {r['trials']:4d} | right-turn share per 50 trials: {blocks} "
              f"| born {r['births']} locked {r['locked']}")
    print(f"\ntotal brain time {total / 60:.1f} min in {wall:.0f} s of real time = {total / wall:.0f}x real time combined")
    print(f"brains saved in {BRAINS}")


if __name__ == "__main__":
    main()
