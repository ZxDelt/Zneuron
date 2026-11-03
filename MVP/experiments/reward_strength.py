"""Does word learning appear with stronger rewards? Compares first vs last 400 lessons across many brains.

    .venv/Scripts/python.exe -m experiments.reward_strength
"""

import statistics
from concurrent.futures import ProcessPoolExecutor

import train

VARIANTS = [  # (label, reward scale, locking on, seeds)
    ("reward x1 (as specced)", 1.0, True, range(100, 110)),
    ("reward x5, locking off", 5.0, False, range(200, 210)),
    ("reward x20, locking off", 20.0, False, range(300, 310)),
]
MINUTES = 30


def summary(blocks):
    vals = [b for b in blocks if b is not None]
    return statistics.mean(vals[:8]), statistics.mean(vals[-8:])


def main():
    jobs = [(label, seed, scale, lock) for label, scale, lock, seeds in VARIANTS for seed in seeds]
    with ProcessPoolExecutor(max_workers=min(30, len(jobs))) as pool:
        futures = [(label, pool.submit(train.train_one, seed, MINUTES, True, False, scale, lock))
                   for label, seed, scale, lock in jobs]
        results = {}
        for label, f in futures:
            results.setdefault(label, []).append(summary(f.result()["blocks"]))
    print(f"correct head turns for the heard word (chance = 50%), {MINUTES} brain-minutes per brain:")
    for label, rows in results.items():
        first = statistics.mean(r[0] for r in rows)
        last = [r[1] for r in rows]
        print(f"  {label:26s} first 400 lessons {first:.1%} -> last 400 {statistics.mean(last):.1%}"
              f"   (best brain {max(last):.0%}, worst {min(last):.0%}, {len(rows)} brains)")


if __name__ == "__main__":
    main()
