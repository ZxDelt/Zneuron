"""Word lessons with the learning fixes: rewarded brains vs identical brains whose rewards are switched off.

    .venv/Scripts/python.exe -m experiments.word_fixes
"""

import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

import train

MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 30
GROUPS = [("rewarded", False, range(500, 515)), ("no reward (control)", True, range(500, 515))]
LFUEL = 0.0      # resting Lfuel: learning only sticks when a reward arrives (three-factor rule, #30)


def blocks(r):
    vals = [b for b in r["blocks"] if b is not None]
    return statistics.mean(vals[:8]), statistics.mean(vals[-8:])


def main():
    with ProcessPoolExecutor(max_workers=30) as pool:
        futures = [(label, pool.submit(train.train_one, seed, MINUTES, True, control, 1.0, True, LFUEL))
                   for label, control, seeds in GROUPS for seed in seeds]
        out = {}
        for label, f in futures:
            out.setdefault(label, []).append(f.result())
    print(f"correct head turns for the heard word (chance = 50%), {MINUTES} brain-minutes each, same seeds in both groups:")
    for label, rs in out.items():
        fl = [blocks(r) for r in rs]
        last = [b for _, b in fl]
        print(f"  {label:20s} first 400 lessons {statistics.mean(f for f, _ in fl):.1%} -> last 400 "
              f"{statistics.mean(last):.1%}  (best {max(last):.0%}, worst {min(last):.0%}) | "
              f"babbling at end {statistics.mean(r['babble'] for r in rs) * 1000:.1f}/s")
    a, b = ([blocks(r)[1] for r in out[label]] for label, _, _ in GROUPS)
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    z = (statistics.mean(a) - statistics.mean(b)) / se
    p = math.erfc(abs(z) / math.sqrt(2))
    print(f"  difference {statistics.mean(a) - statistics.mean(b):+.1%} (Welch z = {z:.2f}, two-sided p ~ {p:.3f})")


if __name__ == "__main__":
    main()
