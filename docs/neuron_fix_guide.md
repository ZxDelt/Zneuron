# SuperNeuron fix guide: the 10 problems, their root causes, where to change them

Written 2026-11-02 after the fair bee test (Hub SuperNeuron vs textbook LIF, 18 tests: LIF won 5, Hub won 2, 11 ties).
Code links point at `MVP/superneuron/engine.py` (the compiled neuron) and `MVP/superneuron/params.py` (its numbers).
Spec IDs (#20, F1, …) are the items in `neuron_spec.html`: change the spec there first, then the code.

---

## How the neuron works (the 30-second picture)

Think of each neuron as a **bucket (hub) fed by small cups (pools), one cup per line**.

1. A spike arrives down a line and pours charge into that line's cup.
2. A cup only tips into the bucket in fixed-size chunks (chunk = 1 / number of lines).
3. When the bucket reaches the threshold (0.5), the neuron fires and sends spikes down all its output lines.
4. If 20 ms pass without firing, **the bucket and all cups are emptied** (the "dump").
5. Around this sit the helpers: fatigue (a line gets weaker each time it fires), fuels (Lfuel = learn, Rfuel = calm,
   Cfuel = curiosity, Efuel = energy), homeostasis (aim for 2 spikes per second), glia (calm busy neurons), and
   growth and pruning of lines (max 8 lines per neuron).

The standard LIF is just **one leaky bucket**: charge goes straight in and slowly leaks out (half gone in about 14 ms),
and there are no helpers.

Most of the problems below come from one theme: **charge gets lost or weakened on its way to the bucket, and the
helpers keep shrinking lines faster than learning can grow them.**

---

## 1. Weak signal passing

**In one sentence:** a spike delivers less charge than its line strength promises, so signals fade after 1–2 layers.

**Picture it:** a 0.45-strong line should add 0.45 to the bucket. On a busy line, fatigue has already halved it to about
0.22, and with 2 lines the cup only tips in 0.5 chunks, so nothing reaches the bucket until a second spike comes.

**Root causes (three stacked losses):**
- **Fatigue (#20):** every spike multiplies the line's energy by 0.7, and it recovers with a 50 ms time constant.
  A sensor firing at 80 Hz settles at **about 49% energy**, so it delivers half its charge.
  Code: [engine.py:445](../MVP/superneuron/engine.py) (charge = thickness × energy),
  [engine.py:448](../MVP/superneuron/engine.py) (energy × 0.7), `fatigue_mult`/`fatigue_tau_ms` in [params.py:25-26](../MVP/superneuron/params.py).
- **Cup chunks (F1):** a cup tips only whole chunks of 1/lines ([engine.py:548-553](../MVP/superneuron/engine.py)).
  The leftover waits in the cup and dies at the 20 ms dump (problem 2). Fewer lines means bigger chunks and more charge stuck.
  The chunk size also counts *output* lines, so a neuron that sends a lot ends up listening with smaller chunks.
- **Homeostasis thinning (#31):** any neuron above 2 Hz has **all its input lines × 0.99 every 100 ms, about −10% per
  second** ([engine.py:647-652](../MVP/superneuron/engine.py), `homeo_thin` in [params.py:51](../MVP/superneuron/params.py)).
  Ten seconds of being useful shrinks its inputs to about 35%.

**My advice (in order):**
1. **Make fatigue tell "news" instead of eating signal:** keep it on sensor→hub lines only, or soften it to 0.9 with
   20 ms recovery. Fatigue is a good novelty detector but a bad default on every line.
2. **Cups pass charge continuously, not in chunks:** pour pool → hub at a rate (e.g. 50% of the cup per ms). The cups
   still give each input its own "voice" (your F1 idea) but nothing gets stuck.
3. **Size chunks by input lines only**, not total lines (if you keep chunks).
4. Fix homeostasis thinning as part of problem 5. It is the biggest silent killer.

**How to check it worked:** in the bee lab at weights ×1, the Kenyon and MBON layers fire above 0 Hz (Activity tab),
and Yellow's ×1 deliveries come up from 1.8 toward LIF's 14.2.

---

## 2. The 20 ms dump throws away information

**In one sentence:** charge that hasn't fired the neuron within 20 ms is wiped out completely, all at once.

**Picture it:** two weak inputs 25 ms apart. LIF still holds about 30% of the first one when the second arrives and
can fire. Yours has already emptied the bucket at 20 ms, so the first input never counted.

**Root cause:** a hard cliff instead of a slope. The window (#5) starts at the first input and erases the hub and every cup
at 20 ms ([engine.py:557-562](../MVP/superneuron/engine.py), `window_ms` in [params.py:18](../MVP/superneuron/params.py)).
The neuron can only add up inputs that land inside one 20 ms box, and the box starts at an arbitrary moment.

**My advice:**
1. **Replace the dump with a leak:** hub × e^(−1/τ) every tick with τ ≈ 20 ms. Same "forget old stuff" idea, no cliff.
   This one change likely closes most of the ×1 gap with LIF.
2. If you want to keep "no leak" as an identity of the neuron: use a **sliding** window (charge older than 20 ms drops
   out per input), not a box that wipes everything.
3. Optional and unique: make τ **per neuron and learnable** (your old "dynamic clock" idea). Sensory neurons get short τ,
   Kenyon/decision neurons long τ. That's a real advantage over LIF, which uses one fixed τ.

**How to check:** a unit test firing two 0.3 inputs 25 ms apart. Today: no spike; after the fix: a spike.

---

## 3. The 8-line limit squeezes the wiring

**In one sentence:** each neuron can have only 8 lines *in total* (inputs + outputs), so brains must be deep and narrow.

**Picture it:** a motor neuron in the bee can listen to only 6 MBONs + 2 senses. A real fly descending neuron listens to
hundreds. To reach many inputs you must stack layers, and every layer loses signal (problem 1).

**Root cause:** each line owns a cup and a slot, and the memory is sized by a fixed `MAXL = 8`
([engine.py:17](../MVP/superneuron/engine.py)), checked in `connect` ([engine.py:282](../MVP/superneuron/engine.py)),
`max_lines` in [params.py:24](../MVP/superneuron/params.py), spec R14 / #21.

**My advice:**
1. **Separate inputs from outputs:** limit inputs (cups) only. Outputs are just "send the spike there", so they don't need
   a cup or a slot, and a neuron can then send to 50+ targets.
2. **Raise the input limit to 32–64** and make it a param (`MAXL` from params, arrays sized from it). Memory is cheap:
   1000 neurons × 64 slots is tiny.
3. Keep a limit at all (unlike LIF) because it forces sparse, efficient wiring, but make it a soft cost (Efuel per line)
   instead of a hard wall.

**How to check:** rebuild the bee brain with 16–32 inputs per motor and fewer layers; middle-layer firing and
deliveries at ×1 should rise.

---

## 4. Pruning eats the brain

**In one sentence:** lines are removed for being quiet, and in a sparse brain most useful lines *are* quiet most of the time.

**Picture it:** the "square on the far left" line only fires when a square is on the far left, maybe once a minute. After
10 s of silence, once homeostasis has thinned it below 0.3, it is deleted. About half the bee's wiring was gone after an hour.

**Root causes:**
- Prune rule: remove if unused for 10 s **and** thinner than 0.3, or thinner than 0.1 at any time
  ([engine.py:701-702](../MVP/superneuron/engine.py), `unused_ms`/`unused_thin` in [params.py:39-40](../MVP/superneuron/params.py), spec #22).
  "Unused" measures **time**, not **usefulness**.
- Homeostasis thinning (problem 1) pushes lines below the 0.3 line, which then makes them prunable.
- Growth (#21) picks partners from the **nearest 50 neurons in space** ([engine.py:655-683](../MVP/superneuron/engine.py),
  `compute_nearest`). In the bee's grid layout, space has nothing to do with layers, so new lines go eye→eye or Kenyon→Kenyon:
  useless.

**My advice:**
1. **Prune by usefulness, not silence:** remove a line only if it has been *weakened by learning* (negative eligibility or
   reward) or is below 0.1. Never for being quiet alone.
2. **Protect rewarded lines:** a line that took part in a rewarded act gets a "keep" stamp. Your lock idea (#35) exists
   but needs 0.8 thickness for 60 s, which almost never happens. Lock on reward instead.
3. **Grow toward activity, not distance:** pick candidates from neurons that fired just before this one (the Hebbian
   co-fire list you already keep), from any layer, not only the 50 nearest in space.

**How to check:** "lines kept %" in the comparison stays above 90% over an hour, and Wiring-tab "grown" lines connect
the right layers.

---

## 5. Homeostasis and glia fight the body

**In one sentence:** every neuron is pushed toward 2 spikes per second, even neurons whose job is to fire fast or to stay silent.

**Picture it:** a motor neuron that should fire 30 Hz while walking gets its threshold raised and its inputs shrunk until it
goes deaf. We had to exempt motors and sensors by hand
([engine.py:633](../MVP/superneuron/engine.py), the `S.sensor[i] or S.motor[i]` skip).

**Root causes:**
- One fixed target for all: `target_hz = 2.0` ([params.py:48](../MVP/superneuron/params.py), #31).
- Two strong push-backs at once: threshold up (+0.001 per check, [engine.py:645-648](../MVP/superneuron/engine.py))
  **plus** input thinning ×0.99 per check (about −10%/s, [engine.py:649-652](../MVP/superneuron/engine.py)), plus the glia
  calming threshold ×Rfuel up to ×2 ([engine.py:636-639](../MVP/superneuron/engine.py), #34).
  Three brakes on the same pedal.
- It runs every 100 ms, far faster than learning (rewards add about 0.01 per event).

**My advice:**
1. **Per-neuron target rate**, learned or set by role (sensor/motor high, Kenyon low). Remove the hard-coded exemption.
2. **One brake, not three:** keep threshold adjustment only; remove input thinning (`homeo_thin = 1.0`) or make it 1000×
   slower (×0.9999). Let glia calm only real runaway (for example over 50 Hz), not anything above 2 Hz.
3. **Homeostasis slower than learning:** time window of minutes, not 5 s. The brain should stabilize over minutes and
   learn over seconds, not the reverse.

**How to check:** with the exemption code deleted, motors keep firing from their inputs (msyn/mall share in the robot)
and the bee lab still runs stable.

---

## 6. The learning rule doesn't work as specified

**In one sentence:** with Lfuel always on, timing learning rewires everything all the time, so a reward's small push
gets drowned out. With Lfuel off it's correct but too weak and too short-sighted.

**Picture it:** a delivery reward arrives 2–5 s after the turns that found the square. Eligibility fades with 250 ms, so by
then the memory of "which lines did it" is about 0.0003 of its size. The reward reaches almost nothing that mattered.

**Root causes:**
- **Lfuel = 1 at rest (#30 v1):** every STDP change is applied immediately ([engine.py:348](../MVP/superneuron/engine.py)),
  so random co-firing rewires the brain regardless of reward. We switched to Lfuel 0 (proposed, not confirmed).
- **Credit-assignment gap:** eligibility time constant 250 ms ([params.py:75](../MVP/superneuron/params.py)) versus
  seconds between action and reward.
- **Reward is too small and gets erased:** reward adds about eligibility × r ≈ 0.01 per line
  ([engine.py:363-391](../MVP/superneuron/engine.py)), while homeostasis removes about 10% per second (problem 5).
- Without targeting, every eligible line in the brain moves for every reward, a "smear".

**My advice (this is where you can beat every standard neuron):**
1. **Confirm resting Lfuel = 0** (three-factor learning) in the spec (#30).
2. **Two eligibility timescales:** fast (250 ms, for timing) plus slow (2–5 s, for "what I did lately"). Reward uses both.
   This closes the credit gap.
3. **Reward prediction error, not raw reward:** Lfuel = reward − expected reward (a running average). Standard
   R-STDP learns badly without this; it's the single biggest known fix (as dopamine does in real brains).
4. **Scale learning so it outweighs homeostasis:** learning rate around 0.05–0.1 per rewarded event, after problem 5's slower homeostasis.
5. Keep targeted reward as an *official* spec item (lines into neurons that just acted), now that you own it.

**How to check:** the "rewards off vs on" control in Verdict turns significant (green p) for Yellow, and the learning
curve (Conditions tab) rises over the hour.

---

## 7. Big luck swings

**In one sentence:** results vary as much between seeds as between neurons, so most tests end in "tie".

**Root causes:** a tiny brain (331 neurons), random wiring per seed, behavior mostly driven by random babbling, and
only 10 matches per test (`--seeds` in [compare.py](../MVP/bees/compare.py)).

**My advice:**
1. Fix problems 1–6 first: a brain that actually learns produces steadier results than one that wanders.
2. Fade babbling as control grows (the robot already does this: `Agent._fade_babbling` in `MVP/superneuron/world.py`).
3. Run 30 matches per test for final claims (the app's "matches" box). The paired test stays valid.

---

## 8. Its efficiency is partly just doing less

**In one sentence:** fewer spikes partly means silent middle layers, not smart coding.

**Root cause:** problems 1, 2 and 5 keep Kenyon and MBON layers near 0 Hz, so "energy saved" includes "thinking skipped".

**My advice:** judge efficiency only as **spikes per delivery** (already a metric in Conditions) and **delivered per
1000 spikes**. After the signal fixes, middle layers will fire more; the goal is to stay ahead on spikes per delivery
while they do. Your sparse ideas (Kenyon sparsity, Efuel cost) are the right tools for that.

---

## 9. Speed

**In one sentence:** your neuron does more work per tick, and some of it touches every line in the brain.

**Root causes:**
- `apply_reward` loops over **all lines** for every reward ([engine.py:363](../MVP/superneuron/engine.py)), and rewards
  arrive every 200 ms (shaping).
- `glia_check` loops over all lines every 100 ms for pruning and locking ([engine.py:687-711](../MVP/superneuron/engine.py)).
- Python runs each 20 ms body step (`bees/brains.py` `Match.step`), with a Numba call per bee per step.

**My advice:**
1. Keep a short list of **eligible lines** (touched in the last ~1 s); reward only loops those. Likely 10–50× faster rewards.
2. Prune and lock every 1–5 s on a rotating slice of lines, not all lines every 100 ms.
3. Later: move the whole body step into Numba, and run many brains on the GPU (the RTX 4060). That's where 1000× lives:
   thousands of brains in parallel, not one brain faster.

---

## 10. The test itself has limits

**In one sentence:** we beat or tied the *simplest* standard neuron; the strongest one wasn't in the ring yet.

**What's being done:** a separate session is adding **ALIF** (adaptive LIF, Bellec et al. 2018/2020), the strongest
widely used spiking neuron for learning tasks, as a third bee (`MVP/bees/alif.py`). The textbook LIF stays.
Once both are set up, all three get hooked into the bee lab.

**Your part:** confirm spec items #30 (resting Lfuel 0), #32 (motor babbling) and #37 (neurogenesis) in `neuron_spec.html`,
so "your neuron" is fixed before the next fight.

---

## About "10× better than the best neuron in the world"

Blunt truth: "10× better" at *everything* isn't a real target, because nothing is 10× better than ALIF at everything. Pick
the measures where 10× is possible and prove those:

| Measure | Where you stand now | 10× is… |
|---|---|---|
| Spikes per second (energy) | already about 4–14× fewer than LIF (spikes per delivery: won 8 of 18 tests) | **realistic**: your strongest card |
| Robustness to noisy senses | already 3–4× more deliveries | **realistic** after the signal fixes |
| Learning speed (reward events to learn a task) | no real learning yet | possible with fixes 6.2 + 6.3; that would be a genuine research result |
| Raw task score vs ALIF | unknown | unlikely as 10×; aim to *match or beat*, then win on the measures above |
| Simulation speed | 2× slower than LIF | not a neuron property; won with GPU batching |

## Fix order I recommend

1. **Problem 5 (homeostasis thinning off/slower)**: one number, biggest effect (`homeo_thin` in params.py).
2. **Problem 2 (leak instead of dump)**: small code change in `tick`, closes most of the weak-signal gap.
3. **Problem 1 (fatigue + chunks)**.
4. **Problem 6 (two eligibility timescales + reward prediction error)**: the part that can make it *better*, not just equal.
5. Problems 4 and 3 (prune by usefulness, inputs-only limit).
6. Problem 9 (speed), then rerun the 3-way fight with 30 matches per test.

Change one thing at a time and rerun `python -m bees.compare --minutes 20 --seeds 10 --scales 1 2` after each (about 6 min),
so you know which fix did what.
