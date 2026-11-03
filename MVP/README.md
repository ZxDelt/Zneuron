# SuperNeuron MVP

Hub Neuron brain built from `../neuron_spec.json` (edit it with `../neuron_spec.html`), living in a ZX-style robot
in an endless 3D world. Work is steered through `../NEXT.md`.

## Run
```bash
.venv/Scripts/python.exe run_live.py                    # robot + world, 1000-spot cube brain (900 neurons at start)
.venv/Scripts/python.exe run_live.py --speed 10         # start at 10x real time
.venv/Scripts/python.exe run_live.py --no-body          # no robot: fixed sensor test pattern (older demo)
.venv/Scripts/python.exe train.py                       # train ~30 brains in parallel (word lessons), saves to brains/
.venv/Scripts/python.exe -m experiments.word_fixes 90   # rewarded vs no-reward brains, 90 brain-min each
.venv/Scripts/python.exe run_live.py --reward-only     # app where only rewards (👍/👎, flowers) teach
.venv/Scripts/python.exe -m pytest                      # tests
.venv/Scripts/python.exe -m experiments.tune_grid       # compare tuning options (no body)
```
Opens http://127.0.0.1:8765 in your browser. Ctrl+C in the terminal stops it. The first start after a code change
takes ~30–60 s while Numba compiles the engine (cached afterwards).

## Layout
- `superneuron/engine.py` — every neuron rule, compiled with Numba; all state in numpy arrays.
- `superneuron/network.py` — builds the brain (cube / grid / random), friendly views (`net.neurons[i].hub`, ...).
- `superneuron/params.py` — every number from the spec, in one place.
- `superneuron/world.py` — world, robot body, senses, the `Agent` that couples brain and body, `make_brain()`.
- `superneuron/teacher.py` — word lessons: says "left"/"right", rewards the matching head turn.
- `train.py` — trains many brains at once (one per CPU core), each in its own world.
- `superneuron/insight.py` — why a neuron fired, thinking groups, chains, learning, alarms, brain-state teller.
- `superneuron/server.py`, `saving.py` — live stream to the page, chat, rewards, save/load brains (`brains/`).
- `web/app.html`, `web/js/` — the app (Three.js 3D + dockview panels, kept offline in `web/vendor/`).
- `bees/` — bee lab: fly wiring, arena, LIF baseline, comparison runner, live server; `bee-lab/` — its Next.js app.
- `tests/` — tests (incl. `test_v2.py`: single-neuron XOR; `test_gpu.py`, skipped without CUDA): one per neuron rule, engine features, world/body, teacher, server.

## The app
Unity-style panels: drag tabs, resize, **F** or "⛶ Panel" = full-screen the active panel, "Reset layout".
- **Body & World** — the robot on an endless grass field with flowers and rocks; sun, day/night (2 min), stars.
  Drag to orbit; the camera follows the robot.
- **Brain 3D** — cube neurons (body, hub, one frustum pool per line), lines, arrows, zigzag spikes. Click to inspect.
- **Output** — brain-state teller: senses, motors, thinking groups, learning, drives, what the body is doing.
- **Chat** — your words are *heard*: each letter is 60 ms of sound in one pitch band on the ear sensors.
  👍/👎 or typing good/bad = reward.
- **Senses / Motors** — the two retinas, both ears, skin touch, recent rewards; each motor pair's push.
- **Alarms, Thinking, Neuron, Why it fired, Learned, Drives** — reading the brain.
- Top bar: brain time (days:hours:min:sec), real speed reached, speed (× real time), Save / Load brain, GPU in use.

## How it learns
- STDP on every line (#19), boosted by curiosity Cfuel (#29).
- **Reward (#30):** recent learning stays "eligible" for ~1 s. A reward repeats it × r: + strengthens what just
  happened, − undoes it. Rewards: picking a flower +0.6, getting closer to one +0.15, bumping a rock −0.5, your 👍/👎.
- **Motor babbling (#32):** motor neurons fire at random ~10 times/s, so the body moves and has something to learn from.
- **Reflex arcs (wiring choice):** each motor neuron gets 3 weak lines straight from sensor neurons (one from the ears),
  so senses can reach the motors without crossing the whole cube. With babbling off, 17 of 20 motors fire from senses.
- **Word lessons (`train.py`):** the teacher says "left" or "right" and rewards a head turn the matching way.
  What it took to make it learn (all measured with 15 rewarded vs 15 identical no-reward brains):
  1. Motor neurons are exempt from homeostasis and glia calming (like sensors) — otherwise babbling made them look
     "too busy", their thresholds hit the cap and their input lines were thinned until they went deaf.
  2. Babbling fades (10/s -> 2/s) as motor neurons start firing from their input lines.
  3. Targeted reward: only lines into neurons that fired in the last 300 ms are changed.
  4. Ears have 8 pitch bands; speech uses bands 1–6, apart from wind/footsteps (0) and birds (7).
  5. Orienting reflex wiring: the head-turn motors get 4 extra lines from the speech bands.
  6. **Resting Lfuel = 0** (`--lfuel 0`, default in `train.py`; `--reward-only` in the app): timing learning only sticks
     when a reward arrives. With resting Lfuel = 1 (v1 spec) nothing was learned. Proposed for #30 in the spec page.
  **Result:** rewarded brains 58% -> **64%** correct over 90 brain-minutes (best brain 100%); no-reward control flat at
  ~53%. Difference +11.5 points, p ~ 0.06 with 15+15 brains — promising, not yet proven.

## Bee lab: SuperNeuron v2 vs v1 vs LIF vs ALIF
Bees share one 2D arena and compete for the same cyan squares, carrying them to the magenta spot. All have the same
mini fly brain (84 senses -> 147 Kenyon cells -> 84 MBONs -> 16 motor neurons). Only the neuron differs:
**v2** = SuperNeuron v2 (`params.v2()`: slow leak, threshold 0.3, no line limit, mini-buckets with the choke rule,
gentle brakes, reward-guided growth, ...), **v1** = the old SuperNeuron, **LIF** = textbook, **ALIF** = adaptive LIF
(Bellec 2018/2020). Fair: same wiring and start weights (tested at x1, x2, x3), same senses, rewards and learning rule
(reward-gated STDP with a 250 ms + 2 s memory, driven by reward surprise), starts rotate, paired tests per match.
```bash
.venv/Scripts/python.exe run_bees.py                     # live sim server on :8766
cd bee-lab && npm run dev                                # the app: http://localhost:3000 (Next.js)
.venv/Scripts/python.exe -m bees.compare                 # full 4-bee fight on the CPU (30 cores)
.venv/Scripts/python.exe -m bees.compare --gpu           # the same fight, all matches at once on the GPU
.venv/Scripts/python.exe -m bees.research --over '{"reward_grow": 0}' --name test   # try a v2 variant vs LIF/ALIF
.venv/Scripts/python.exe -m bees.probe --scale 1         # why a bee wins: turning, speed, steering toward targets
```
Research 2026-11-02 (`bees/results/research.jsonl` keeps every variant tried):
- Before reward-guided growth no bee steered better than chance (~50%); v2 tied LIF overall.
- Reward-guided growth (when a reward comes, wire up the neurons that just caused it): v2 learns to steer
  (~56% -> 61%) and is **ahead of LIF and ALIF in all 18 tests** (6 conditions x 3 weight levels), 16 matches each:
  avg squares v2 8.7, LIF 5.0, ALIF 4.0; significant wins 11/18 vs LIF and 13/18 vs ALIF; **0 losses**.
- Control: ALIF given the same growth rule only goes 3.2 -> 3.8 and does not learn to steer; v2 still ahead 18/18.
  The win comes from the growth rule *combined with* the SuperNeuron (mini-buckets, branch credit, leak, fuels).
- Tried and dropped: stronger growth (8 lines/reward), protecting rewarded lines from thinning, "how close to firing"
  eligibility, no line thinning / glia only at 50 Hz (ran away to ~38 Hz everywhere).
App: the arena on the left half; on the right two panels, each with a bee picker and the tabs
Brain (live wiring and firing), Activity, Senses, Wiring, Score, Race, Verdict, Conditions.

## GPU
- `superneuron/gpu.py`: engine.py compiled for CUDA, one brain per GPU thread. Spike-exact with the CPU, but slow
  (a GPU thread is weak at branchy code).
- `superneuron/gpu_par.py`, `bees/gpu_bees.py`: one GPU thread per neuron, one block per brain, pull delivery, serial
  step for growth/pruning. Same averages as the CPU, not spike for spike. The CPU keeps the sleep method.
- Speed on the RTX 4060 laptop (bee brains, combined): v2 ~390x, LIF/ALIF ~4000x. One CPU core: v2 ~19x.
- `bees/gpu_world.py`, `bees/gpu_match.py`: arenas and bodies on the GPU too, so whole fights run there.

## Speed (measured, × real time)
| | engine only | with body | live app |
|---|---|---|---|
| 100 neurons | ~480x | – | ~170x (no body) |
| 1000-spot cube | ~70–90x | ~18x (one brain) | ~1x at the 1x setting; higher settings reach about the "with body" speed |
| 30 brains in parallel (`train.py`, 32-core CPU) | | **~315x combined** (each ~12x) | |

One brain with its body is limited by the brain itself (~0.7 ms of compute per 20 ms of brain time with all
86 senses streaming), so 1000x for a single brain is not reachable on this machine. Brain *time* scales with cores:
30 brains gave 15 hours of brain time in under 3 minutes. 1000x combined would need ~80 such cores.

## Dedicated GPU
The top bar shows which graphics card draws the 3D views. Windows picks it per app, so to use the dedicated one:
Windows Settings → System → Display → Graphics → Google Chrome → Options → **High performance** → Save,
then close every Chrome window and reopen. (NVIDIA laptops: NVIDIA Control Panel → Manage 3D settings →
Program settings → chrome.exe → High-performance NVIDIA processor.) Check at chrome://gpu → GL_RENDERER.

## Notes
- Saved brains are pickle files: only load brains you saved yourself. Load needs the same `run_live.py` settings.
- The robot is an original design in the Mega Man ZX style (blue armour, red accents, crested helmet), not Capcom's model.
- The older 2D view is at http://127.0.0.1:8765/classic (works with `--no-body`).
