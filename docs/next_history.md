# NEXT history

Finished NEXT runs, newest first.

---

# Run 3 — 2026-11-01 (learning fixes)

## Your requests
- Try the fixes for word learning.

## What changed
1. Motor neurons exempt from homeostasis + glia calming (babbling had made them deaf: threshold capped, inputs thinned).
2. Babbling fades from 10/s toward 2/s as motors fire from their input lines.
3. Targeted reward: only lines into neurons that fired in the last 300 ms.
4. Ears: 8 pitch bands; speech in bands 1–6, separate from wind/footsteps and birds (94 sensor neurons now).
5. Orienting reflex wiring: head-turn motors get 4 extra lines from the speech bands.
6. Resting Lfuel 0 (three-factor rule): timing learning only sticks when rewarded. `train.py` default; app `--reward-only`.

## Results (15 rewarded vs 15 identical no-reward brains)
| step | rewarded | control |
|---|---|---|
| fixes 2–5 only, 30 brain-min | 50.8% | 50.5% |
| + fix 1, 30 brain-min | 48.2% (brains respond to words, but reward doesn't steer) | 50.6% |
| + fix 6, 30 brain-min | 56.6% -> 60.8% | 53.3% |
| + fix 6, 90 brain-min | 58.1% -> **64.1%** (best brain 100%) | 52.6% |
Difference +11.5 points, p ~ 0.06: the first real sign of learning; more brains/time needed to be sure.

## Waiting for you (spec page)
- #30 proposed change (resting Lfuel 0), #32 (babbling for motors), #37 (neurogenesis) — check and Submit.
- Brains saved before this run (86 sensors) can't be loaded any more (94 sensors now).

---

# Run 2 — 2026-11-01 (the left-overs from run 1)

## Your requests
- Complete the 4 left-over items from run 1 (speed, neurogenesis rule in the spec, motors reached only by babbling,
  words not taught yet) and shout when done.

## Done
- **Speed.** World step compiled/cached (0.81 → 0.34 ms per 20 ms), one brain with body ~18.6x. New `train.py` trains
  one brain per CPU core: **30 brains x 30 brain-minutes = 15 h of brain time in 2 min 52 s = 315x combined**.
- **Spec page.** #37 Neurogenesis added to `neuron_spec.html` (Growth & structure) with the rule as built; #32 changed
  (babbling ON for motor neurons). Both are unconfirmed — waiting for your Submit.
- **Senses → motors.** Reflex arcs: each motor neuron gets 3 weak lines straight from sensor neurons (one from the ears).
  With babbling off, 17 of 20 motors now fire from the senses alone (was 0). Babbling stays on for exploration.
- **Teaching words.** `superneuron/teacher.py`: says "left"/"right", rewards a matching head turn immediately.
  Trained 30 brains (~1200 lessons each) + a no-reward control, then 30 more with rewards x5 / x20 and locking off.

## Result (honest)
- **No word learning yet**: every variant stays at 48–51% correct = chance, same as the control.
- Likely causes: random babbling drives the head far more than learned lines; reward is global (strengthens every
  recently active line, not just ear → head); "left"/"right" share pitch bands with each other and the wind.
- Ideas for a next run: babbling that fades as learned control grows; reward only lines that were active in the
  ~300 ms before the move; distinct sound patterns per word; dedicated ear → head-turn reflex wiring.

---

# Run 1 — 2026-11-01

## Your requests (run 1, 2026-11-01)

From the layout sketch:
- Mega Man 3D show window, below infinite walkable quad in 3D, with sun and night and stars, grass on quad, with flowers.
- 3D brain map, can move, zoom and rotate the camera.
- Output text window.
- Input text window, chat history and chat style text input box.
- Whatever you wanna add in white space, but make sure no scrolling the main but the sub panels can scroll, make it like unity style ui, resizeable, modular and full screen a particular window support.
- Make sure you are using the tech which utilizes the device well, with making sure the model works with performance and efficiency both.

From chat:
- Time to give 3D cube map. And also how can we train it.
- Make a body for this brain, Mega Man ZX body: hands, legs, torso, all controllable, and eyes, and sound only as senses, and skin senses. And a total brain state teller.
- 3D robot: like Mega Man ZX, please.
- Neurons in 3D: think of the neuron in 3D, square → cube, trapezium → some polygon.
- They can grow more new super neurons now as per the needs.
- I need 1000x simulation time and 10000x simulation time.
- Make this NEXT doc; refresh it after every run; reset it to the template when done.
- Q1 answer: Numba.
- Since it can learn while live, I'll try teaching it words and stuff through chat, so it might learn.

---

## Tasks (expanded by Claude)

### A. Architecture (needed for 1000x / 10000x time)
- The brain, the world and the body all run in **Python** (one clock). The browser only **draws** what is happening and sends your input.
  Reason: a browser can't render a 3D world 1000x faster than real time; a headless Python world can.
- Speed modes: 1x (real time), 10x, 100x, 1000x, 10000x, and "max". When faster than ~30x, the page shows
  snapshots (it can't draw every ms); the brain still runs every ms.
- Measured today: 1000 neurons run ~13x real time (mostly asleep). With a body sending senses every ms it will be slower.
  1000x on 1000 neurons needs roughly a 100x faster engine. See Questions Q1.

### B. Brain: 3D cube + new neurons
- Layout `cube`: neurons on a 3D grid (default 10x10x10 = 1000). "Nearest 50" uses 3D distance (F7 already allows any shape).
- Sensor neurons on one face of the cube, motor neurons on the opposite face, so signals flow through the cube.
- **Neurogenesis (new spec item #37, needs your confirm, Q2):** grow a new super neuron when a region needs it, for example
  when neurons nearby are full (8 lines) and still starving. Also a rule to remove dead neurons.
- Engine rewrite for speed (same rules, same tests): arrays instead of Python objects, plus a compiled core (Q1).

### C. 3D neuron look (your Design 3 in 3D)
- Body = cube. Hub = small cube in the centre.
- Pools = the space between hub and body split evenly per line; each pool is a 3D trapezium shape (a frustum, like a
  pyramid with its top cut off) facing its line. It lights up as it fills.
- Lines leave from the body surface; thickness = strength; travelling spike = zigzag; arrow at the receiving end.
- Drawn with GPU instancing so 1000 neurons stay smooth. Far away: simple coloured cubes; close up: full detail.

### D. World
- Infinite walkable ground (generated in chunks around the robot), grass, flowers, objects to bump into.
- Day/night cycle: moving sun, sky colour, stars at night. Sounds in the world: birds near flowers, wind, footsteps.

### E. Body: Mega Man ZX-style robot
- Blue armour with red accents, helmet with crest and gem, chest core, arm buster, boots. Built from 3D shapes.
  It will be **in the ZX style**, not a copy of Capcom's exact model (their character).
- Controllable by motor neurons: head (turn, nod), torso (twist, bend), each shoulder and elbow, walk speed and turn.
  Walking is steered (no falling over); arms, head and torso move freely.

### F. Senses (only these)
- **Eyes:** small retina (8x8 per eye) seen from the robot's head. Reports brightness and motion → eye sensor neurons.
- **Ears:** left and right loudness, a few pitch bands. Your chat text also arrives as sound.
- **Skin:** touch on hands, feet, torso, head (ground, flowers, objects).

### G. Training
- **Reward (Lfuel):** 👍/👎 buttons, or type "good"/"bad" in chat.
- **Automatic task rewards:** + for turning toward a sound or movement, + for reaching a flower, − for bumping into things.
- **Curriculum:** reflexes (look toward sound/motion) → reach → walk toward targets.
- Save / load brains to compare training runs.
- Honest: a new brain will twitch randomly at first. Learning to walk from reward alone is hard research.

### G2. Teaching words through chat
- Chat text is turned into "heard sound": a timed pattern on the ear sensor neurons (one channel per letter/sound,
  played in order, like hearing the word). Same word = same pattern every time.
- 👍/👎 right after its reaction teaches it (reward-timed learning, todo 15).
- Realistic goal: simple word → action links (hear "left" → move left arm). It will not understand or speak language;
  that needs an LLM, as the roadmap says.

### H. UI (your sketch)
- Unity-style docking: tabs, drag, resize, close/reopen, full-screen any panel, layout remembered. Page never scrolls;
  panels scroll inside.
- Panels: **Body/World 3D** (top-left), **Brain 3D** (top-centre), **Output text = brain-state teller** (bottom-left),
  **Chat** (bottom-middle). Right side: Senses monitor, Motor monitor, Alarms, Thinking window, Neuron inspector,
  Drives, Run controls (speed modes, pause, save/load).
- Brain-state teller: plain-language narration of senses, active groups, which limbs it drives, drives, alarms.
  It is a readout of the brain, not the brain talking. Real language would need an LLM later.

### I. Tech
- 3D: Three.js (WebGL on the GPU), instanced meshes.
- Docking: dockview (modern docking-panel library, no framework).
- Both downloaded once into `MVP/web/vendor/` so it works offline. Python side stays built-in + numpy (+ Q1).

---

## Questions for you
- **Q1 Speed engine.** For 1000x+ the brain core needs compiling. Options:
  (a) **Numba** — one `pip install numba` into the MVP venv; Python code compiled to machine code; ~50–200x faster. Recommended.
  (b) Rust/C++ core — fastest, but a second language to maintain.
  (c) No install — numpy only; maybe 3–10x faster; 1000x on 1000 neurons likely out of reach.
  Honest: **10000x on 1000 neurons is probably not reachable on one PC** even compiled (that is 10 million neuron-ms per
  second of heavy rules). 10000x should work for small brains (~50–100 neurons). The UI will show the real speed reached.
- **Q2 Neurogenesis rule (#37).** Proposed: a glia region grows 1 new neuron when, for 1 s, at least 3 of its neurons are
  starving (below target rate) and already at 8 lines, and there is an empty spot in the cube. It starts with 2 input lines
  from the busiest neighbours (80% excitatory / 20% inhibitory). A neuron with no lines left and no spike for 60 s dies.
  Max brain size is a setting. OK?

---

## Todo (final)
- [x] 1. NEXT doc and workflow
- [x] 2. Q1 answered (Numba). Q2 not answered — the proposed neurogenesis rule was used (see Left over).
- [x] 3. Three.js 0.186.1 + dockview-core 8.5.1 in `MVP/web/vendor/`
- [x] 4. Docking UI: tabs, drag, resize, full-screen panel (F), reset layout, saved layout, page never scrolls
- [x] 5. 3D cube brain: 1000 spots, 900 neurons at start, 86 sensors on one face, 20 motors on the opposite face
- [x] 6. Brain 3D: Design 3 cubes, hub, frustum pools, lines, arrows, zigzag spikes, click to inspect
- [x] 7. World: endless 16 m chunks, grass, flowers, rocks, day/night (2 min), sun, stars, birds/wind/footsteps
- [x] 8. Body: ZX-style robot (original design), head/torso/shoulders/elbows/walk/turn driven by 10 motor pairs
- [x] 9. Senses: eyes 6x6 per eye (brightness + motion), ears L/R x 4 bands, skin x 6 → 86 sensor neurons
- [x] 10. Body & World 3D panel
- [x] 11. Chat (heard as sound), brain-state teller (incl. body), 👍/👎 reward
- [x] 12. Senses and Motors panels (+ Alarms, Thinking, Neuron, Why, Learned, Drives)
- [x] 13. Speed: compiled Numba engine, speed menu 0.03x … 10000x, real speed shown (targets not reached, see below)
- [x] 13b. Chat words → timed sound patterns on the ear sensors
- [x] 14. Neurogenesis + neuron death (#37, proposed rule)
- [x] 15. Reward timing (eligibility ~1 s, reward repeats/undoes it), task rewards, save/load brains
- [x] 16. Tests (34 pass), performance check, README
- [x] 17. Archived here; NEXT reset

## Done (summary)
- Brain speed: 100 neurons 76x → ~480x; 1000-spot cube 13x → ~70–90x (engine), ~18x with the body.
- Live app holds ~1x real time at 30 frames/s with the full robot + 1000-spot brain on the RTX 4060.
- Fixed on the way: glia checks were 95% of the cost; the page frame took 136 ms (now 18 ms); motor babbling was
  clock-like so opposing motors cancelled (now random); walking had no momentum (now builds up over ~0.6 s).
- 3-minute check: ~7 neurons born per minute, none died, mean rate ~1.4 Hz, no alarms; the robot wanders a few metres
  per minute and earns "getting closer to a flower" rewards.

## Left over (not done / not reached)
- **1000x and 10000x are not reached** for the 1000-spot brain: ~70–90x engine only, ~18x with the body. The body/world
  step (Python, every 20 ms) is now the main limit. Options: move the world step into Numba, run many brains in parallel.
- **Q2 / neurogenesis rule (#37)** was never confirmed by you; the proposed rule is in use. It is not in
  `neuron_spec.html` yet.
- **Motor babbling (#32 spontaneous firing ON for motors)** was switched on because signals still don't reach the motor
  face from the senses. Learned control will need many hours of training (or more direct sensor→motor wiring).
- Words through chat are heard, but nothing has been taught yet; expect only simple word → action links, not language.
