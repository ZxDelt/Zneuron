# NEXT

How this works:
1. You write what you want in **Your requests**, in plain words. Paste images by file path if you like.
2. Tell Claude "check next doc". Claude copies your words, expands them into **Tasks**, makes a **Todo**, and asks anything unclear under **Questions**.
3. Claude does the todo one by one, ticking items and writing results under **Done this run**.
4. When everything is done, the run is archived to `docs/next_history.md` and this file goes back to this empty template.

---

## Your requests

- That means we need to make sure it's efficient as well as it can learn fast.
- I want to make it talk to me, but with 4 GB of my VRAM used.
- I want to make it learn letters, then a dictionary, and then see if it can talk to me.
- After the next doc is done, let me review it.

---

## Where we stand (from the bee lab, 2026-11-02)

- v2 beats LIF/ALIF ~2x on the bee task, but it learns slowly and only from one good/bad reward for the whole brain.
- Memory: 16 MB per 371-neuron brain today (~44 KB/neuron); 4 GB as-is = ~90k neurons.
- To learn language it needs: letters in and out, and a faster "you were right / wrong" teacher signal.

---

## Tasks (expanded by Claude) — 3 stages, each must pass before the next

### Stage 1. Letters
- **Ears:** one group of sensor neurons per letter (a–z, space, plus a few marks ~ 32 groups). It "hears" one letter
  every 20 ms.
- **Mouth:** one group of motor neurons per letter. The group that fires most in its 20 ms is the letter it "says".
- **Teacher:** right after it says a letter, the right letter's mouth neurons get "+", the wrong ones that fired get
  "−" (per-neuron teacher signal, a new spec item, Q3). Faster than one reward for the whole brain.
- Lessons, in order:
  1. **Echo:** hear "a" → say "a" (all 26 letters).
  2. **Alphabet:** hear "a" → say "b", hear "b" → say "c" … (next letter).
  3. **Recite:** given "a", say the whole alphabet by itself.
- Pass: ≥ 95% correct on each lesson. Report how many repeats it needed per letter (learning speed).

### Stage 2. Dictionary
- Words from a simple dictionary list (Q1): first **spelling**, then **word → meaning**.
  1. **Spell:** hear a word, say it back; then hear the first 1–2 letters, finish the word.
  2. **Meaning:** hear "cat", say its short meaning ("small animal"); and back: hear the meaning, say the word.
- Grow the list step by step: 10 → 50 → 200 → 1000 words.
- Pass per step: ≥ 90% correct words; also check it still remembers the old words (no forgetting).
- The brain grows neurons/connections as the list grows; stop growing at the 4 GB VRAM cap.

### Stage 3. Talk
- New **Talk** tab in the bee lab: you type, it answers letter by letter, and keeps learning from what you type
  (you can mark a reply right/wrong).
- First it answers with dictionary words and meanings ("what is cat" → "small animal"), then short simple sentences
  if it gets there.
- Honest: real conversation like a local LLM is very unlikely at this size. Words and meanings are the realistic goal;
  short phrases would be a big win.

### Support work (only as much as the stages need)
- **Efficiency:** slim the brain state (32-bit numbers, buffers sized to the brain) so the dictionary brain fits in
  4 GB; profile and speed up the GPU engine.
- **GPU limits:** hard cap **4 GB VRAM**, GPU duty ~50% by default (you work in Unity); progress + VRAM logged to a file.
- **Fair check:** LIF and ALIF take the same lessons with the same teacher signal, plus a simple letter-pair counter
  as the floor. Shown side by side: correct %, repeats needed, VRAM, spikes per letter.
- **Learning page:** a Letters / Dictionary progress view in the bee lab (correct % per lesson, per word, over time).

---

## Questions for you

- **Q1 Dictionary.** (a) A simple children's dictionary I write (~1000 common words, each with a 2–5 word meaning;
  no download). Recommended. (b) A big free public dictionary (Webster's 1913, ~100k words, old-fashioned meanings;
  ~30 MB download).
- **Q2 Letters only, or letters + digits** (0–9) in stage 1?
- **Q3 New spec item:** per-neuron teacher signal (right letter "+", wrong letter "−"). OK to add to
  `neuron_spec.html` as unconfirmed and use it?
- **Q4 Time budget.** How long can one training run use the GPU (e.g. ≤ 2 h, overnight)?
- **Q5 Rivals.** Keep LIF/ALIF in the same lessons (fair check), or v2 only to go faster?

---

## Todo (waiting for your review — nothing started)

- [ ] 1. Answer Q1–Q5
- [ ] 2. Letter ears + mouth groups, per-neuron teacher signal (+ tests), lesson runner with 4 GB cap and duty
- [ ] 3. Stage 1 lessons: echo → alphabet → recite (v2 + rivals), results
- [ ] 4. Slim brain state + GPU speed-up (as needed for stage 2)
- [ ] 5. Dictionary list (Q1) and stage 2: spell → finish word → meaning, 10 → 1000 words, forgetting check
- [ ] 6. Letters / Dictionary progress view in the bee lab
- [ ] 7. Stage 3: Talk tab, live learning from chat
- [ ] 8. Report, README, spec, archive this run

## Done this run
