# SuperNeuron

An experimental spiking neuron model — the **Hub neuron** — that learns while it runs, grows and prunes its own
connections, and splits its input into mini-buckets (dendrite-like branches). Includes a fair test bench against the
two standard spiking neurons, **LIF** and **ALIF**, and a live 2D "bee lab" to watch them compete.

> Research prototype. Results below come from one small task; treat them as a starting point, not a proof.

## Results so far (bee lab, 168 GPU matches)

Four bees share one arena and must carry squares to a spot. Same fly-like wiring (84 sensors → 147 → 84 → 16 motors),
same start weights, same reward-gated learning rule; **only the neuron model differs**.

| | SuperNeuron v2 | LIF | ALIF | SuperNeuron v1 |
|---|---|---|---|---|
| Squares delivered (20 min, avg) | **9.05** | 4.81 | 3.75 | 1.34 |
| Spikes per square delivered | **379k** | 879k | 771k | 394k |
| Head-to-head over 18 conditions (sig. wins / ties / losses) | — | 9 / 9 / 0 | 11 / 7 / 0 | 18 / 0 / 0 |

Honest limits:
- v2 is ~10x slower than LIF/ALIF in software (more work per neuron), and ~44 KB of memory per neuron (unoptimised).
- Learning is slow (first delivery ~2 min) and steering accuracy tops out around 60%.
- LIF/ALIF are textbook versions with the same learning rule, not tuned library versions.

## What's in here

| Path | What |
|---|---|
| `neuron_spec.html` / `neuron_spec.json` | The neuron's rules, item by item (open the HTML in a browser to read/edit). |
| `01_model_rules.md` … `05_roadmap.md` | Design notes: rules, function map, learning and growth, biology, roadmap. |
| `docs/` | Fix guide and run history. |
| `MVP/superneuron/` | The engine (Numba, CPU) plus two CUDA engines (`gpu.py`, `gpu_par.py`). |
| `MVP/bees/` | Bee lab: arena, fly wiring, LIF/ALIF, fair comparison runner, live server. |
| `MVP/bee-lab/` | Next.js app to watch the bees and see the verdict. |
| `MVP/web/`, `run_live.py` | 3D robot-in-a-world demo with a 1000-spot cube brain. |
| `MVP/tests/` | pytest suite (GPU tests skip without CUDA). |

## Quick start

Needs Python 3.12 and Node 20+. An NVIDIA GPU is optional (for `--gpu` runs).

```bash
cd MVP
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest                              # tests
```

Bee lab (two terminals):

```bash
cd MVP && .venv/Scripts/python run_bees.py      # simulation server on :8766
cd MVP/bee-lab && npm install && npm run dev    # app on http://localhost:3000
```

Fair comparison (writes `MVP/bees/results/compare.json`, shown in the app's Verdict tab):

```bash
cd MVP
.venv/Scripts/python -m bees.compare --minutes 20 --seeds 8                     # CPU
.venv/Scripts/python -m bees.compare --gpu --gpu-duty 0.5 --minutes 20 --seeds 8  # GPU, half duty
```

More commands and the code layout are in [MVP/README.md](MVP/README.md).

## Ideas welcome

Open an issue or PR, for example:
- a stronger or tuned rival neuron (fairness matters more than winning),
- new arena conditions or tasks,
- memory and speed improvements for the GPU engines,
- language tasks (letters → words), which is the next planned step.

## License

MIT — see [LICENSE](LICENSE).
