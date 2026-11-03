# Hub Neuron Model — Rules (current version)

Goal: build AI from realistic neuron patterns. Start with a fruit-fly-scale spiking network, then scale up.
Images: `images/design_pack_v1.jpg` (full pack), `images/neuron_and_network.png`, `images/design_evolution_sketch.jpg`.

## Anatomy
- **Hub**: central charge store, range 0–1, fires at ≥ 0.5. Biology match: soma / axon hillock.
- **Lines (L1..Ln)**: connector lines radiating from the body. Biology match: dendrites + axon.
- **Line pools (P1..Pn)**: each line has its own charge pool inside the neuron so inputs don't mix before the hub. Biology match: dendritic compartments.

## Rules
| ID | Rule |
|----|------|
| R1 Body | Neuron = central charge hub + N radiating connector lines. |
| R2 Line pools | Every line has its own charge pool (0–1) inside the neuron. |
| R3 Synapse | A sends to B → charge stored in B's receiving line pool. |
| R4 Pool → hub | Line pool passes its charge to the hub when it reaches 0.5. **Conflict with R14, see Flaws.** |
| R5 Fire | Hub fires an all-or-nothing spike when hub charge ≥ 0.5. |
| R6 Output | Spike goes out through line(s) into the next neuron's line pool. |
| R7 Two-way | Each line can send or receive, but only one at a time. Lines of one neuron work in parallel. |
| R8 Conflict | If both ends want to send, the first to send wins; the other receives. |
| R9 Rest | After firing, the neuron can receive (charge builds) but cannot send. |
| R10 No leak | Charge stays indefinitely while < 0.5 (user choice; see Flaws). |
| R11 Strength | Line thickness = synapse strength (weight) = long-term memory. |
| R12 Signals | Spike = 1, silent = 0. Sign (+/−) belongs to the line (excitatory/inhibitory), not the signal. |
| R13 Visual | Zigzag = active spike, plain line = weak synapse, thick line = strong synapse, ○ = unconnected connector. |
| R14 Growth (new) | A basic neuron starts with 2 lines; each pool takes half the internal space, hub fixed in the centre. A new line is added when a connection is needed and the space re-divides evenly. Hub size never changes. |
| R15 Thickness (new) | 0 = no connection. 0.1–1 = connected, thin to thick (strongest). Thickness sets how much charge passes: delivered = spike × thickness. |
| R16 Dump (new, temporary) | Once charge is used, leftover charge is dumped. User will revise later. |

## Design evolution (sketch)
1. Round body with 8 radiating lines, internal pool segments, central hub.
2. Square body, one line in and one out, three internal compartments.
3. (Current) Square body, 2 lines, space split in half, hub in the centre. Lines added on demand.

## Open decisions
- Fan-out: hub sends to all free lines (neuron-like) or one (router-like)? Recommended: all.
- Tie-break when two lines send in the same step: lower ID or random.
- Rest length in time steps. Recommended: 2.
- Inhibition: minus lines (needed because there is no leak).
- When does a neuron need a new connection? See `03_learning_and_growth.md`.
- Learning rule for thickness: STDP (see `03_learning_and_growth.md`).

## Flaws (blunt)
- **R4 vs R14:** with 3+ lines each pool holds ≤ 1/3, so it can never reach 0.5 to pass to the hub. Fix: pool passes when full, or the threshold scales with line count.
- **No leak + no inhibition** → charge creeps to 0.5 everywhere → runaway firing. Recommended: small leak per step.
- **Two-way lines** add complexity with no proven benefit; real chemical synapses and the FlyWire data are one-way.
- Hub routing to a single line = router, not neuron.
- Neuron-like ≠ smarter: spiking nets mainly win on power, not accuracy.
