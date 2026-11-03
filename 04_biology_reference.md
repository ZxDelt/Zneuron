# Biology Reference Facts

## Charge in a real neuron
- Limited by voltage, not storage. Rest ≈ −70 mV, threshold ≈ −55 mV, spike peak ≈ +40 mV (ceiling set by sodium reversal potential). Spike ≈ 1–2 ms.
- Typical neuron capacitance ≈ 100 pF. Rest → threshold ≈ 1.5 pC ≈ 9 million ions. Full swing to +40 mV ≈ 11 pC ≈ 70 million ions.
- Fly neurons: a few pF, roughly 10–50× less charge.
- Charge cannot be held: it leaks back to rest in ~10–20 ms (membrane time constant).
- Model mapping: hub 0 = −70 mV, 0.5 = −55 mV (threshold), 1 = +40 mV peak. Real neurons never sit above 0.5 without firing.

## Scale
- Fruit fly (FlyWire): ~140,000 neurons (central brain), ~50 million synapses.
- Full male fly connectome (Google Research + HHMI Janelia, Cell, Sept 2026): brain + ventral nerve cord, 166,000+ neurons.
- Human: ~86 billion neurons, ~100 trillion synapses.

## Standard neuron models
- Leaky integrate-and-fire (LIF): simplest; the Hub Neuron Model is close to a two-compartment LIF.
- Hodgkin–Huxley: detailed ion channels, far heavier to run.
- Multi-compartment models: dendrites as separate compartments (like line pools).

## Prior work to build on
- Shiu et al. 2024 (Nature): whole fly brain on LIF neurons + FlyWire connectome predicted feeding and grooming circuits, confirmed in the lab. Open code (Brian2), runs on a normal PC.
- Lappalainen et al. 2024 (Nature): fly visual system, connectome wiring + deep-learned neuron parameters predicted real responses.
- 2026 viral demos: simulated fly brain driving characters in Minecraft, Doom, Super Mario 64, Beat Saber. Fun, not evidence of a "playing" fly.

## Limits of the connectome
- Gives wiring and synapse counts, not exact strengths, neuromodulators (dopamine), electrical synapses or timing.
- A fancier neuron on incomplete wiring can perform worse.
- A fly brain does navigation, smell and reflexes, not language: an assistant still needs an LLM.
