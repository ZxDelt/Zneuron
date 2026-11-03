# Roadmap

1. Code one hub neuron in Python (~100 lines); plot charge and spikes.
2. Add inhibition, reset, weight math, time steps, leak.
3. Add STDP learning; test recall of stored patterns.
4. Scale to 1k–100k neurons (Brian2 / NumPy).
5. Load a FlyWire fruit-fly connectome subset; drive virtual behaviour.
6. Compare against a standard neural network on the same task.

## Fly-brain path (alternative entry)
1. Run Shiu et al.'s open whole-fly-brain model as-is; reproduce one result (sugar taste → feeding).
2. Swap in the Hub Neuron, keeping the same wiring and inputs.
3. Compare on the same tests: hits known behaviours? more right? speed?
4. Measurable win = real result; otherwise a fun build.

## Pattern finding
- Load FlyWire into Python; graph tools find hubs, motifs and circuits on a normal PC.
- Copy fly wiring motifs into an artificial network: research-grade; results usually trail standard AI.

## Assistant goal (realistic version)
- A connectome-inspired network steers a VR agent / creature in Unity; an LLM handles language.
- Embodied fly in a Unity 3D world = engineering, most achievable part.

## Recommended picks for open decisions
- Hub resets to 0 after a spike; fire on all outgoing lines.
- Charge delivered = spike × thickness, thickness −1..1 (minus = inhibitory).
- Rest 2 steps; line pool empties after passing charge to the hub.
- Prefer one-way synapses (matches FlyWire data).
