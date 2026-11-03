# Learning, Memory and Connection Growth

## Memory = synapses, not charge
- Charge is wiped by every spike (many per second). Charge = RAM (now); thickness = storage (learned).
- One neuron holds one charge value but up to ~10,000 synapses.
- A memory is a path: strong lines make the same group of neurons fire together again.
- Evidence: blocking LTP blocks new memories; brain activity can stop (deep cold) and memories survive.

## Recall = rebuilding the pattern
- No file is read: a cue fires a few neurons, strong lines complete the rest (pattern completion, Hopfield network).
- Each recall can alter the memory (reconsolidation).

## What happens when we learn (biology)
- A neuron does not detect that something is "missing". It only reacts to local signals: what arrived, when, and whether it fired.
- **Coincidence detector:** NMDA receptors open only when input arrives while the neuron is already excited. Calcium floods in; calcium is the "learn now" signal.
- **Trial and selection:** neurons constantly push out tiny exploratory branches (filopodia) at random. Branches that catch coincident activity stabilise; the rest retract within hours.
- **Third factor:** dopamine / acetylcholine broadcast "this mattered" (reward, surprise). With it, changes stick; without it, most fade.

## When a neuron makes a new connection
- **Development:** young neurons grow branches guided by chemical signals and connect almost blindly; many are cut later.
- **Hebbian:** two unconnected neurons often active together → new contacts form ("fire together, wire together").
- **Homeostatic:** too little input / rarely fires → grows more input contacts; fires too much → removes some.

## Proposed rules for the model
- **STDP:** sender fired just before receiver → thicken line; sender fired just after → thin line. Optional reward signal scales the change.
- **Growth:** each neuron grows random thin (0.1) test lines. A line becomes permanent if sender and receiver fire within a few steps of each other, × a global reward value.
- **Starvation growth:** add a line if a hub has not fired for N steps.
- **Pruning:** remove lines that stay unused or at 0.1 for a long time; otherwise growth makes a jungle and pool space (R14) keeps shrinking.
