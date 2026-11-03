# Function Map — 36 real-neuron functions vs. the Hub Neuron Model

Current count after rules R14–R16: **15 done · 5 half done · 16 missing** (was 13 · 6 · 17 in design pack v1).

## Done (15)
1. Receive signals (lines)
2. Sum inputs (charge hub)
3. Spatial summation (many pools → hub)
4. Temporal summation (no leak keeps charge)
6. Threshold firing (0.5)
7. All-or-nothing spike
8. Reset after spike — via R16 dump (moved from half)
9. Refractory rest (receive, no send)
10. Output (any free line acts as axon)
13. Pass signal on (line → line)
14. Excitation (+ charge)
16. Synapse strength (thickness)
16b. Weight math: delivered = spike × thickness (R15) (moved from half)
23. Dendritic computation (line pools)
25. Rate coding (emerges when running)

## Half done (5)
12. Fan-out: all lines or one, undecided
21. Grow connections: R14 gives the mechanism, but no trigger rule yet
26. Timing coding: needs a clock / time steps
31. Homeostasis: even space split normalises inputs (moved from missing)
33. Gap junctions: two-way lines similar, but real ones are both ways at once

## Missing (16)
5. Leak (rejected → need inhibition)
11. Conduction delay on lines
15. Inhibition (minus lines)
17. LTP (strengthen useful lines)
18. LTD (weaken useless lines)
19. STDP (spike-order learning)
20. Short-term fatigue / boost
22. Pruning unused lines
24. Back-propagating spike (learning signal)
27. Bursts
28. Oscillations / rhythms (network level)
29. Neuromodulation (dopamine etc.)
30. Reward learning
32. Spontaneous firing
34. Glia support cells
35. Gene-based permanent memory
36. Energy limits

## Priority
- Most important missing: leak, inhibition (15), STDP/LTP/LTD (17–19). Without them the network can't stay stable or learn.
- Fine to skip: glia (34), gene memory (35), energy (36), neuromodulation (29). Even good brain simulations leave most out.
