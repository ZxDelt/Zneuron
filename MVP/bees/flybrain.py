"""Mini fly brain wiring, shared by both bees so only the neuron model differs.

Fly-like layers (sizes kept so every neuron has at most 8 lines, the Hub neuron limit #21):
  sensors (compound-eye sectors + body sense)  ->  Kenyon cells (mushroom body, sparse mixing of sight + context)
  ->  MBONs (mushroom body output)  ->  descending motor neurons.
  Plus an innate path (lateral horn): a few sensors wired straight to motors.

Each Kenyon cell gets 1 context input (carrying / empty hand / touching / on spot / bump) and 3 eye inputs, so a
KC can mean "square on the left while my hands are empty" — the combination the task needs.
"""

import random
from dataclasses import dataclass, field

SECTORS = 7                                 # eye sectors over 210 degrees, 0 = far right .. 6 = far left
KINDS = ("square", "spot", "red")           # what the eyes tell apart: cyan squares, the magenta drop spot, red decoys
EYE_COPIES = 3                              # ommatidia per sector and kind
CONTEXT = (("carrying", 6), ("empty", 6), ("touch", 3), ("on_spot", 3), ("bump", 3))
ACTIONS = ("left", "right", "forward", "grab")
MOTOR_COPIES = 4

N_EYE = SECTORS * len(KINDS) * EYE_COPIES   # 63
N_CTX = sum(k for _, k in CONTEXT)          # 21
N_SENSORS = N_EYE + N_CTX                   # 84
N_MOTORS = len(ACTIONS) * MOTOR_COPIES      # 16
N_KC = N_SENSORS * 7 // 4                   # 147: each sensor feeds 7 KCs, each KC hears 4 sensors
N_MBON = N_KC * 4 // 7                      # 84: each KC feeds 4 MBONs, each MBON hears 7 KCs
N = N_SENSORS + N_MOTORS + N_KC + N_MBON    # 331
INNATE_PER_MOTOR = 2
MBON_INHIBITORY = 0.3


def eye_id(kind, sector, copy):
    return (KINDS.index(kind) * SECTORS + sector) * EYE_COPIES + copy


def context_ids(name):
    start = N_EYE
    for nm, k in CONTEXT:
        if nm == name:
            return list(range(start, start + k))
        start += k
    raise KeyError(name)


def motor_ids(action):
    a = ACTIONS.index(action)
    return list(range(N_SENSORS + a * MOTOR_COPIES, N_SENSORS + (a + 1) * MOTOR_COPIES))


KC0 = N_SENSORS + N_MOTORS
MBON0 = KC0 + N_KC
GROUPS = {"eyes": (0, N_EYE), "body": (N_EYE, N_SENSORS), "motors": (N_SENSORS, KC0),
          "kenyon": (KC0, MBON0), "mbon": (MBON0, N)}


@dataclass
class FlyBrain:
    n: int
    sensors: int
    motors: int
    sign: list                               # +1 / -1 per neuron (Dale's law, #15)
    edges: list = field(default_factory=list)   # (src, dst, start weight 0..1)


def build(seed=1):
    rng = random.Random(seed)
    sign = [1] * N
    edges = []
    w = lambda: rng.uniform(0.2, 0.45)
    # sensors -> KCs: each KC gets 1 context + 3 eye inputs; every sensor ends up feeding exactly 7 KCs
    ctx = [i for i in range(N_EYE, N_SENSORS) for _ in range(7)]
    eye = [i for i in range(N_EYE) for _ in range(7)]
    rng.shuffle(ctx)
    rng.shuffle(eye)
    for k in range(N_KC):
        kc = KC0 + k
        srcs = {ctx[k]}
        for s in eye[3 * k:3 * k + 3]:
            srcs.add(s)                       # a repeat just means one line
        for s in srcs:
            edges.append((s, kc, w()))
    # KCs -> MBONs: each KC feeds 4 MBONs, each MBON hears 7 KCs
    slots = [m for m in range(MBON0, N) for _ in range(7)]
    rng.shuffle(slots)
    for k in range(N_KC):
        for m in set(slots[4 * k:4 * k + 4]):
            edges.append((KC0 + k, m, w()))
    for m in rng.sample(range(MBON0, N), round((N - MBON0) * MBON_INHIBITORY)):
        sign[m] = -1
    # MBONs -> motors: each MBON drives one motor neuron (6 per motor at most)
    motors = list(range(N_SENSORS, KC0))
    targets = [motors[i % N_MOTORS] for i in range(N - MBON0)]
    rng.shuffle(targets)
    for m, tgt in zip(range(MBON0, N), targets):
        edges.append((m, tgt, w()))
    # innate path (lateral horn): random sensors straight to each motor, weak
    edges = list({(a, b): (a, b, x) for a, b, x in edges}.values())
    deg = [0] * N
    for a, b, _ in edges:
        deg[a] += 1
        deg[b] += 1
    for tgt in motors:
        free = [s for s in range(N_SENSORS) if deg[s] < 8]
        for s in rng.sample(free, min(INNATE_PER_MOTOR, len(free), 8 - deg[tgt])):
            edges.append((s, tgt, 0.25))
            deg[s] += 1
            deg[tgt] += 1
    return FlyBrain(N, N_SENSORS, N_MOTORS, sign, edges)
