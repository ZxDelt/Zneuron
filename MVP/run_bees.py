"""Bee lab: yellow bee (Hub Zneuron) vs pink bee (textbook LIF) in one shared arena, live.

    .venv/Scripts/python.exe run_bees.py                        (simulation server on :8766)
    cd bee-lab && npm run dev                                   (the app on http://localhost:3000)
    .venv/Scripts/python.exe -m bees.compare                    (full comparison, also from the app's button)
"""

import argparse

from bees.arena import CONDITIONS
from bees.server import Lab, serve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="basic", choices=list(CONDITIONS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--speed", type=float, default=5)
    ap.add_argument("--scale", type=float, default=2.0, help="shared start-weight level (both bees)")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    serve(Lab(a.condition, a.seed, a.speed, a.scale), a.port)
