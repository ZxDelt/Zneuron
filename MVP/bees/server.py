"""Live bee lab: both bees in one shared arena, streamed to the Next.js app (bee-lab/) over Server-Sent Events.

GET  /stream   frames (~15 per second)
GET  /results  latest comparison (bees/results/compare.json)
GET  /meta     conditions, speeds, brain layout
POST /control  {"cmd": "pause" | "speed" (value) | "reset" (condition, seed, scale) | "reward" (value, bee)
                | "compare" (minutes, seeds)}
"""

import json
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import arena as A
from . import flybrain as F
from .brains import DEFAULT_SCALE, FREE_SPOTS, MODELS, SCALES, Match
from .compare import OUT

SPEEDS = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 1000]
GROUPS = dict(F.GROUPS, new=(F.N, F.N + FREE_SPOTS))        # "new": spots where SuperNeuron v2 grows neurons


def group_of(i):
    for g, (a, b) in GROUPS.items():
        if a <= i < b:
            return g


class Lab:
    def __init__(self, condition="basic", seed=1, speed=1, scale=DEFAULT_SCALE):
        self.lock = threading.Lock()
        self.speed = speed
        self.paused = False
        self.compare = dict(running=False, progress=0, total=0, last="", error=None)
        self.frame_n = 0
        self.reset(condition, seed, scale)

    def reset(self, condition, seed, scale=None):
        with self.lock:
            self.condition, self.seed = condition, int(seed)
            self.scale = float(scale if scale is not None else getattr(self, "scale", DEFAULT_SCALE))
            self.match = Match(condition, self.seed, minutes=20, scale=self.scale)
            self.bees = self.match.bees
            self.history = {m: deque(maxlen=600) for m in MODELS}      # (t s, deliveries, reward) every 2 s
            self.rewards = {m: deque(maxlen=40) for m in MODELS}
            self.log = deque(maxlen=80)
            self.prev_fires = {m: self.bees[m].brain.fires_n().copy() for m in MODELS}
            self.prev_t = 0
            self.rates = {m: {g: 0.0 for g in GROUPS} for m in MODELS}
            self.rate_hist = {m: deque(maxlen=120) for m in MODELS}     # per-layer rates over time
            self.start_paths = {m: self._paths(m) for m in MODELS}
            self.version = time.time()                                  # tells the page the brains were rebuilt
            self.real_x = 0.0

    def _paths(self, m):
        src, dst, w = self.bees[m].brain.lines()
        ends = [b for _, (_, b) in GROUPS.items()]
        g = np.searchsorted(ends, src, side="right")
        names = list(GROUPS)
        out = {}
        gd = np.searchsorted(ends, dst, side="right")
        for a, b, x in zip(g.tolist(), gd.tolist(), w.tolist()):
            k = f"{names[a]}>{names[b]}"
            s = out.setdefault(k, [0.0, 0])
            s[0] += x
            s[1] += 1
        return {k: [round(v[0] / v[1], 4), v[1]] for k, v in out.items()}

    def step(self):
        with self.lock:
            before = {m: bee.body.stats["reward"] for m, bee in self.bees.items()}
            self.match.step()
            t = self.match.t
            for te, m, e in self.match.events:
                if e != "bump":
                    self.log.append(dict(t=round(te / 1000, 1), bee=m, event=e))
            for m, bee in self.bees.items():
                r = bee.body.stats["reward"] - before[m]
                if abs(r) > 1e-9:
                    self.rewards[m].append((round(t / 1000, 2), round(r, 2)))
            if t % 2000 == 0:
                for m, bee in self.bees.items():
                    s = bee.body.stats
                    self.history[m].append((t // 1000, s["deliveries"], round(s["reward"], 2)))

    def frame(self):
        with self.lock:
            t = self.match.t
            dt = max(1, t - self.prev_t)
            self.frame_n += 1
            send_lines = self.frame_n % 4 == 1
            out = dict(t=t, speed=self.speed, paused=self.paused, condition=self.condition, seed=self.seed,
                       scale=self.scale, version=self.version, real_x=round(self.real_x, 1),
                       world=self.match.world.snapshot(), log=list(self.log)[-30:], compare=self.compare, bees={})
            for m, bee in self.bees.items():
                f = bee.brain.fires_n()
                d = f - self.prev_fires[m]
                if t > self.prev_t:
                    self.rates[m] = {g: round(float(d[a:b].sum()) / max(1, min(b, len(d)) - a) / (dt / 1000), 2)
                                     for g, (a, b) in GROUPS.items()}
                    self.rate_hist[m].append([round(t / 1000, 1)] + [self.rates[m][g] for g in GROUPS])
                self.prev_fires[m] = f.copy()
                ts, ids = bee.brain.recent_fires(t - 1000)
                hot = sorted(set(ids[ts >= t - 60].tolist()))           # fired in the last 60 ms (brain view)
                if len(ts) > 900:
                    keep = np.linspace(0, len(ts) - 1, 900).astype(int)
                    ts, ids = ts[keep], ids[keep]
                s = bee.body.stats
                sec = max(1e-9, t / 1000)
                b = dict(
                    label=MODELS[m]["label"],
                    stats=dict(s, reward=round(s["reward"], 2), distance=round(s["distance"], 1),
                               per_min=round(s["deliveries"] / (sec / 60), 3),
                               carry_success=round(100 * s["deliveries"] / s["pickups"], 1) if s["pickups"] else 0),
                    spikes=bee.spikes(), spikes_per_s=round(bee.spikes() / sec, 1),
                    compute_ms_per_s=round(1000 * bee.compute_s / sec, 2), line_count=bee.brain.line_count(),
                    neurons=bee.neurons(),
                    start_lines=bee.start_lines, rates=self.rates[m], rate_hist=list(self.rate_hist[m]),
                    history=list(self.history[m]), rewards=list(self.rewards[m]), hot=hot,
                    senses=[round(float(v), 4) for v in bee.body.senses],
                    raster=[(ts - t).tolist(), ids.tolist()], damaged=bee.damaged)
                if send_lines:
                    src, dst, w = bee.brain.lines()
                    b["lines"] = [src.tolist(), dst.tolist(), np.round(w, 3).tolist()]
                    b["paths"] = self._paths(m)
                    b["start_paths"] = self.start_paths[m]
                    b["sign"] = bee.fly.sign
                out["bees"][m] = b
            self.prev_t = t
            return out

    def run_forever(self):
        window = deque(maxlen=40)                     # (brain ms, wall s) per loop, for the real speed reached
        last = time.perf_counter()
        while True:
            if self.paused:
                time.sleep(0.05)
                last = time.perf_counter()
                continue
            t0 = time.perf_counter()
            steps = max(1, min(500, round(self.speed * 0.05 * 1000 / A.BODY_MS)))   # ~50 ms of wall time per loop
            for _ in range(steps):
                self.step()
            want = steps * A.BODY_MS / 1000 / self.speed
            took = time.perf_counter() - t0
            if took < want:
                time.sleep(want - took)
            now = time.perf_counter()
            window.append((steps * A.BODY_MS, now - last))
            last = now
            self.real_x = sum(w[0] for w in window) / 1000 / max(1e-9, sum(w[1] for w in window))

    def start_compare(self, minutes, seeds):
        if self.compare["running"]:
            return
        self.compare = dict(running=True, progress=0, total=0, last="starting...", error=None)

        def go():
            cmd = [sys.executable, "-m", "bees.compare", "--minutes", str(minutes), "--seeds", str(seeds)]
            p = subprocess.Popen(cmd, cwd=Path(__file__).parent.parent, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if "progress" in msg:
                    self.compare.update(progress=msg["progress"], total=msg["total"], last=msg["last"])
            p.wait()
            self.compare["running"] = False
            if p.returncode:
                self.compare["error"] = f"comparison failed (exit {p.returncode})"
            else:
                self.compare["last"] = "done"
        threading.Thread(target=go, daemon=True).start()


def make_handler(lab):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, body):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self._cors()
                self.end_headers()
                try:
                    while True:
                        self.wfile.write(b"data: " + json.dumps(lab.frame()).encode() + b"\n\n")
                        self.wfile.flush()
                        time.sleep(1 / 15)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
            elif self.path.startswith("/results"):
                f = OUT / "compare.json"
                self._json(f.read_bytes() if f.exists() else b"null")
            elif self.path.startswith("/meta"):
                self._json(json.dumps(dict(
                    conditions={k: dict(label=v["label"], about=v["about"]) for k, v in A.CONDITIONS.items()},
                    speeds=SPEEDS, scales=list(SCALES), models=MODELS, groups=GROUPS, kinds=F.KINDS,
                    sectors=F.SECTORS, eye_copies=F.EYE_COPIES, context=F.CONTEXT, actions=F.ACTIONS,
                    motor_copies=F.MOTOR_COPIES, n=F.N + FREE_SPOTS)).encode())
            else:
                self.send_response(404)
                self._cors()
                self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            msg = json.loads(self.rfile.read(n) or b"{}")
            cmd = msg.get("cmd")
            if cmd == "pause":
                lab.paused = not lab.paused
            elif cmd == "speed":
                lab.speed = float(msg["value"])
            elif cmd == "reset":
                lab.reset(msg.get("condition", lab.condition), msg.get("seed", lab.seed), msg.get("scale", lab.scale))
            elif cmd == "reward":
                with lab.lock:
                    for m, bee in lab.bees.items():
                        if msg.get("bee") in (None, m):
                            bee.brain.reward(float(msg.get("value", 1)))
            elif cmd == "compare":
                lab.start_compare(int(msg.get("minutes", 60)), int(msg.get("seeds", 10)))
            self._json(b'{"ok":true}')
    return H


def serve(lab, port=8766):
    threading.Thread(target=lab.run_forever, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(lab))
    print(f"bee lab server on http://127.0.0.1:{port}  (open the app at http://localhost:3000)", flush=True)
    srv.serve_forever()
