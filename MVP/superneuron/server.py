"""Runs the network in a background thread and streams frames to the browser page.

Built-in Python only: http.server + Server-Sent Events. The page (web/index.html) draws
with canvas and sends control commands back with POST /control.
"""

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np



from . import insight, saving

WEB = Path(__file__).resolve().parent.parent / "web"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml"}
SPEEDS = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 10000]   # × real time (1 = real time)


class Sim:
    def __init__(self, net, stimulus=None, speed=1, fps=30, agent=None, save_dir=None):
        """stimulus: test input pattern (no body); agent: brain + body + world (senses and motors)."""
        self.net, self.stimulus, self.fps, self.agent = net, stimulus, fps, agent
        self.save_dir = Path(save_dir) if save_dir else None
        self.speed = speed if speed in SPEEDS else 1
        self._carry = 0.0                  # leftover fraction of a tick between frames
        self.paused = False
        self.step_request = 0
        self.selected = next((n.id for n in net.neurons if not n.sensor and n.alive), 0)
        net.select(self.selected)
        # sensor input as a schedule inside the compiled engine, so whole batches run in one call
        self.batched = stimulus is not None and hasattr(stimulus, "install")
        if self.batched:
            stimulus.install(net)
        self.tracker = insight.GroupTracker()
        self.insight = {}
        self.chat = deque(maxlen=200)     # (who, text, sim time)
        self.frame, self.frame_no = "{}", 0
        self.cond = threading.Condition()
        self.running = True

    # ---------- simulation thread ----------

    def _advance(self, ticks):
        # the whole frame in one compiled call (drives and reward timing are handled inside the engine)
        if self.agent is not None:
            self.agent.run(ticks)
        elif self.batched:
            self.net.run(ticks)
        else:
            for _ in range(ticks):
                self.stimulus(self.net)
                self.net.step()

    def tick_once(self):
        with self.cond:
            if not self.paused:
                # ms of brain time per frame for the chosen × real time; fractions carry over
                self._carry += self.speed * 1000.0 / self.fps
                ticks = int(self._carry)
                self._carry -= ticks
                self._advance(ticks)
            elif self.step_request:
                self._advance(self.step_request)
                self.step_request = 0
            if self.frame_no % 8 == 0:
                self._update_insight()
            self.frame = json.dumps(self.build_frame(), separators=(",", ":"))
            self.frame_no += 1
            self.cond.notify_all()

    def loop(self):
        while self.running:
            start = time.perf_counter()
            self.tick_once()
            time.sleep(max(0.0, 1.0 / self.fps - (time.perf_counter() - start)))

    def _update_insight(self):
        net = self.net
        groups = self.tracker.update(net)
        alarm_list = insight.alarms(net)
        self.insight = {
            "groups": groups,
            "chains": insight.chains(net),
            "capacity": insight.capacity(net, groups),
            "why": insight.why_fired(net, self.selected),
            "learned": insight.learned(net),
            "drives": net.drives(),
            "alarms": alarm_list,
            "teller": insight.tell(net, groups, alarm_list) + self._body_lines(),
        }

    def _body_lines(self):
        """Body part of the brain-state teller."""
        if self.agent is None:
            return []
        w = self.agent.world
        moving = [f"{name} {'+' if v > 0 else '−'}" for name, v in w.cmd.items() if abs(v) > 0.15]
        out = [f"Body: {'moving — ' + ', '.join(moving) if moving else 'still'}; "
               f"{'walking' if abs(w.speed) > 0.1 else 'standing'}; "
               f"{'day' if w.daylight(self.net.t) > 0.05 else 'night'}."]
        eyes, ears, skin = w.senses["eyes"], w.senses["ears"], w.senses["skin"]
        out.append(f"Eyes see {'bright' if eyes.mean() > 0.4 else 'dim'} light; strongest sound in the "
                   f"{'left' if ears[0].sum() >= ears[1].sum() else 'right'} ear; touching: "
                   + (", ".join(name for name, v in zip(("left hand", "right hand", "left foot", "right foot",
                                                         "chest", "back"), skin) if v > 0.5) or "nothing") + ".")
        recent = [f"{text} ({(t / 1000):.0f} s)" for t, text in self.agent.events[-3:]]
        if recent:
            out.append("Recently: " + "; ".join(recent) + ".")
        return out

    # ---------- frame for the page ----------

    def _neuron_rows(self):
        """[id, x, y, sign, sensor, hub, threshold, awake, pools, z, motor, alive] per neuron, from whole arrays.
        pools = one [other neuron, pool fill 0-1] per line, for drawing the pools."""
        net = self.net
        S = net.S
        if not hasattr(self, "_static"):
            self._static = [(n.id, round(n.x, 3), round(n.y, 3), round(n.z, 3), int(n.motor)) for n in net.neurons]
        hub = np.round(S.hub, 3).tolist()
        thr = np.round(net.thresholds(), 3).tolist()
        awake = S.awake.astype(np.int8).tolist()
        sign = S.sign.tolist()
        sensor = S.sensor.astype(np.int8).tolist()
        nl = S.nlines.tolist()
        slot = S.slot_line.tolist()
        fill = np.round(np.minimum(1.0, S.pool * np.maximum(S.nlines, 1)[:, None]), 2).tolist()
        la, lb = S.line_a.tolist(), S.line_b.tolist()
        alive = S.alive.astype(np.int8).tolist()
        rows = []
        for i, x, y, z, motor in self._static:
            pools = [[lb[L] if la[L] == i else la[L], fill[i][s]] for s, L in enumerate(slot[i][:nl[i]])]
            rows.append([i, x, y, sign[i], sensor[i], hub[i], thr[i], awake[i], pools, z, motor, alive[i]])
        return rows

    def _line_rows(self):
        """[src, dst, thickness, trial, locked] for every connected mini-line."""
        S = self.net.S
        alive = np.repeat(S.line_alive, 2)
        mi = np.flatnonzero(alive & (S.thick != 0.0))
        L = mi // 2
        even = mi % 2 == 0
        src = np.where(even, S.line_a[L], S.line_b[L])
        dst = np.where(even, S.line_b[L], S.line_a[L])
        return [list(row) for row in zip(src.tolist(), dst.tolist(), np.round(S.thick[mi], 3).tolist(),
                                         S.trial[mi].astype(np.int8).tolist(), S.locked[mi].astype(np.int8).tolist())]

    def build_frame(self):
        net = self.net
        r = lambda v: round(v, 3)  # noqa: E731
        sel = net.neurons[self.selected]
        cap = 1.0 / len(sel.lines) if sel.lines else 0.0
        pools = []
        for lid in sel.lines:
            line = net.lines[lid]
            inp = line.into(sel.id)
            pools.append([line.other(sel.id), r(sel.pools.get(lid, 0.0)), r(inp.thickness)])
        state = "resting" if net.resting(sel) else ("awake" if sel.id in net.awake else "asleep")
        window = net.p.window_ms - (net.t - sel.window_start) if sel.window_start is not None else None
        S, t = net.S, net.t
        return {
            "t": t, "paused": self.paused, "speed": self.speed, "speeds": SPEEDS,
            "size": net.size, "selected": self.selected,
            "neurons": self._neuron_rows(),
            "chat": list(self.chat)[-60:],
            "lfuel": net.lfuel,
            "layout": net.layout,
            "counts": dict(net.counts),
            "lines": self._line_rows(),
            "inflight": [[s, d, r(f), r(c)] for s, d, f, c in net.inflight()],
            "raster": [[ft, k] for ft, k in net.fire_log if ft >= t - 500],
            "detail": {
                "id": sel.id, "sensor": sel.sensor, "sign": sel.sign, "state": state,
                "window": window, "rate": r(net.rate_hz(sel)), "hub": r(sel.hub),
                "thr": r(net.final_threshold(sel)), "cap": r(cap), "pools": pools,
                "efuel": r(net.efuel(sel)), "cfuel": r(sel.cfuel), "rfuel": r(sel.rfuel_local),
            },
            "trace": self.net.trace()[-500:],
            "insight": self.insight,
            "world": self.agent.world.snapshot(t) if self.agent is not None else None,
            "rewards": [[rt, round(v, 2), why] for rt, v, why in self.agent.rewards[-10:]] if self.agent else [],
            "saves": self._saves(),
        }

    # ---------- commands from the page ----------

    def command(self, cmd):
        with self.cond:
            kind = cmd.get("cmd")
            if kind == "pause":
                self.paused = not self.paused
            elif kind == "step":
                self.paused = True
                self.step_request += int(cmd.get("ticks", 1))
            elif kind == "speed":
                if float(cmd.get("value", 1)) in SPEEDS:
                    self.speed = float(cmd["value"])
            elif kind == "faster":
                self.speed = SPEEDS[min(len(SPEEDS) - 1, SPEEDS.index(self.speed) + 1)]
            elif kind == "slower":
                self.speed = SPEEDS[max(0, SPEEDS.index(self.speed) - 1)]
            elif kind == "chat":
                text = str(cmd.get("text", "")).strip()[:500]
                if text:
                    self.chat.append(("you", text, self.net.t))
                    word = text.lower().strip(" !.")
                    if word in ("good", "bad"):
                        self._reward(1.0 if word == "good" else -1.0)
                    elif self.agent is not None:
                        self.agent.world.hear(text, self.net.t)
                        self.chat.append(("brain", f"(heard as sound: {len([c for c in text if c.isalnum()])} "
                                          "letters on the ear sensors)", self.net.t))
            elif kind == "save":
                self.chat.append(("system", self._save(str(cmd.get("name", "")).strip()), self.net.t))
            elif kind == "load":
                self.chat.append(("system", self._load(str(cmd.get("name", ""))), self.net.t))
            elif kind == "reward":
                self._reward(max(-1.0, min(1.0, float(cmd.get("value", 0)))))
            elif kind == "select":
                nid = int(cmd.get("id", 0))
                if 0 <= nid < len(self.net.neurons):
                    self.selected = nid
                    self.net.select(nid)
                    self._update_insight()

    # ---------- save / load brains ----------

    def _saves(self):
        if not self.save_dir or not self.save_dir.exists():
            return []
        return sorted(f.stem for f in self.save_dir.glob("*.brain"))

    def _save(self, name):
        if not self.save_dir:
            return "Saving is off (no save folder)."
        name = "".join(c for c in name if c.isalnum() or c in "-_ ")[:40] or f"brain-{self.net.t // 1000}s"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"{name}.brain"
        saving.save(self.net, path, self.agent.world if self.agent else None)
        return f"Saved brain as '{name}' ({path.stat().st_size // 1024} KB)."

    def _load(self, name):
        path = self.save_dir / f"{name}.brain" if self.save_dir else None
        if not path or not path.exists():
            return f"No saved brain called '{name}'."
        try:
            saving.load(self.net, path, self.agent.world if self.agent else None)
        except ValueError as err:
            return f"Can't load '{name}': {err}"
        self.net.select(self.selected)
        self._update_insight()
        return f"Loaded brain '{name}' (brain time {self.net.t // 1000} s)."

    def _reward(self, value):
        """Reward: learning from the last ~1 s is applied again × value (+ keeps it stronger, − undoes it)."""
        changed = self.net.reward(value)
        self.chat.append(("system", f"Reward {value:+.0f}: {changed} recently-learned lines "
                          f"{'strengthened' if value > 0 else 'undone'}", self.net.t))


def make_handler(sim):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/stream":
                self._stream()
                return
            name = {"/": "app.html", "/classic": "classic.html"}.get(path, path.lstrip("/"))
            target = (WEB / name).resolve()
            # only files inside web/ are served
            if WEB.resolve() not in target.parents or not target.is_file():
                self.send_error(404)
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", TYPES.get(target.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = -1
            try:
                while sim.running:
                    with sim.cond:
                        sim.cond.wait_for(lambda: sim.frame_no != last, timeout=5)
                        data, last = sim.frame, sim.frame_no
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return

        def do_POST(self):
            if self.path != "/control":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                sim.command(json.loads(self.rfile.read(length) or b"{}"))
            except (ValueError, KeyError):
                self.send_error(400)
                return
            self.send_response(204)
            self.end_headers()

    return Handler


def serve(sim, host="127.0.0.1", port=8765):
    threading.Thread(target=sim.loop, daemon=True).start()
    server = ThreadingHTTPServer((host, port), make_handler(sim))
    server.daemon_threads = True
    return server
