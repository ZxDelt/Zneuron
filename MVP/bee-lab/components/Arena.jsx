"use client";
import { useEffect, useRef } from "react";
import { BEES, COLOR, NAME, RGB, fmt } from "@/lib/sim";

// Minecraft-style bee seen from above, 11 x 8 "pixels", nose pointing +x. Letters map to colours per bee.
const BEE = [
  "...WWW.....",
  "..WWWWW....",
  ".BBSBBSBF..",
  "TBBSBBSBEA.",
  "TBBSBBSBFA.",
  ".BBSBBSBE..",
  "..WWWWW....",
  "...WWW.....",
];
const SKIN = {
  hub2: { B: "#ffd21f", S: "#3b2a12", F: "#3b2a12", E: "#ffffff", T: "#9a9a9a", A: "#2a1d0c" },
  hub1: { B: "#b388ff", S: "#2a1650", F: "#2a1650", E: "#ffffff", T: "#9a9a9a", A: "#1c0e38" },
  lif: { B: "#ff86dc", S: "#4a1240", F: "#4a1240", E: "#ffffff", T: "#9a9a9a", A: "#2a0c26" },
  alif: { B: "#ff9f43", S: "#4a2608", F: "#4a2608", E: "#ffffff", T: "#9a9a9a", A: "#2e1704" },
};

function drawBee(ctx, kind, x, y, heading, px, flap, carrying) {
  const skin = SKIN[kind];
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-heading);                       // world y points up, canvas y points down
  const w = BEE[0].length, h = BEE.length;
  ctx.translate(-w * px / 2, -h * px / 2);
  if (carrying) {
    ctx.shadowColor = "#00f0ff";
    ctx.shadowBlur = 12;
    ctx.fillStyle = "#00f0ff";
    ctx.fillRect((w - 0.6) * px, (h / 2 - 1.6) * px, 3.2 * px, 3.2 * px);
  }
  ctx.shadowColor = COLOR[kind];
  ctx.shadowBlur = 14;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const ch = BEE[r][c];
      if (ch === "." || ch === "W") continue;
      ctx.fillStyle = skin[ch];
      ctx.fillRect(c * px, r * px, px + 0.5, px + 0.5);
    }
  }
  ctx.shadowBlur = 0;
  ctx.fillStyle = `rgba(190, 235, 255, ${flap ? 0.55 : 0.32})`;   // wings on top, flapping
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      if (BEE[r][c] === "W") ctx.fillRect(c * px, r * px, px + 0.5, px + 0.5);
    }
  }
  ctx.fillStyle = "#000";                                           // pupils
  ctx.fillRect(8.5 * px, 3 * px, px / 2, px);
  ctx.fillRect(8.5 * px, 5 * px, px / 2, px);
  ctx.restore();
}

function drawWorld(ctx, W, H, d, now) {
  ctx.clearRect(0, 0, W, H);
  const world = d.world;
  if (!world) return;
  const pad = 14;
  const s = Math.min(W - 2 * pad, H - 2 * pad) / world.size;
  if (!(s > 0)) return;                                             // canvas too small to draw (e.g. while resizing)
  const ox = (W - world.size * s) / 2, oy = (H - world.size * s) / 2;
  const X = (x) => ox + x * s, Y = (y) => oy + (world.size - y) * s;

  ctx.strokeStyle = "rgba(0, 240, 255, 0.07)";                     // floor grid
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let g = 0; g <= world.size; g += 2) {
    ctx.moveTo(X(g), Y(0)); ctx.lineTo(X(g), Y(world.size));
    ctx.moveTo(X(0), Y(g)); ctx.lineTo(X(world.size), Y(g));
  }
  ctx.stroke();
  ctx.save();                                                       // walls
  ctx.shadowColor = "#00f0ff";
  ctx.shadowBlur = 16;
  ctx.strokeStyle = "#00f0ff";
  ctx.lineWidth = 2;
  ctx.strokeRect(X(0), Y(world.size), world.size * s, world.size * s);
  ctx.restore();

  const fov = (210 / 180) * Math.PI;                               // fields of view
  for (const k of BEES) {
    const b = world.bees[k];
    ctx.fillStyle = `rgba(${RGB[k]}, 0.045)`;
    ctx.beginPath();
    ctx.moveTo(X(b.x), Y(b.y));
    ctx.arc(X(b.x), Y(b.y), world.eye_range * s, -b.heading - fov / 2, -b.heading + fov / 2);
    ctx.closePath();
    ctx.fill();
  }

  const pulse = 0.6 + 0.4 * Math.sin(now / 300);                   // drop spot
  ctx.save();
  ctx.shadowColor = "#ff2bd6";
  ctx.shadowBlur = 22 * pulse;
  ctx.strokeStyle = "#ff2bd6";
  ctx.lineWidth = 2.5;
  ctx.setLineDash([6, 5]);
  ctx.lineDashOffset = -now / 60;
  ctx.beginPath();
  ctx.arc(X(world.spot[0]), Y(world.spot[1]), world.spot[2] * s, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = `rgba(255, 43, 214, ${0.08 + 0.06 * pulse})`;
  ctx.fill();
  ctx.restore();

  const sq = 0.8 * s;                                               // squares and decoys
  ctx.save();
  ctx.shadowBlur = 14;
  ctx.lineWidth = 2;
  for (const [x, y] of world.squares) {
    ctx.shadowColor = "#00f0ff";
    ctx.fillStyle = "rgba(0, 240, 255, 0.25)";
    ctx.strokeStyle = "#00f0ff";
    ctx.fillRect(X(x) - sq / 2, Y(y) - sq / 2, sq, sq);
    ctx.strokeRect(X(x) - sq / 2, Y(y) - sq / 2, sq, sq);
  }
  for (const [x, y] of world.reds) {
    ctx.shadowColor = "#ff3b5c";
    ctx.strokeStyle = "#ff3b5c";
    ctx.fillStyle = "rgba(255, 59, 92, 0.2)";
    ctx.fillRect(X(x) - sq / 2, Y(y) - sq / 2, sq, sq);
    ctx.strokeRect(X(x) - sq / 2, Y(y) - sq / 2, sq, sq);
    ctx.beginPath();
    ctx.moveTo(X(x) - sq / 3, Y(y) - sq / 3); ctx.lineTo(X(x) + sq / 3, Y(y) + sq / 3);
    ctx.moveTo(X(x) + sq / 3, Y(y) - sq / 3); ctx.lineTo(X(x) - sq / 3, Y(y) + sq / 3);
    ctx.stroke();
  }
  ctx.restore();

  for (const k of BEES) {                                           // trails
    const tr = d.trails[k];
    ctx.lineWidth = 1.5;
    for (let i = 1; i < tr.length; i++) {
      ctx.strokeStyle = `rgba(${RGB[k]}, ${(i / tr.length) * 0.5})`;
      ctx.beginPath();
      ctx.moveTo(X(tr[i - 1][0]), Y(tr[i - 1][1]));
      ctx.lineTo(X(tr[i][0]), Y(tr[i][1]));
      ctx.stroke();
    }
  }

  for (const f of d.flashes) {                                      // pick-up / delivery rings
    const age = (now - f.at) / 900;
    if (age > 1) continue;
    const col = f.kind === "deliver" ? "57, 255, 159" : f.kind === "decoy" ? "255, 59, 92" : RGB[f.bee];
    ctx.strokeStyle = `rgba(${col}, ${1 - age})`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(X(f.x), Y(f.y), Math.max(0, (0.8 + age * 2.5) * s), 0, Math.PI * 2);
    ctx.stroke();
    if (f.kind === "deliver") {
      ctx.fillStyle = `rgba(${RGB[f.bee]}, ${1 - age})`;
      ctx.font = `bold ${Math.max(12, s * 0.9)}px ui-monospace, monospace`;
      ctx.textAlign = "center";
      ctx.fillText("+1", X(f.x), Y(f.y) - (1.2 + age * 2) * s);
    }
  }

  const flap = Math.floor(now / 70) % 2 === 0;
  for (const k of BEES) {
    const b = world.bees[k];
    drawBee(ctx, k, X(b.x), Y(b.y), b.heading, Math.max(2, (2.4 * s) / 11), flap, b.carrying);
  }
}

/** The shared arena: both bees, one set of squares, one drop spot. */
export default function Arena({ frame }) {
  const wrap = useRef(null), cvs = useRef(null);
  const data = useRef({ world: null, trails: Object.fromEntries(BEES.map((k) => [k, []])), flashes: [], seen: new Set(), version: null });

  useEffect(() => {
    const d = data.current;
    if (!frame?.world) return;
    if (d.version !== frame.version) {                 // new match: forget trails and old events
      d.version = frame.version;
      d.trails = Object.fromEntries(BEES.map((k) => [k, []]));
      d.seen = new Set((frame.log || []).map((e) => `${e.t}-${e.bee}-${e.event}`));
    }
    d.world = frame.world;
    for (const k of BEES) {
      const b = frame.world.bees[k];
      if (!b) continue;
      d.trails[k].push([b.x, b.y]);
      if (d.trails[k].length > 160) d.trails[k].shift();
    }
    for (const e of frame.log || []) {
      const key = `${e.t}-${e.bee}-${e.event}`;
      if (d.seen.has(key)) continue;
      d.seen.add(key);
      if (e.bee && (e.event === "deliver" || e.event === "pickup" || e.event === "decoy")) {
        const b = frame.world.bees[e.bee];
        d.flashes.push({ kind: e.event, bee: e.bee, x: b.x, y: b.y, at: performance.now() });
      }
    }
    if (d.seen.size > 600) d.seen = new Set([...d.seen].slice(-300));
    d.flashes = d.flashes.filter((f) => performance.now() - f.at < 1000);
  }, [frame]);

  useEffect(() => {
    let raf;
    const draw = () => {
      const c = cvs.current, w = wrap.current;
      if (c && w) {
        const dpr = window.devicePixelRatio || 1;
        const W = w.clientWidth, H = w.clientHeight;
        if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
          c.width = Math.round(W * dpr);
          c.height = Math.round(H * dpr);
        }
        const ctx = c.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawWorld(ctx, W, H, data.current, performance.now());
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  const scores = BEES.map((k) => frame?.bees?.[k]?.stats.deliveries ?? 0);
  const top = Math.max(...scores);
  const lead = top > 0 && scores.filter((v) => v === top).length === 1 ? BEES[scores.indexOf(top)] : null;
  return (
    <div className="arena-full">
      <div className="hud">
        {BEES.map((k) => {
          const b = frame?.bees?.[k], s = b?.stats;
          const glow = { color: COLOR[k], textShadow: `0 0 8px rgba(${RGB[k]}, 0.55)` };
          return (
            <div key={k} className="chip" style={{ borderColor: `rgba(${RGB[k]}, 0.45)` }}>
              <div className="chip-row">
                <span className="who" style={glow}>{NAME[k]}</span>
                {lead === k && <span className="badge" style={{ color: COLOR[k] }}>leading</span>}
                {b?.damaged && <span className="badge tie r">damaged</span>}
                <span className="big stat" style={glow}>{s?.deliveries ?? 0}</span>
              </div>
              <div className="chip-row dim small stat">
                <span>{frame?.world?.bees[k]?.carrying ? "carrying" : "searching"}</span>
                <span className="push">picked {s?.pickups ?? 0} · reward {fmt(s?.reward, 1)}</span>
              </div>
            </div>
          );
        })}
      </div>
      <div className="canvas-wrap" ref={wrap}><canvas ref={cvs} /></div>
    </div>
  );
}
