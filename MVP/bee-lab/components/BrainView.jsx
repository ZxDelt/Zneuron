"use client";
import { useEffect, useRef } from "react";
import { COLOR, RGB } from "@/lib/sim";

const COLS = [   // group, columns of dots in its block, label
  ["eyes", 3, "Eyes"], ["body", 1, "Body"], ["kenyon", 5, "Kenyon cells"], ["mbon", 3, "MBONs"], ["new", 2, "Grown"],
  ["motors", 1, "Motors"],
];

/** Neuron positions: one block per layer, left to right, dots in a small grid inside each block. */
function layout(meta, W, H) {
  const top = 26, bottom = 16, left = 12, right = 64;
  const pos = new Array(meta.n);
  const cols_ = COLS.filter(([g]) => meta.groups[g]);
  const blockW = (W - left - right) / cols_.length;
  const blocks = [];
  cols_.forEach(([g, cols, label], gi) => {
    const [a, b] = meta.groups[g];
    const n = b - a, rows = Math.ceil(n / cols);
    const cx = left + blockW * gi + blockW / 2;
    const gapX = Math.min(12, (blockW * 0.6) / Math.max(1, cols));
    const gapY = (H - top - bottom) / rows;
    for (let k = 0; k < n; k++) {
      const c = k % cols, r = Math.floor(k / cols);
      pos[a + k] = [cx + (c - (cols - 1) / 2) * gapX, top + gapY * (r + 0.5)];
    }
    blocks.push({ g, label, cx, n });
  });
  return { pos, blocks, gap: (H - top - bottom) };
}

export default function BrainView({ bee, kind, meta, version }) {
  // neurons in the "Grown" block are free spots until Zneuron v2 grows a neuron there
  const wrap = useRef(null), cvs = useRef(null);
  const data = useRef({ bee: null, lines: null, sign: null, start: null, version: null });

  useEffect(() => {
    const d = data.current;
    if (!bee) return;
    d.bee = bee;
    if (bee.lines) {
      d.lines = bee.lines;
      d.sign = bee.sign;
      if (!d.start || d.version !== version) {      // remember the starting wiring to spot grown lines
        d.start = new Set(bee.lines[0].map((s, i) => s * 100000 + bee.lines[1][i]));
        d.version = version;
      }
    }
  }, [bee, version]);

  useEffect(() => {
    let raf;
    const draw = () => {
      const c = cvs.current, w = wrap.current, d = data.current;
      if (c && w && meta && d.bee) {
        const dpr = window.devicePixelRatio || 1;
        const W = w.clientWidth, H = w.clientHeight;
        if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
          c.width = Math.round(W * dpr);
          c.height = Math.round(H * dpr);
        }
        const ctx = c.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, W, H);
        const { pos, blocks } = layout(meta, W, H);
        const hot = new Set(d.bee.hot);

        if (d.lines) {                                         // lines: strength = opacity
          const [src, dst, wt] = d.lines;
          ctx.lineWidth = 1;
          for (let i = 0; i < src.length; i++) {
            const a = pos[src[i]], b = pos[dst[i]];
            if (!a || !b) continue;
            const grown = d.start && !d.start.has(src[i] * 100000 + dst[i]);
            const firing = hot.has(src[i]);
            const inh = d.sign && d.sign[src[i]] < 0;
            const alpha = firing ? 0.35 + 0.5 * wt[i] : 0.03 + 0.22 * wt[i];
            ctx.strokeStyle = grown ? `rgba(57,255,159,${Math.max(alpha, 0.35)})`
              : inh ? `rgba(255,59,92,${alpha})` : firing ? `rgba(${RGB[kind]},${alpha})`
              : `rgba(0,240,255,${alpha})`;
            ctx.beginPath();
            ctx.moveTo(a[0], a[1]);
            ctx.lineTo(b[0], b[1]);
            ctx.stroke();
          }
        }
        const born = new Set([...(d.lines ? d.lines[0] : []), ...(d.lines ? d.lines[1] : [])]);
        const g0 = meta.groups.new ? meta.groups.new[0] : meta.n;
        for (let i = 0; i < meta.n; i++) {                     // neurons
          const p = pos[i];
          if (!p || (i >= g0 && !born.has(i))) continue;       // free spot, no neuron grown there yet
          const on = hot.has(i);
          const inh = d.sign && d.sign[i] < 0;
          ctx.beginPath();
          ctx.arc(p[0], p[1], on ? 3.2 : 2.2, 0, Math.PI * 2);
          if (on) {
            ctx.shadowColor = COLOR[kind];
            ctx.shadowBlur = 10;
            ctx.fillStyle = COLOR[kind];
          } else {
            ctx.shadowBlur = 0;
            ctx.fillStyle = inh ? "rgba(255,59,92,0.55)" : "rgba(160,170,230,0.45)";
          }
          ctx.fill();
        }
        ctx.shadowBlur = 0;
        ctx.font = "11px Segoe UI, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillStyle = "#8f88b8";
        for (const b of blocks) ctx.fillText(`${b.label} (${b.n})`, b.cx, 14);
        if (meta.actions) {                                    // motor labels on the right
          ctx.textAlign = "left";
          const [a] = meta.groups.motors;
          meta.actions.forEach((name, k) => {
            const ids = Array.from({ length: meta.motor_copies }, (_, j) => a + k * meta.motor_copies + j);
            const y = ids.reduce((s, i) => s + pos[i][1], 0) / ids.length;
            const on = ids.some((i) => hot.has(i));
            ctx.fillStyle = on ? COLOR[kind] : "#8f88b8";
            ctx.fillText(name, pos[ids[0]][0] + 10, y + 4);
          });
        }
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [meta, kind]);

  return (
    <div className="brainview">
      <div className="canvas-wrap" ref={wrap}><canvas ref={cvs} /></div>
      <div className="legend brain-legend">
        <span><i style={{ background: COLOR[kind] }} />firing now</span>
        <span><i style={{ background: "#00f0ff" }} />excitatory line</span>
        <span><i style={{ background: "#ff3b5c" }} />inhibitory</span>
        <span><i style={{ background: "#39ff9f" }} />grown since start</span>
        <span className="push">brighter = stronger · {bee?.neurons ?? 0} neurons · {bee?.line_count ?? 0} lines (start {bee?.start_lines ?? 0})</span>
      </div>
    </div>
  );
}
