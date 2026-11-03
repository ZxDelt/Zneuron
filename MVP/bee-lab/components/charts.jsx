"use client";
import { useSize, fmt } from "@/lib/sim";

function niceMax(v) {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  const n = v / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
}

/** Line chart drawn at real pixel width. series: [{ name, color, points: [[x, y], ...], dash }] */
export function LineChart({ series, height = 150, xLabel = "", yLabel = "", yMin = null }) {
  const [ref, { w }] = useSize();
  const L = 40, R = 10, T = 8, B = 24;
  const all = series.flatMap((s) => s.points);
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const x0 = xs.length ? Math.min(...xs) : 0, x1 = xs.length ? Math.max(...xs, x0 + 1) : 1;
  const lo = yMin !== null ? yMin : Math.min(0, ...(ys.length ? ys : [0]));
  const hi = niceMax(Math.max(...(ys.length ? ys : [1]), lo + 1e-9) - Math.min(lo, 0)) + Math.min(lo, 0);
  const iw = Math.max(10, w - L - R), ih = height - T - B;
  const X = (x) => L + ((x - x0) / (x1 - x0 || 1)) * iw;
  const Y = (y) => T + ih - ((y - lo) / (hi - lo || 1)) * ih;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => lo + f * (hi - lo));
  const xt = [0, 0.25, 0.5, 0.75, 1].map((f) => x0 + f * (x1 - x0));
  return (
    <div ref={ref} style={{ width: "100%" }}>
      {w > 0 && (
        <svg className="chart" width={w} height={height}>
          {ticks.map((t, i) => (
            <g key={i}>
              <line x1={L} x2={L + iw} y1={Y(t)} y2={Y(t)} stroke="rgba(0,240,255,0.09)" />
              <text x={L - 6} y={Y(t) + 3} fill="#8f88b8" fontSize="10" textAnchor="end" fontFamily="ui-monospace,monospace">
                {Math.abs(hi - lo) < 5 ? fmt(t, 1) : Math.round(t)}
              </text>
            </g>
          ))}
          {xt.map((t, i) => (
            <text key={i} x={X(t)} y={height - 8} fill="#8f88b8" fontSize="10" fontFamily="ui-monospace,monospace"
              textAnchor={i === 0 ? "start" : i === 4 ? "end" : "middle"}>{Math.round(t)}</text>
          ))}
          {lo < 0 && <line x1={L} x2={L + iw} y1={Y(0)} y2={Y(0)} stroke="rgba(255,255,255,0.25)" />}
          {series.map((s) =>
            s.points.length > 1 ? (
              <polyline key={s.name} fill="none" stroke={s.color} strokeWidth={s.dash ? 1.5 : 2}
                strokeDasharray={s.dash ? "5 4" : undefined} opacity={s.dash ? 0.75 : 1}
                style={{ filter: s.dash ? undefined : `drop-shadow(0 0 4px ${s.color})` }}
                points={s.points.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ")} />
            ) : null
          )}
          <text x={L + iw} y={T + 10} fill="#8f88b8" fontSize="10" textAnchor="end">{yLabel}</text>
          <text x={L + iw / 2} y={height - 8} fill="transparent" fontSize="10">{xLabel}</text>
        </svg>
      )}
      <div className="legend" style={{ paddingLeft: L }}>
        {series.map((s) => (
          <span key={s.name}><i style={{ background: s.color, opacity: s.dash ? 0.7 : 1 }} />{s.name}</span>
        ))}
        {xLabel && <span style={{ marginLeft: "auto" }}>x: {xLabel}</span>}
      </div>
    </div>
  );
}

/** Two bars per row (yellow vs pink) on a shared scale. rows: [{ label, hub, lif, unit }] */
export function PairBars({ rows, digits = 1 }) {
  const max = Math.max(1e-9, ...rows.flatMap((r) => [r.hub || 0, r.lif || 0]));
  return (
    <div className="bars">
      {rows.map((r) => [
        <span key={r.label + "l"} className="dim" style={{ gridRow: "span 2" }}>{r.label}</span>,
        <div key={r.label + "hb"} className="bar"><div className="yb" style={{ width: `${(100 * (r.hub || 0)) / max}%` }} /></div>,
        <span key={r.label + "hv"} className="stat y" style={{ textAlign: "right" }}>{fmt(r.hub, digits)}</span>,
        <div key={r.label + "kb"} className="bar"><div className="kb" style={{ width: `${(100 * (r.lif || 0)) / max}%` }} /></div>,
        <span key={r.label + "kv"} className="stat k" style={{ textAlign: "right" }}>{fmt(r.lif, digits)}</span>,
      ])}
    </div>
  );
}
