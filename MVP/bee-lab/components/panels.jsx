"use client";
import { useEffect, useRef, useState } from "react";
import { BEES, COLOR, FOCUS, NAME, RGB, SHORT as SHORT_NAME, fmt, post } from "@/lib/sim";
import { LineChart } from "./charts";

const SHORT = {
  deliveries: "Delivered", pickups: "Picked up", carry_success: "Pick-ups delivered %", decoys: "Decoys grabbed",
  drops: "Wasted grabs", bumps: "Wall bumps", reward: "Total reward", gain: "Learning gain /min",
  first_delivery_s: "First delivery (s)", after_damage: "After damage /min", spikes_per_s: "Spikes / s",
  spikes_per_delivery: "Spikes / delivery", speed_x: "Sim speed ×", lines_kept: "Lines kept %",
};
const GROUPS = ["eyes", "body", "kenyon", "mbon", "motors", "new"];
const GROUP_LABEL = { eyes: "Eyes", body: "Body", kenyon: "Kenyon", mbon: "MBON", motors: "Motors", new: "Grown" };
const GROUP_COLOR = { eyes: "#00f0ff", body: "#39ff9f", kenyon: "#b388ff", mbon: "#ff9f43", motors: "#ff2bd6", new: "#ffffff" };
const isHub = (k) => k === "hub2" || k === "hub1";
const PATH_LABEL = {
  "eyes>kenyon": "Eyes → Kenyon", "body>kenyon": "Body → Kenyon", "kenyon>mbon": "Kenyon → MBON",
  "mbon>motors": "MBON → Motor", "eyes>motors": "Eyes → Motor", "body>motors": "Body → Motor",
};

/* ================= per-bee tabs ================= */

function Raster({ bee, meta, color, height = 150 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !bee) return;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    c.width = W * dpr;
    c.height = H * dpr;
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!meta) return;
    // one band per layer, in brain order, each at least 18 px tall so labels never overlap
    const n = meta.n, MIN = 18, L = 52;
    const groups = GROUPS.filter((g) => meta.groups[g]);
    const sizes = groups.map((g) => meta.groups[g][1] - meta.groups[g][0]);
    const free = H - MIN * groups.length;
    let y = 0;
    const band = {};
    groups.forEach((g, gi) => {
      const h = MIN + (free * sizes[gi]) / n;
      band[g] = [y, h];
      ctx.fillStyle = gi % 2 ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0)";
      ctx.fillRect(0, y, W, h);
      ctx.fillStyle = "#8f88b8";
      ctx.font = "10px Segoe UI, sans-serif";
      ctx.fillText(GROUP_LABEL[g], 4, y + h / 2 + 3);
      y += h;
    });
    const rowOf = (id) => {
      for (const g of groups) {
        const [a, b] = meta.groups[g];
        if (id >= a && id < b) return band[g][0] + 2 + ((id - a) / (b - a)) * (band[g][1] - 4);
      }
      return 0;
    };
    ctx.fillStyle = color;
    const [ts, ids] = bee.raster;
    for (let i = 0; i < ts.length; i++) ctx.fillRect(L + ((ts[i] + 1000) / 1000) * (W - L), rowOf(ids[i]), 1.6, 1.6);
  }, [bee, meta, color]);
  return <canvas ref={ref} className="raster" style={{ height }} />;
}

export function Activity({ bee, kind, meta }) {
  if (!bee) return <div className="empty">Waiting…</div>;
  const present = GROUPS.filter((g) => bee.rates[g] !== undefined && (g !== "new" || isHub(kind)));
  const rows = present.map((g) => ({ g, v: bee.rates[g] }));
  const max = Math.max(1, ...rows.map((r) => r.v));
  const series = present.map((g) => ({
    name: GROUP_LABEL[g], color: GROUP_COLOR[g], points: bee.rate_hist.map((r) => [r[0], r[GROUPS.indexOf(g) + 1]]),
  }));
  return (
    <>
      <div className="kpis">
        <div><b className="stat">{fmt(bee.spikes_per_s, 0)}</b><span>spikes / s (energy)</span></div>
        <div><b className="stat">{fmt(bee.compute_ms_per_s, 1)}</b><span>compute ms / brain-s</span></div>
        <div><b className="stat">{bee.line_count}</b><span>lines (start {bee.start_lines}) · {bee.neurons} neurons</span></div>
        <div><b className="stat">{bee.hot.length}</b><span>neurons firing now</span></div>
      </div>
      <div className="split even">
        <div>
          <div className="sub">Firing rate per layer now (Hz)</div>
          <div className="bars one">
            {rows.map((r) => [
              <span key={r.g + "l"} className="dim">{GROUP_LABEL[r.g]}</span>,
              <div key={r.g + "b"} className="bar"><div style={{ width: `${(100 * r.v) / max}%`, background: GROUP_COLOR[r.g],
                boxShadow: `0 0 8px ${GROUP_COLOR[r.g]}` }} /></div>,
              <span key={r.g + "v"} className="stat" style={{ textAlign: "right" }}>{fmt(r.v, 1)}</span>,
            ])}
          </div>
        </div>
        <div>
          <div className="sub">Firing rate over time (Hz)</div>
          <LineChart series={series} height={120} xLabel="brain s" />
        </div>
      </div>
      <div className="sub">Every spike in the last second (rows = neurons)</div>
      <Raster bee={bee} meta={meta} color={COLOR[kind]} />
    </>
  );
}

export function Senses({ bee, meta }) {
  if (!bee || !meta) return <div className="empty">Waiting…</div>;
  const s = bee.senses;
  const per = meta.sectors * meta.eye_copies;
  const tint = { square: "0,240,255", spot: "255,43,214", red: "255,59,92" };
  let off = meta.kinds.length * per;
  const ctx = meta.context.map(([name, k]) => {
    const on = s[off] > 0;
    off += k;
    return [name, on];
  });
  return (
    <>
      <div className="sub">Compound eye: what each of the 7 sectors sees (left → right, 210°)</div>
      <div className="eyegrid">
        <span />
        {Array.from({ length: meta.sectors }, (_, i) => (
          <span key={i} className="dim" style={{ textAlign: "center" }}>
            {i === 0 ? "far L" : i === meta.sectors - 1 ? "far R" : i === 3 ? "front" : ""}
          </span>
        ))}
        {meta.kinds.map((kind, ki) => [
          <span key={kind} className="dim">{kind === "red" ? "decoy" : kind}</span>,
          ...Array.from({ length: meta.sectors }, (_, sec) => {
            const v = s[ki * per + (meta.sectors - 1 - sec) * meta.eye_copies] / 0.08;
            return <div key={kind + sec} className="cell" style={{ background: `rgba(${tint[kind]},${Math.min(1, v) * 0.9})`,
              boxShadow: v > 0.05 ? `0 0 6px rgba(${tint[kind]},${v})` : "none" }} />;
          }),
        ])}
      </div>
      <div className="sub">Body sense</div>
      <div className="ctx">{ctx.map(([n, on]) => <span key={n} className={on ? "on" : ""}>{n.replace("_", " ")}</span>)}</div>
      <p className="dim small" style={{ marginTop: 10 }}>
        Each sector has 3 sensor neurons per kind; brighter = closer. Both bees get the same kind of senses and
        the same sense noise.
      </p>
    </>
  );
}

export function Wiring({ bee, kind }) {
  if (!bee?.paths) return <div className="empty">Waiting…</div>;
  const keys = Object.keys(PATH_LABEL).filter((k) => bee.paths[k]);
  const other = Object.keys(bee.paths).filter((k) => !PATH_LABEL[k]);
  const max = Math.max(1e-9, ...keys.flatMap((k) => [bee.paths[k][0], bee.start_paths[k]?.[0] || 0]));
  const fillStyle = { background: COLOR[kind], boxShadow: `0 0 8px ${COLOR[kind]}` };
  return (
    <>
      <div className="sub">Average line strength per pathway: now (bar) vs at start (tick)</div>
      <div className="bars three">
        <span className="dim small">Pathway</span><span className="dim small">Strength</span>
        <span className="dim small" style={{ textAlign: "right" }}>Now</span>
        <span className="dim small" style={{ textAlign: "right" }}>Lines</span>
        {keys.map((k) => {
          const [now, n] = bee.paths[k];
          const st = bee.start_paths[k]?.[0] || 0;
          return [
            <span key={k + "l"} className="dim">{PATH_LABEL[k]}</span>,
            <div key={k + "b"} className="bar tick"><div style={{ ...fillStyle, width: `${(100 * now) / max}%` }} />
              <i style={{ left: `${(100 * st) / max}%` }} /></div>,
            <span key={k + "v"} className="stat" style={{ textAlign: "right" }}>{fmt(now, 3)}</span>,
            <span key={k + "n"} className="stat dim" style={{ textAlign: "right" }}>{n}</span>,
          ];
        })}
      </div>
      <div className="sub">Lines the brain made itself</div>
      {other.length ? (
        <div className="ctx">{other.map((k) => <span key={k} className="on">{k.replace(">", " → ")} · {bee.paths[k][1]}</span>)}</div>
      ) : <p className="dim small">None: {isHub(kind) ? "no new lines grown yet." : "LIF and ALIF networks never grow lines."}</p>}
      <p className="dim small" style={{ marginTop: 10 }}>
        {bee.line_count} active lines now, {bee.start_lines} at start.
        {isHub(kind) ? " The Hub neuron grows lines between neurons that fire together and prunes unused ones (spec #21, #22)."
          : " LIF/ALIF have a fixed wiring; only line strengths change with reward."}
      </p>
    </>
  );
}

/* ================= shared tabs ================= */

const who = (k) => (k === "tie" || !k ? "tie" : SHORT_NAME[k] || k);
const tint = (k) => ({ color: COLOR[k], textShadow: `0 0 6px rgba(${RGB[k]}, 0.5)` });
const Badge = ({ k }) => (
  <span className="badge" style={k && k !== "tie" ? { ...tint(k), borderColor: COLOR[k] } : undefined}>{who(k)}</span>
);

const LIVE = [
  ["Delivered", (b) => b.stats.deliveries, 0, true],
  ["Delivered / min", (b) => b.stats.per_min, 2, true],
  ["Picked up", (b) => b.stats.pickups, 0, true],
  ["Pick-ups delivered %", (b) => b.stats.carry_success, 0, true],
  ["Decoys grabbed", (b) => b.stats.decoys, 0, false],
  ["Wasted grabs", (b) => b.stats.drops, 0, false],
  ["Wall bumps", (b) => b.stats.bumps, 0, false],
  ["Total reward", (b) => b.stats.reward, 1, true],
  ["Distance flown", (b) => b.stats.distance, 0, null],
  ["First delivery (s)", (b) => (b.stats.first_delivery_ms ? b.stats.first_delivery_ms / 1000 : null), 1, false],
  ["Spikes / s (energy)", (b) => b.spikes_per_s, 0, false],
  ["Compute ms / brain-s", (b) => b.compute_ms_per_s, 2, false],
  ["Lines active", (b) => b.line_count, 0, null],
  ["Neurons", (b) => b.neurons, 0, null],
];

export function Score({ frame }) {
  if (!frame) return <div className="empty">Waiting for the simulation server…</div>;
  const wins = Object.fromEntries(BEES.map((k) => [k, 0]));
  const rows = LIVE.map(([label, get, d, higher]) => {
    const vals = BEES.map((k) => (frame.bees[k] ? get(frame.bees[k]) : null));
    let best = null;
    if (higher !== null && vals.every((v) => v !== null)) {
      const target = higher ? Math.max(...vals) : Math.min(...vals);
      if (vals.filter((v) => v === target).length === 1) best = BEES[vals.indexOf(target)];
    }
    if (best) wins[best] += 1;
    return { label, vals, d, best };
  });
  return (
    <>
      <table className="t">
        <colgroup><col style={{ width: "32%" }} />{BEES.map((k) => <col key={k} />)}</colgroup>
        <thead><tr><th>This match · live</th>{BEES.map((k) => <th key={k} style={tint(k)}>{SHORT_NAME[k]}</th>)}</tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <td>{r.label}</td>
              {r.vals.map((v, i) => (
                <td key={BEES[i]} className={`n ${r.best === BEES[i] ? "lead" : ""}`} style={r.best === BEES[i] ? tint(BEES[i]) : undefined}>
                  {fmt(v, r.d)}
                </td>
              ))}
            </tr>
          ))}
          <tr><td className="dim">Rows leading</td>{BEES.map((k) => <td key={k} className="n lead" style={tint(k)}>{wins[k]}</td>)}</tr>
        </tbody>
      </table>
      <p className="dim small" style={{ margin: "8px 2px 0" }}>
        Best value per row is highlighted. One match is luck-heavy; Verdict and Conditions use many matches and paired tests.
      </p>
    </>
  );
}

export function Race({ frame }) {
  const [what, setWhat] = useState("deliveries");
  if (!frame) return <div className="empty">Waiting…</div>;
  const idx = what === "reward" ? 2 : 1;
  const series = BEES.filter((m) => frame.bees[m]).map((m) => ({
    name: NAME[m], color: COLOR[m], points: frame.bees[m].history.map((h) => [h[0], h[idx]]),
  }));
  return (
    <div className="split even">
      <div>
        <div className="toolbar">
          <div className="seg">
            <button className={what === "deliveries" ? "on" : ""} onClick={() => setWhat("deliveries")}>Delivered</button>
            <button className={what === "reward" ? "on" : ""} onClick={() => setWhat("reward")}>Reward</button>
          </div>
        </div>
        <LineChart series={series} height={170} xLabel="brain seconds" yLabel={what === "reward" ? "total reward" : "squares delivered"} />
      </div>
      <div>
        <div className="sub">Events</div>
        <div className="log">
          {[...frame.log].reverse().map((e, i) => (
            <div key={i}>
              <span className="dim">{fmt(e.t, 1)}s</span>
              <span style={{ color: e.bee ? COLOR[e.bee] : "#8f88b8" }}>●</span>
              <span className={e.event === "deliver" ? "g" : e.event === "decoy" || e.event === "brain damage" ? "r" : ""}>
                {e.bee ? `${SHORT_NAME[e.bee]} ` : ""}
                {e.event === "deliver" ? "delivered a square" : e.event === "pickup" ? "picked up a square"
                  : e.event === "decoy" ? "grabbed a red decoy" : e.event}
              </span>
            </div>
          ))}
          {!frame.log.length && <div className="empty">Nothing yet.</div>}
        </div>
      </div>
    </div>
  );
}

function ScalePick({ results, scale, setScale }) {
  return (
    <div className="seg" title="All bees always start from the same weights. The comparison is repeated at each level.">
      {(results?.scales || []).map((k) => (
        <button key={k} className={scale === k ? "on" : ""} onClick={() => setScale(k)}>×{k}</button>
      ))}
    </div>
  );
}

const rivals = (results) => (results?.models ? Object.keys(results.models) : BEES).filter((k) => k !== FOCUS);

export function Verdict({ results, scale, setScale }) {
  if (!results?.by_scale || !results.focus) return <div className="empty">No 4-bee comparison yet. Start one in the Conditions tab.</div>;
  const opps = rivals(results);
  const tally = Object.fromEntries(opps.map((o) => [o, { win: 0, loss: 0, tie: 0 }]));
  const perScale = results.scales.map((k) => {
    const conds = results.by_scale[k].conditions;
    const avg = Object.fromEntries(BEES.map((m) => [m, 0]));
    for (const c of conds) {
      for (const o of opps) {
        const w = c.vs[o];
        tally[o][w === FOCUS ? "win" : w === o ? "loss" : "tie"] += 1;
      }
      for (const m of BEES) avg[m] += (c.metrics.deliveries.bees[m]?.mean || 0) / conds.length;
    }
    return [k, avg];
  });
  const cells = results.scales.length * (results.conditions_order?.length || 6);
  const where = {};
  for (const k of results.scales) for (const c of results.by_scale[k].conditions) {
    for (const [key, m] of Object.entries(c.metrics)) {
      if (m.higher === null) continue;
      where[key] = where[key] || { label: SHORT[key] || m.label, win: 0, loss: 0 };
      for (const o of opps) {
        const w = m.vs[o]?.winner;
        if (w === FOCUS) where[key].win += 1;
        else if (w === o) where[key].loss += 1;
      }
    }
  }
  const total = cells * opps.length;
  const strong = (side) => Object.values(where).filter((w) => w[side] >= Math.ceil(total / 2)).map((w) => `${w.label} (${w[side]}/${total})`);
  const cur = results.by_scale[scale] || results.by_scale[results.scales[0]];
  const ctl = cur.control || {};
  return (
    <>
      <div className="sub">SuperNeuron v2 against each rival · squares delivered · {cells} tests each</div>
      <div className="vs4">
        {opps.map((o) => (
          <div key={o} className="vsbox">
            <div className="vsname"><span style={tint(FOCUS)}>v2</span> vs <span style={tint(o)}>{SHORT_NAME[o]}</span></div>
            <div className="vsnums stat">
              <span style={tint(FOCUS)}>{tally[o].win}</span>
              <span className="dim">{tally[o].tie}</span>
              <span style={tint(o)}>{tally[o].loss}</span>
            </div>
            <div className="lbl">won · tie · lost</div>
          </div>
        ))}
      </div>
      <div className="box">
        <h4>Average squares delivered per match</h4>
        <table className="t">
          <colgroup><col style={{ width: "22%" }} />{BEES.map((m) => <col key={m} />)}</colgroup>
          <thead><tr><th>Weights</th>{BEES.map((m) => <th key={m} style={tint(m)}>{SHORT_NAME[m]}</th>)}</tr></thead>
          <tbody>
            {perScale.map(([k, avg]) => {
              const best = BEES.reduce((a, m) => (avg[m] > avg[a] ? m : a), BEES[0]);
              return (
                <tr key={k} className={`click ${k === scale ? "sel" : ""}`} onClick={() => setScale(k)}>
                  <td>×{k}</td>
                  {BEES.map((m) => <td key={m} className={`n ${m === best ? "lead" : ""}`} style={m === best ? tint(m) : undefined}>{fmt(avg[m], 1)}</td>)}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="box">
        <h4>Where v2 is clearly better / worse (at least half of all its tests)</h4>
        <p><span style={tint(FOCUS)}>Better:</span> {strong("win").join(", ") || "nothing consistent"}</p>
        <p><span className="r">Worse:</span> {strong("loss").join(", ") || "nothing consistent"}</p>
      </div>
      <div className="box">
        <h4>Does reward learning help? (Basic, weights ×{scale})</h4>
        <div className="kv">
          <span className="dim">Bee</span><span className="dim">rewards off → on</span><span className="dim">p</span>
          {BEES.map((m) => ctl[m] && [
            <span key={m + "a"} style={tint(m)}>{NAME[m]}</span>,
            <span key={m + "b"}>{fmt(ctl[m].learning_off, 1)} → {fmt(ctl[m].learning_on, 1)}</span>,
            <span key={m + "c"} className={ctl[m].p < 0.05 ? "g" : "dim"}>{fmt(ctl[m].p, 3)}</span>,
          ])}
        </div>
        <p className="dim small" style={{ marginTop: 6 }}>Squares delivered per match. Green p = rewards made a real difference.</p>
      </div>
      <div className="box">
        <h4>Fair-test rules</h4>
        <ul className="dim">
          <li>All bees share one arena, the same squares and spot; starts rotate between bees; who acts first rotates every step.</li>
          <li>Same fly wiring and the same start weights for every bee (no per-bee tuning), tested at ×{results.scales.join(", ×")}.</li>
          <li>Same senses, noise, rewards, babbling and the same learning rule: reward-gated STDP with a fast (250 ms) and a slow (2 s) memory, driven by reward surprise.</li>
          <li>Only the neuron differs. v2 = new SuperNeuron, v1 = old one, LIF = textbook, ALIF = adaptive LIF (Bellec 2018/2020).</li>
          <li>Paired test inside each match (sign-flip permutation, p &lt; 0.05). v2's activity brakes were set by activity checks, not task score.</li>
        </ul>
      </div>
      <p className="dim small" style={{ margin: "0 2px" }}>
        Made {results.made} · {results.minutes} brain-min × {results.seeds} matches per test · took {Math.round(results.wall_s / 60)} min
      </p>
    </>
  );
}

export function Compare({ results, frame, reload, scale, setScale }) {
  const [sel, setSel] = useState(null);
  const [minutes, setMinutes] = useState(60);
  const [seeds, setSeeds] = useState(10);
  const cmp = frame?.compare;
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && cmp && !cmp.running) reload();
    wasRunning.current = !!cmp?.running;
  }, [cmp, reload]);

  const block = results?.focus ? results?.by_scale?.[scale] : null;
  const conds = block?.conditions || [];
  const cur = conds.find((c) => c.condition === sel) || conds[0];
  const opps = rivals(results);
  return (
    <>
      <div className="toolbar">
        <button className="btn" disabled={cmp?.running} onClick={() => post({ cmd: "compare", minutes, seeds })}>
          {cmp?.running ? "Running…" : "Run comparison"}
        </button>
        <label className="dim small">brain-min</label>
        <input className="num" type="number" min={2} max={240} value={minutes} onChange={(e) => setMinutes(+e.target.value)} />
        <label className="dim small">matches</label>
        <input className="num" type="number" min={2} max={30} value={seeds} onChange={(e) => setSeeds(+e.target.value)} />
        <span className="dim small" style={{ marginLeft: "auto" }}>start weights</span>
        <ScalePick results={results} scale={scale} setScale={setScale} />
      </div>
      {cmp?.running && (
        <div style={{ marginBottom: 8 }}>
          <div className="progress"><div style={{ width: `${(100 * cmp.progress) / Math.max(1, cmp.total)}%` }} /></div>
          <div className="dim stat small" style={{ marginTop: 2 }}>{cmp.progress}/{cmp.total} · {cmp.last}</div>
        </div>
      )}
      {cmp?.error && <p className="r">{cmp.error}</p>}
      {!block ? <div className="empty">No 4-bee comparison results yet.</div> : (
        <>
          <table className="t">
            <colgroup><col style={{ width: "17%" }} />{BEES.map((m) => <col key={m} />)}{opps.map((o) => <col key={o} style={{ width: "11%" }} />)}</colgroup>
            <thead>
              <tr>
                <th>Condition</th>
                {BEES.map((m) => <th key={m} style={tint(m)}>{SHORT_NAME[m]}</th>)}
                {opps.map((o) => <th key={o}>vs {SHORT_NAME[o]}</th>)}
              </tr>
            </thead>
            <tbody>
              {conds.map((c) => {
                const d = c.metrics.deliveries;
                return (
                  <tr key={c.condition} className={`click ${cur?.condition === c.condition ? "sel" : ""}`} onClick={() => setSel(c.condition)} title={c.about}>
                    <td>{c.label}</td>
                    {BEES.map((m) => <td key={m} className="n">{fmt(d.bees[m]?.mean, 1)} <span className="sd">±{fmt(d.bees[m]?.sd, 0)}</span></td>)}
                    {opps.map((o) => <td key={o} title={`p = ${d.vs[o]?.p}`}><Badge k={d.vs[o]?.winner} /></td>)}
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="dim small" style={{ margin: "6px 2px" }}>
            Squares delivered per {results.minutes}-min match (mean ± sd, {results.seeds} matches), weights ×{scale}.
            “vs” columns = winner of v2 against that bee (p &lt; 0.05), else tie. Click a row.
          </p>
          {cur && (
            <div className="split">
              <div>
                <div className="sub">{cur.label}: deliveries per minute while learning</div>
                <LineChart height={150} xLabel="brain minute" series={[
                  ...BEES.filter((m) => cur.curve[m]).map((m) => ({ name: SHORT_NAME[m], color: COLOR[m], points: cur.curve[m].map((v, i) => [i + 1, v]) })),
                  ...(cur.condition === "basic" && block.control ? BEES.filter((m) => block.control[m]).map((m) => ({
                    name: `${SHORT_NAME[m]} off`, color: COLOR[m], dash: true,
                    points: block.control[m].curve_off.map((v, i) => [i + 1, v]),
                  })) : []),
                ]} />
              </div>
              <div>
                <table className="t">
                  <colgroup><col style={{ width: "32%" }} />{BEES.map((m) => <col key={m} />)}<col style={{ width: "15%" }} /></colgroup>
                  <thead><tr><th>{cur.label}</th>{BEES.map((m) => <th key={m} style={tint(m)}>{SHORT_NAME[m]}</th>)}<th>v2 beats</th></tr></thead>
                  <tbody>
                    {Object.entries(cur.metrics).map(([k, m]) => {
                      const beats = opps.filter((o) => m.vs[o]?.winner === FOCUS).length;
                      const loses = opps.filter((o) => m.vs[o]?.winner === o).length;
                      return (
                        <tr key={k} title={`${m.label} · ` + opps.map((o) => `vs ${SHORT_NAME[o]} p=${m.vs[o]?.p}`).join(", ")}>
                          <td>{SHORT[k] || m.label}{m.higher === false ? <span className="dim"> ↓</span> : null}</td>
                          {BEES.map((b) => {
                            const v = m.bees[b]?.mean;
                            return <td key={b} className="n">{fmt(v, v > 100 ? 0 : 2)}</td>;
                          })}
                          <td className="n">{m.higher === null ? <span className="dim">info</span>
                            : <><span style={tint(FOCUS)}>{beats}</span><span className="dim">/{opps.length}</span>{loses ? <span className="r"> −{loses}</span> : null}</>}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className="dim small" style={{ margin: "6px 2px" }}>↓ = lower is better · “v2 beats” = rivals v2 beats on this metric (p &lt; 0.05), −n = rivals that beat v2. {cur.about}.</p>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}
