"use client";
import { useEffect, useState } from "react";
import Panel from "@/components/Panel";
import Arena from "@/components/Arena";
import BrainView from "@/components/BrainView";
import { Activity, Compare, Race, Score, Senses, Verdict, Wiring } from "@/components/panels";
import { BEES, COLOR, SHORT, clock, fmt, post, useJson, useStream } from "@/lib/sim";

const TABS = ["Brain", "Activity", "Senses", "Wiring", "Score", "Race", "Verdict", "Conditions"];

/** One half of the right side: pick a bee, then its brain tabs, plus the shared comparison tabs. */
function BeePanel({ start, startBee, frame, meta, results, reload, scale, setScale, className }) {
  const [tab, setTab] = useState(start);
  const [kind, setKind] = useState(startBee);
  const bee = frame?.bees?.[kind];
  const picker = (
    <span className="picker" title="Which bee this panel shows">
      {BEES.map((k) => (
        <button key={k} className={k === kind ? "on" : ""} onClick={() => setKind(k)}
          style={k === kind ? { color: COLOR[k], borderColor: COLOR[k], boxShadow: `0 0 8px ${COLOR[k]}` } : { color: COLOR[k] }}>
          {SHORT[k]}
        </button>
      ))}
    </span>
  );
  return (
    <Panel className={className} title={picker} tabs={TABS} tab={tab} onTab={setTab} flush={tab === "Brain"}>
      {tab === "Brain" && <BrainView bee={bee} kind={kind} meta={meta} version={frame?.version} />}
      {tab === "Activity" && <Activity bee={bee} kind={kind} meta={meta} />}
      {tab === "Senses" && <Senses bee={bee} meta={meta} />}
      {tab === "Wiring" && <Wiring bee={bee} kind={kind} />}
      {tab === "Score" && <Score frame={frame} />}
      {tab === "Race" && <Race frame={frame} />}
      {tab === "Verdict" && <Verdict results={results} scale={scale} setScale={setScale} />}
      {tab === "Conditions" && <Compare results={results} frame={frame} reload={reload} scale={scale} setScale={setScale} />}
    </Panel>
  );
}

export default function Home() {
  const { frame, ok } = useStream();
  const [meta] = useJson("/meta", [ok]);
  const [results, reload] = useJson("/results", [ok]);
  const [seed, setSeed] = useState(1);
  const [scale, setScale] = useState("2");                 // start-weight level shown in Verdict / Conditions
  useEffect(() => {
    if (results?.scales && !results.scales.includes(scale)) setScale(results.scales[0]);
  }, [results, scale]);

  const conds = meta?.conditions || {};
  const speeds = meta?.speeds || [1, 5, 20, 100];
  const reset = (over) => post({ cmd: "reset", condition: frame?.condition || "basic", seed, scale: frame?.scale ?? 2, ...over });
  const shared = { frame, meta, results, reload, scale, setScale };
  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><b>BEE</b><i>LAB</i></div>
        <span className="dim tagline">
          <span className="y">Zneuron v2</span> vs v1, LIF, ALIF · one arena · fair test
        </span>
        <span className="sep" />
        <div className="group">
          <label>Condition</label>
          <select className="sel" value={frame?.condition || "basic"} onChange={(e) => reset({ condition: e.target.value })}>
            {Object.entries(conds).map(([k, c]) => <option key={k} value={k}>{c.label}</option>)}
          </select>
        </div>
        <div className="group" title="Start weights, the same for both bees">
          <label>Weights</label>
          <select className="sel" value={frame?.scale ?? 2} onChange={(e) => reset({ scale: +e.target.value })}>
            {(meta?.scales || [1, 2, 3]).map((k) => <option key={k} value={k}>×{k}</option>)}
          </select>
        </div>
        <div className="group">
          <label>Seed</label>
          <input className="num" type="number" min={1} value={seed} onChange={(e) => setSeed(+e.target.value || 1)} />
          <button className="btn" onClick={() => reset({})}>Restart</button>
        </div>
        <div className="group">
          <label>Speed ×</label>
          <div className="seg">
            {speeds.map((s) => (
              <button key={s} className={frame?.speed === s ? "on" : ""} onClick={() => post({ cmd: "speed", value: s })}>{s}</button>
            ))}
          </div>
          <button className={`btn ${frame?.paused ? "on" : ""}`} onClick={() => post({ cmd: "pause" })}>
            {frame?.paused ? "▶ Play" : "❚❚ Pause"}
          </button>
        </div>
        <span className="grow" />
        <div className="group stat" title="brain time d:hh:mm:ss · real speed reached">
          <span className="c">{clock(frame?.t || 0)}</span>
          <span className="dim">{fmt(frame?.real_x, 1)}×</span>
        </div>
        <div className="group" title={ok ? "connected to the simulation" : "start: .venv/Scripts/python.exe run_bees.py"}>
          <span className={`dot ${ok ? "ok" : ""}`} />
          <span className="dim">{ok ? "live" : "no sim"}</span>
        </div>
      </header>

      <div className="grid">
        <Panel className="p-arena" title="Arena" flush
          right={<span className="dim small">{conds[frame?.condition]?.about}</span>}>
          <Arena frame={frame} />
        </Panel>
        <BeePanel startBee="hub2" className="p-top" start="Brain" {...shared} />
        <BeePanel startBee="alif" className="p-bottom" start="Brain" {...shared} />
      </div>
    </main>
  );
}
