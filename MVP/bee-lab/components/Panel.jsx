"use client";
import { useEffect, useState } from "react";

/** A dock panel: title, optional tabs, a maximise button; the body scrolls, the page never does. */
export default function Panel({ className = "", title, tabs, tab, onTab, flush, right, children }) {
  const [max, setMax] = useState(false);
  useEffect(() => {
    if (!max) return;
    const k = (e) => e.key === "Escape" && setMax(false);
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [max]);
  return (
    <section className={`panel ${className} ${max ? "max" : ""}`}>
      <header className="phead">
        <span className="ptitle">{title}</span>
        {tabs && (
          <div className="tabs">
            {tabs.map((t) => (
              <button key={t} className={t === tab ? "on" : ""} onClick={() => onTab(t)}>{t}</button>
            ))}
          </div>
        )}
        {right && <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>{right}</div>}
        <button className="pbtn" style={right ? { marginLeft: 8 } : undefined} title={max ? "Restore (Esc)" : "Full screen"}
          onClick={() => setMax(!max)}>{max ? "▣" : "⛶"}</button>
      </header>
      <div className={`pbody ${flush ? "flush" : ""}`}>{children}</div>
    </section>
  );
}
