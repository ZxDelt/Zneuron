"use client";
import { useEffect, useRef, useState, useCallback } from "react";

export const SIM = process.env.NEXT_PUBLIC_SIM || "http://127.0.0.1:8766";
export const BEES = ["hub2", "hub1", "lif", "alif"];
export const COLOR = { hub2: "#ffe14d", hub1: "#b388ff", lif: "#ff6fd8", alif: "#ff9f43" };
export const RGB = { hub2: "255, 225, 77", hub1: "179, 136, 255", lif: "255, 111, 216", alif: "255, 159, 67" };
export const NAME = { hub2: "Zneuron v2", hub1: "Zneuron v1", lif: "LIF (textbook)", alif: "ALIF (adaptive)" };
export const SHORT = { hub2: "v2", hub1: "v1", lif: "LIF", alif: "ALIF" };
export const FOCUS = "hub2";

export function post(body) {
  return fetch(`${SIM}/control`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    .catch(() => null);
}

/** Live frames over Server-Sent Events, reconnecting on its own. */
export function useStream() {
  const [frame, setFrame] = useState(null);
  const [ok, setOk] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    let es, timer, dead = false;
    const open = () => {
      es = new EventSource(`${SIM}/stream`);
      es.onopen = () => setOk(true);
      es.onmessage = (e) => {
        const f = JSON.parse(e.data);
        ref.current = f;
        setFrame(f);
      };
      es.onerror = () => {
        setOk(false);
        es.close();
        if (!dead) timer = setTimeout(open, 1500);
      };
    };
    open();
    return () => { dead = true; clearTimeout(timer); es && es.close(); };
  }, []);
  return { frame, ok, ref };
}

export function useJson(path, deps = []) {
  const [data, setData] = useState(null);
  const load = useCallback(() => {
    fetch(`${SIM}${path}`).then((r) => r.json()).then(setData).catch(() => {});
  }, [path]);
  useEffect(() => { load(); }, [load, ...deps]); // eslint-disable-line react-hooks/exhaustive-deps
  return [data, load];
}

/** Element size, for charts and canvases that draw at real pixels. */
export function useSize() {
  const ref = useRef(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => {
      const r = e.contentRect;
      setSize({ w: Math.floor(r.width), h: Math.floor(r.height) });
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

export const fmt = (v, d = 1) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (a >= 1e5) return `${(v / 1e3).toFixed(0)}k`;
  return Number(v).toFixed(d);
};
export const clock = (ms) => {
  const s = Math.floor(ms / 1000);
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  const p = (x) => String(x).padStart(2, "0");
  return `${d}:${p(h)}:${p(m)}:${p(s % 60)}`;
};
