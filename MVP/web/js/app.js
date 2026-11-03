// App shell: Unity-style docking panels (dockview), live stream from the brain, top-bar controls.
import { makePanels } from './panels.js';
import { GROUP_COLORS } from './util.js';

const { createDockview, themeAbyss } = window['dockview-core'];
const $ = id => document.getElementById(id);
const LAYOUT_KEY = 'superneuron.layout.v2';

const app = {
  F: null, I: {}, groupOf: new Map(), views: new Map(),
  send: cmd => fetch('/control', { method: 'POST', body: JSON.stringify(cmd) }),
};
const PANELS = makePanels(app);

// ---------- docking ----------
const dock = createDockview($('dock'), {
  theme: themeAbyss,
  createComponent: ({ id, name }) => {
    const spec = PANELS[name];
    const view = spec.make();
    app.views.set(id, view);
    return { element: view.el, init() {}, dispose() { app.views.delete(id); } };
  },
});

// default layout, following the sketch: world | brain | side panels, and output | chat underneath
function defaultLayout() {
  dock.clear();
  const add = (id, position) => dock.addPanel({ id, component: id, title: PANELS[id].title, position });
  add('world');
  add('brain', { referencePanel: 'world', direction: 'right' });
  add('alarms', { referencePanel: 'brain', direction: 'right' });
  for (const id of ['senses', 'motors', 'thinking', 'inspector', 'why', 'learned', 'drives']) add(id, { referencePanel: 'alarms', direction: 'within' });
  dock.getPanel('alarms').api.setActive();
  add('output', { referencePanel: 'world', direction: 'below' });
  add('chat', { referencePanel: 'output', direction: 'right' });
  try {
    dock.getPanel('world').group.api.setSize({ width: Math.round(window.innerWidth * 0.3) });
    dock.getPanel('brain').group.api.setSize({ width: Math.round(window.innerWidth * 0.42) });
    dock.getPanel('world').group.api.setSize({ height: Math.round(window.innerHeight * 0.6) });
  } catch { /* sizes are a nicety */ }
}

let restored = false;
try {
  const saved = localStorage.getItem(LAYOUT_KEY);
  if (saved) { dock.fromJSON(JSON.parse(saved)); restored = dock.panels.length > 0; }
} catch { restored = false; }
if (!restored) defaultLayout();
dock.onDidLayoutChange(() => { try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(dock.toJSON())); } catch { /* private mode */ } });

function toggleMaximize() {
  const panel = dock.activePanel; if (!panel) return;
  if (panel.api.isMaximized()) panel.api.exitMaximized(); else panel.api.maximize();
}
$('maximize').onclick = toggleMaximize;
$('resetLayout').onclick = () => { try { localStorage.removeItem(LAYOUT_KEY); } catch { /* ignore */ } defaultLayout(); };

// ---------- controls ----------
$('pause').onclick = () => app.send({ cmd: 'pause' });
$('save').onclick = () => { const name = prompt('Save this brain as:', `brain-${Math.floor((app.F?.t || 0) / 1000)}s`); if (name) app.send({ cmd: 'save', name }); };
$('load').onchange = e => { if (e.target.value && confirm(`Load '${e.target.value}'? The current brain is replaced.`)) app.send({ cmd: 'load', name: e.target.value }); e.target.value = ''; };
$('step').onclick = () => app.send({ cmd: 'step', ticks: 1 });
$('speed').onchange = e => app.send({ cmd: 'speed', value: +e.target.value });
document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
  if (e.key === ' ') { e.preventDefault(); app.send({ cmd: 'pause' }); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); app.send({ cmd: 'step', ticks: 1 }); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); app.send({ cmd: 'faster' }); }
  else if (e.key === 'ArrowDown') { e.preventDefault(); app.send({ cmd: 'slower' }); }
  else if (e.key === 'f' || e.key === 'F') toggleMaximize();
});
document.addEventListener('click', e => {
  const id = e.target.dataset && e.target.dataset.nid;
  if (id !== undefined) app.send({ cmd: 'select', id: +id });
});

// brain time as days:hours:minutes:seconds (ms in the tooltip)
function clock(ms) {
  const s = Math.floor(ms / 1000), p2 = n => String(n).padStart(2, '0');
  $('t').title = `${ms.toLocaleString()} ms`;
  return `${Math.floor(s / 86400)}:${p2(Math.floor(s / 3600) % 24)}:${p2(Math.floor(s / 60) % 60)}:${p2(s % 60)}`;
}

// ---------- stream ----------
let lastT = null, lastWall = null, realSpeed = null;
function connect() {
  const es = new EventSource('/stream');
  es.onopen = () => { $('conn').textContent = 'live'; $('conn').style.color = '#3fb950'; };
  es.onerror = () => { $('conn').textContent = 'disconnected — is run_live.py running?'; $('conn').style.color = '#f85149'; };
  es.onmessage = e => {
    const F = JSON.parse(e.data);
    const now = performance.now();
    if (lastT !== null && now - lastWall > 500) {
      realSpeed = (F.t - lastT) / (now - lastWall);   // simulated ms per real ms
      lastT = F.t; lastWall = now;
    } else if (lastT === null) { lastT = F.t; lastWall = now; }
    app.F = F;
    if (F.insight && F.insight.alarms) app.I = F.insight;
  };
}

// ---------- draw loop ----------
let lastPanels = 0;
function frame(now) {
  const F = app.F;
  if (F) {
    for (const v of app.views.values()) v.frame?.();
    if (now - lastPanels > 100) {
      lastPanels = now;
      app.groupOf = new Map();
      (app.I.groups || []).forEach((g, i) => g.members.forEach(m => app.groupOf.set(m, GROUP_COLORS[i % GROUP_COLORS.length])));
      for (const v of app.views.values()) { try { v.update(F, app.I); } catch (err) { console.error(err); } }
      $('t').textContent = clock(F.t);
      const ld = $('load'), names = ['', ...(F.saves || [])];
      if (ld.options.length !== names.length) { ld.innerHTML = ''; names.forEach(n => ld.add(new Option(n || 'Load brain…', n))); }
      $('nneurons').textContent = F.neurons.filter(n => n[11]).length;
      $('nlines').textContent = F.lines.length;
      $('nawake').textContent = `${F.neurons.filter(n => n[7]).length}/${F.neurons.length}`;
      $('realspeed').textContent = F.paused ? 'paused' : realSpeed === null ? '–' : `${realSpeed.toFixed(realSpeed < 1 ? 2 : realSpeed < 10 ? 1 : 0)}×`;
      $('pause').textContent = F.paused ? 'Play' : 'Pause';
      const sp = $('speed');
      if (!sp.options.length) F.speeds.forEach(v => sp.add(new Option(`${v}×`, v)));
      if (document.activeElement !== sp) sp.value = F.speed;
    }
  }
  requestAnimationFrame(frame);
}
connect();
requestAnimationFrame(frame);
