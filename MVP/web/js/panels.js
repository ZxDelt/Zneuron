// Panel contents. Each factory returns { el, update(F, I), frame?() }.
//  F = latest frame from the brain, I = latest insight (updated a few times a second).
import { Brain3D } from './brain3d.js';
import { World3D } from './world3d.js';
import { C, GROUP_COLORS, fit, esc, chip } from './util.js';

const div = (cls, html = '') => { const d = document.createElement('div'); d.className = cls; d.innerHTML = html; return d; };

export function makePanels(app) {
  return {
    world: { title: 'Body & World', make: () => {
      const el = div('pane fill');
      let view = null, last = null;
      const note = div('overlay', 'drag = orbit around the robot · wheel = zoom · the camera follows the robot');
      const empty = div('placeholder', '<div>No body: started with --no-body.</div>');
      return { el, update() {}, frame() {
        const F = app.F;
        if (!F) return;
        if (!F.world) { if (!empty.parentNode) el.replaceChildren(empty); return; }
        if (!view) { view = new World3D(el); el.appendChild(note); }
        if (F !== last) { last = F; view.update(F.world); }
        view.render();
      } };
    } },

    senses: { title: 'Senses', make: () => {
      const el = div('pane');
      el.innerHTML = '<h3>Eyes (left · right) — what each retina cell sees</h3><canvas class="plot" style="height:120px"></canvas>'
        + '<h3>Ears (left · right) — pitch bands, low → high (speech = 1–6)</h3><canvas class="plot" style="height:70px"></canvas>'
        + '<h3>Skin</h3><div data-k="skin"></div><h3>Rewards (automatic + yours)</h3><div data-k="rewards"></div>';
      const [eyes, ears] = el.querySelectorAll('canvas'), skin = el.querySelector('[data-k=skin]'), rew = el.querySelector('[data-k=rewards]');
      const SKIN = ['left hand', 'right hand', 'left foot', 'right foot', 'chest', 'back'];
      return { el, update(F) {
        const W = F.world;
        if (!W) { skin.innerHTML = '<span class="sub">No body.</span>'; return; }
        drawRetina(eyes, W.senses.eyes);
        drawEars(ears, W.senses.ears);
        skin.innerHTML = SKIN.map((n, i) => `<span class="chip" style="${W.senses.skin[i] > 0.5 ? 'border-color:#ffd33d;color:#ffd33d' : ''}">${n}</span>`).join('');
        rew.innerHTML = F.rewards.length ? F.rewards.slice().reverse().map(([t, v, why]) =>
          `<div class="row"><span class="sub">${(t / 1000).toFixed(1)} s</span> <span class="${v > 0 ? 'plus' : 'minus'}">${v > 0 ? '+' : ''}${v}</span> ${esc(why)}</div>`).join('')
          : '<div class="sub">No rewards yet. They come from picking flowers (+), getting closer to one (+), bumping rocks (−), and your 👍/👎.</div>';
      } };
    } },

    motors: { title: 'Motors', make: () => {
      const el = div('pane');
      return { el, update(F) {
        const W = F.world;
        if (!W) { el.innerHTML = '<span class="sub">No body.</span>'; return; }
        el.innerHTML = '<h3>What the motor neurons are doing (−1 … +1)</h3>' + Object.entries(W.motors).map(([name, v]) => {
          const pct = Math.abs(v) * 50, left = v < 0 ? 50 - pct : 50;
          return `<div class="meter"><span>${name}</span><div class="track" style="position:relative">
            <div class="fill" style="position:absolute;left:${left}%;width:${pct}%;background:${v >= 0 ? '#3fb950' : '#f85149'}"></div>
            <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#8b949e"></div></div><span class="num">${v.toFixed(2)}</span></div>`;
        }).join('') + '<div class="sub" style="margin-top:8px">Each row is a pair of motor neurons pulling opposite ways. '
          + 'They babble (fire randomly ~10 times/s, spec #32) so the body moves and has something to learn from.</div>'
          + `<h3>Robot</h3><div class="row">position ${W.robot.x.toFixed(1)}, ${W.robot.z.toFixed(1)} m · heading ${(W.robot.heading * 57.3 % 360).toFixed(0)}° · `
          + `speed ${W.robot.speed.toFixed(2)} m/s · ${W.daylight > 0.05 ? 'day' : 'night'}</div>`;
      } };
    } },

    brain: { title: 'Brain 3D', make: () => {
      const el = div('pane fill');
      const view = new Brain3D(el, id => app.send({ cmd: 'select', id }));
      const note = div('overlay', 'drag = rotate · right-drag = pan · wheel = zoom · click a neuron to inspect it · '
        + 'blue = sensor · orange = motor · green/red = excitatory/inhibitory');
      el.appendChild(note);
      let last = null;
      return { el, update() {}, frame() {
        if (app.F && app.F !== last) { last = app.F; view.update(app.F, app.groupOf); }
        view.render();
      } };
    } },

    output: { title: 'Output — brain state', make: () => {
      const el = div('pane teller');
      return { el, update(F, I) {
        if (!I.teller) return;
        el.innerHTML = '<h3>What the brain is doing</h3>' + I.teller.map(line =>
          `<p class="${line.startsWith('⚠') ? 'warn' : ''}">${esc(line)}</p>`).join('')
          + '<p class="sub">This is a readout of the brain\'s state, not the brain talking.</p>';
      } };
    } },

    chat: { title: 'Chat', make: () => {
      const el = div('pane fill');
      el.innerHTML = `<div class="chat"><div class="log"></div>
        <form><input placeholder="Talk to the brain… (type good / bad to reward)" autocomplete="off">
        <button type="submit">Send</button><button type="button" data-r="1" title="Reward +1">👍</button>
        <button type="button" data-r="-1" title="Reward −1">👎</button></form></div>`;
      const log = el.querySelector('.log'), input = el.querySelector('input');
      el.querySelector('form').onsubmit = e => {
        e.preventDefault();
        if (input.value.trim()) app.send({ cmd: 'chat', text: input.value });
        input.value = '';
      };
      el.querySelectorAll('[data-r]').forEach(b => b.onclick = () => app.send({ cmd: 'reward', value: +b.dataset.r }));
      let shown = '';
      return { el, update(F) {
        const key = JSON.stringify(F.chat);
        if (key === shown) return;
        shown = key;
        const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
        log.innerHTML = F.chat.length ? F.chat.map(([who, text, t]) =>
          `<div class="msg ${who}"><span class="when">${who} · ${(t / 1000).toFixed(1)} s</span>${esc(text)}</div>`).join('')
          : '<div class="sub">No messages yet. Rewards: 👍 / 👎 or type good / bad.</div>';
        if (atBottom) log.scrollTop = log.scrollHeight;
      } };
    } },

    alarms: { title: 'Alarms', make: () => {
      const el = div('pane');
      return { el, update(F, I) {
        if (I.alarms) el.innerHTML = I.alarms.map(([lvl, msg]) => `<div class="alarm ${lvl}">${esc(msg)}</div>`).join('');
      } };
    } },

    thinking: { title: 'Thinking', make: () => {
      const el = div('pane');
      const meter = (label, v, max, text) => `<div class="meter"><span>${label}</span><div class="track">
        <div class="fill" style="width:${Math.min(100, 100 * v / Math.max(1, max))}%"></div></div><span class="num">${text}</span></div>`;
      return { el, update(F, I) {
        if (!I.capacity) return;
        const cap = I.capacity, n = F.counts || {};
        el.innerHTML = '<h3>Capacity</h3>'
          + meter('Awake neurons', cap.awake, cap.neurons, `${cap.awake}/${cap.neurons}`)
          + meter('Line space used', cap.lines, cap.line_slots, `${cap.lines}/${cap.line_slots}`)
          + meter('Thoughts at once', cap.groups, cap.max_groups, `${cap.groups}/${cap.max_groups}`)
          + `<div class="row sub">Since start: <b class="plus">${n.grow || 0}</b> grown · <b class="minus">${n.prune || 0}</b> pruned ·
             <b style="color:${C.lock}">${n.lock || 0}</b> locked · net <b>${(n.grow || 0) - (n.prune || 0)}</b></div>`
          + '<h3>Active groups</h3>' + (I.groups.length ? I.groups.map((g, i) =>
            `<div class="row"><b style="color:${GROUP_COLORS[i % GROUP_COLORS.length]}">${g.name}</b> ${g.members.map(m => chip(m)).join('')}</div>`).join('')
            : '<div class="sub">No group is firing together right now.</div>')
          + '<h3>Live chains (last 60 ms)</h3>' + (I.chains.length
            ? I.chains.map(ch => `<div class="row">${ch.map(id => chip(id)).join(' → ')}</div>`).join('')
            : '<div class="sub">No cause-and-effect chains right now.</div>');
      } };
    } },

    inspector: { title: 'Neuron', make: () => {
      const el = div('pane');
      el.innerHTML = '<div class="row" data-k="title"></div><canvas class="plot" style="height:140px"></canvas>'
        + '<canvas class="plot" style="height:110px;margin-top:8px"></canvas>'
        + '<div class="sub">yellow = hub charge · dashed = threshold · dotted = pool capacity</div>';
      const [bars, trace] = el.querySelectorAll('canvas'), title = el.querySelector('[data-k=title]');
      return { el, update(F) {
        const d = F.detail;
        const kind = d.sensor ? 'sensor' : F.neurons[d.id][10] ? 'motor' : d.sign > 0 ? 'excitatory' : 'inhibitory';
        title.innerHTML = `<b>Neuron ${d.id}</b> · ${kind} · ${d.state} · threshold ${d.thr} · window ${d.window === null ? 'off' : d.window + ' ms'}
          · ${d.pools.length} lines · ${d.rate} Hz · local Rfuel ${d.rfuel}`;
        drawBars(bars, d); drawTrace(trace, F);
      } };
    } },

    why: { title: 'Why it fired', make: () => {
      const el = div('pane');
      return { el, update(F, I) { if ('why' in I) el.innerHTML = `<h3>Neuron ${F.selected}, last spike</h3>` + whyHtml(I.why, true); } };
    } },

    learned: { title: 'Learned', make: () => {
      const el = div('pane');
      return { el, update(F, I) {
        const L = I.learned; if (!L) return;
        const row = ([s, d, v]) => `<div class="row">${chip(s)}→${chip(d)} <span class="${v > 0 ? 'plus' : 'minus'}">${v > 0 ? '+' : ''}${v.toFixed(3)}</span></div>`;
        el.innerHTML = '<div class="cols"><div><h3>Strengthened (2 s)</h3>' + (L.up.map(row).join('') || '<div class="sub">nothing</div>')
          + '</div><div><h3>Weakened (2 s)</h3>' + (L.down.map(row).join('') || '<div class="sub">nothing</div>') + '</div></div>'
          + '<h3>Grow / prune / lock (10 s)</h3>' + (L.events.length ? L.events.slice().reverse().map(([t, kind, s, d]) =>
            `<div class="row"><span class="sub">${t} ms</span> ${kind} ${chip(s)}→${chip(d)}</div>`).join('') : '<div class="sub">none</div>');
      } };
    } },

    drives: { title: 'Drives', make: () => {
      const el = div('pane');
      el.innerHTML = '<canvas class="plot" style="height:110px"></canvas><div class="sub">'
        + '<span class="plus">Lfuel</span> = reward · <span style="color:#58a6ff">Rfuel</span> = calm (threshold ×) · '
        + '<span style="color:#d2a8ff">Cfuel</span> = curiosity (learning speed)</div>';
      const cv = el.querySelector('canvas');
      return { el, update(F, I) { drawDrives(cv, I.drives || []); } };
    } },
  };
}

function drawRetina(canvas, eyes) {
  const [ctx, w, h] = fit(canvas);
  const size = Math.min(h, (w - 20) / 2) / eyes[0].length;
  eyes.forEach((eye, e) => eye.forEach((row, r) => row.forEach((v, c) => {
    const g = Math.round(255 * v);
    ctx.fillStyle = `rgb(${g},${g},${Math.min(255, g + 20)})`;
    ctx.fillRect(e * (size * row.length + 20) + c * size, r * size, size - 1, size - 1);
  })));
}

function drawEars(canvas, ears) {
  const [ctx, w, h] = fit(canvas);
  const nb = ears[0].length, bw = (w - 20) / (2 * nb);
  ears.forEach((ear, e) => ear.forEach((v, b) => {
    ctx.fillStyle = C.blue; ctx.fillRect(e * (bw * nb + 20) + b * bw + 2, h - 12 - (h - 16) * v, bw - 4, (h - 16) * v);
    ctx.fillStyle = C.muted; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
    ctx.fillText(`${e ? 'R' : 'L'}${b}`, e * (bw * nb + 20) + b * bw + bw / 2, h - 2);
  }));
}

function drawBars(canvas, d) {
  const [ctx, w, h] = fit(canvas);
  const items = [['hub', d.hub, C.spike]]
    .concat(d.pools.map(([from, v, th]) => [`←${from}`, v, th > 0 ? C.exc : th < 0 ? C.inh : '#484f58']))
    .concat([['Efuel', d.efuel, C.blue], ['Cfuel', d.cfuel, C.lock]]);
  const top = 6, bottom = 18, H = h - top - bottom, bw = w / items.length;
  const y = v => top + H * (1 - Math.min(1, v));
  items.forEach(([label, v, col], i) => {
    ctx.fillStyle = col; ctx.fillRect(i * bw + 4, y(v), bw - 8, H * Math.min(1, v));
    ctx.fillStyle = C.muted; ctx.font = '10px system-ui'; ctx.textAlign = 'center'; ctx.fillText(label, i * bw + bw / 2, h - 4);
  });
  ctx.strokeStyle = '#fff'; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(0, y(d.thr)); ctx.lineTo(w, y(d.thr)); ctx.stroke();
  if (d.cap) { ctx.strokeStyle = C.muted; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(bw, y(d.cap)); ctx.lineTo(bw * (1 + d.pools.length), y(d.cap)); ctx.stroke(); }
  ctx.setLineDash([]);
}

function drawTrace(canvas, F) {
  const [ctx, w, h] = fit(canvas);
  const tr = F.trace; if (!tr.length) return;
  const t0 = tr[0][0], span = Math.max(10, tr[tr.length - 1][0] - t0);
  const X = t => (t - t0) / span * w, Y = v => h - 4 - (h - 8) * Math.min(1, v);
  for (const [t, id] of F.raster) if (id === F.selected && t >= t0) { ctx.fillStyle = 'rgba(255,211,61,.25)'; ctx.fillRect(X(t), 0, 2, h); }
  const line = (k, col, dash) => {
    ctx.strokeStyle = col; ctx.setLineDash(dash); ctx.lineWidth = 1.4; ctx.beginPath();
    tr.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[k])) : ctx.moveTo(X(p[0]), Y(p[k]))); ctx.stroke();
  };
  line(2, '#fff', [5, 4]); line(1, C.spike, []); ctx.setLineDash([]);
}

function drawDrives(canvas, d) {
  const [ctx, w, h] = fit(canvas);
  if (d.length < 2) return;
  const t0 = d[0][0], span = Math.max(1, d[d.length - 1][0] - t0);
  const X = t => (t - t0) / span * w, Y = v => h / 2 - (h / 2 - 4) * Math.max(-1, Math.min(1, (v - 1)));
  ctx.strokeStyle = C.line; ctx.beginPath(); ctx.moveTo(0, Y(1)); ctx.lineTo(w, Y(1)); ctx.stroke();
  [[1, C.exc], [2, C.blue], [3, C.lock]].forEach(([k, col]) => {
    ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath();
    d.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[k])) : ctx.moveTo(X(p[0]), Y(p[k]))); ctx.stroke();
  });
}

function whyHtml(node, root) {
  if (!node) return '<span class="sub">This neuron has not fired yet.</span>';
  const head = root ? `${chip(node.id)} fired at ${node.t} ms` : '';
  const items = (node.causes || []).map(c => {
    if (c.outside) return `<li><span class="out">← outside input</span> +${c.charge.toFixed(2)} at ${c.t} ms</li>`;
    const sign = c.charge >= 0 ? `<span class="plus">+${c.charge.toFixed(2)}</span>` : `<span class="minus">${c.charge.toFixed(2)}</span>`;
    const more = c.forgotten ? ' <span class="sub">(older history not kept)</span>' : whyHtml(c, false);
    return `<li>← ${chip(c.id)} ${sign} (sent at ${c.t} ms)${more}</li>`;
  }).join('');
  return head + (items ? `<ul class="why${root ? ' root' : ''}">${items}</ul>` : (root ? '<div class="sub">No inputs recorded.</div>' : ''));
}
