// Small shared helpers for the app panels.

// charge colour: dark purple -> red -> orange -> pale yellow (0..1)
const HEAT = [[20, 11, 52], [120, 28, 109], [229, 92, 48], [252, 255, 164]];
export function heat(v) {
  v = Math.max(0, Math.min(1, v || 0)) * (HEAT.length - 1);
  const i = Math.min(HEAT.length - 2, Math.floor(v)), f = v - i;
  return HEAT[i].map((c, k) => Math.round(c + (HEAT[i + 1][k] - c) * f));
}

export const C = { exc: '#3fb950', inh: '#f85149', spike: '#ffd33d', lock: '#d2a8ff', blue: '#58a6ff',
  muted: '#8b949e', line: '#30363d', bg: '#0d1117', ink: '#e6edf3', motor: '#ffa657' };
export const GROUP_COLORS = ['#58a6ff', '#ff7b72', '#d2a8ff', '#ffa657', '#7ee787', '#79c0ff', '#f778ba', '#e3b341'];

// size a canvas to its CSS box (sharp on high-DPI screens) and return [ctx, width, height]
export function fit(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return [ctx, w, h];
}

export const esc = s => String(s).replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
export const chip = (id, color) => `<span class="chip" data-nid="${id}" ${color ? `style="border-color:${color}"` : ''}>${id}</span>`;
