// 3D brain map. Each neuron is Design 3 in 3D: a cube body, a small cube hub in the centre,
// and one frustum ("3D trapezium") pool per line between hub and body, facing that line.
// Everything is drawn with instanced meshes so 1000+ neurons stay smooth on the GPU.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { heat } from './util.js';

const EXC = new THREE.Color('#3fb950'), INH = new THREE.Color('#f85149'), LOCK = new THREE.Color('#d2a8ff');
const SPIKE = new THREE.Color('#ffd33d'), DIM = new THREE.Color('#1c2128'), WHITE = new THREE.Color('#ffffff');
const SENSOR = new THREE.Color('#58a6ff'), MOTOR = new THREE.Color('#ffa657');
const MAX_LINES = 12000, MAX_SPIKES = 4000;
const UP = new THREE.Vector3(0, 1, 0), FWD = new THREE.Vector3(0, 0, 1);

// truncated pyramid along +Z: small square (hub face) at z = 0 to a big square (body face) at z = 1
function frustumGeometry() {
  const s = 0.5, b = 0.5;
  const v = [[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0], [-b, -b, 1], [b, -b, 1], [b, b, 1], [-b, b, 1]];
  const faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]];
  const pos = [];
  for (const [a, b2, c, d] of faces) for (const i of [a, b2, c, a, c, d]) pos.push(...v[i]);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return g;
}

// zigzag spike along +Y, length 1 (R13: zigzag = active spike)
function zigzagGeometry() {
  const pts = [];
  for (let i = 0; i <= 6; i++) pts.push(new THREE.Vector3(i % 2 ? 0.12 : -0.12, -0.5 + i / 6, 0));
  pts[0].x = 0; pts[6].x = 0;
  const path = new THREE.CurvePath();
  for (let i = 0; i < 6; i++) path.add(new THREE.LineCurve3(pts[i], pts[i + 1]));
  return new THREE.TubeGeometry(path, 24, 0.035, 5, false);
}

export class Brain3D {
  constructor(el, onSelect) {
    this.el = el;
    this.onSelect = onSelect;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    el.appendChild(this.renderer.domElement);
    // show which graphics card the browser gave us (Windows decides; see README "Dedicated GPU")
    const gl = this.renderer.getContext();
    const info = gl.getExtension('WEBGL_debug_renderer_info');
    const gpu = document.getElementById('gpu');
    if (gpu) gpu.textContent = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#0d1117');
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
    this.camera.position.set(13, 9, 15);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.userMoved = false;
    this.controls.addEventListener('start', () => { this.userMoved = true; });
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(8, 14, 10);
    this.scene.add(sun);

    const mat = (opts) => new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.1, ...opts });
    this.bodyMat = mat({ transparent: true, opacity: 0.16, depthWrite: false });
    this.hubMat = new THREE.MeshBasicMaterial();
    this.poolMat = mat({ transparent: true, opacity: 0.85 });
    this.lineMat = mat({ transparent: true, opacity: 0.55, depthWrite: false });
    this.spikeMat = new THREE.MeshBasicMaterial();
    this.geo = {
      box: new THREE.BoxGeometry(1, 1, 1), frustum: frustumGeometry(),
      cyl: new THREE.CylinderGeometry(1, 1, 1, 6, 1, true), cone: new THREE.ConeGeometry(1, 1, 8),
      zig: zigzagGeometry(),
    };
    this.lines = new THREE.InstancedMesh(this.geo.cyl, this.lineMat, MAX_LINES);
    this.arrows = new THREE.InstancedMesh(this.geo.cone, this.lineMat, MAX_LINES);
    this.spikes = new THREE.InstancedMesh(this.geo.zig, this.spikeMat, MAX_SPIKES);
    for (const m of [this.lines, this.arrows, this.spikes]) { m.count = 0; m.frustumCulled = false; this.scene.add(m); }

    this.selBox = new THREE.LineSegments(new THREE.EdgesGeometry(this.geo.box), new THREE.LineBasicMaterial({ color: 0xffffff }));
    this.selBox.visible = false;
    this.scene.add(this.selBox);

    this.n = 0;
    this.tmp = { m: new THREE.Matrix4(), q: new THREE.Quaternion(), s: new THREE.Vector3(), p: new THREE.Vector3(),
      a: new THREE.Vector3(), b: new THREE.Vector3(), d: new THREE.Vector3(), c: new THREE.Color() };

    this.raycaster = new THREE.Raycaster();
    let down = null;
    this.renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
    this.renderer.domElement.addEventListener('pointerup', e => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;   // a drag, not a click
      this.pick(e);
    });
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
  }

  resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    // keep the whole cube in view in narrow or short panels, until the user moves the camera
    if (!this.userMoved) this.camera.position.setLength(24 / Math.min(1, this.camera.aspect));
  }

  // neuron world position (µm) -> scene units: the brain fits in a 10-unit cube centred on 0
  toScene(n, out) { return out.set((n[1] - this.half) * this.k, (n[9] - this.half) * this.k, -(n[2] - this.half) * this.k); }

  rebuildNeurons(F) {
    for (const m of [this.bodies, this.hubs, this.pools, this.edges]) if (m) { this.scene.remove(m); m.dispose?.(); }
    const N = F.neurons.length;
    this.n = N;
    this.k = 10 / F.size;
    this.half = F.size / 2;
    const flat = F.neurons.every(n => n[9] === F.neurons[0][9]);
    const side = flat ? Math.ceil(Math.sqrt(N)) : Math.ceil(Math.cbrt(N));
    this.spacing = 10 / side;
    this.B = this.spacing * 0.3;            // body half-size
    this.H = this.B * 0.38;                 // hub half-size
    this.bodies = new THREE.InstancedMesh(this.geo.box, this.bodyMat, N);
    this.hubs = new THREE.InstancedMesh(this.geo.box, this.hubMat, N);
    this.maxPools = N * 8;
    this.pools = new THREE.InstancedMesh(this.geo.frustum, this.poolMat, this.maxPools);
    this.pools.frustumCulled = false;
    const { m, p, s, q } = this.tmp;
    this.pos = [];
    const edgePos = [], edgeCol = [];
    const cube = new THREE.EdgesGeometry(this.geo.box).attributes.position.array;
    F.neurons.forEach((n, i) => {
      const c = this.toScene(n, new THREE.Vector3());
      this.pos.push(c);
      // empty spots (no neuron yet, #37) are drawn at size 0
      const k = n[11] ? 1 : 0;
      this.bodies.setMatrixAt(i, m.compose(c, q.identity(), s.setScalar(this.B * 2 * k)));
      this.hubs.setMatrixAt(i, m.compose(c, q.identity(), s.setScalar(this.H * 2 * k)));
      const col = n[4] ? SENSOR : n[10] ? MOTOR : n[3] > 0 ? EXC : INH;
      this.bodies.setColorAt(i, col);
      for (let j = 0; j < cube.length * (n[11] ? 1 : 0); j += 3) {
        edgePos.push(c.x + cube[j] * this.B * 2, c.y + cube[j + 1] * this.B * 2, c.z + cube[j + 2] * this.B * 2);
        edgeCol.push(col.r, col.g, col.b);
      }
    });
    const eg = new THREE.BufferGeometry();
    eg.setAttribute('position', new THREE.Float32BufferAttribute(edgePos, 3));
    eg.setAttribute('color', new THREE.Float32BufferAttribute(edgeCol, 3));
    this.edges = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8 }));
    for (const mesh of [this.bodies, this.hubs, this.pools, this.edges]) this.scene.add(mesh);
    this.signature = `${N}:${F.layout}:${F.neurons.reduce((a, n) => a + n[11], 0)}`;
  }

  update(F, groupOf) {
    if (`${F.neurons.length}:${F.layout}:${F.neurons.reduce((a, n) => a + n[11], 0)}` !== this.signature) this.rebuildNeurons(F);
    const { m, q, s, p, a, b, d, c } = this.tmp;
    const N = F.neurons, P = this.pos;
    const fired = new Set(F.raster.filter(r => r[0] >= F.t - 3).map(r => r[1]));

    // hubs: charge vs threshold; flash yellow when the neuron just fired
    N.forEach((n, i) => {
      if (fired.has(i)) c.copy(SPIKE);
      else { const [r, g, bl] = heat(n[5] / n[6]); c.setRGB(r / 255, g / 255, bl / 255, THREE.SRGBColorSpace); }
      if (!n[7] && !fired.has(i)) c.lerp(DIM, 0.55);
      this.hubs.setColorAt(i, c);
    });
    this.hubs.instanceColor.needsUpdate = true;

    // pools: one frustum per line from the hub face out to the body face, facing the line
    const showPools = this.camera.position.length() < 40;
    let k = 0;
    if (showPools) {
      N.forEach((n, i) => {
        for (const [other, fill] of n[8]) {
          if (k >= this.maxPools) break;
          d.subVectors(P[other], P[i]).normalize();
          q.setFromUnitVectors(FWD, d);
          // starts at the hub surface, ends just inside the body surface
          p.copy(P[i]).addScaledVector(d, this.H);
          s.set(this.B * 1.1, this.B * 1.1, this.B - this.H);
          this.pools.setMatrixAt(k, m.compose(p, q, s));
          if (fill > 0) { const [r, g, bl] = heat(0.25 + 0.75 * fill); c.setRGB(r / 255, g / 255, bl / 255, THREE.SRGBColorSpace); }
          else c.copy(DIM);
          this.pools.setColorAt(k, c);
          k++;
        }
      });
    }
    this.pools.count = k;
    this.pools.instanceMatrix.needsUpdate = true;
    if (this.pools.instanceColor) this.pools.instanceColor.needsUpdate = true;

    // lines: thin cylinders from body surface to body surface, thickness = strength, cone at the receiving end
    let li = 0;
    const offset = this.B * 0.18;
    for (const [src, dst, th, trial, locked] of F.lines) {
      if (li >= MAX_LINES) break;
      a.copy(P[src]); b.copy(P[dst]);
      d.subVectors(b, a); const len = d.length(); d.normalize();
      // shift A→B and B→A sideways so both directions are visible
      const side = new THREE.Vector3().crossVectors(d, Math.abs(d.y) > 0.9 ? FWD : UP).normalize().multiplyScalar(offset);
      a.addScaledVector(d, this.B).add(side); b.addScaledVector(d, -this.B).add(side);
      const span = Math.max(0.01, len - 2 * this.B);
      q.setFromUnitVectors(UP, d);
      const r = 0.012 + 0.05 * Math.abs(th);
      p.addVectors(a, b).multiplyScalar(0.5);
      this.lines.setMatrixAt(li, m.compose(p, q, s.set(r, span, r)));
      c.copy(locked ? LOCK : th > 0 ? EXC : INH);
      if (trial) c.multiplyScalar(0.5);
      this.lines.setColorAt(li, c);
      p.copy(b).addScaledVector(d, -0.06);
      this.arrows.setMatrixAt(li, m.compose(p, q, s.set(r * 2.6, 0.12, r * 2.6)));
      this.arrows.setColorAt(li, c);
      li++;
    }
    for (const mesh of [this.lines, this.arrows]) {
      mesh.count = li; mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }

    // spikes in flight: a zigzag moving along its line
    let si = 0;
    for (const [src, dst, f, charge] of F.inflight) {
      if (si >= MAX_SPIKES) break;
      d.subVectors(P[dst], P[src]);
      p.copy(P[src]).addScaledVector(d, f);
      q.setFromUnitVectors(UP, d.normalize());
      this.spikes.setMatrixAt(si, m.compose(p, q, s.set(this.spacing * 0.5, this.spacing * 0.35, this.spacing * 0.5)));
      this.spikes.setColorAt(si, charge >= 0 ? SPIKE : INH);
      si++;
    }
    this.spikes.count = si;
    this.spikes.instanceMatrix.needsUpdate = true;
    if (this.spikes.instanceColor) this.spikes.instanceColor.needsUpdate = true;

    // selected neuron: white outline; thinking groups tint the body
    const sel = P[F.selected];
    this.selBox.visible = !!sel;
    if (sel) { this.selBox.position.copy(sel); this.selBox.scale.setScalar(this.B * 2.6); }
    N.forEach((n, i) => {
      const g = groupOf && groupOf.get(i);
      c.copy(g ? new THREE.Color(g) : n[4] ? SENSOR : n[10] ? MOTOR : n[3] > 0 ? EXC : INH);
      this.bodies.setColorAt(i, c);
    });
    this.bodies.instanceColor.needsUpdate = true;
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  pick(e) {
    if (!this.bodies) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(v, this.camera);
    const hit = this.raycaster.intersectObject(this.bodies)[0];
    if (hit && hit.instanceId !== undefined) this.onSelect(hit.instanceId);
  }
}
