// Body & World panel: endless grass field, flowers, rocks, sun/sky/stars day-night cycle,
// and a ZX-style armoured robot whose joints follow the brain's motor neurons.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const TILE = 24;                 // grass tile size (m); 3x3 tiles follow the robot = endless field
const BLADES = 2600;             // grass blades per tile

function rand(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

// ---------- the robot (original design in the Mega Man ZX style: blue armour, red accents, crested helmet) ----------
function makeRobot() {
  const M = (color, opts = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness: 0.45, ...opts });
  const blue = M('#2563eb'), dark = M('#1e3a8a'), red = M('#dc2626'), white = M('#e5e7eb', { metalness: 0.2 });
  const gold = M('#facc15', { emissive: '#a16207', emissiveIntensity: 0.6 }), skin = M('#f1c7a6', { metalness: 0, roughness: 0.8 });
  const gem = M('#38bdf8', { emissive: '#0ea5e9', emissiveIntensity: 1.2 });
  const box = (w, h, d, mat) => new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  const cyl = (rt, rb, h, mat, seg = 16) => new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, seg), mat);
  const sph = (r, mat) => new THREE.Mesh(new THREE.SphereGeometry(r, 20, 16), mat);
  const at = (o, x, y, z) => { o.position.set(x, y, z); return o; };

  const robot = new THREE.Group();
  const hips = at(new THREE.Group(), 0, 0.9, 0); robot.add(hips);
  hips.add(at(box(0.36, 0.14, 0.22, dark), 0, 0, 0));

  // legs (animated by walking), chunky boots with red trim
  const legs = [];
  for (const side of [1, -1]) {
    const hip = at(new THREE.Group(), 0.11 * side, -0.05, 0); hips.add(hip);
    hip.add(at(cyl(0.07, 0.065, 0.36, blue), 0, -0.2, 0));
    const knee = at(new THREE.Group(), 0, -0.4, 0); hip.add(knee);
    knee.add(at(sph(0.07, white), 0, 0, 0));
    knee.add(at(cyl(0.075, 0.09, 0.3, blue), 0, -0.17, 0));
    const boot = at(box(0.17, 0.16, 0.28, blue), 0, -0.38, 0.04); knee.add(boot);
    boot.add(at(box(0.18, 0.04, 0.29, red), 0, 0.07, 0));
    legs.push({ hip, knee });
  }

  // torso: bends and twists at the waist
  const waist = at(new THREE.Group(), 0, 0.08, 0); hips.add(waist);
  waist.add(at(cyl(0.13, 0.15, 0.16, white), 0, 0.08, 0));
  const chest = at(box(0.46, 0.34, 0.28, blue), 0, 0.34, 0); waist.add(chest);
  chest.add(at(box(0.3, 0.22, 0.05, red), 0, 0.0, 0.14));
  chest.add(at(sph(0.06, gem), 0, 0.0, 0.17));                 // chest core
  chest.add(at(box(0.5, 0.06, 0.3, dark), 0, -0.16, 0));

  // head: helmet with crest and gem, face visor, ear guards
  const neck = at(new THREE.Group(), 0, 0.6, 0); waist.add(neck);
  neck.add(at(cyl(0.06, 0.07, 0.08, dark), 0, 0.02, 0));
  const head = at(new THREE.Group(), 0, 0.15, 0); neck.add(head);
  const helmet = sph(0.17, blue); helmet.scale.set(1, 1.05, 1.05); head.add(helmet);
  head.add(at(box(0.2, 0.12, 0.05, skin), 0, -0.03, 0.15));      // face
  head.add(at(box(0.05, 0.03, 0.02, M('#0ea5e9', { emissive: '#0369a1' })), 0.05, -0.01, 0.18));
  head.add(at(box(0.05, 0.03, 0.02, M('#0ea5e9', { emissive: '#0369a1' })), -0.05, -0.01, 0.18));
  head.add(at(sph(0.04, gold), 0, 0.09, 0.16));                  // forehead gem
  const crest = at(box(0.04, 0.16, 0.26, red), 0, 0.17, -0.02); crest.rotation.x = -0.35; head.add(crest);
  for (const side of [1, -1]) {
    const ear = at(cyl(0.06, 0.06, 0.06, white, 12), 0.17 * side, 0, 0); ear.rotation.z = Math.PI / 2; head.add(ear);
    head.add(at(box(0.03, 0.14, 0.03, red), 0.2 * side, 0.08, -0.02));
  }

  // arms: shoulder pads, elbows; the right forearm is a buster
  const arms = {};
  for (const [name, side] of [['left', 1], ['right', -1]]) {
    const shoulder = at(new THREE.Group(), 0.3 * side, 0.45, 0); waist.add(shoulder);
    shoulder.add(at(sph(0.11, blue), 0, 0, 0));
    shoulder.add(at(box(0.16, 0.08, 0.2, red), 0.02 * side, 0.08, 0));
    shoulder.add(at(cyl(0.055, 0.055, 0.26, white), 0, -0.16, 0));
    const elbow = at(new THREE.Group(), 0, -0.3, 0); shoulder.add(elbow);
    elbow.add(at(sph(0.06, dark), 0, 0, 0));
    if (name === 'right') {
      elbow.add(at(cyl(0.085, 0.1, 0.3, blue), 0, -0.17, 0));
      elbow.add(at(cyl(0.05, 0.05, 0.04, gold), 0, -0.33, 0));
    } else {
      elbow.add(at(cyl(0.07, 0.08, 0.24, blue), 0, -0.14, 0));
      elbow.add(at(sph(0.065, white), 0, -0.3, 0));
    }
    arms[name] = { shoulder, elbow };
  }
  robot.traverse(o => { if (o.isMesh) o.castShadow = true; });
  return { robot, waist, neck, head, arms, legs };
}

export class World3D {
  constructor(el) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x87ceeb, 25, 60);
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.05, 400);
    this.camera.position.set(3.2, 2.4, -3.6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 1, 0);
    this.controls.maxPolarAngle = Math.PI * 0.49;

    this.hemi = new THREE.HemisphereLight(0xbfe3ff, 0x2d4a1f, 0.8);
    this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xfff4d6, 2.2);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(1024, 1024);
    Object.assign(this.sun.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8, near: 1, far: 60 });
    this.scene.add(this.sun, this.sun.target);
    this.sunDisc = new THREE.Mesh(new THREE.SphereGeometry(4, 16, 12), new THREE.MeshBasicMaterial({ color: 0xfff1a8, fog: false }));
    this.scene.add(this.sunDisc);

    // stars: a big sphere of points, visible at night
    const sp = [], r = rand(7);
    for (let i = 0; i < 1500; i++) {
      const u = r() * 2 - 1, a = r() * Math.PI * 2, s = Math.sqrt(1 - u * u);
      if (u > -0.05) sp.push(Math.cos(a) * s * 180, u * 180, Math.sin(a) * s * 180);
    }
    const sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3));
    this.stars = new THREE.Points(sg, new THREE.PointsMaterial({ color: 0xffffff, size: 0.9, sizeAttenuation: false, transparent: true, fog: false }));
    this.scene.add(this.stars);

    // ground + grass (3x3 repeating tiles that follow the robot)
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({ color: '#3f7d2a', roughness: 1 }));
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);
    const blade = new THREE.ConeGeometry(0.025, 0.32, 3); blade.translate(0, 0.16, 0);
    this.grass = new THREE.InstancedMesh(blade, new THREE.MeshStandardMaterial({ roughness: 0.9 }), BLADES * 9);
    this.grassBase = [];
    const gr = rand(11), m = new THREE.Matrix4(), q = new THREE.Quaternion(), c = new THREE.Color();
    for (let i = 0; i < BLADES; i++) {
      this.grassBase.push([gr() * TILE, gr() * TILE, gr() * Math.PI, 0.6 + gr() * 0.9, (gr() - 0.5) * 0.4]);
    }
    for (let t = 0; t < 9; t++) for (let i = 0; i < BLADES; i++) {
      const [, , , , tilt] = this.grassBase[i];
      c.setHSL(0.27 + tilt * 0.08, 0.55, 0.28 + Math.abs(tilt) * 0.3);
      this.grass.setColorAt(t * BLADES + i, c);
    }
    this.grass.frustumCulled = false;
    this.scene.add(this.grass);
    this.grassTile = null;

    // flowers (stem + head) and rocks
    this.stems = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.015, 0.02, 0.35, 5).translate(0, 0.175, 0),
      new THREE.MeshStandardMaterial({ color: '#2f6b1f' }), 600);
    this.heads = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.11, 0), new THREE.MeshStandardMaterial({ roughness: 0.5 }), 600);
    this.rocks = new THREE.InstancedMesh(new THREE.DodecahedronGeometry(1, 0), new THREE.MeshStandardMaterial({ color: '#8a8f98', roughness: 0.95, flatShading: true }), 300);
    for (const mesh of [this.stems, this.heads, this.rocks]) { mesh.count = 0; mesh.castShadow = true; mesh.frustumCulled = false; this.scene.add(mesh); }

    this.bot = makeRobot();
    this.scene.add(this.bot.robot);
    this.follow = true;
    this.controls.addEventListener('start', () => { this.dragging = true; });
    this.controls.addEventListener('end', () => { this.dragging = false; });

    this.tmp = { m, q, c, p: new THREE.Vector3(), s: new THREE.Vector3(), e: new THREE.Euler() };
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
  }

  resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    Object.assign(this.renderer.domElement.style, { width: '100%', height: '100%' });
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  placeGrass(x, z) {
    const tx = Math.floor(x / TILE), tz = Math.floor(z / TILE);
    if (this.grassTile === `${tx},${tz}`) return;
    this.grassTile = `${tx},${tz}`;
    const { m, q, p, s, e } = this.tmp;
    let k = 0;
    for (let dx = -1; dx <= 1; dx++) for (let dz = -1; dz <= 1; dz++) {
      for (const [gx, gz, yaw, h, tilt] of this.grassBase) {
        p.set((tx + dx) * TILE + gx, 0, (tz + dz) * TILE + gz);
        q.setFromEuler(e.set(tilt, yaw, tilt * 0.5));
        this.grass.setMatrixAt(k++, m.compose(p, q, s.set(1, h, 1)));
      }
    }
    this.grass.instanceMatrix.needsUpdate = true;
    this.grass.instanceColor.needsUpdate = true;
  }

  update(W) {
    if (!W) return;
    const R = W.robot, J = R.joints, { m, q, c, p, s, e } = this.tmp;

    // robot pose: world position/heading, then joints from the motor neurons
    const bot = this.bot;
    bot.robot.position.set(R.x, 0, R.z);
    bot.robot.rotation.y = R.heading;
    bot.waist.rotation.set(J['torso bend'], J['torso twist'], 0);
    bot.head.rotation.set(-J['head nod'], J['head turn'], 0);
    for (const side of ['left', 'right']) {
      bot.arms[side].shoulder.rotation.x = -J[`${side} shoulder`];
      bot.arms[side].elbow.rotation.x = -J[`${side} elbow`];
    }
    const swing = Math.abs(R.speed) > 0.05 ? Math.sin(R.step * 2) * 0.5 : 0;
    bot.legs[0].hip.rotation.x = swing; bot.legs[1].hip.rotation.x = -swing;
    bot.legs[0].knee.rotation.x = Math.max(0, -swing) * 0.8; bot.legs[1].knee.rotation.x = Math.max(0, swing) * 0.8;

    // camera follows the robot (drag to orbit around it)
    const target = new THREE.Vector3(R.x, 1.0, R.z);
    if (this.follow) {
      const delta = target.clone().sub(this.controls.target);
      this.controls.target.add(delta);
      this.camera.position.add(delta);
    }
    this.placeGrass(R.x, R.z);
    this.ground.position.set(R.x, 0, R.z);

    // flowers and rocks near the robot
    let i = 0;
    for (const [fx, fz, color] of W.flowers) {
      if (i >= 600) break;
      p.set(fx, 0, fz);
      this.stems.setMatrixAt(i, m.compose(p, q.identity(), s.set(1, 1, 1)));
      p.set(fx, 0.38, fz);
      this.heads.setMatrixAt(i, m.compose(p, q.setFromEuler(e.set(fx, fz, 0)), s.set(1, 0.7, 1)));
      this.heads.setColorAt(i, c.set(color));
      i++;
    }
    this.stems.count = this.heads.count = i;
    this.stems.instanceMatrix.needsUpdate = this.heads.instanceMatrix.needsUpdate = true;
    if (this.heads.instanceColor) this.heads.instanceColor.needsUpdate = true;
    i = 0;
    for (const [rx, rz, rr] of W.rocks) {
      if (i >= 300) break;
      p.set(rx, rr * 0.45, rz);
      this.rocks.setMatrixAt(i++, m.compose(p, q.setFromEuler(e.set(rx * 3, rz * 5, 0)), s.set(rr, rr * 0.75, rr)));
    }
    this.rocks.count = i;
    this.rocks.instanceMatrix.needsUpdate = true;

    // day / night: sun goes round, sky colour fades, stars come out
    const a = 2 * Math.PI * (W.time % W.day_ms) / W.day_ms, light = W.daylight;
    const sunDir = new THREE.Vector3(Math.cos(a), Math.sin(a), 0.35).normalize();
    this.sun.position.copy(target).addScaledVector(sunDir, 30);
    this.sun.target.position.copy(target);
    this.sun.intensity = 2.4 * light;
    this.sunDisc.position.copy(target).addScaledVector(sunDir, 160);
    this.sunDisc.visible = sunDir.y > -0.1;
    this.hemi.intensity = 0.12 + 0.75 * light;
    const sky = new THREE.Color('#0b1026').lerp(new THREE.Color(light < 0.25 ? '#f59e6b' : '#87ceeb'), Math.min(1, light * 2.2));
    this.scene.background = sky;
    this.scene.fog.color.copy(sky);
    this.stars.position.copy(target);
    this.stars.material.opacity = Math.max(0, 1 - light * 4);
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
