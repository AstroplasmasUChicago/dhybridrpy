// ---------- Reference dimensions & aspect ratio ----------
const REF_W = 650;
const REF_H = 180;
const ASPECT = REF_W / REF_H;

// ---------- Dynamic dimensions ----------
let W = REF_W;
let H = REF_H;
let S = 1; // scale factor

// ---------- Physics ----------
const QM_RATIO = 1.0;
const B0 = 5;
const E = { x: 0, y: 0, z: 0 };
const DT = 0.012;

// ---------- Perspective ----------
const FOV = 200;

// ---------- Depth layering ----------
const TEXT_Z = 0; // text sits at z = 0

// ---------- Tunables ----------
const TRAIL_LIFE = 80;
const SMOOTHING = 0.08;

let rL = 0.5;
let vPar = 0.5;

// ---------- State ----------
let pos, vel;
let trail = [];
let useTrail = true;
let bhatSmooth = { x: Math.cos(Math.PI / 4), y: Math.sin(Math.PI / 4), z: 0 };

// ---------- Theme colors (updated each frame) ----------
let colTitle, colSub, colPrimary, colFg;

function getLineY() { return Math.round(H * 0.68); }

function project(px, py, pz) {
  const fov = FOV * S;
  const scale = fov / (fov + pz);
  return {
    x: W * 0.5 + (px - W * 0.5) * scale,
    y: H * 0.5 + (py - H * 0.5) * scale,
    s: scale
  };
}

function readCssRgb(varName, fallback) {
  const raw = getComputedStyle(document.documentElement)
    .getPropertyValue(varName).trim();
  if (!raw) return fallback;
  const el = document.createElement("span");
  el.style.color = raw;
  el.style.display = "none";
  document.body.appendChild(el);
  const c = getComputedStyle(el).color;
  document.body.removeChild(el);
  const m = c.match(/rgba?\(\s*(\d+),\s*(\d+),\s*(\d+)/);
  if (m) return [parseInt(m[1]), parseInt(m[2]), parseInt(m[3])];
  return fallback;
}

function detectTheme() {
  colPrimary = readCssRgb("--md-primary-fg-color", [63, 81, 181]);
  colFg = readCssRgb("--md-default-fg-color", [0, 0, 0]);

  const scheme = (document.body.getAttribute("data-md-color-scheme") || "default");
  const isDark = scheme === "slate";

  colTitle = isDark ? [230, 240, 255, 230] : [...colFg, 220];
  colSub   = isDark ? [200, 215, 240, 130] : [...colFg, 100];
}

function sizeCanvas() {
  const container = document.getElementById("p5-banner");
  W = container ? container.offsetWidth : REF_W;
  H = Math.round(W / ASPECT);
  S = W / REF_W;
}

function initPhysics() {
  pos = { x: 0, y: getLineY() + rL, z: 0 };
  vel = { x: vPar, y: 0, z: 0 };
  vel.z = rL * QM_RATIO * B0;
  trail = [];
}

function setup() {
  sizeCanvas();
  const canvas = createCanvas(W, H);
  canvas.parent("p5-banner");
  pixelDensity(2);
  textFont("system-ui");

  detectTheme();
  initPhysics();
}

function windowResized() {
  const oldW = W;
  sizeCanvas();
  resizeCanvas(W, H);
  // rescale particle position proportionally
  if (oldW > 0) {
    const r = W / oldW;
    pos.x *= r;
    pos.y *= r;
  }
}

function draw() {
  clear();
  detectTheme();

  const bhat = getBhat();

  borisStep(bhat);
  wrapAllDirections();
  updateTrail();

  // Behind text (z > TEXT_Z = farther from camera)
  drawTrailLayer(TEXT_Z, Infinity);
  if (pos.z > TEXT_Z) drawParticleDot();

  drawTitle();

  // In front of text (z <= TEXT_Z)
  drawTrailLayer(-Infinity, TEXT_Z);
  if (pos.z <= TEXT_Z) drawParticleDot();

  drawBIndicator(bhat);
}

function drawBIndicator(bhat) {
  const cx = W - 10 * S;
  const cy = H - 5 * S;
  const L = 4 * S;

  // Draw line centered on (cx, cy)
  const halfX = bhat.x * L;
  const halfY = bhat.y * L;
  const x1 = cx - halfX;
  const y1 = cy - halfY;
  const x2 = cx + halfX;
  const y2 = cy + halfY;

  stroke(...colPrimary, 160);
  strokeWeight(2 * S);
  line(x1, y1, x2, y2);

  // Arrowhead at tip (x2, y2)
  const nx = -bhat.y;
  const ny = bhat.x;
  line(x2, y2, x2 - bhat.x * 5 * S + nx * 3 * S, y2 - bhat.y * 5 * S + ny * 3 * S);
  line(x2, y2, x2 - bhat.x * 5 * S - nx * 3 * S, y2 - bhat.y * 5 * S - ny * 3 * S);
}

function drawTitle() {
  noStroke();

  fill(...colTitle);
  textSize(50 * S);
  textAlign(LEFT, BASELINE);
  text("dhybridrpy", 22 * S, 68 * S);

  fill(...colSub);
  textSize(20 * S);
  text("A Python package to easily read input + output data from dHybridR.", 22 * S, 104 * S);
}

// Field points along the line from banner center -> mouse
function getBhat() {
  const dx = mouseX - W * 0.5;
  const dy = mouseY - H * 0.5;
  const mag = Math.hypot(dx, dy);

  let target;
  if (mag > 1e-6) {
    target = { x: dx / mag, y: dy / mag, z: 0 };
  } else {
    target = { ...bhatSmooth };
  }

  bhatSmooth.x = lerp(bhatSmooth.x, target.x, SMOOTHING);
  bhatSmooth.y = lerp(bhatSmooth.y, target.y, SMOOTHING);

  const m = Math.hypot(bhatSmooth.x, bhatSmooth.y);
  if (m > 1e-6) {
    bhatSmooth.x /= m;
    bhatSmooth.y /= m;
  }

  return { x: bhatSmooth.x, y: bhatSmooth.y, z: 0 };
}

function wrapAllDirections() {
  let wrapped = false;

  // X wrap
  if (pos.x > W) { pos.x = 0; wrapped = true; }
  else if (pos.x < 0) { pos.x = W; wrapped = true; }

  // Y wrap
  if (pos.y > H) { pos.y = 0; wrapped = true; }
  else if (pos.y < 0) { pos.y = H; wrapped = true; }

  // Z wrap, keep depth within a visible range
  const Z_MAX = FOV * S * 3.0;
  if (pos.z > Z_MAX) { pos.z -= 2 * Z_MAX; wrapped = true; }
  else if (pos.z < -Z_MAX) { pos.z += 2 * Z_MAX; wrapped = true; }

  // Break trail so we don't draw a diagonal across the banner
  if (wrapped && useTrail) trail.push(null);
}

function updateTrail() {
  if (!useTrail) {
    trail.length = 0;
    return;
  }

  trail.push({ x: pos.x, y: pos.y, z: pos.z, age: 0 });

  for (let p of trail) {
    if (p !== null) p.age++;
  }

  trail = trail.filter((p) => p === null || p.age < TRAIL_LIFE);
}

function drawTrailLayer(zMin, zMax) {
  if (!useTrail) return;

  noFill();

  // Collect runs of non-null points
  const runs = [];
  let run = [];
  for (const p of trail) {
    if (p === null) {
      if (run.length > 1) runs.push(run);
      run = [];
    } else {
      run.push(p);
    }
  }
  if (run.length > 1) runs.push(run);

  for (const pts of runs) {
    // Catmull-Rom subdivide: insert interpolated points between each pair
    const SUB = 4;
    const smooth = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[Math.max(i - 1, 0)];
      const p1 = pts[i];
      const p2 = pts[i + 1];
      const p3 = pts[Math.min(i + 2, pts.length - 1)];

      for (let j = 0; j < SUB; j++) {
        const t = j / SUB;
        const t2 = t * t;
        const t3 = t2 * t;
        const sx = 0.5 * (2*p1.x + (-p0.x+p2.x)*t + (2*p0.x-5*p1.x+4*p2.x-p3.x)*t2 + (-p0.x+3*p1.x-3*p2.x+p3.x)*t3);
        const sy = 0.5 * (2*p1.y + (-p0.y+p2.y)*t + (2*p0.y-5*p1.y+4*p2.y-p3.y)*t2 + (-p0.y+3*p1.y-3*p2.y+p3.y)*t3);
        const sz = 0.5 * (2*p1.z + (-p0.z+p2.z)*t + (2*p0.z-5*p1.z+4*p2.z-p3.z)*t2 + (-p0.z+3*p1.z-3*p2.z+p3.z)*t3);
        const sage = p1.age + (p2.age - p1.age) * t;
        smooth.push({ x: sx, y: sy, z: sz, age: sage });
      }
    }
    smooth.push(pts[pts.length - 1]);

    // Draw smooth segments (filtered by z layer)
    for (let i = 1; i < smooth.length; i++) {
      const prev = smooth[i - 1];
      const cur = smooth[i];
      const avgZ = (prev.z + cur.z) * 0.5;
      if (avgZ < zMin || avgZ >= zMax) continue;

      const sp1 = project(prev.x, prev.y, prev.z);
      const sp2 = project(cur.x, cur.y, cur.z);

      const depthScale = (sp1.s + sp2.s) * 0.5;
      const fade = edgeFade(cur.x, cur.y, cur.z);
      const a = map(cur.age, 0, TRAIL_LIFE, 160, 0) * depthScale * fade;
      const w = map(cur.age, 0, TRAIL_LIFE, 2.2 * S, 0.6 * S) * depthScale;

      stroke(...colPrimary, a);
      strokeWeight(w);
      line(sp1.x, sp1.y, sp2.x, sp2.y);
    }
  }
}

function edgeFade(px, py, pz) {
  const margin = 10 * S;
  const Z_MAX = FOV * S * 3.0;
  const fx = Math.min(px, W - px) / margin;
  const fy = Math.min(py, H - py) / margin;
  const fz = (Z_MAX - Math.abs(pz)) / margin;
  return constrain(Math.min(fx, fy, fz), 0, 1);
}

function drawParticleDot() {
  const p = project(pos.x, pos.y, pos.z);
  const fade = edgeFade(pos.x, pos.y, pos.z);

  noStroke();
  fill(...colPrimary, 70 * p.s * fade);
  circle(p.x, p.y, 14 * S * p.s);

  fill(...colPrimary, 230 * p.s * fade);
  circle(p.x, p.y, 5 * S * p.s);
}

// Boris pusher (arbitrary B direction)
function borisStep(bhat) {
  const B = { x: B0 * bhat.x, y: B0 * bhat.y, z: B0 * bhat.z };

  const vMinus = {
    x: vel.x + QM_RATIO * E.x * DT * 0.5,
    y: vel.y + QM_RATIO * E.y * DT * 0.5,
    z: vel.z + QM_RATIO * E.z * DT * 0.5
  };

  const t = {
    x: QM_RATIO * B.x * DT * 0.5,
    y: QM_RATIO * B.y * DT * 0.5,
    z: QM_RATIO * B.z * DT * 0.5
  };

  const vPrime = {
    x: vMinus.x + (vMinus.y * t.z - vMinus.z * t.y),
    y: vMinus.y + (vMinus.z * t.x - vMinus.x * t.z),
    z: vMinus.z + (vMinus.x * t.y - vMinus.y * t.x)
  };

  const t2 = t.x * t.x + t.y * t.y + t.z * t.z;
  const s = {
    x: (2 * t.x) / (1 + t2),
    y: (2 * t.y) / (1 + t2),
    z: (2 * t.z) / (1 + t2)
  };

  const vPlus = {
    x: vMinus.x + (vPrime.y * s.z - vPrime.z * s.y),
    y: vMinus.y + (vPrime.z * s.x - vPrime.x * s.z),
    z: vMinus.z + (vPrime.x * s.y - vPrime.y * s.x)
  };

  vel.x = vPlus.x + QM_RATIO * E.x * DT * 0.5;
  vel.y = vPlus.y + QM_RATIO * E.y * DT * 0.5;
  vel.z = vPlus.z + QM_RATIO * E.z * DT * 0.5;

  pos.x += vel.x;
  pos.y += vel.y;
  pos.z += vel.z;
}

