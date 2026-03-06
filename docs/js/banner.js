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
const B0 = 2;
const E = { x: 0, y: 0, z: 0 };
const DT = 0.012;

// ---------- Tunables ----------
const TRAIL_LIFE = 80;
const SMOOTHING = 0.08;

let rL = 1;
let vPar = 1;

// ---------- State ----------
let pos, vel;
let trail = [];
let useTrail = true;
let bhatSmooth = { x: Math.cos(Math.PI / 4), y: Math.sin(Math.PI / 4), z: 0 };

// ---------- Theme colors (updated each frame) ----------
let colTitle, colSub, colPrimary, colFg;

function getLineY() { return Math.round(H * 0.68); }
function getArrowBase() { return { x: W - 40 * S, y: H - 30 * S }; }

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

  drawTitle();

  const bhat = getBhat();
  drawBArrow(bhat);

  borisStep(bhat);
  wrapAllDirections();
  updateTrail();

  drawParticle();
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

// Field points along the line from arrow base -> mouse
function getBhat() {
  const ab = getArrowBase();
  const dx = mouseX - ab.x;
  const dy = mouseY - ab.y;
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

  // Break trail so we don't draw a diagonal across the banner
  if (wrapped && useTrail) trail.push(null);
}

function updateTrail() {
  if (!useTrail) {
    trail.length = 0;
    return;
  }

  trail.push({ x: pos.x, y: pos.y, age: 0 });

  for (let p of trail) {
    if (p !== null) p.age++;
  }

  trail = trail.filter((p) => p === null || p.age < TRAIL_LIFE);
}

function drawTrail() {
  if (!useTrail) return;

  noFill();
  beginShape();

  for (let p of trail) {
    if (p === null) {
      endShape();
      beginShape();
      continue;
    }

    const a = map(p.age, 0, TRAIL_LIFE, 160, 0);
    const w = map(p.age, 0, TRAIL_LIFE, 2.2 * S, 0.6 * S);

    stroke(...colPrimary, a);
    strokeWeight(w);
    vertex(p.x, p.y);
  }

  endShape();
}

function drawParticle() {
  drawTrail();

  noStroke();
  fill(...colPrimary, 70);
  circle(pos.x, pos.y, 14 * S);

  fill(...colPrimary, 230);
  circle(pos.x, pos.y, 5 * S);
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

function drawBArrow(bhat) {
  const ab = getArrowBase();
  const baseX = ab.x;
  const baseY = ab.y;
  const L = 20 * S;

  stroke(...colPrimary, 160);
  strokeWeight(2 * S);

  const tipX = baseX + bhat.x * L;
  const tipY = baseY + bhat.y * L;

  line(baseX, baseY, tipX, tipY);

  const nx = -bhat.y;
  const ny = bhat.x;

  line(tipX, tipY, tipX - bhat.x * 6 * S + nx * 4 * S, tipY - bhat.y * 6 * S + ny * 4 * S);
  line(tipX, tipY, tipX - bhat.x * 6 * S - nx * 4 * S, tipY - bhat.y * 6 * S - ny * 4 * S);

  noStroke();
  fill(...colPrimary, 200);
  textSize(12 * S);
  textAlign(RIGHT, CENTER);
  text("B", baseX + 30 * S, baseY + 25 * S);
}