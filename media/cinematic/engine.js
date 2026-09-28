/*
 * Shared engine for the Kaggriculture CloudFronts films.
 *
 * A film page loads this file, defines its scene functions, then calls
 * film(DURATION, SCENES).  Every frame is a pure function of time:
 * window.renderAt(t) draws the frame at t seconds.  render_movie.py steps
 * through it frame by frame and encodes the result; opening a film page in a
 * browser simply plays it in real time.
 */
const W = 1920, H = 1080;
const cv = document.getElementById('c');
const ctx = cv.getContext('2d');

const SERIF = '"Liberation Serif", "DejaVu Serif", Georgia, serif';
const SANS = '"DejaVu Sans", "Liberation Sans", Helvetica, sans-serif';
const GOLD = '#f3c969', GOLD_DEEP = '#b8862b', CYAN = '#6fe3ff', LEAF = '#7be38a';

// ---------------------------------------------------------------- utilities
function mulberry(seed) {
  return () => {
    seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
const R = mulberry(20330);
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const lerp = (a, b, k) => a + (b - a) * k;
const smooth = x => { x = clamp(x); return x * x * (3 - 2 * x); };
const easeOut = x => 1 - Math.pow(1 - clamp(x), 3);
const easeInOut = x => { x = clamp(x); return x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
// alpha envelope: fade in over [a, a+fi], hold, fade out over [b-fo, b]
const env = (t, a, b, fi = 1, fo = 1) => smooth((t - a) / fi) * smooth((b - t) / fo);

// ------------------------------------------------------------ shared assets
const stars = Array.from({ length: 700 }, () => ({ x: R() * W, y: R() * H, s: R() * 1.7 + .2, p: R() * 6.283, d: R() }));
const dust = Array.from({ length: 260 }, () => ({ x: R() * W, y: R() * H, s: R() * 2.4 + .4, v: R() * 14 + 4, p: R() * 6.283 }));
const rain = Array.from({ length: 900 }, () => ({ x: R() * (W + 400), y: R() * H, l: R() * 40 + 25, v: R() * 900 + 1400 }));
const clouds = Array.from({ length: 70 }, () => ({ x: R() * (W + 800) - 400, y: R() * 520 + 60, rx: R() * 360 + 200, ry: R() * 90 + 50, v: R() * 22 + 6, sh: R() }));
const embers = Array.from({ length: 320 }, () => ({ x: R() * W, y: R() * H, s: R() * 2.6 + .6, v: R() * 60 + 25, w: R() * 2 - 1, p: R() * 6.283 }));

// Fibonacci-sphere globe; a smooth trig field decides which dots are "land".
const globe = [];
{
  const N = 11000, ga = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < N; i++) {
    const y = 1 - (i / (N - 1)) * 2, lat = Math.asin(y), lon = (i * ga) % (2 * Math.PI);
    const f = Math.sin(3 * lon) * Math.cos(2 * lat) + .6 * Math.sin(5 * lon + 1.3) * Math.sin(3 * lat + .7)
            + .4 * Math.cos(7 * lon - 2) * Math.cos(5 * lat) + .25 * Math.sin(11 * lon + 4 * lat);
    if (f > .12 && Math.abs(lat) < 1.3) globe.push({ lat, lon, h: R(), k: f });
  }
}
function project(p, cx, cy, r, rot, tilt = .32) {
  const lon = p.lon + rot;
  let x = Math.cos(p.lat) * Math.sin(lon), y = Math.sin(p.lat), z = Math.cos(p.lat) * Math.cos(lon);
  const y2 = y * Math.cos(tilt) - z * Math.sin(tilt), z2 = y * Math.sin(tilt) + z * Math.cos(tilt);
  return { x: cx + r * x, y: cy - r * y2, z: z2 };
}
// mode: 'tech' | 'heat' | 'life'
function drawGlobe(t, cx, cy, r, rot, mode, alpha = 1, heat = 0) {
  ctx.save();
  ctx.globalAlpha = alpha;
  const pal = { tech: ['#0b2a45', '#02070f', '#3aa9ff'], heat: ['#3a0d06', '#0a0202', '#ff6a2b'], life: ['#0d3350', '#020a14', '#7fd7ff'] }[mode];
  // atmosphere
  const at = ctx.createRadialGradient(cx, cy, r * .92, cx, cy, r * 1.22);
  at.addColorStop(0, pal[2] + '66'); at.addColorStop(1, pal[2] + '00');
  ctx.fillStyle = at; ctx.beginPath(); ctx.arc(cx, cy, r * 1.22, 0, 7); ctx.fill();
  const g = ctx.createRadialGradient(cx - r * .35, cy - r * .4, r * .1, cx, cy, r);
  g.addColorStop(0, pal[0]); g.addColorStop(1, pal[1]);
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, r, 0, 7); ctx.fill();
  // graticule
  ctx.strokeStyle = pal[2] + '22'; ctx.lineWidth = 1;
  for (let la = -60; la <= 60; la += 30) {
    ctx.beginPath();
    for (let lo = 0; lo <= 360; lo += 6) {
      const q = project({ lat: la * Math.PI / 180, lon: lo * Math.PI / 180 }, cx, cy, r, rot);
      if (q.z > 0) ctx.lineTo(q.x, q.y); else ctx.moveTo(q.x, q.y);
    }
    ctx.stroke();
  }
  // land dots
  for (const p of globe) {
    const q = project(p, cx, cy, r, rot);
    if (q.z <= 0) continue;
    let col, s = (1.1 + q.z * 1.3) * r / 420;
    if (mode === 'tech') col = `rgba(111,227,255,${.25 + .75 * q.z})`;
    else if (mode === 'heat') {
      const k = clamp(heat * (.6 + .8 * p.h) + .15 * Math.sin(t * 6 + p.h * 40));
      col = `rgba(255,${Math.round(lerp(170, 50, k))},${Math.round(lerp(90, 20, k))},${.3 + .7 * q.z})`;
      s *= 1 + k * .6;
    } else {
      const k = clamp(heat * 1.4 - p.h * .5);
      col = `rgba(${Math.round(lerp(120, 123, k))},${Math.round(lerp(200, 227, k))},${Math.round(lerp(255, 138, k))},${.3 + .7 * q.z})`;
      s *= 1 + k * .5;
    }
    ctx.fillStyle = col;
    ctx.fillRect(q.x - s / 2, q.y - s / 2, s, s);
  }
  // terminator shading
  const sh = ctx.createLinearGradient(cx - r, cy - r, cx + r, cy + r);
  sh.addColorStop(0, 'rgba(0,0,0,0)'); sh.addColorStop(.7, 'rgba(0,0,0,.15)'); sh.addColorStop(1, 'rgba(0,0,0,.65)');
  ctx.fillStyle = sh; ctx.beginPath(); ctx.arc(cx, cy, r, 0, 7); ctx.fill();
  ctx.restore();
}

// Film grain tiles
const grain = [];
for (let k = 0; k < 6; k++) {
  const c = document.createElement('canvas'); c.width = 480; c.height = 270;
  const g = c.getContext('2d'), im = g.createImageData(480, 270), r = mulberry(99 + k);
  for (let i = 0; i < im.data.length; i += 4) { const v = r() * 255; im.data[i] = im.data[i + 1] = im.data[i + 2] = v; im.data[i + 3] = 255; }
  g.putImageData(im, 0, 0); grain.push(c);
}

// ------------------------------------------------------------------ type
function wrap(text, font, maxW) {
  ctx.font = font;
  const words = text.split(' '), lines = [];
  let cur = '';
  for (const w of words) {
    const tr = cur ? cur + ' ' + w : w;
    if (ctx.measureText(tr).width > maxW && cur) { lines.push(cur); cur = w; } else cur = tr;
  }
  if (cur) lines.push(cur);
  return lines;
}
// Words resolve one after another out of a soft blur, like a line being spoken.
// o.hi: word fragments to pick out in o.hiColor (gold by default).
function speak(t, text, a, b, o = {}) {
  const alpha = env(t, a, b, .01, o.fo ?? 1.1);
  if (alpha <= 0) return;
  const size = o.size ?? 62, font = `${o.italic ? 'italic ' : ''}${o.weight ?? 400} ${size}px ${o.family ?? SERIF}`;
  const lines = wrap(text, font, o.maxW ?? 1500);
  const lh = size * (o.lh ?? 1.28), y0 = (o.y ?? H / 2) - (lines.length - 1) * lh / 2;
  const stagger = o.stagger ?? .11;
  let wi = 0;
  ctx.save();
  ctx.font = font; ctx.textBaseline = 'middle';
  if (o.spacing) ctx.letterSpacing = o.spacing + 'px';
  lines.forEach((ln, li) => {
    const words = ln.split(' ');
    const total = ctx.measureText(ln).width, space = ctx.measureText(' ').width;
    let x = (o.x ?? W / 2) - total / 2;
    const drift = (1 - easeOut((t - a) / 3)) * 10;
    for (const w of words) {
      const k = easeOut((t - a - wi * stagger) / .9);
      ctx.globalAlpha = alpha * k;
      ctx.shadowColor = o.glow ?? 'rgba(0,0,0,.85)';
      ctx.shadowBlur = o.glowBlur ?? 18;
      ctx.fillStyle = o.hi?.some(h => w.includes(h)) ? (o.hiColor ?? GOLD) : (o.color ?? '#f4efe4');
      ctx.fillText(w, x, y0 + li * lh + drift + (1 - k) * 14);
      x += ctx.measureText(w).width + space; wi++;
    }
  });
  ctx.restore();
}
function label(text, x, y, size, color, alpha, spacing = 6, align = 'center', weight = 400, family = SANS) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalAlpha = alpha; ctx.font = `${weight} ${size}px ${family}`; ctx.letterSpacing = spacing + 'px';
  ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillStyle = color;
  ctx.fillText(text, x + (align === 'center' ? spacing / 2 : 0), y);
  ctx.restore();
}

// ================================================================= FILM
// SCENES: [[start, end, draw(t, alpha)], ...] — overlapping ranges cross-fade.
function film(DURATION, SCENES) {
  function renderAt(t) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1; ctx.letterSpacing = '0px';
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
    for (const [a, b, fn] of SCENES) {
      if (t < a || t > b) continue;
      const A = smooth((t - a) / .9) * smooth((b - t) / .9);
      if (A > 0) fn(t, A);
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; ctx.shadowBlur = 0; ctx.letterSpacing = '0px';
    }
    // vignette
    const v = ctx.createRadialGradient(W / 2, H / 2, H * .35, W / 2, H / 2, H * 1.05);
    v.addColorStop(0, 'rgba(0,0,0,0)'); v.addColorStop(1, 'rgba(0,0,0,.72)');
    ctx.fillStyle = v; ctx.fillRect(0, 0, W, H);
    // grain
    ctx.globalAlpha = .055; ctx.globalCompositeOperation = 'overlay';
    ctx.drawImage(grain[Math.floor(t * 24) % grain.length], 0, 0, W, H);
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
    // 2.39:1 letterbox
    const bar = Math.round((H - W / 2.39) / 2);
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, bar); ctx.fillRect(0, H - bar, W, bar);
    // fade from / to black
    const fade = 1 - smooth(t / 1.2) * smooth((DURATION - t) / 1.8);
    if (fade > 0) { ctx.fillStyle = `rgba(0,0,0,${fade})`; ctx.fillRect(0, 0, W, H); }
  }
  window.renderAt = renderAt;
  window.DURATION = DURATION;

  // Real-time playback when opened directly (render_movie.py sets ?capture).
  if (!location.search.includes('capture')) {
    const t0 = performance.now();
    const loop = () => { renderAt(((performance.now() - t0) / 1000) % DURATION); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }
}
