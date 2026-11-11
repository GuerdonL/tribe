'use strict';
/* tribe run visualizer. Data comes from viz/build.py: data/index.json (runs, models, statistics),
   data/worlds.json (maps), data/log/<run>.json (training logs), data/ep/<model>.json (recorded episodes).
   All names that come from the data are put in the page with textContent, never innerHTML. */

const $ = (s) => document.querySelector(s);
const SVGNS = 'http://www.w3.org/2000/svg';
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) e.append(kid);
  return e;
}
function s(tag, attrs, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

// ------------------------------------------------------------------ state ---
const S = {
  index: null, worlds: null, terrain: [], field: {},
  runColor: new Map(), slot: new Map(), known: new Set(), hidden: new Set(), metric: null, model: null, built: 0, builtText: '',
  logs: new Map(), eps: new Map(),
  tracks: [], focus: 0, mode: 'single', t: 0, playing: false, addMode: false, last: 0,
};
const SEQ_LIGHT = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];
const SEQ_DARK = ['#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4', '#b7d3f6'];
const isDark = () => matchMedia('(prefers-color-scheme: dark)').matches;
const CAUSE_SLOT = { alive: 1, thirst: 3, hunger: 2, cold: 7, hazard: 8, predator: 5 };
const CAUSE_LABEL = { alive: 'survived', hunger: 'died of hunger', thirst: 'died of thirst', cold: 'died of cold',
  hazard: 'fell in a hazard', predator: 'taken by the predator' };
const causeColor = (c) => cssVar(`--series-${CAUSE_SLOT[c] || 1}`);

// ----------------------------------------------------------------- labels ---
const LABEL = {
  ep_steps: 'Steps survived (mean, out of 320)', ep_survived: 'Survived the whole episode',
  ep_return: 'Return per episode', ep_hunger: 'Mean hunger meter', ep_thirst: 'Mean thirst meter',
  ep_warmth: 'Mean warmth meter', ep_night_sheltered: 'Night steps spent in the shelter',
  ep_eats: 'Meals per episode', ep_drinks: 'Drinks per episode', ep_adult_dist: 'Mean distance to the adult (cells)',
  ep_cells: 'Distinct cells visited', ep_heard_frac: 'Steps on which a word was heard',
  train_gaze_gap: 'Gaze following above chance', train_gaze_follow_rate: 'Gaze follow rate', train_gaze_chance: 'Gaze chance rate',
  train_gaze_cues: 'Gaze cues per update', train_deaths: 'Deaths per update', train_heard: 'Word prediction loss',
  train_world: 'World prediction loss', train_reward_per_step: 'Reward per step', train_ent_action: 'Action entropy',
  train_clipfrac: 'PPO clip fraction', train_approx_kl: 'PPO approximate KL', train_loss: 'Total loss',
  train_pg: 'Policy loss', train_vf: 'Value loss', train_episodes_ended: 'Episodes ended per update',
  eval_died_normal: 'Death rate', eval_steps_normal: 'Steps survived',
  eval_flip_all: 'Effect of flipping the adult\u2019s facing (all cues)', eval_flip_point: 'Facing-flip effect while the adult points',
  eval_flip_nopoint: 'Facing-flip effect while the adult is not pointing', eval_flip_fw: 'Facing-flip effect on "food" / "water"',
  eval_flip_home: 'Facing-flip effect on "home"', eval_flip_danger: 'Facing-flip effect on "danger"', eval_flip_quiet: 'Facing-flip effect when the adult is silent',
  eval_dgap_no_gaze: 'Extra deaths with gaze cues hidden', eval_dgap_no_words: 'Extra deaths with words hidden',
  eval_dgap_no_adult: 'Extra deaths with the adult hidden', eval_adult_visible: 'Share of steps with the adult in view',
  eval_p_turn: 'Share of actions that are turns', eval_p_forward: 'Share of actions that are forward', eval_ent_action: 'Action entropy',
};
function label(k) {
  if (LABEL[k]) return LABEL[k];
  if (k.startsWith('ep_died_')) return 'Died of ' + k.slice(8);
  if (k.startsWith('ep_act_')) return 'Share of actions: ' + k.slice(7).replace('_', ' ');
  if (k.startsWith('train_')) return k.slice(6).replace(/_/g, ' ');
  if (k.startsWith('eval_')) return k.slice(5).replace(/_/g, ' ');
  return k;
}
const isPct = (k) => ['ep_survived', 'ep_night_sheltered', 'ep_heard_frac'].includes(k) || k.startsWith('ep_died_') || k.startsWith('ep_act_');
function fmt(v, k) {
  if (v == null || !isFinite(v)) return '–';
  if (k && isPct(k)) return (v * 100).toFixed(v > 0 && v < 0.1 ? 1 : 0) + '%';
  if (k === 'train_gaze_gap') return (v >= 0 ? '+' : '') + v.toFixed(3);
  const a = Math.abs(v);
  if (a >= 10000) return (v / 1000).toFixed(1) + 'K';
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return Number(v.toPrecision(3)).toString();
}
function tickFmt(v, pct) {
  if (pct) return Math.round(v * 100) + '%';
  const a = Math.abs(v);
  if (a >= 10000) return (v / 1000) + 'K';
  return Number(v.toPrecision(6)).toString();
}
const modelLabel = (m) => `${m.run} · update ${m.update}${m.final ? ' (latest)' : ''}`;
const modelById = (id) => S.index.models.find((m) => m.id === id);
const visibleModels = () => S.index.models.filter((m) => !S.hidden.has(m.run));

// ------------------------------------------------------------------- data ---
async function getJSON(path) {
  const r = await fetch(path, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
function b64bytes(b) {
  const bin = atob(b), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function loadEp(id) {
  if (!S.eps.has(id)) {
    S.eps.set(id, getJSON(`data/ep/${encodeURIComponent(id)}.json`).then((j) => {
      const F = j.fields.length;
      return j.episodes.map((e) => ({ ...e, F, u8: b64bytes(e.data), data: undefined }));
    }));
  }
  return S.eps.get(id);
}
function loadLog(run) {
  if (!S.logs.has(run)) {
    S.logs.set(run, null);
    getJSON(`data/log/${encodeURIComponent(run)}.json`).then((j) => { S.logs.set(run, j); renderA(); }).catch(() => {});
  }
  return S.logs.get(run);
}

// ---------------------------------------------------------------- tooltip ---
function showTip(x, y, title, rows) {
  const tip = $('#tip');
  tip.replaceChildren();
  if (title) tip.append(h('div', { class: 't' }, title));
  for (const r of rows) {
    const key = h('span', { class: r.dot ? 'dot' : 'key', style: r.dot ? `background:${r.color}` : `border-color:${r.color}` });
    tip.append(h('div', { class: 'r' }, r.color ? key : null, h('b', {}, r.value), h('span', { class: 'name' }, r.name || '')));
  }
  tip.hidden = false;
  const w = tip.offsetWidth, hh = tip.offsetHeight;
  tip.style.left = Math.max(4, Math.min(innerWidth - w - 4, x + 14)) + 'px';
  tip.style.top = Math.max(4, y - hh - 12 < 4 ? y + 16 : y - hh - 12) + 'px';
}
const hideTip = () => { $('#tip').hidden = true; };

function nearest(pts, x, y, max) {
  let best = null, bd = max * max;
  for (const p of pts) {
    const d = (p.x - x) ** 2 + (p.y - y) ** 2;
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}
function niceTicks(lo, hi, n) {
  if (!(hi > lo)) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw)), norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

// ----------------------------------------------------------------- legend ---
function renderLegend() {
  const box = $('#legend');
  box.replaceChildren();
  for (const r of S.index.runs) {
    const n = S.index.models.filter((m) => m.run === r.name).length;
    const item = h('button', { type: 'button', class: 'item' + (S.hidden.has(r.name) ? ' off' : ''), 'aria-pressed': String(!S.hidden.has(r.name)),
      title: 'Show or hide this run', onclick: () => { S.hidden.has(r.name) ? S.hidden.delete(r.name) : S.hidden.add(r.name); assignColors(); renderAll(); } },
      h('span', { class: 'key', style: `border-color:${S.runColor.get(r.name)}` }), r.name,
      h('span', { class: 'n' }, `${n} model${n === 1 ? '' : 's'}${r.finished ? '' : r.has_log ? ', running or unfinished' : ''}`));
    box.append(item);
  }
}

// ---------------------------------------------------------------- chart A ---
function logSeries(run, metric) {
  if (!metric.startsWith('train_')) return null;
  const log = loadLog(run);
  if (!log || !log.update) return null;
  const col = metric.slice(6);
  let ys;
  if (col === 'gaze_gap') {
    if (!log.gaze_follow_rate || !log.gaze_chance) return null;
    ys = log.gaze_follow_rate.map((v, i) => (v == null || log.gaze_chance[i] == null ? null : v - log.gaze_chance[i]));
  } else ys = log[col];
  if (!ys) return null;
  const w = Number($('#smooth').value), out = [];
  let sum = 0, cnt = 0;
  const q = [];
  for (let i = 0; i < ys.length; i++) {
    if (ys[i] != null) { q.push(ys[i]); sum += ys[i]; cnt++; } else q.push(null);
    if (q.length > w) { const d = q.shift(); if (d != null) { sum -= d; cnt--; } }
    if (cnt && log.update[i] != null) out.push([log.update[i], sum / cnt]);
  }
  const stride = Math.max(1, Math.ceil(out.length / 700));
  return out.filter((_, i) => i % stride === 0 || i === out.length - 1);
}

function renderA() {
  const box = $('#chartA'), metric = S.metric;
  $('#titleA').textContent = `${label(metric)}, over training`;
  const models = visibleModels().filter((m) => m.stats[metric] != null);
  box.replaceChildren();
  if (!models.length) { box.append(h('div', { class: 'empty' }, 'No visible model has this metric.')); return; }
  const W = box.clientWidth || 600, H = 270, L = 52, R = 14, T = 10, B = 30;
  const logx = $('#logx').checked, pct = isPct(metric);
  const lines = [];
  for (const r of S.index.runs) {
    if (S.hidden.has(r.name)) continue;
    const ms = models.filter((m) => m.run === r.name).sort((a, b) => a.update - b.update);
    if (!ms.length) continue;
    const ls = logSeries(r.name, metric);
    lines.push({ run: r.name, color: S.runColor.get(r.name), pts: ls || ms.map((m) => [m.update, m.stats[metric]]), fromLog: !!ls });
  }
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const m of models) { x0 = Math.min(x0, m.update); x1 = Math.max(x1, m.update); y0 = Math.min(y0, m.stats[metric]); y1 = Math.max(y1, m.stats[metric]); }
  for (const l of lines) for (const [u, v] of l.pts) {
    if (logx && u < 1) continue;
    x0 = Math.min(x0, u); x1 = Math.max(x1, u); y0 = Math.min(y0, v); y1 = Math.max(y1, v);
  }
  if (logx) x0 = Math.max(1, x0);
  if (!(x1 > x0)) { x0 = logx ? x0 / 2 : x0 - 1; x1 = logx ? x1 * 2 : x1 + 1; }
  const yt = niceTicks(y0, y1, 5);
  y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]);
  if (!(y1 > y0)) { y0 -= 1; y1 += 1; }
  const tx = (u) => logx ? Math.log10(Math.max(u, x0)) : u;
  const X = (u) => L + (tx(u) - tx(x0)) / (tx(x1) - tx(x0)) * (W - L - R);
  const Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);

  const svg = s('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${label(metric)} over training updates, one line per run` });
  for (const v of yt) {
    svg.append(s('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: cssVar('--grid'), 'stroke-width': 1 }));
    svg.append(s('text', { x: L - 6, y: Y(v) + 4, 'text-anchor': 'end' }, tickFmt(v, pct)));
  }
  let xt = logx ? [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000].filter((v) => v >= x0 && v <= x1) : niceTicks(x0, x1, 6).filter((v) => v >= x0 && v <= x1);
  let lastPx = -100;
  xt = xt.filter((v) => { const p = X(v); if (p - lastPx < 44) return false; lastPx = p; return true; });
  for (const v of xt) svg.append(s('text', { x: X(v), y: H - B + 16, 'text-anchor': 'middle' }, tickFmt(v, false)));
  svg.append(s('line', { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: cssVar('--axis'), 'stroke-width': 1 }));
  svg.append(s('text', { x: W - R, y: H - 2, 'text-anchor': 'end' }, 'training update'));

  for (const l of lines) {
    const pts = l.pts.filter(([u]) => !logx || u >= 1);
    if (pts.length > 1) svg.append(s('path', { d: pts.map(([u, v], i) => `${i ? 'L' : 'M'}${X(u).toFixed(1)},${Y(v).toFixed(1)}`).join(''),
      fill: 'none', stroke: l.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
  }
  const cross = s('line', { y1: T, y2: H - B, stroke: cssVar('--axis'), 'stroke-width': 1, visibility: 'hidden' });
  svg.append(cross);
  const marks = [];
  for (const m of models) {
    const sel = S.model && m.id === S.model.id;
    const p = { x: X(m.update), y: Y(m.stats[metric]), m };
    marks.push(p);
    svg.append(s('circle', { cx: p.x, cy: p.y, r: sel ? 7 : 4.5, fill: S.runColor.get(m.run), stroke: sel ? cssVar('--text') : cssVar('--surface'), 'stroke-width': 2 }));
  }
  svg.addEventListener('pointermove', (ev) => {
    const r = svg.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    const hit = nearest(marks, x, y, 26);
    if (hit) {
      cross.setAttribute('visibility', 'hidden');
      showTip(ev.clientX, ev.clientY, modelLabel(hit.m), [{ color: S.runColor.get(hit.m.run), dot: true, value: fmt(hit.m.stats[metric], metric), name: label(metric) }]);
      svg.style.cursor = 'pointer';
      return;
    }
    svg.style.cursor = '';
    if (x < L || x > W - R) { hideTip(); cross.setAttribute('visibility', 'hidden'); return; }
    const rows = [];
    let snapU = null;
    for (const l of lines) {
      let best = null, bd = Infinity;
      for (const p of l.pts) { const d = Math.abs(X(p[0]) - x); if (d < bd) { bd = d; best = p; } }
      if (best && bd < 40) { rows.push({ color: l.color, value: fmt(best[1], metric), name: l.run + (l.fromLog ? '' : ` (model at ${best[0]})`), u: best[0] }); if (snapU == null || l.fromLog) snapU = best[0]; }
    }
    if (!rows.length) { hideTip(); cross.setAttribute('visibility', 'hidden'); return; }
    cross.setAttribute('x1', X(snapU)); cross.setAttribute('x2', X(snapU)); cross.setAttribute('visibility', 'visible');
    showTip(ev.clientX, ev.clientY, `update ${snapU}`, rows);
  });
  svg.addEventListener('pointerleave', () => { hideTip(); cross.setAttribute('visibility', 'hidden'); });
  svg.addEventListener('click', (ev) => {
    const r = svg.getBoundingClientRect();
    const hit = nearest(marks, ev.clientX - r.left, ev.clientY - r.top, 40);
    if (hit) selectModel(hit.m.id);
  });
  box.append(svg);
}

// --------------------------------------------------------------- dot plot ---
/* items: [{value, color, ring, tipTitle, tipRows, onClick}] stacked in bins along x. */
function dotPlot(box, items, opt) {
  box.replaceChildren();
  if (!items.length) { box.append(h('div', { class: 'empty' }, opt.empty || 'Nothing to show.')); return; }
  const W = box.clientWidth || 600, L = 14, R = 14, T = 8, B = 30;
  let lo = opt.domain ? opt.domain[0] : Math.min(...items.map((i) => i.value));
  let hi = opt.domain ? opt.domain[1] : Math.max(...items.map((i) => i.value));
  const ticks = niceTicks(lo, hi, 6);
  if (!opt.domain) { lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]); }
  if (!(hi > lo)) { lo -= 1; hi += 1; }
  // bins a few dots wide: dots fill a bin left to right, then stack upward
  let r = 5;
  const nb = Math.max(4, Math.min(opt.bins || 24, Math.floor((W - L - R) / (2 * r + 2))));
  const binW = (W - L - R) / nb;
  const bins = Array.from({ length: nb }, () => []);
  for (const it of items) bins[Math.min(nb - 1, Math.max(0, Math.floor((it.value - lo) / (hi - lo) * nb)))].push(it);
  const most = Math.max(...bins.map((b) => b.length));
  const maxH = opt.maxHeight || 190;
  let cols = Math.max(1, Math.floor((binW - 2) / (2 * r + 2)));
  let rows = Math.ceil(most / cols);
  if (rows * (2 * r + 2) > maxH - T - B) {          // still too tall: smaller dots, more columns
    r = 3; cols = Math.max(1, Math.floor((binW - 2) / (2 * r + 2))); rows = Math.ceil(most / cols);
  }
  const pitch = Math.min(2 * r + 2, (maxH - T - B) / rows);
  const H = Math.max(opt.minHeight || 90, Math.ceil(T + B + rows * pitch + 4));
  const X = (v) => L + (v - lo) / (hi - lo) * (W - L - R);
  const svg = s('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': opt.aria || 'dot plot' });
  svg.append(s('line', { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: cssVar('--axis'), 'stroke-width': 1 }));
  let lastPx = -100;
  for (const v of ticks) {
    if (v < lo || v > hi) continue;
    const p = X(v);
    if (p - lastPx < 40) continue;
    lastPx = p;
    svg.append(s('text', { x: p, y: H - B + 16, 'text-anchor': p < 30 ? 'start' : p > W - 30 ? 'end' : 'middle' }, tickFmt(v, opt.pct)));
  }
  if (opt.xlabel) svg.append(s('text', { x: W - R, y: H - 2, 'text-anchor': 'end' }, opt.xlabel));
  const pts = [];
  bins.forEach((b, bi) => {
    b.sort(opt.sort || (() => 0));
    b.forEach((it, k) => {
      const used = Math.min(cols, b.length), col = k % cols, row = Math.floor(k / cols);
      const p = { x: L + (bi + 0.5) * binW + (col - (used - 1) / 2) * (2 * r + 2), y: H - B - pitch / 2 - 1 - row * pitch, it };
      pts.push(p);
      svg.append(s('circle', { cx: p.x, cy: p.y, r: it.ring ? r + 1.5 : r, fill: it.color, stroke: it.ring ? cssVar('--text') : cssVar('--surface'), 'stroke-width': it.ring ? 2 : 1.5 }));
    });
  });
  svg.addEventListener('pointermove', (ev) => {
    const rc = svg.getBoundingClientRect();
    const hit = nearest(pts, ev.clientX - rc.left, ev.clientY - rc.top, 22);
    svg.style.cursor = hit ? 'pointer' : '';
    if (hit) showTip(ev.clientX, ev.clientY, hit.it.tipTitle, hit.it.tipRows); else hideTip();
  });
  svg.addEventListener('pointerleave', hideTip);
  svg.addEventListener('click', (ev) => {
    const rc = svg.getBoundingClientRect();
    const hit = nearest(pts, ev.clientX - rc.left, ev.clientY - rc.top, 30);
    if (hit && hit.it.onClick) hit.it.onClick();
  });
  box.append(svg);
}

function renderB() {
  const metric = S.metric;
  $('#titleB').textContent = `${label(metric)}, distribution across models`;
  const order = new Map(S.index.runs.map((r, i) => [r.name, i]));
  const items = visibleModels().filter((m) => m.stats[metric] != null).map((m) => ({
    value: m.stats[metric], color: S.runColor.get(m.run), ring: S.model && S.model.id === m.id, m,
    tipTitle: modelLabel(m), tipRows: [{ color: S.runColor.get(m.run), dot: true, value: fmt(m.stats[metric], metric), name: label(metric) }],
    onClick: () => selectModel(m.id),
  }));
  dotPlot($('#chartB'), items, { pct: isPct(metric), xlabel: label(metric), aria: `Distribution of ${label(metric)} across saved models`,
    empty: 'No visible model has this metric.', sort: (a, b) => order.get(a.m.run) - order.get(b.m.run) || a.m.update - b.m.update });
}

function renderTable() {
  const metric = S.metric, box = $('#table');
  const tb = h('tbody');
  for (const m of visibleModels()) {
    tb.append(h('tr', { class: 'row' + (S.model && S.model.id === m.id ? ' sel' : ''), tabindex: 0, onclick: () => selectModel(m.id),
      onkeydown: (e) => { if (e.key === 'Enter') selectModel(m.id); } },
      h('td', {}, h('span', { class: 'dot', style: `display:inline-block;margin-right:6px;background:${S.runColor.get(m.run)}` }), m.run),
      h('td', { class: 'num' }, String(m.update) + (m.final ? ' (latest)' : '')),
      h('td', { class: 'num' }, fmt(m.stats[metric], metric)),
      h('td', { class: 'num' }, fmt(m.stats.ep_steps, 'ep_steps')),
      h('td', { class: 'num' }, fmt(m.stats.ep_survived, 'ep_survived'))));
  }
  box.replaceChildren(h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Run'), h('th', { class: 'num' }, 'Update'),
    h('th', { class: 'num' }, label(metric)), h('th', { class: 'num' }, 'Mean steps'), h('th', { class: 'num' }, 'Survived'))), tb));
}

// ------------------------------------------------------------- model card ---
function renderModel() {
  const m = S.model;
  if (!m) return;
  $('#modelTitle').textContent = modelLabel(m);
  const drain = m.world.night_drain ? `1/${Math.round(1 / m.world.night_drain)}` : '?';
  $('#modelSub').textContent = `seed ${m.seed} · learning rate ${m.lr} · night drain ${drain} per step · warmth reward "${m.world.warmth_reward}"`;
  const tiles = [['ep_survived', 'Survived all 320 steps'], ['ep_steps', 'Mean steps survived'], ['ep_return', 'Mean return'],
    ['ep_night_sheltered', 'Night steps sheltered'], ['train_gaze_gap', 'Gaze following above chance (training)'], ['train_deaths', 'Deaths per update (training)']];
  $('#tiles').replaceChildren(...tiles.filter(([k]) => m.stats[k] != null).map(([k, l]) =>
    h('div', { class: 'tile' }, h('div', { class: 'v' }, fmt(m.stats[k], k)), h('div', { class: 'l' }, l))));

  const counts = {};
  for (const e of m.episodes) counts[e.cause] = (counts[e.cause] || 0) + 1;
  $('#epLegend').replaceChildren(...S.index.schema.causes.filter((c) => counts[c]).map((c) =>
    h('span', { class: 'item' }, h('span', { class: 'dot', style: `background:${causeColor(c)}` }), `${CAUSE_LABEL[c]} `, h('span', { class: 'n' }, String(counts[c])))));
  const onMap = new Set(S.tracks.filter((t) => t.model.id === m.id).map((t) => t.world));
  dotPlot($('#chartE'), m.episodes.map((e) => ({
    value: e.steps, color: causeColor(e.cause), ring: onMap.has(e.world), e,
    tipTitle: `world ${e.world}`, tipRows: [{ color: causeColor(e.cause), dot: true, value: `${e.steps} steps`, name: CAUSE_LABEL[e.cause] },
      { value: fmt(e.ret), name: 'return' }, { value: `${e.eats} / ${e.drinks}`, name: 'meals / drinks' }],
    onClick: () => (S.addMode ? addTrack(m.id, e.world) : playSingle(m.id, e.world, true)),
  })), { domain: [0, S.index.schema.episode_steps], xlabel: 'steps survived', maxHeight: 170, minHeight: 80, bins: 16,
    aria: 'Steps survived in each of this model’s episodes', sort: (a, b) => a.e.cause.localeCompare(b.e.cause) });

  const tb = h('tbody');
  for (const e of m.episodes) {
    tb.append(h('tr', { class: 'row' + (onMap.has(e.world) ? ' sel' : ''), tabindex: 0, onclick: () => playSingle(m.id, e.world, true),
      onkeydown: (ev) => { if (ev.key === 'Enter') playSingle(m.id, e.world, true); } },
      h('td', { class: 'num' }, String(e.world)), h('td', { class: 'num' }, String(e.steps)), h('td', {}, CAUSE_LABEL[e.cause]),
      h('td', { class: 'num' }, fmt(e.ret)), h('td', { class: 'num' }, String(e.eats)), h('td', { class: 'num' }, String(e.drinks)),
      h('td', { class: 'num' }, fmt(e.night_sheltered, 'ep_night_sheltered')), h('td', { class: 'num' }, String(e.cells))));
  }
  $('#epTable').replaceChildren(h('table', {}, h('thead', {}, h('tr', {}, ...['World', 'Steps', 'Outcome', 'Return', 'Meals', 'Drinks', 'Night sheltered', 'Cells visited']
    .map((t, i) => h('th', { class: i === 2 ? '' : 'num' }, t)))), tb));

  const sb = h('tbody');
  const groups = [['ep_', 'From its 32 recorded episodes'], ['train_', 'Training log, mean of the 10 updates before this snapshot'], ['eval_', 'Snapshot evaluation (scripts/snapshot_eval.py, 2048 episodes per model)']];
  for (const [pre, title] of groups) {
    const keys = Object.keys(m.stats).filter((k) => k.startsWith(pre));
    if (!keys.length) continue;
    sb.append(h('tr', {}, h('th', { colspan: 2 }, title)));
    for (const k of keys) sb.append(h('tr', {}, h('td', {}, label(k)), h('td', { class: 'num' }, fmt(m.stats[k], k))));
  }
  $('#statTable').replaceChildren(h('table', {}, sb));
  const siblings = S.index.models.filter((x) => x.run === m.run).length;
  $('#btnSnaps').disabled = siblings < 2;
  $('#btnAdd').classList.toggle('on', S.addMode);
}

function selectModel(id, keepTracks) {
  const m = modelById(id);
  if (!m) return;
  S.model = m;
  renderAll();
  if (!keepTracks) {
    const w = S.tracks[S.focus] ? S.tracks[S.focus].world : 0;
    playSingle(id, w, false);
  }
}

// ----------------------------------------------------------------- player ---
function seqColor(i, n) {
  const ramp = isDark() ? SEQ_DARK : SEQ_LIGHT;
  return ramp[n <= 1 ? ramp.length - 3 : Math.round(i / (n - 1) * (ramp.length - 1))];
}
async function makeTrack(id, world) {
  const eps = await loadEp(id), m = modelById(id);
  return { model: m, world, ep: eps[world], color: cssVar('--series-1'), label: `${m.run} u${m.update} · world ${world}` };
}
function setTracks(tracks, mode, focus, autoplay) {
  S.tracks = tracks; S.mode = mode; S.focus = Math.max(0, Math.min(focus || 0, tracks.length - 1));
  S.t = 0; S.playing = !!autoplay; S.last = 0;
  const f = tracks[S.focus];
  if (f) history.replaceState(null, '', `#m=${encodeURIComponent(f.model.id)}&w=${f.world}`);
  renderPlayerChrome();
  if (S.model) renderModel();
}
async function playSingle(id, world, autoplay) {
  const t = await makeTrack(id, world);
  setTracks([t], 'single', 0, autoplay);
}
async function playWorlds(id) {
  const eps = await loadEp(id), m = modelById(id);
  const cur = S.tracks[S.focus] ? S.tracks[S.focus].world : 0;
  const tracks = eps.map((e, w) => ({ model: m, world: w, ep: e, color: cssVar('--series-1'), label: `world ${w}` }));
  setTracks(tracks, 'worlds', cur, true);
}
async function playSnapshots(run, world) {
  const ms = S.index.models.filter((m) => m.run === run).sort((a, b) => a.update - b.update);
  const tracks = await Promise.all(ms.map((m) => makeTrack(m.id, world)));
  tracks.forEach((t, i) => { t.color = seqColor(i, tracks.length); t.label = `update ${t.model.update}`; });
  const cur = S.model ? ms.findIndex((m) => m.id === S.model.id) : tracks.length - 1;
  setTracks(tracks, 'snapshots', cur < 0 ? tracks.length - 1 : cur, true);
}
async function addTrack(id, world) {
  if (S.tracks.some((t) => t.model.id === id && t.world === world)) return;
  if (S.tracks.length >= 8) { $('#playNote').textContent = 'Eight agents is the most this mode shows (one colour each). Remove one first.'; return; }
  const t = await makeTrack(id, world);
  const tracks = [...S.tracks, t];
  tracks.forEach((x, i) => { x.color = cssVar(`--series-${i + 1}`); x.label = `${x.model.run} u${x.model.update} · world ${x.world}`; });
  const keepT = S.t;
  setTracks(tracks, 'mixed', S.focus, false);
  S.t = keepT;
}
const maxSteps = () => Math.max(1, ...S.tracks.map((t) => t.ep.steps));
const speedOf = () => 0.5 * 2 ** (Number($('#speed').value) / 10);

function renderPlayerChrome() {
  const f = S.tracks[S.focus];
  const title = { single: 'One episode', worlds: 'All 32 episodes of one model, superimposed', snapshots: 'One world, every saved model of a run', mixed: 'Chosen episodes, superimposed' }[S.mode];
  $('#playTitle').textContent = f ? `Playback: ${title}` : 'Playback';
  const same = S.tracks.every((t) => t.world === (f && f.world));
  let note = '';
  if (f) {
    note = `Map: world ${f.world}. `;
    if (S.mode === 'snapshots') note += 'Trail colour runs from early training (pale) to late (strong). The adult and predator shown belong to the highlighted agent.';
    else if (!same) note += 'The other agents are in different worlds, so only their trails are drawn and this map is dimmed. Click an agent below to see its world.';
    else if (S.tracks.length > 1) note += 'The adult and predator shown belong to the highlighted agent.';
  }
  $('#playNote').textContent = note;
  $('#scrub').max = maxSteps() - 1;
  const box = $('#tracks');
  box.replaceChildren();
  if (S.tracks.length > 1) {
    S.tracks.forEach((t, i) => {
      box.append(h('span', { class: 'item' + (i === S.focus ? ' focus' : ''), role: 'button', tabindex: 0, title: 'Highlight this agent',
        onclick: () => { S.focus = i; if (t.model.id !== (S.model && S.model.id)) selectModel(t.model.id, true); renderPlayerChrome(); if (S.model) renderModel(); } },
        h('span', { class: 'dot', style: `background:${t.color}` }), `${t.label} (${t.ep.steps})`,
        h('button', { class: 'x', type: 'button', title: 'Remove from the map', 'aria-label': 'Remove', onclick: (ev) => {
          ev.stopPropagation();
          const rest = S.tracks.filter((_, k) => k !== i);
          if (rest.length) { const keepT = S.t; setTracks(rest, rest.length === 1 ? 'single' : S.mode, Math.min(S.focus, rest.length - 1), false); S.t = keepT; }
        } }, '×')));
    });
  }
  $('#btnPlay').textContent = S.playing ? 'Pause' : 'Play';
}

function renderMapKey() {
  const items = [['--map-tree', 'tree'], ['--map-rock', 'rock'], ['--map-water', 'water'], ['--map-food', 'food (size = stock)'],
    ['--map-shelter', 'shelter'], ['--map-hazard', 'hazard'], ['--series-1', 'learner'], ['--map-adult', 'adult (line = pointing)'], ['--map-predator', 'predator (night)']];
  $('#mapKey').replaceChildren(...items.map(([v, l]) => h('span', { class: 'item' }, h('span', { class: 'sw', style: `background:${cssVar(v)}` }), l)));
}

const F = (t, i, name) => t.ep.u8[i * t.ep.F + S.field[name]];
function posAt(t, time, a, b) {
  const last = t.ep.steps - 1, i = Math.min(Math.floor(time), last), j = Math.min(i + 1, last), fr = time >= last ? 0 : time - i;
  const x0 = F(t, i, a), y0 = F(t, i, b), x1 = F(t, j, a), y1 = F(t, j, b);
  if (Math.abs(x1 - x0) + Math.abs(y1 - y0) > 1) return [x0, y0];
  return [x0 + (x1 - x0) * fr, y0 + (y1 - y0) * fr];
}

function draw() {
  const cv = $('#map'), ctx = cv.getContext('2d'), size = cv.width;
  const f = S.tracks[S.focus], sc = S.index.schema, G = sc.grid;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (!f) { ctx.fillStyle = cssVar('--map-day'); ctx.fillRect(0, 0, size, size); return; }
  const time = Math.min(S.t, maxSteps() - 1);
  const fi = Math.min(Math.floor(time), f.ep.steps - 1);
  const tod = Math.floor(time) % sc.day_length;
  const phase = tod >= sc.night_start ? 2 : tod >= sc.dusk_start ? 1 : 0;
  const same = S.tracks.every((t) => t.world === f.world);

  // camera
  const [flx, fly] = posAt(f, time, 'lx', 'ly');
  const V = $('#follow').checked ? 15 : G;
  const cx = V === G ? G / 2 : Math.max(V / 2, Math.min(G - V / 2, flx + 0.5));
  const cy = V === G ? G / 2 : Math.max(V / 2, Math.min(G - V / 2, fly + 0.5));
  const c = size / V, ox = cx - V / 2, oy = cy - V / 2;
  const px = (x) => (x - ox) * c, py = (y) => (y - oy) * c;

  ctx.fillStyle = cssVar(['--map-day', '--map-dusk', '--map-night'][phase]);
  ctx.fillRect(0, 0, size, size);

  // view cone of the highlighted learner (before occlusion by trees and rocks)
  if ($('#cone').checked && time < f.ep.steps) {
    const lx = F(f, fi, 'lx'), ly = F(f, fi, 'ly'), lf = F(f, fi, 'lf');
    ctx.fillStyle = cssVar('--map-cone');
    for (let ry = 0; ry <= sc.view_depth; ry++) for (let rx = -Math.min(ry, sc.view_half); rx <= Math.min(ry, sc.view_half); rx++) {
      const [dx, dy] = lf === 0 ? [rx, -ry] : lf === 1 ? [ry, rx] : lf === 2 ? [-rx, ry] : [-ry, -rx];
      const x = lx + dx, y = ly + dy;
      if (x >= 0 && x < G && y >= 0 && y < G) ctx.fillRect(px(x), py(y), c, c);
    }
  }

  // terrain of the highlighted agent's world
  const ter = S.terrain[f.world], W = S.worlds.worlds[f.world];
  ctx.globalAlpha = same ? 1 : 0.35;
  const stock = new Map(W.food.map((cell, k) => [cell, F(f, fi, 'food' + k) / 255]));
  for (let y = Math.max(0, Math.floor(oy)); y < Math.min(G, Math.ceil(oy + V)); y++) for (let x = Math.max(0, Math.floor(ox)); x < Math.min(G, Math.ceil(ox + V)); x++) {
    const k = ter[y * G + x];
    if (!k) continue;
    const X0 = px(x), Y0 = py(y), m = c / 2;
    if (k === 1) { const st = stock.has(y * G + x) ? stock.get(y * G + x) : 1; ctx.fillStyle = cssVar('--map-food'); ctx.beginPath(); ctx.arc(X0 + m, Y0 + m, c * (0.12 + 0.3 * st), 0, 7); ctx.fill(); }
    else if (k === 2) { ctx.fillStyle = cssVar('--map-water'); ctx.fillRect(X0 + 0.5, Y0 + 0.5, c - 1, c - 1); }
    else if (k === 3) { ctx.fillStyle = cssVar('--map-tree'); ctx.beginPath(); ctx.arc(X0 + m, Y0 + m, c * 0.42, 0, 7); ctx.fill(); }
    else if (k === 4) { ctx.fillStyle = cssVar('--map-rock'); ctx.fillRect(X0 + c * 0.1, Y0 + c * 0.1, c * 0.8, c * 0.8); }
    else if (k === 5) { ctx.fillStyle = cssVar('--map-shelter'); ctx.beginPath(); ctx.moveTo(X0 + m, Y0 + c * 0.02); ctx.lineTo(X0 + c * 0.98, Y0 + c * 0.45); ctx.lineTo(X0 + c * 0.85, Y0 + c * 0.45);
      ctx.lineTo(X0 + c * 0.85, Y0 + c * 0.95); ctx.lineTo(X0 + c * 0.15, Y0 + c * 0.95); ctx.lineTo(X0 + c * 0.15, Y0 + c * 0.45); ctx.lineTo(X0 + c * 0.02, Y0 + c * 0.45); ctx.closePath(); ctx.fill(); }
    else if (k === 7) { ctx.strokeStyle = cssVar('--map-hazard'); ctx.lineWidth = Math.max(1.5, c * 0.14); ctx.beginPath(); ctx.moveTo(X0 + c * 0.2, Y0 + c * 0.2); ctx.lineTo(X0 + c * 0.8, Y0 + c * 0.8);
      ctx.moveTo(X0 + c * 0.8, Y0 + c * 0.2); ctx.lineTo(X0 + c * 0.2, Y0 + c * 0.8); ctx.stroke(); }
  }
  ctx.globalAlpha = 1;
  if (c >= 14) { ctx.strokeStyle = cssVar('--map-line'); ctx.lineWidth = 1; ctx.beginPath();
    for (let i = Math.ceil(ox); i <= ox + V; i++) { ctx.moveTo(px(i), 0); ctx.lineTo(px(i), size); }
    for (let i = Math.ceil(oy); i <= oy + V; i++) { ctx.moveTo(0, py(i)); ctx.lineTo(size, py(i)); }
    ctx.stroke(); }

  const tri = (x, y, facing, r, fill, stroke) => {
    const [dx, dy] = sc.facing_dxy[facing], X0 = px(x) + c / 2, Y0 = py(y) + c / 2;
    ctx.beginPath(); ctx.moveTo(X0 + dx * r, Y0 + dy * r); ctx.lineTo(X0 - dx * r * 0.7 - dy * r * 0.75, Y0 - dy * r * 0.7 + dx * r * 0.75);
    ctx.lineTo(X0 - dx * r * 0.7 + dy * r * 0.75, Y0 - dy * r * 0.7 - dx * r * 0.75); ctx.closePath();
    ctx.fillStyle = fill; ctx.fill();
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = Math.max(1.5, c * 0.08); ctx.stroke(); }
  };

  // trails, then agents; the highlighted one last so it is on top
  const order = S.tracks.map((_, i) => i).filter((i) => i !== S.focus).concat([S.focus]);
  const many = S.tracks.length > 1;
  for (const i of order) {
    const t = S.tracks[i], isF = i === S.focus, last = t.ep.steps - 1, ti = Math.min(Math.floor(time), last);
    const [x, y] = posAt(t, time, 'lx', 'ly');
    if ($('#trails').checked) {
      ctx.strokeStyle = t.color; ctx.globalAlpha = isF || !many ? 0.9 : S.mode === 'worlds' ? 0.3 : 0.6;
      ctx.lineWidth = Math.max(1.5, c * (isF ? 0.16 : 0.1)); ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(px(F(t, 0, 'lx')) + c / 2, py(F(t, 0, 'ly')) + c / 2);
      for (let k = 1; k <= ti; k++) ctx.lineTo(px(F(t, k, 'lx')) + c / 2, py(F(t, k, 'ly')) + c / 2);
      ctx.lineTo(px(x) + c / 2, py(y) + c / 2); ctx.stroke();
    }
    ctx.globalAlpha = isF || !many ? 1 : 0.75;
    const dead = time >= last && t.ep.cause !== 'alive';
    if (dead) {
      const X0 = px(x) + c / 2, Y0 = py(y) + c / 2, r = c * 0.36;
      ctx.strokeStyle = t.color; ctx.lineWidth = Math.max(2, c * 0.16); ctx.beginPath();
      ctx.moveTo(X0 - r, Y0 - r); ctx.lineTo(X0 + r, Y0 + r); ctx.moveTo(X0 + r, Y0 - r); ctx.lineTo(X0 - r, Y0 + r); ctx.stroke();
    } else {
      tri(x, y, F(t, ti, 'lf'), c * 0.46, t.color, cssVar('--surface'));
      if (F(t, ti, 'sheltered')) { ctx.strokeStyle = t.color; ctx.lineWidth = Math.max(1.5, c * 0.08); ctx.beginPath(); ctx.arc(px(x) + c / 2, py(y) + c / 2, c * 0.62, 0, 7); ctx.stroke(); }
    }
    ctx.globalAlpha = 1;
  }

  // the highlighted agent's adult and predator
  if (time < f.ep.steps) {
    const [ax, ay] = posAt(f, time, 'ax', 'ay'), af = F(f, fi, 'af'), adult = cssVar('--map-adult');
    const [dx, dy] = sc.facing_dxy[af], X0 = px(ax) + c / 2, Y0 = py(ay) + c / 2;
    if (F(f, fi, 'apoint')) { ctx.strokeStyle = adult; ctx.lineWidth = Math.max(1.5, c * 0.09); ctx.beginPath(); ctx.moveTo(X0, Y0); ctx.lineTo(X0 + dx * c * 2.6, Y0 + dy * c * 2.6); ctx.stroke();
      ctx.beginPath(); ctx.arc(X0 + dx * c * 2.6, Y0 + dy * c * 2.6, Math.max(2, c * 0.13), 0, 7); ctx.fillStyle = adult; ctx.fill(); }
    tri(ax, ay, af, c * 0.5, adult, cssVar('--surface'));
    const word = sc.words[F(f, fi, 'asay')];
    if (F(f, fi, 'asay') && word) {
      ctx.font = `600 ${Math.max(11, Math.min(15, Math.round(c * 0.62)))}px system-ui, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      const tw = ctx.measureText(word).width + 8, th = Math.max(14, Math.min(20, c * 0.8)), bx = Math.max(2, Math.min(size - tw - 2, X0 - tw / 2)), by = Math.max(2, Y0 - c * 0.6 - th);
      ctx.fillStyle = cssVar('--surface'); ctx.globalAlpha = 0.92; ctx.fillRect(bx, by, tw, th); ctx.globalAlpha = 1;
      ctx.fillStyle = cssVar('--text'); ctx.fillText(word, bx + tw / 2, by + th - 1);
    }
    if (F(f, fi, 'pactive')) {
      const X1 = px(F(f, fi, 'px')) + c / 2, Y1 = py(F(f, fi, 'py')) + c / 2, r = c * 0.5;
      ctx.fillStyle = cssVar('--map-predator'); ctx.beginPath(); ctx.moveTo(X1, Y1 - r); ctx.lineTo(X1 + r, Y1); ctx.lineTo(X1, Y1 + r); ctx.lineTo(X1 - r, Y1); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = cssVar('--surface'); ctx.lineWidth = Math.max(1.5, c * 0.08); ctx.stroke();
    }
  }
  renderHud(f, fi, time, phase);
}

let hudKey = '';
function renderHud(f, fi, time, phase) {
  const sc = S.index.schema, step = Math.floor(time);
  const key = `${S.focus}|${f.model.id}|${f.world}|${step}`;
  $('#scrub').value = step;
  $('#stepLabel').textContent = `${step + 1} / ${maxSteps()}`;
  if (key === hudKey) return;
  hudKey = key;
  const ended = step >= f.ep.steps - 1;
  const kv = (k, v) => h('div', { class: 'kv' }, h('span', { class: 'k' }, k), h('span', { class: 'v' }, v));
  const meter = (k, name) => { const v = F(f, fi, name) / 255; return h('div', { class: 'kv' }, h('span', { class: 'k' }, k),
    h('span', { class: 'meter' + (v < 0.25 ? ' low' : ''), role: 'img', 'aria-label': `${k} ${Math.round(v * 100)}%` }, h('i', { style: `width:${Math.round(v * 100)}%` })), h('span', { class: 'num' }, Math.round(v * 100) + '%')); };
  const dirs = ['ahead', 'to the right', 'behind', 'to the left'];
  const heard = F(f, fi, 'heard');
  $('#hud').replaceChildren(
    h('div', { class: 'kv wide' }, h('span', { class: 'k' }, 'Agent'), h('span', { class: 'v' }, `${f.model.run} \u00b7 update ${f.model.update} \u00b7 world ${f.world}`)),
    kv('Time', `day ${Math.floor(step / sc.day_length) + 1}, ${['day', 'dusk', 'night'][phase]}`),
    kv('Status', ended ? `${CAUSE_LABEL[f.ep.cause]} at step ${f.ep.steps}` : F(f, fi, 'sheltered') ? 'in the shelter' : 'alive'),
    kv('Action', sc.actions[F(f, fi, 'act')].replace('_', ' ')),
    kv('Hears', heard ? `"${sc.words[heard]}" from ${dirs[F(f, fi, 'heard_dir')] || 'nearby'}` : 'nothing'),
    kv('Adult', (F(f, fi, 'asay') ? `says "${sc.words[F(f, fi, 'asay')]}"` : 'silent') + (F(f, fi, 'apoint') ? ', pointing' : '')),
    meter('Hunger', 'hunger'), meter('Thirst', 'thirst'), meter('Warmth', 'warmth'));
}

function frame(now) {
  if (S.playing) {
    const dt = S.last ? Math.min(0.1, (now - S.last) / 1000) : 0;
    S.t += dt * speedOf();
    if (S.t >= maxSteps() - 1) { S.t = maxSteps() - 1; S.playing = false; $('#btnPlay').textContent = 'Play'; }
  }
  S.last = now;
  if (S.index) draw();
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------- live ---
/* Rows arrive from /live/state.json, which the server fills by reading each running instance's log.csv. */
const LIVE = { runs: new Map(), hover: false, dirty: false, sig: '' };
const LIVE_METRICS = ['loss', 'world', 'heard', 'deaths', 'reward_per_step', 'gaze_gap', 'ent_action', 'clipfrac'];
const STATE_LABEL = { starting: 'starting', loading: 'loading the image', setup: 'setting up', training: 'training', ended: 'ended' };

function liveCol(r, k) {
  if (k === 'gaze_gap') {
    const a = r.cols.gaze_follow_rate, b = r.cols.gaze_chance;
    return a && b ? a.map((v, i) => (v == null || b[i] == null ? null : v - b[i])) : null;
  }
  return r.cols[k] || null;
}
const tailMean = (ys, n) => { if (!ys) return null; const v = ys.slice(-n).filter((x) => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
function liveColor(name) {
  const i = [...LIVE.runs.keys()].sort().indexOf(name);
  return i >= 0 && i < 8 ? cssVar(`--series-${i + 1}`) : cssVar('--other');
}
function duration(sec) {
  if (sec == null || !isFinite(sec)) return '–';
  const m = Math.round(sec / 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : m >= 1 ? `${m} min` : `${Math.round(sec)} s`;
}

async function pollLive() {
  let delay = document.hidden ? 20000 : 4000;
  try {
    const have = [...LIVE.runs.values()].map((r) => `${encodeURIComponent(r.name)}:${r.n}`).join(',');
    const d = await getJSON('live/state.json?have=' + have);
    if (d.disabled) return;
    const seen = new Set();
    let changed = false;
    for (const r of d.runs) {
      seen.add(r.name);
      let loc = LIVE.runs.get(r.name);
      if (!loc || r.start !== loc.n || r.instance !== loc.instance) {
        if (r.start !== 0) { LIVE.runs.set(r.name, { name: r.name, n: 0, cols: {}, instance: r.instance }); changed = true; continue; }   // out of step: ask for everything
        loc = { name: r.name, n: 0, cols: {} };
        LIVE.runs.set(r.name, loc);
      }
      for (const [k, v] of Object.entries(r.cols)) loc.cols[k] = (loc.cols[k] || []).concat(v);
      if (r.n !== loc.n || r.state !== loc.state || r.detail !== loc.detail) changed = true;
      Object.assign(loc, { n: r.n, state: r.state, detail: r.detail, gpu: r.gpu, dph: r.dph, total: r.total, instance: r.instance, age: r.last_row_age_s });
    }
    for (const k of [...LIVE.runs.keys()]) if (!seen.has(k)) { LIVE.runs.delete(k); changed = true; }
    LIVE.error = d.error;
    // the server records a run's models as soon as it ends; when that build lands, take the new data in place
    if (d.build) {
      $('#built').textContent = d.build.running ? `${S.builtText} \u00b7 recording new models (${d.build.why})\u2026` : S.builtText;
      if (S.built && d.build.built > S.built + 0.5) { S.built = d.build.built; await applyIndex(); }
      else if (!S.built) S.built = d.build.built;
    }
    if (changed) { if (LIVE.hover) LIVE.dirty = true; else renderLive(); }
    if (![...LIVE.runs.values()].some((r) => r.state !== 'ended')) delay = 20000;
  } catch (e) { delay = 15000; }
  setTimeout(pollLive, delay);
}

function renderLive() {
  LIVE.dirty = false;
  const runs = [...LIVE.runs.values()].sort((a, b) => a.name.localeCompare(b.name));
  const box = $('#live');
  box.hidden = !runs.length;
  if (!runs.length) return;
  const active = runs.filter((r) => r.state !== 'ended').length;
  $('#liveTitle').replaceChildren(active ? h('span', { class: 'pulse', 'aria-hidden': 'true' }) : null,
    active ? `Training now: ${active} run${active === 1 ? '' : 's'}` : 'Recently ended runs');
  $('#liveLegend').replaceChildren(...runs.map((r) => h('span', { class: 'item' }, h('span', { class: 'key', style: `border-color:${liveColor(r.name)}` }), r.name)));

  const tb = h('tbody');
  let cost = 0;
  for (const r of runs) {
    const el = r.cols.elapsed_s, n = r.n;
    const k = Math.min(20, n - 1);
    const rate = el && k > 0 && el[n - 1] != null && el[n - 1 - k] != null ? (el[n - 1] - el[n - 1 - k]) / k : null;
    const upd = r.cols.update && n ? r.cols.update[n - 1] : 0;
    const frac = r.total ? Math.min(1, upd / r.total) : 0;
    if (r.state !== 'ended' && r.dph) cost += r.dph;
    const eta = r.state === 'training' && rate && r.total ? (r.total - upd) * rate : null;
    tb.append(h('tr', { class: r.state === 'ended' ? 'ended' : '' },
      h('td', {}, h('span', { class: 'key', style: `display:inline-block;vertical-align:middle;margin-right:6px;border-color:${liveColor(r.name)}` }), r.name),
      h('td', {}, STATE_LABEL[r.state] || r.state, r.detail ? h('span', { class: 'sub' }, r.detail) : null),
      h('td', { class: 'num' }, h('span', { class: 'meter', role: 'img', 'aria-label': `${Math.round(frac * 100)}% done` }, h('i', { style: `width:${(frac * 100).toFixed(1)}%` })), `${upd} / ${r.total || '?'}`),
      h('td', { class: 'num' }, rate ? rate.toFixed(1) + ' s' : '–'),
      h('td', { class: 'num' }, duration(eta)),
      h('td', { class: 'num' }, fmt(tailMean(liveCol(r, 'loss'), 10))),
      h('td', { class: 'num' }, fmt(tailMean(liveCol(r, 'deaths'), 10))),
      h('td', { class: 'num' }, fmt(tailMean(liveCol(r, 'gaze_gap'), 10), 'train_gaze_gap')),
      h('td', { class: 'num' }, r.dph ? '$' + r.dph.toFixed(2) : '–')));
  }
  $('#liveTable').replaceChildren(h('table', {}, h('thead', {}, h('tr', {}, ...[['Run'], ['State'], ['Update', 1], ['Per update', 1], ['Time left', 1], ['Loss', 1], ['Deaths per update', 1], ['Gaze above chance', 1], ['Per hour', 1]]
    .map(([t, nm]) => h('th', { class: nm ? 'num' : '' }, t)))), tb),
    LIVE.error ? h('p', { class: 'hint' }, `Live data problem: ${LIVE.error}.`) : cost ? h('p', { class: 'hint' }, `Running instances cost $${cost.toFixed(2)} per hour in total.`) : null);

  const grid = $('#liveCharts');
  grid.replaceChildren();
  for (const k of LIVE_METRICS) {
    const series = runs.map((r) => ({ r, ys: liveCol(r, k) })).filter((x) => x.ys && x.ys.length);
    if (!series.length) continue;
    const cell = h('div', {}, h('h3', {}, label('train_' + k)), h('div', { class: 'chart' }));
    grid.append(cell);
    liveChart(cell.lastChild, k, series);
  }
}

function liveChart(box, metric, series) {
  const W = box.clientWidth || 300, H = 150, L = 44, R = 8, T = 8, B = 20, win = 10;
  let y0 = Infinity, y1 = -Infinity, x1 = 1;
  const lines = series.map(({ r, ys }) => {
    const pts = [];
    let sum = 0, cnt = 0;
    const q = [];
    for (let i = 0; i < ys.length; i++) {
      q.push(ys[i]); if (ys[i] != null) { sum += ys[i]; cnt++; }
      if (q.length > win) { const d = q.shift(); if (d != null) { sum -= d; cnt--; } }
      const u = r.cols.update && r.cols.update[i] != null ? r.cols.update[i] : i + 1;
      if (cnt && i >= Math.min(win, 4) - 1) { const v = sum / cnt; pts.push([u, v]); y0 = Math.min(y0, v); y1 = Math.max(y1, v); }
    }
    x1 = Math.max(x1, r.total || 0, pts.length ? pts[pts.length - 1][0] : 0);
    return { r, pts, color: liveColor(r.name) };
  }).filter((l) => l.pts.length);
  if (!lines.length) { box.append(h('div', { class: 'empty' }, 'Waiting for the first updates.')); return; }
  const yt = niceTicks(y0, y1, 3);
  y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]);
  if (!(y1 > y0)) { y0 -= 1; y1 += 1; }
  const X = (u) => L + u / x1 * (W - L - R), Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const svg = s('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${label('train_' + metric)} for the runs training now` });
  for (const v of yt) {
    svg.append(s('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: cssVar('--grid'), 'stroke-width': 1 }));
    svg.append(s('text', { x: L - 5, y: Y(v) + 4, 'text-anchor': 'end' }, tickFmt(v, false)));
  }
  svg.append(s('line', { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: cssVar('--axis'), 'stroke-width': 1 }));
  for (const v of niceTicks(0, x1, 3)) if (v <= x1) svg.append(s('text', { x: X(v), y: H - 5, 'text-anchor': v === 0 ? 'start' : X(v) > W - 24 ? 'end' : 'middle' }, tickFmt(v, false)));
  for (const l of lines) {
    svg.append(s('path', { d: l.pts.map(([u, v], i) => `${i ? 'L' : 'M'}${X(u).toFixed(1)},${Y(v).toFixed(1)}`).join(''), fill: 'none', stroke: l.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    const [u, v] = l.pts[l.pts.length - 1];
    if (l.r.state !== 'ended') svg.append(s('circle', { cx: X(u), cy: Y(v), r: 4, fill: l.color, stroke: cssVar('--surface'), 'stroke-width': 2 }));
  }
  const cross = s('line', { y1: T, y2: H - B, stroke: cssVar('--axis'), 'stroke-width': 1, visibility: 'hidden' });
  svg.append(cross);
  svg.addEventListener('pointermove', (ev) => {
    LIVE.hover = true;
    const rc = svg.getBoundingClientRect(), x = ev.clientX - rc.left;
    const u = Math.max(1, Math.round((x - L) / (W - L - R) * x1));
    const rows = [];
    for (const l of lines) {
      let best = null;
      for (const p of l.pts) if (!best || Math.abs(p[0] - u) < Math.abs(best[0] - u)) best = p;
      if (best && Math.abs(best[0] - u) <= Math.max(3, x1 / 60)) rows.push({ color: l.color, value: fmt(best[1], metric === 'gaze_gap' ? 'train_gaze_gap' : null), name: l.r.name });
    }
    if (!rows.length) { hideTip(); cross.setAttribute('visibility', 'hidden'); return; }
    cross.setAttribute('x1', X(u)); cross.setAttribute('x2', X(u)); cross.setAttribute('visibility', 'visible');
    showTip(ev.clientX, ev.clientY, `update ${u}`, rows);
  });
  svg.addEventListener('pointerleave', () => { LIVE.hover = false; hideTip(); cross.setAttribute('visibility', 'hidden'); if (LIVE.dirty) renderLive(); });
  box.append(svg);
}

// ------------------------------------------------------------------- wire ---
function renderAll() { renderLegend(); renderA(); renderB(); renderTable(); renderModel(); }
function sizeCanvas() {
  const cv = $('#map'), w = Math.round(cv.clientWidth * Math.min(2, devicePixelRatio || 1));
  if (w && cv.width !== w) { cv.width = w; cv.height = w; }
}
function togglePlay() {
  if (!S.tracks.length) return;
  if (!S.playing && S.t >= maxSteps() - 1) S.t = 0;
  S.playing = !S.playing; S.last = 0;
  $('#btnPlay').textContent = S.playing ? 'Pause' : 'Play';
}
function stepBy(d) { S.playing = false; $('#btnPlay').textContent = 'Play'; S.t = Math.max(0, Math.min(maxSteps() - 1, Math.floor(S.t) + d)); }

/* Colour follows the run for as long as it is shown. There are eight colours: a visible run takes the lowest
   free one and keeps it; hiding a run frees its colour for a run that has none (those are grey meanwhile). */
function assignColors() {
  for (const [name] of [...S.slot]) if (S.hidden.has(name) || !S.index.runs.some((r) => r.name === name)) S.slot.delete(name);
  for (const r of S.index.runs) {
    if (S.hidden.has(r.name) || S.slot.has(r.name)) continue;
    const used = new Set(S.slot.values());
    for (let k = 1; k <= 8; k++) if (!used.has(k)) { S.slot.set(r.name, k); break; }
  }
  for (const r of S.index.runs) S.runColor.set(r.name, S.slot.has(r.name) ? cssVar(`--series-${S.slot.get(r.name)}`) : cssVar('--other'));
}

/* Load (or load again, after the server rebuilt the data) the index and the maps, keeping what is selected. */
async function applyIndex() {
  const [index, worlds] = await Promise.all([getJSON('data/index.json'), getJSON('data/worlds.json')]);
  const first = !S.index;
  S.index = index; S.worlds = worlds;
  S.terrain = worlds.worlds.map((w) => b64bytes(w.terrain));
  index.fields.forEach((f, i) => { S.field[f] = i; });
  S.builtText = `${index.models.length} models in ${index.runs.length} runs \u00b7 data built ${index.built_utc.replace('T', ' ').replace('Z', ' UTC')}`;
  $('#built').textContent = S.builtText;
  for (const r of index.runs) {
    if (!S.known.has(r.name) && r.name.startsWith('smoke')) S.hidden.add(r.name);     // smoke tests start hidden
    S.known.add(r.name);
  }
  assignColors();
  S.logs.clear(); S.eps.clear();

  const keys = new Set();
  for (const m of index.models) for (const k of Object.keys(m.stats)) keys.add(k);
  const sel = $('#metric');
  sel.replaceChildren();
  for (const [pre, title] of [['train_', 'Training log (mean of the 10 updates before each snapshot)'], ['ep_', 'From each model\u2019s 32 recorded episodes'], ['eval_', 'Snapshot evaluation (scripts/snapshot_eval.py, 2048 episodes per model)']]) {
    const ks = [...keys].filter((k) => k.startsWith(pre));
    if (!ks.length) continue;
    const g = h('optgroup', { label: title });
    ks.sort((a, b) => (LABEL[b] ? 1 : 0) - (LABEL[a] ? 1 : 0) || label(a).localeCompare(label(b)));
    for (const k of ks) g.append(h('option', { value: k }, label(k)));
    sel.append(g);
  }
  if (!S.metric || !keys.has(S.metric)) S.metric = keys.has('train_gaze_gap') ? 'train_gaze_gap' : keys.has('ep_steps') ? 'ep_steps' : [...keys][0];
  sel.value = S.metric;
  if (!first) {
    if (S.model) S.model = modelById(S.model.id) || null;
    renderAll();
  }
}

async function init() {
  await applyIndex();
  const index = S.index, sel = $('#metric');
  sel.addEventListener('change', () => { S.metric = sel.value; renderA(); renderB(); renderTable(); });
  $('#smooth').addEventListener('change', renderA);
  $('#logx').addEventListener('change', renderA);

  $('#btnPlay').addEventListener('click', togglePlay);
  $('#btnBack').addEventListener('click', () => stepBy(-1));
  $('#btnFwd').addEventListener('click', () => stepBy(1));
  $('#scrub').addEventListener('input', (e) => { S.playing = false; $('#btnPlay').textContent = 'Play'; S.t = Number(e.target.value); });
  const speedLabel = () => { const v = speedOf(); $('#speedLabel').textContent = `${v >= 10 ? v.toFixed(0) : v.toFixed(1)} steps/s`; };
  $('#speed').addEventListener('input', speedLabel); speedLabel();
  $('#btnWorlds').addEventListener('click', () => S.model && playWorlds(S.model.id));
  $('#btnSnaps').addEventListener('click', () => S.model && playSnapshots(S.model.run, S.tracks[S.focus] ? S.tracks[S.focus].world : 0));
  $('#btnAdd').addEventListener('click', () => { S.addMode = !S.addMode; $('#btnAdd').classList.toggle('on', S.addMode);
    $('#btnAdd').textContent = S.addMode ? 'Adding: click episodes (click here to stop)' : 'Add next clicked episode to the map'; });
  document.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT', 'BUTTON', 'TEXTAREA', 'SUMMARY'].includes(e.target.tagName) && !(e.target.type === 'range')) return;
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    else if (e.key === 'ArrowLeft' && e.target.type !== 'range') stepBy(-1);
    else if (e.key === 'ArrowRight' && e.target.type !== 'range') stepBy(1);
  });
  let rt = 0;
  addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { sizeCanvas(); renderA(); renderB(); renderModel(); renderLive(); }, 120); });
  pollLive();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => location.reload());

  sizeCanvas(); renderMapKey();
  const hash = new URLSearchParams(location.hash.slice(1));
  let start = modelById(hash.get('m'));
  if (!start) {
    const shown = index.models.filter((m) => !S.hidden.has(m.run));
    // the latest model of the run with the most saved models
    const count = (run) => shown.filter((m) => m.run === run).length;
    start = [...shown].sort((a, b) => count(b.run) - count(a.run) || a.run.localeCompare(b.run) || b.update - a.update)[0] || index.models[index.models.length - 1];
  }
  if (start) {
    if (S.hidden.has(start.run)) { S.hidden.delete(start.run); assignColors(); }
    S.model = start;
    renderAll();
    const w = Math.max(0, Math.min(index.n_worlds - 1, Number(hash.get('w')) || 0));
    await playSingle(start.id, w, false);
  } else renderAll();
  requestAnimationFrame(frame);
}

init().catch((e) => {
  document.querySelector('main').replaceChildren(h('div', { class: 'card' }, h('h2', {}, 'The data could not be loaded'),
    h('p', { class: 'hint' }, String(e.message || e) + '. Run viz/build.py on the server, then reload.')));
});
