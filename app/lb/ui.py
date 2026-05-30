from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>eased admin</title>
<style>
:root{
  --bg:#0d1117;--panel:#161b22;--panel2:#1c2128;
  --border:#21262d;--border2:#30363d;
  --text:#c9d1d9;--muted:#8b949e;--dim:#484f58;
  --green:#3fb950;--amber:#e3b341;--red:#f85149;
  --blue:#79c0ff;--orange:#f0883e;--purple:#bc8cff;
  --green-bg:#3fb95018;--amber-bg:#e3b34118;--red-bg:#f8514918;
  --blue-bg:#79c0ff18;--orange-bg:#f0883e18;--purple-bg:#bc8cff18;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{background:var(--bg);color:var(--text);font-family:ui-monospace,'Cascadia Code','JetBrains Mono',monospace;font-size:12px;display:flex;flex-direction:column}

/* ── Header ── */
header{display:flex;align-items:center;gap:10px;padding:8px 14px;border-bottom:1px solid var(--border);background:var(--panel);flex-shrink:0;min-height:42px}
.brand{font-size:13px;font-weight:700;letter-spacing:.5px;color:var(--text)}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--dim);flex-shrink:0;transition:background .4s}
.live-dot.ok{background:var(--green);box-shadow:0 0 6px var(--green)}
.live-dot.err{background:var(--red);box-shadow:0 0 6px var(--red)}
.spacer{flex:1}
#last-upd{color:var(--muted);font-size:11px}
.tok-area{display:flex;gap:6px;align-items:center}
.tok-area input{background:var(--bg);border:1px solid var(--border2);color:var(--text);padding:4px 8px;border-radius:4px;font:inherit;font-size:11px;width:190px}
.tok-area input:focus{outline:none;border-color:var(--blue)}

/* ── Buttons ── */
button{background:var(--panel2);border:1px solid var(--border2);color:var(--muted);padding:4px 10px;border-radius:4px;cursor:pointer;font:inherit;font-size:11px;transition:all .15s}
button:hover{border-color:var(--blue);color:var(--blue)}
button.g{color:var(--green);border-color:#3fb95050;background:var(--green-bg)}
button.g:hover{background:#3fb95030}
button.b{color:var(--blue);border-color:#79c0ff50;background:var(--blue-bg)}
button.b:hover{background:#79c0ff30}
button.r{color:var(--red);border-color:transparent;background:transparent}
button.r:hover{border-color:var(--red);background:var(--red-bg)}

/* ── Summary strip ── */
#summary{display:flex;gap:0;border-bottom:1px solid var(--border);background:var(--panel);flex-shrink:0;overflow-x:auto}
.stat{padding:8px 16px;border-right:1px solid var(--border);flex-shrink:0;min-width:80px}
.stat:last-child{border-right:none}
.sl{font-size:9px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim);margin-bottom:2px}
.sv{font-size:18px;font-weight:700}
.sv.g{color:var(--green)}.sv.b{color:var(--blue)}.sv.a{color:var(--amber)}.sv.r{color:var(--red)}.sv.p{color:var(--purple)}

/* ── Main layout ── */
.main{display:flex;flex:1;overflow:hidden}

/* ── Workers pane ── */
.wpane{flex:1;display:flex;flex-direction:column;overflow:hidden;border-right:1px solid var(--border)}
.ph{display:flex;align-items:center;gap:8px;padding:7px 12px;border-bottom:1px solid var(--border);flex-shrink:0;background:var(--panel)}
.pt{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.6px;color:var(--dim)}
.tscroll{overflow-y:auto;flex:1}
table{width:100%;border-collapse:collapse}
thead th{padding:5px 10px;text-align:left;font-size:9px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim);border-bottom:1px solid var(--border);white-space:nowrap;position:sticky;top:0;background:var(--panel);z-index:1}
tbody tr{border-bottom:1px solid #ffffff06;cursor:pointer;transition:background .1s}
tbody tr:hover{background:#ffffff07}
tbody tr.sel{background:#79c0ff0d;border-left:2px solid var(--blue)}
td{padding:6px 10px;vertical-align:middle;white-space:nowrap}

/* ── Badges ── */
.badge{display:inline-flex;align-items:center;gap:3px;padding:1px 6px;border-radius:9px;font-size:10px;font-weight:500}
.badge::before{content:'';display:inline-block;width:5px;height:5px;border-radius:50%}
.bd-running{background:var(--green-bg);color:var(--green);border:1px solid #3fb95030}.bd-running::before{background:var(--green)}
.bd-starting{background:var(--blue-bg);color:var(--blue);border:1px solid #79c0ff30}.bd-starting::before{background:var(--blue);animation:pulse 1.2s infinite}
.bd-pending{background:var(--amber-bg);color:var(--amber);border:1px solid #e3b34130}.bd-pending::before{background:var(--amber);animation:pulse 1.2s infinite}
.bd-bidding{background:var(--purple-bg);color:var(--purple);border:1px solid #bc8cff30}.bd-bidding::before{background:var(--purple);animation:pulse 1.5s infinite}
.bd-unhealthy{background:var(--orange-bg);color:var(--orange);border:1px solid #f0883e30}.bd-unhealthy::before{background:var(--orange)}
.bd-draining{background:var(--amber-bg);color:var(--amber);border:1px solid #e3b34130}.bd-draining::before{background:var(--amber)}
.bd-terminated{background:#6e768118;color:#6e7681;border:1px solid #6e768130}.bd-terminated::before{background:#6e7681}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

.prov-tag{font-size:9px;color:var(--muted);border:1px solid var(--border2);padding:0 5px;border-radius:3px}
.type-spot{font-size:9px;color:var(--muted)}
.type-od{font-size:9px;color:var(--orange)}

/* ── Right pane ── */
.rpane{width:340px;flex-shrink:0;display:flex;flex-direction:column;overflow:hidden}

/* ── Tab bar ── */
.tabs{display:flex;border-bottom:1px solid var(--border);flex-shrink:0;background:var(--panel)}
.tab{padding:8px 14px;cursor:pointer;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:var(--dim);border-bottom:2px solid transparent;transition:all .15s}
.tab:hover{color:var(--muted)}
.tab.active{color:var(--blue);border-bottom-color:var(--blue)}

/* ── Tab panels ── */
.tpanel{display:none;flex:1;overflow-y:auto;flex-direction:column}
.tpanel.active{display:flex}

/* ── Deploy panel ── */
.dpanel{padding:12px}
.dlabel{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;margin-top:10px}
.dlabel:first-child{margin-top:0}

/* Provider grid */
.prov-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.prov-card{background:var(--panel2);border:1px solid var(--border);border-radius:6px;padding:8px 10px;cursor:pointer;transition:all .15s;position:relative}
.prov-card:hover{border-color:var(--border2)}
.prov-card.active{border-color:var(--blue);background:var(--blue-bg)}
.prov-card.has-workers{border-color:#3fb95030}
.prov-card .pname{font-size:11px;font-weight:600;color:var(--text)}
.prov-card .pcount{font-size:10px;color:var(--muted);margin-top:2px}
.prov-card .pcfg{font-size:9px;margin-top:3px}
.pcfg.yes{color:var(--green)}.pcfg.no{color:var(--dim)}
.prov-card .psel-dot{position:absolute;top:7px;right:8px;width:7px;height:7px;border-radius:50%;background:var(--border2)}
.prov-card.active .psel-dot{background:var(--blue)}

/* Image & launch */
.img-input{width:100%;background:var(--bg);border:1px solid var(--border2);color:var(--text);padding:6px 8px;border-radius:4px;font:inherit;font-size:11px}
.img-input:focus{outline:none;border-color:var(--blue)}
.launch-type{display:flex;gap:6px;margin-top:4px}
.launch-type button{flex:1}
.launch-type button.active{background:var(--blue-bg);border-color:var(--blue);color:var(--blue)}

/* ── Queue panel ── */
.qpanel{padding:12px}
.util-track{background:var(--border);border-radius:3px;height:6px;overflow:hidden;margin:6px 0}
.util-fill{height:100%;border-radius:3px;background:var(--blue);transition:width .4s}
.util-fill.hi{background:var(--amber)}
.util-fill.max{background:var(--red)}
.qworkers{display:flex;flex-direction:column;gap:4px;margin-top:8px}
.qworker{background:var(--panel2);border:1px solid var(--border);border-radius:4px;padding:5px 8px;display:flex;align-items:center;gap:8px}
.qw-id{font-size:10px;color:var(--text)}
.qw-meta{font-size:9px;color:var(--muted);flex:1}
.qw-state{font-size:9px;padding:0 6px;border-radius:9px}
.qws-avail{color:var(--green);background:var(--green-bg)}
.qws-leased{color:var(--blue);background:var(--blue-bg)}
.qws-drain{color:var(--amber);background:var(--amber-bg)}
.qdivider{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;margin:8px 0 4px}

/* ── Events panel ── */
.epanel{flex:1;overflow-y:auto;display:flex;flex-direction:column}
.evhdr{padding:8px 12px;font-size:10px;color:var(--muted);border-bottom:1px solid var(--border);flex-shrink:0;display:flex;align-items:center;gap:6px}
.evwid{color:var(--blue);font-size:10px}
.evlist{overflow-y:auto;flex:1}
.ev{padding:6px 12px;border-bottom:1px solid #ffffff06}
.ev:hover{background:#ffffff04}
.ev-ts{font-size:9px;color:var(--dim)}
.ev-type{font-size:11px;font-weight:500;margin:1px 0}
.ev-msg{font-size:10px;color:var(--muted);word-break:break-word;white-space:pre-wrap}
.ev-meta{font-size:9px;color:var(--dim);margin-top:2px;font-style:italic}
.log-block{margin-top:4px;background:var(--bg);border:1px solid var(--border);border-radius:3px;padding:5px 7px;font-size:9px;color:#6e7681;max-height:150px;overflow:hidden;white-space:pre;cursor:pointer;position:relative}
.log-block.open{max-height:none}
.log-block:not(.open)::after{content:'▸ expand';display:block;position:sticky;bottom:0;text-align:center;color:var(--blue);background:linear-gradient(transparent,var(--bg) 70%);padding-top:8px}

/* ── Empty states ── */
.empty{color:var(--dim);text-align:center;padding:24px 12px;font-size:11px}
</style>
</head>
<body>

<header>
  <div class="live-dot" id="dot"></div>
  <div class="brand">eased admin</div>
  <div class="spacer"></div>
  <span id="last-upd"></span>
  <div class="tok-area">
    <input id="tok" type="password" placeholder="admin token" onkeydown="if(e=>e.key==='Enter')(event),connect()">
    <button onclick="connect()">Connect</button>
    <button onclick="doRefresh()" title="Refresh now">⟳</button>
  </div>
</header>

<div id="summary">
  <div class="stat"><div class="sl">Running</div><div class="sv g" id="sr">—</div></div>
  <div class="stat"><div class="sl">Starting</div><div class="sv b" id="ss">—</div></div>
  <div class="stat"><div class="sl">Pending</div><div class="sv a" id="sp">—</div></div>
  <div class="stat"><div class="sl">Unhealthy</div><div class="sv r" id="su">—</div></div>
  <div class="stat"><div class="sl">$/hr</div><div class="sv" id="srate">—</div></div>
  <div class="stat"><div class="sl">Total spent</div><div class="sv" id="stotal">—</div></div>
  <div class="stat"><div class="sl">Queue util</div><div class="sv" id="squtil">—</div></div>
  <div class="stat"><div class="sl">503s</div><div class="sv r" id="s503">—</div></div>
</div>

<div class="main">

  <!-- Workers table -->
  <div class="wpane">
    <div class="ph">
      <span class="pt">Workers</span>
      <div class="spacer"></div>
      <button class="g" onclick="switchTab('deploy')">↑ Deploy</button>
    </div>
    <div class="tscroll">
      <table>
        <thead>
          <tr>
            <th>ID</th><th>Provider</th><th>GPU</th><th>Type</th>
            <th>Status</th><th>$/hr</th><th>vs market</th>
            <th>Uptime</th><th>Cost</th><th></th>
          </tr>
        </thead>
        <tbody id="wtbody">
          <tr><td colspan="10" class="empty">Enter token to connect</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Right pane -->
  <div class="rpane">
    <div class="tabs">
      <div class="tab active" id="tab-deploy"  onclick="switchTab('deploy')">Deploy</div>
      <div class="tab"        id="tab-queue"   onclick="switchTab('queue')">Queue</div>
      <div class="tab"        id="tab-events"  onclick="switchTab('events')">Events</div>
    </div>

    <!-- Deploy tab -->
    <div class="tpanel active" id="tp-deploy">
      <div class="dpanel">
        <div class="dlabel">Provider</div>
        <div class="prov-grid" id="prov-grid">
          <!-- populated by JS -->
        </div>

        <div class="dlabel">Image</div>
        <input class="img-input" id="img-input" placeholder="ghcr.io/org/image:tag" autocomplete="off">

        <div class="dlabel">Launch type</div>
        <div class="launch-type">
          <button id="lt-spot" class="active" onclick="setLaunchType('spot')">⚡ Spot (bid)</button>
          <button id="lt-od" onclick="setLaunchType('od')">💰 On-demand</button>
        </div>

        <div style="margin-top:14px;display:flex;gap:6px">
          <button class="g" style="flex:1;padding:8px" onclick="launch()">↑ Launch</button>
          <button onclick="resetDeploy()" title="Reset selections">↺</button>
        </div>

        <div id="launch-status" style="margin-top:8px;font-size:10px;color:var(--muted)"></div>
      </div>
    </div>

    <!-- Queue tab -->
    <div class="tpanel" id="tp-queue">
      <div class="qpanel">
        <div style="display:flex;align-items:center;justify-content:space-between">
          <span style="font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px">Queue utilization</span>
          <span id="qpct" style="font-size:11px;color:var(--muted)">—</span>
        </div>
        <div class="util-track"><div class="util-fill" id="qbar" style="width:0"></div></div>
        <div id="qworkers"></div>
      </div>
    </div>

    <!-- Events tab -->
    <div class="tpanel" id="tp-events">
      <div class="evhdr">
        <span>Events</span>
        <span id="ev-wid" class="evwid"></span>
        <div class="spacer"></div>
        <button onclick="clearEvents()" style="padding:2px 7px;font-size:10px">✕</button>
      </div>
      <div class="evlist" id="evlist">
        <div class="empty">Click a worker row to view events</div>
      </div>
    </div>
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let _tok = sessionStorage.getItem('eased_tok') || '';
let _cfg = {};           // from /admin/config
let _selWorker = null;   // selected worker_id
let _selProvider = null; // selected provider in deploy panel
let _launchType = 'spot';
let _timer = null;
let _cd = 15;

const PROVIDERS = ['vastai','runpod','salad','lambdalabs','tensordock'];
const PROV_LABELS = {vastai:'Vast.ai',runpod:'RunPod',salad:'Salad',lambdalabs:'Lambda',tensordock:'TensorDock'};
const PROV_SHORT  = {vastai:'VA',runpod:'RP',salad:'SL',lambdalabs:'LL',tensordock:'TD'};

// ── Auth ───────────────────────────────────────────────────────────────────
function connect() {
  const v = document.getElementById('tok').value.trim();
  if (v) { _tok = v; sessionStorage.setItem('eased_tok', v); }
  doRefresh();
}

async function api(path, opts={}) {
  try {
    const r = await fetch(path, {
      ...opts,
      headers:{ 'Authorization':'Bearer '+_tok, 'Content-Type':'application/json', ...(opts.headers||{}) },
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
    if (r.status===401) { dot('err'); setUpd('⚠ invalid token'); return null; }
    if (!r.ok) { const t=await r.text().catch(()=>''); dot('err'); setUpd(`⚠ ${r.status} ${t.slice(0,60)}`); return null; }
    return r.json();
  } catch(e) { dot('err'); setUpd('⚠ network error'); return null; }
}

// ── Refresh loop ───────────────────────────────────────────────────────────
async function doRefresh() {
  clearTimeout(_timer);
  if (!_tok) return;

  const [health, queue, cfg] = await Promise.all([
    api('/admin/health'),
    api('/admin/queue'),
    _cfg.worker_image ? null : api('/admin/config'),
  ]);

  if (cfg) { _cfg = cfg; syncDeployPanel(); }
  if (health) renderFleet(health);
  if (queue)  renderQueue(queue);
  if (health || queue) { dot('ok'); setUpd('updated '+new Date().toLocaleTimeString()); }
  if (_selWorker) loadEvents(_selWorker);

  _cd = 15; tick();
}

function tick() {
  _timer = setTimeout(()=>{
    _cd--;
    if (_cd<=0) doRefresh();
    else { setUpd(`updated — next in ${_cd}s`); tick(); }
  }, 1000);
}

// ── Fleet ──────────────────────────────────────────────────────────────────
function renderFleet(d) {
  const c = d.counts||{};
  set('sr', c.running??0);
  set('ss', (c.starting??0)+(c.bidding??0));
  set('sp', c.pending??0);
  set('su', c.unhealthy??0);
  set('srate', '$'+(d.spend_rate_per_hr||0).toFixed(4)+'/hr');
  set('stotal','$'+(d.total_spent_usd||0).toFixed(4));

  // Per-provider worker counts for the deploy panel
  const provCounts = {};
  (d.workers||[]).forEach(w=>{ provCounts[w.provider]=(provCounts[w.provider]||0)+1; });
  renderProvGrid(provCounts);

  const ws = d.workers||[];
  if (!ws.length) {
    set('wtbody','<tr><td colspan="10" class="empty">No active workers</td></tr>');
    return;
  }
  document.getElementById('wtbody').innerHTML = ws.map(w=>{
    const vs = w.bid_price && w.market_price && w.market_price>0
      ? Math.round(w.bid_price/w.market_price*100)+'%'
      : '—';
    return `<tr class="${w.worker_id===_selWorker?'sel':''}" onclick="selWorker('${w.worker_id}')">
      <td><code style="font-size:11px">${w.worker_id}</code></td>
      <td><span class="prov-tag">${PROV_SHORT[w.provider]||w.provider}</span></td>
      <td style="color:var(--text)">${w.gpu_name||'—'}</td>
      <td><span class="${w.worker_type==='on_demand'?'type-od':'type-spot'}">${w.worker_type==='on_demand'?'on-demand':'spot'}</span></td>
      <td>${sbadge(w.status)}</td>
      <td>${w.bid_price?'$'+w.bid_price.toFixed(4):'—'}</td>
      <td style="color:var(--muted);font-size:10px">${vs}</td>
      <td style="color:var(--muted)">${fmtUp(w.uptime_hr)}</td>
      <td style="color:var(--muted)">${w.cost_usd>0?'$'+w.cost_usd.toFixed(4):'—'}</td>
      <td><button class="r" onclick="terminate(event,'${w.worker_id}')">✕</button></td>
    </tr>`;
  }).join('');
}

// ── Queue ──────────────────────────────────────────────────────────────────
function renderQueue(d) {
  const av=(d.available||[]).length, ls=(d.leased||[]).length, dr=(d.draining||[]).length;
  const tot=av+ls;
  const pct=tot?Math.round(ls/tot*100):0;
  set('squtil', tot ? pct+'%' : '—');
  const bar=document.getElementById('qbar');
  bar.style.width=pct+'%';
  bar.className='util-fill'+(pct>=90?' max':pct>=70?' hi':'');
  set('qpct', tot?`${pct}% busy · ${av} avail · ${ls} leased`:'no workers');

  let html='';
  if (av){ html+=`<div class="qdivider">Available (${av})</div>`; }
  (d.available||[]).forEach(w=>{
    const idle=w.idle_sec!=null?` · idle ${fmtSec(w.idle_sec)}`:'';
    html+=`<div class="qworker"><span class="qw-id">${w.worker_id}</span><span class="qw-meta">${idle}</span><span class="qw-state qws-avail">ready</span></div>`;
  });
  if (ls){ html+=`<div class="qdivider">Leased (${ls})</div>`; }
  (d.leased||[]).forEach(w=>{
    const ttl=w.lease_ttl_remaining>0?` ttl:${w.lease_ttl_remaining}s`:'';
    html+=`<div class="qworker"><span class="qw-id">${w.worker_id}</span><span class="qw-meta">${w.lb_id||''}${ttl}</span><span class="qw-state qws-leased">busy</span></div>`;
  });
  if (dr){ html+=`<div class="qdivider">Draining (${dr})</div>`; }
  (d.draining||[]).forEach(id=>{
    html+=`<div class="qworker"><span class="qw-id">${id}</span><span class="qw-meta">tombstoned</span><span class="qw-state qws-drain">drain</span></div>`;
  });
  if (!html) html='<div class="empty">Queue empty</div>';
  set('qworkers',html);
}

// ── Deploy panel ───────────────────────────────────────────────────────────
function syncDeployPanel() {
  if (_cfg.worker_image && !document.getElementById('img-input').value) {
    document.getElementById('img-input').value = _cfg.worker_image;
  }
  renderProvGrid({});
}

function renderProvGrid(counts) {
  const configured = _cfg.providers || [];
  const grid = document.getElementById('prov-grid');
  if (!grid) return;
  grid.innerHTML = PROVIDERS.map(p=>{
    const isCfg = configured.includes(p);
    const cnt   = counts[p] || 0;
    const isActive = _selProvider === p;
    const haswk = cnt > 0;
    return `<div class="prov-card${isActive?' active':''}${haswk?' has-workers':''}" onclick="selProv('${p}')">
      <div class="pname">${PROV_LABELS[p]}</div>
      <div class="pcount">${cnt ? cnt+' worker'+(cnt>1?'s':'') : 'no workers'}</div>
      <div class="pcfg ${isCfg?'yes':'no'}">${isCfg?'✓ configured':'not configured'}</div>
      <div class="psel-dot"></div>
    </div>`;
  }).join('');
}

function selProv(p) {
  _selProvider = _selProvider === p ? null : p;
  renderProvGrid({}); // re-render without counts (counts come from next refresh)
  doRefresh();
}

function setLaunchType(t) {
  _launchType = t;
  document.getElementById('lt-spot').classList.toggle('active', t==='spot');
  document.getElementById('lt-od').classList.toggle('active', t==='od');
}

function resetDeploy() {
  _selProvider = null;
  _launchType = 'spot';
  setLaunchType('spot');
  document.getElementById('img-input').value = _cfg.worker_image || '';
  document.getElementById('launch-status').textContent = '';
  renderProvGrid({});
}

async function launch() {
  const img = document.getElementById('img-input').value.trim();
  const od  = _launchType === 'od';
  const body = {};
  if (_selProvider) body.provider = _selProvider;
  if (img && img !== _cfg.worker_image) body.image = img;

  const what = od ? 'on-demand instance' : 'spot bid campaign';
  const prov = _selProvider ? ` on ${PROV_LABELS[_selProvider]}` : '';
  if (!confirm(`Launch ${what}${prov}?`)) return;

  const endpoint = od ? '/admin/scale/on-demand' : '/admin/scale/bid';
  const r = await api(endpoint, { method:'POST', body });
  const el = document.getElementById('launch-status');
  if (r) {
    el.style.color = 'var(--green)';
    el.textContent = `✓ Signal sent — orchestrator will act on next tick (~30s)`;
    setTimeout(()=>{ el.textContent=''; doRefresh(); }, 3000);
  }
}

// ── Worker selection / events ──────────────────────────────────────────────
async function selWorker(id) {
  _selWorker = id;
  document.querySelectorAll('#wtbody tr').forEach(r=>{
    r.classList.toggle('sel', r.innerHTML.includes(`selWorker('${id}')`));
  });
  switchTab('events');
  await loadEvents(id);
}

async function loadEvents(id) {
  set('ev-wid', id);
  const d = await api(`/admin/events/worker/${id}?limit=60`);
  if (!d) return;
  const evs = d.events||[];
  if (!evs.length){ set('evlist','<div class="empty">No events</div>'); return; }
  document.getElementById('evlist').innerHTML = evs.map((e,i)=>{
    const ts  = e.ts ? new Date(e.ts).toLocaleTimeString() : '—';
    const col = evColor(e.event_type);
    let h = `<div class="ev">
      <div class="ev-ts">${ts}</div>
      <div class="ev-type" style="color:${col}">${esc(e.event_type)}</div>
      <div class="ev-msg">${esc(e.message)}</div>`;
    if (e.log_text) {
      h += `<pre class="log-block" id="lb${i}" onclick="toggleLog('lb${i}')">${esc(e.log_text.slice(-3000))}</pre>`;
    }
    if (e.meta && typeof e.meta==='object' && Object.keys(e.meta).length) {
      h += `<div class="ev-meta">${esc(JSON.stringify(e.meta))}</div>`;
    }
    return h+'</div>';
  }).join('');
}

function clearEvents() {
  _selWorker = null;
  set('ev-wid','');
  set('evlist','<div class="empty">Click a worker row to view events</div>');
  document.querySelectorAll('#wtbody tr').forEach(r=>r.classList.remove('sel'));
}

function toggleLog(id) { document.getElementById(id)?.classList.toggle('open'); }

// ── Terminate ──────────────────────────────────────────────────────────────
async function terminate(event, id) {
  event.stopPropagation();
  if (!confirm(`Terminate worker ${id}?\nThis will destroy the instance immediately.`)) return;
  const r = await api(`/admin/workers/${id}/terminate`, {method:'POST'});
  if (r) { if (_selWorker===id) clearEvents(); doRefresh(); }
}

// ── Tab switching ──────────────────────────────────────────────────────────
function switchTab(name) {
  ['deploy','queue','events'].forEach(t=>{
    document.getElementById('tab-'+t)?.classList.toggle('active', t===name);
    document.getElementById('tp-'+t)?.classList.toggle('active', t===name);
  });
}

// ── Helpers ────────────────────────────────────────────────────────────────
function sbadge(s) {
  return `<span class="badge bd-${s||'terminated'}">${s||'?'}</span>`;
}
function fmtUp(h) {
  if (h==null) return '—';
  if (h<1/60) return '<1m';
  if (h<1) return Math.round(h*60)+'m';
  if (h<24) return h.toFixed(1)+'h';
  return (h/24).toFixed(1)+'d';
}
function fmtSec(s) {
  if (s<60) return s+'s';
  if (s<3600) return Math.round(s/60)+'m';
  return (s/3600).toFixed(1)+'h';
}
function esc(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function set(id,html){ const e=document.getElementById(id); if(e) e.innerHTML=html; }
function dot(cls){ document.getElementById('dot').className='live-dot '+cls; }
function setUpd(t){ set('last-upd',t); }
function evColor(t) {
  if (!t) return 'var(--muted)';
  if (/ready|success|recover|creat/.test(t)) return 'var(--green)';
  if (/fail|error|fatal|terminat|orphan/.test(t)) return 'var(--red)';
  if (/warn|outbid|unhealthy|preempt/.test(t)) return 'var(--orange)';
  if (/start|bid|pending|init/.test(t)) return 'var(--blue)';
  return 'var(--muted)';
}

// ── Init ───────────────────────────────────────────────────────────────────
renderProvGrid({});
if (_tok) { document.getElementById('tok').value='••••••••'; doRefresh(); }
else { document.getElementById('tok').focus(); }
</script>
</body>
</html>"""


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def admin_ui():
    return HTMLResponse(_HTML)
