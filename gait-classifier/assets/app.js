const ui = new WebUI({ path: '/socket.io', transports: ['polling', 'websocket'], autoConnect: true });

const $ = id => document.getElementById(id);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const LABELS = ['still', 'walk', 'run', 'bike'];
const pct = v => `${Math.round(v * 100)}%`;

// ---------- connection ----------

ui.on_connect(() => {
  $('conn').textContent = 'connected';
  $('conn').classList.remove('bad');
});
ui.on_disconnect(() => {
  $('conn').textContent = 'disconnected';
  $('conn').classList.add('bad');
});

ui.on_message('rate', d => {
  $('rate').textContent = d.imu_address === 0 ? '· IMU NOT FOUND, check wiring' : `· ${d.hz.toFixed(1)} Hz`;
});

// ---------- live prediction ----------

function renderVotes(el, votes) {
  el.innerHTML = LABELS.map(l => `
    <div class="vote">
      <span>${l}</span>
      <div class="track"><div class="fill" style="width:${pct(votes[l] || 0)};background:${css('--' + l)}"></div></div>
      <span>${pct(votes[l] || 0)}</span>
    </div>`).join('');
}

ui.on_message('prediction', p => {
  for (const m of ['model1', 'model2']) {
    $(`${m === 'model1' ? 'm1' : 'm2'}-label`).textContent = p[m].label;
    $(`${m === 'model1' ? 'm1' : 'm2'}-label`).style.color = css('--' + p[m].label);
    renderVotes($(`${m === 'model1' ? 'm1' : 'm2'}-votes`), p[m].votes);
  }
  $('m1-axis').textContent = p.axis;
  $('peak').textContent = p.peak.toFixed(2);
});

// ---------- live plots ----------

const PLOT_POINTS = 250; // 10 s at 25 Hz
const history = [];

ui.on_message('samples', batch => {
  history.push(...batch);
  if (history.length > PLOT_POINTS) history.splice(0, history.length - PLOT_POINTS);
});

function drawPlot(canvas, offset) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!history.length) return;

  let max = 0;
  for (const s of history) for (let i = 0; i < 3; i++) max = Math.max(max, Math.abs(s[offset + i]));
  max = Math.max(max * 1.1, offset === 0 ? 2 : 100);

  const y = v => h / 2 - (v / max) * (h / 2 - 4);
  ctx.strokeStyle = css('--line');
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
  ctx.fillStyle = css('--muted');
  ctx.font = '11px system-ui';
  ctx.fillText(`±${max.toFixed(offset === 0 ? 1 : 0)}`, 4, 12);

  ['--x', '--y', '--z'].forEach((color, i) => {
    ctx.strokeStyle = css(color);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    history.forEach((s, j) => {
      const x = (j / (PLOT_POINTS - 1)) * w;
      j ? ctx.lineTo(x, y(s[offset + i])) : ctx.moveTo(x, y(s[offset + i]));
    });
    ctx.stroke();
  });
}

(function frame() {
  drawPlot($('acc'), 0);
  drawPlot($('gyro'), 3);
  requestAnimationFrame(frame);
})();

// ---------- recording ----------

$('record-buttons').innerHTML = LABELS.map(l => `
  <button data-label="${l}" style="background:${css('--' + l)}">
    Record ${l}<span class="count" id="count-${l}"></span>
  </button>`).join('');

$('record-buttons').addEventListener('click', e => {
  const btn = e.target.closest('button');
  if (!btn) return;
  ui.send_message('start_recording', {
    label: btn.dataset.label,
    seconds: Number($('seconds').value),
    countdown: Number($('countdown').value),
  });
});
$('cancel').onclick = () => ui.send_message('cancel_recording');

function renderRecording(rec) {
  const busy = rec.state !== 'idle';
  $('rec-status').classList.toggle('hidden', !busy);
  document.querySelectorAll('#record-buttons button').forEach(b => (b.disabled = busy));
  if (rec.state === 'countdown') {
    $('rec-text').textContent = `Get ready to ${rec.label}… starting in ${Math.ceil(rec.seconds_left)} s`;
    $('rec-bar').style.width = '0';
  } else if (rec.state === 'recording') {
    const left = Math.ceil(rec.seconds * (1 - rec.progress));
    $('rec-text').textContent = `Recording ${rec.label}: ${left} s left`;
    $('rec-bar').style.width = pct(rec.progress);
  }
}

// ---------- model accuracy ----------

function confusionTable(cm) {
  const head = `<tr><th>true ↓ / pred →</th>${LABELS.map(l => `<th>${l}</th>`).join('')}</tr>`;
  const rows = cm.map((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    return `<tr><th>${LABELS[i]}</th>${row.map((v, j) => {
      const alpha = v / total;
      const color = i === j ? `rgba(42,157,143,${alpha})` : `rgba(229,56,59,${alpha})`;
      return `<td style="background:${color}">${v}</td>`;
    }).join('')}</tr>`;
  }).join('');
  return `<table class="cm">${head}${rows}</table>`;
}

function renderModel(m) {
  $('untrained').classList.toggle('hidden', m.trained);
  if (document.activeElement !== $('k')) $('k').value = m.k;
  $('axis').value = m.axis_setting;
  if (!m.trained) {
    $('accuracy').innerHTML = '<p class="hint">Not trained yet.</p>';
    return;
  }
  const axisAcc = Object.entries(m.model1.axis_accuracy).map(([a, v]) => `${a}: ${pct(v)}`).join(' · ');
  const peaks = Object.entries(m.peak_by_label)
    .map(([l, s]) => `${l}: ${s.min.toFixed(1)}–${s.max.toFixed(1)} g (median ${s.median.toFixed(1)})`).join('<br>');
  $('accuracy').innerHTML = `
    <div class="acc-grid">
      <div>
        <div class="model-name">Model 1 · peak ${m.axis} accel only</div>
        <div class="acc-value">${pct(m.model1.accuracy)}</div>
        <div class="small">per axis: ${axisAcc}</div>
        ${confusionTable(m.model1.confusion)}
        <div class="small" style="margin-top:6px">Peak |a<sub>${m.axis}</sub>| per window:<br>${peaks}</div>
      </div>
      <div>
        <div class="model-name">Model 2 · all IMU outputs</div>
        <div class="acc-value">${pct(m.model2.accuracy)}</div>
        <div class="small">34 features: mean / std / min / max of 6 channels, |a| and |ω| stats, dominant frequencies</div>
        ${confusionTable(m.model2.confusion)}
      </div>
    </div>`;
}

$('apply').onclick = () => ui.send_message('settings', { k: Number($('k').value), axis: $('axis').value });

// ---------- recordings list ----------

function renderRecordings(list) {
  for (const l of LABELS) {
    const recs = list.filter(r => r.label === l);
    const secs = recs.reduce((a, r) => a + r.seconds, 0);
    $(`count-${l}`).textContent = `${recs.length} rec · ${Math.round(secs)} s`;
  }
  if (!list.length) {
    $('recordings').innerHTML = '<p class="hint">No recordings yet.</p>';
    return;
  }
  $('recordings').innerHTML = `<table class="rec-list">${list.map(r => `
    <tr>
      <td><span class="tag" style="background:${css('--' + r.label)}">${r.label}</span></td>
      <td>${r.name}</td>
      <td>${r.seconds} s</td>
      <td><a href="api/download/${r.name}">csv</a><button class="secondary" data-name="${r.name}">delete</button></td>
    </tr>`).join('')}</table>`;
}

$('recordings').addEventListener('click', e => {
  const name = e.target.dataset.name;
  if (name && confirm(`Delete ${name}?`)) ui.send_message('delete_recording', { name });
});

// ---------- status ----------

ui.on_message('status', s => {

  renderRecording(s.recording);
  renderModel(s.model);
  renderRecordings(s.recordings);
});
