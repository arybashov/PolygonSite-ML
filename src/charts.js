const RESULTS_URL = 'results.json';
const REFRESH_MS  = 5000;
const ROLL_WIN    = 10;

const RUN_COLORS = [
  '#4ecb71', '#5ba3e8', '#e8a830', '#3ecaa5',
  '#c87aad', '#e87a5b', '#9a8fe0', '#e8d85b',
];

export function initResultsPanel(openBtn, overlay, closeBtn) {
  let charts    = [];
  let timer     = null;
  let lastCount = 0;

  openBtn.addEventListener('click', () => {
    overlay.classList.remove('hidden');
    load();
    timer = setInterval(load, REFRESH_MS);
  });

  closeBtn.addEventListener('click', close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });

  function close() {
    overlay.classList.add('hidden');
    clearInterval(timer);
    timer = null;
  }

  async function load() {
    let runs;
    try {
      const res = await fetch(`${RESULTS_URL}?_=${Date.now()}`);
      if (!res.ok) { showEmpty('Нет данных. Запусти runner с --out results.json'); return; }
      runs = await res.json();
    } catch {
      showEmpty('Нет данных. Запусти runner с --out results.json');
      return;
    }

    if (!Array.isArray(runs) || runs.length === 0) {
      showEmpty('results.json пуст.');
      return;
    }

    if (runs.length === lastCount) return;
    lastCount = runs.length;

    charts.forEach((c) => c.destroy());
    charts = [];
    charts.push(buildRewardChart(runs));
    charts.push(buildRatesChart(runs));
    buildTable(runs);
  }

  function showEmpty(msg) {
    document.getElementById('chartEmpty').textContent = msg;
    document.getElementById('chartEmpty').classList.remove('hidden');
    document.getElementById('chartsGrid').classList.add('hidden');
    document.getElementById('runsTable').innerHTML = '';
  }
}

function rolling(values, win) {
  return values.map((_, i) => {
    const slice = values.slice(Math.max(0, i - win + 1), i + 1);
    return slice.reduce((s, v) => s + v, 0) / slice.length;
  });
}

function buildRewardChart(runs) {
  const ctx = document.getElementById('chartReward').getContext('2d');
  document.getElementById('chartEmpty').classList.add('hidden');
  document.getElementById('chartsGrid').classList.remove('hidden');

  const datasets = runs.map((run, i) => ({
    label:           run.label ?? run.policy,
    data:            rolling(run.records.map((r) => r.reward), ROLL_WIN),
    borderColor:     RUN_COLORS[i % RUN_COLORS.length],
    backgroundColor: 'transparent',
    borderWidth:     i === runs.length - 1 ? 2 : 1,
    pointRadius:     0,
    tension:         0.3,
  }));

  return new window.Chart(ctx, {
    type: 'line',
    data: { labels: runs[0].records.map((_, i) => i + 1), datasets },
    options: chartOptions('Reward (rolling avg)'),
  });
}

function buildRatesChart(runs) {
  const ctx = document.getElementById('chartRates').getContext('2d');
  const datasets = [];

  runs.forEach((run, i) => {
    const col  = RUN_COLORS[i % RUN_COLORS.length];
    const thin = i < runs.length - 1;

    datasets.push({
      label:           `${run.label ?? run.policy} hit`,
      data:            rolling(run.records.map((r) => r.hits / r.total), ROLL_WIN),
      borderColor:     col,
      backgroundColor: 'transparent',
      borderWidth:     thin ? 1 : 2,
      borderDash:      [],
      pointRadius:     0,
      tension:         0.3,
    });

    datasets.push({
      label:           `${run.label ?? run.policy} intercept`,
      data:            rolling(run.records.map((r) => r.intercepted / r.total), ROLL_WIN),
      borderColor:     col,
      backgroundColor: 'transparent',
      borderWidth:     thin ? 1 : 2,
      borderDash:      [4, 3],
      pointRadius:     0,
      tension:         0.3,
    });
  });

  return new window.Chart(ctx, {
    type: 'line',
    data: { labels: runs[0].records.map((_, i) => i + 1), datasets },
    options: chartOptions('Hit rate (—) / Intercept rate (- -)'),
  });
}

function buildTable(runs) {
  const tbody = runs.map((run, i) => {
    const col = RUN_COLORS[i % RUN_COLORS.length];
    const s   = run.summary;
    return `<tr>
      <td><span class="run-dot" style="background:${col}"></span>${run.label ?? run.policy}</td>
      <td>${run.params.ndrones}v${run.params.nanti}</td>
      <td>${run.records.length}</td>
      <td>${s.mean_reward}</td>
      <td>${s.hit_rate}</td>
      <td>${s.intercept_rate}</td>
      <td>${s.mean_time}s</td>
    </tr>`;
  }).join('');

  document.getElementById('runsTable').innerHTML = `
    <table class="runs-table">
      <thead><tr>
        <th>Policy</th><th>Params</th><th>Ep</th>
        <th>Reward</th><th>Hit</th><th>Int</th><th>Time</th>
      </tr></thead>
      <tbody>${tbody}</tbody>
    </table>`;
}

function chartOptions(title) {
  return {
    responsive:          true,
    maintainAspectRatio: false,
    animation:           false,
    plugins: {
      legend: { labels: { color: '#c8d8c0', font: { family: 'Share Tech Mono', size: 10 } } },
      title:  { display: true, text: title, color: '#4ecb71', font: { family: 'Share Tech Mono', size: 11 } },
    },
    scales: {
      x: { ticks: { color: 'rgba(200,216,192,0.45)', font: { size: 9 } }, grid: { color: 'rgba(80,200,100,0.08)' } },
      y: { ticks: { color: 'rgba(200,216,192,0.45)', font: { size: 9 } }, grid: { color: 'rgba(80,200,100,0.08)' } },
    },
  };
}
