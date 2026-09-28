/**
 * Skill-trends view: reproduces dashboard/views/skill_trends.py's real
 * behavior with a hand-rolled SVG multi-series line chart (per
 * docs/frontend-migration-scope.md's locked charting decision -- no JS
 * charting library).
 *
 * `load_skill_trends_data()`'s own docstring documents a real, checked
 * finding: tier3's emos_graphcast/emos_ifs_ens/bma columns are bit-for-bit
 * duplicates of tier2's own values (same held-out window), so that
 * function deliberately does not re-emit them -- tier3 only contributes
 * its own new `regime_conditioned` method. This view just plots every
 * `method` the API actually returns, one line per method; it does not
 * group or re-derive tiers client-side in a way that would reintroduce
 * the duplicate.
 *
 * Method display colours are a small local qualitative palette, same
 * situation as dashboard-web/js/charts/weightMap.js's per-source colours:
 * this project never defined per-method display colours in Python either
 * (the Streamlit view let Plotly pick its own default categorical
 * palette), so there is no dashboard.colors constant to reuse here.
 *
 * SVG helpers below are named with a `skillTrends`/`SKILL_TRENDS` prefix
 * rather than shared with weightMap.js's own `el`/`SVG_NS` -- both files
 * load as plain <script> tags sharing one global scope, and an earlier
 * version of this file collided with weightMap.js's identically-named
 * globals, breaking the whole page's script load. Each chart module stays
 * self-contained instead of depending on another module's load order.
 */

const SKILL_METHOD_COLORS = {
  tier0_ensemble_mean: "#9c9c9c",
  tier1_equal_weight: "#c2c2c2",
  tier1_regional_blend: "#4c78a8",
  tier2_emos_graphcast: "#f58518",
  tier2_emos_ifs_ens: "#54a24b",
  tier2_bma: "#e45756",
  tier3_regime_conditioned: "#72b7b2",
};
const SKILL_METHOD_FALLBACK_COLOR = "#333333";
const SKILL_TRENDS_SVG_NS = "http://www.w3.org/2000/svg";

function skillTrendsEl(tag, attrs) {
  const node = document.createElementNS(SKILL_TRENDS_SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    node.setAttribute(key, value);
  }
  return node;
}

/**
 * Renders one line per real `method` in `rows` (x = lead_hours, y = value)
 * into `container`. Returns the sorted list of methods actually plotted.
 */
function renderSkillTrendsChart(container, rows) {
  container.innerHTML = "";

  const leadHours = [...new Set(rows.map((r) => r.lead_hours))].sort((a, b) => a - b);
  const methods = [...new Set(rows.map((r) => r.method))].sort();

  const width = 760;
  const height = 380;
  const margin = { top: 16, right: 16, bottom: 44, left: 52 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const values = rows.map((r) => r.value);
  const maxValue = Math.max(...values);
  const minValue = Math.min(0, ...values);
  const yFor = (value) =>
    plotHeight - ((value - minValue) / (maxValue - minValue || 1)) * plotHeight;
  const xFor = (lead) =>
    leadHours.length > 1 ? (leadHours.indexOf(lead) / (leadHours.length - 1)) * plotWidth : 0;

  const svg = skillTrendsEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Skill trend line chart",
  });
  svg.style.width = "100%";
  svg.style.height = "auto";

  const plot = skillTrendsEl("g", { transform: `translate(${margin.left},${margin.top})` });
  svg.appendChild(plot);

  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const value = minValue + ((maxValue - minValue) / yTicks) * i;
    const y = yFor(value);
    plot.appendChild(
      skillTrendsEl("line", { x1: 0, x2: plotWidth, y1: y, y2: y, stroke: "#e5e8ec" })
    );
    const label = skillTrendsEl("text", {
      x: -8,
      y: y + 4,
      "text-anchor": "end",
      "font-size": 11,
      fill: "#5b6472",
    });
    label.textContent = value.toFixed(2);
    plot.appendChild(label);
  }

  leadHours.forEach((lead) => {
    const x = xFor(lead);
    const label = skillTrendsEl("text", {
      x,
      y: plotHeight + 20,
      "text-anchor": "middle",
      "font-size": 12,
      fill: "#1a1f27",
    });
    label.textContent = lead;
    plot.appendChild(label);
  });

  methods.forEach((method) => {
    const color = SKILL_METHOD_COLORS[method] || SKILL_METHOD_FALLBACK_COLOR;
    const methodRows = rows
      .filter((r) => r.method === method)
      .sort((a, b) => a.lead_hours - b.lead_hours);

    const points = methodRows.map((r) => [xFor(r.lead_hours), yFor(r.value)]);
    const pathData = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ");
    plot.appendChild(
      skillTrendsEl("path", { d: pathData, fill: "none", stroke: color, "stroke-width": 2 })
    );

    methodRows.forEach((row, i) => {
      const [x, y] = points[i];
      const circle = skillTrendsEl("circle", { cx: x, cy: y, r: 4, fill: color });
      const title = document.createElementNS(SKILL_TRENDS_SVG_NS, "title");
      title.textContent = `${method} @ ${row.lead_hours}h: ${row.value.toFixed(3)}`;
      circle.appendChild(title);
      plot.appendChild(circle);
    });
  });

  container.appendChild(svg);

  const legend = document.createElement("div");
  legend.className = "legend";
  methods.forEach((method) => {
    const item = document.createElement("span");
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = SKILL_METHOD_COLORS[method] || SKILL_METHOD_FALLBACK_COLOR;
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(method));
    legend.appendChild(item);
  });
  container.appendChild(legend);

  return methods;
}

const SKILL_TREND_CAPTION =
  "The real, checked outcome across tiers is mixed, not monotonically " +
  "improving: tier1 beats tier0 domain-wide at every lead but not in " +
  "every region x lead cell; tier2's EMOS-CSG and BMA trade off which " +
  "wins; tier3's regime-conditioned reweighting does not beat tier2 at " +
  "any lead (0 of 5). This chart plots every real method's own " +
  "numbers -- it is not smoothed into a single improving trend line.";

const SkillTrendsView = {
  /** Fetches real data for `metric` and renders chart + caption into `container`. */
  async render(container, metric) {
    container.innerHTML = '<p class="loading-state">Loading skill trends…</p>';

    let rows;
    try {
      rows = await fetchSkillTrends(metric);
    } catch (err) {
      container.innerHTML = `<p class="error-state">Failed to load skill trends: ${err.message}</p>`;
      return;
    }

    container.innerHTML = "";

    const caption = document.createElement("p");
    caption.className = "view-caption";
    caption.textContent = SKILL_TREND_CAPTION;
    container.appendChild(caption);

    if (rows.length === 0) {
      // Real, checked case in dashboard/views/skill_trends.py: tier0/tier1
      // are deterministic blends with no CRPS, so this triggers if a
      // metric ever has zero rows across every tier -- not exercised by
      // today's committed data (crps_mm still has tier2/tier3 rows), but
      // a real defensive path this view must reproduce, not drop.
      const info = document.createElement("p");
      info.className = "not-built";
      info.textContent =
        "No tier reports this metric here (tier0/tier1 are deterministic " +
        "blends with no CRPS).";
      container.appendChild(info);
      return;
    }

    const chartContainer = document.createElement("div");
    container.appendChild(chartContainer);
    renderSkillTrendsChart(chartContainer, rows);
  },
};
