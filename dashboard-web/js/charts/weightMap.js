/**
 * Weight-map view: reproduces dashboard/views/weight_map.py's real
 * behavior with a hand-rolled SVG grouped bar chart (per
 * docs/frontend-migration-scope.md's locked charting decision -- no JS
 * charting library).
 *
 * This project's real weighting scheme, per
 * docs/phase6-operational-scope.md, is Phase 3's `fit_region_weights` --
 * a per-region ordinary-least-squares fit -- not a softmax/GBM gate. This
 * view exists to explain *that* real, already-fitted scheme.
 *
 * Fallback regions (is_fallback=true: not enough real training points to
 * fit a weight, so the region got an equal split instead) are rendered
 * with a diagonal-hatch overlay, matching the Streamlit view's
 * `pattern_shape="fit_status"`, plus a warning banner listing them --
 * both real, checked findings from Phase 3's fit, not decoration.
 *
 * Weight sources (graphcast / hres / ifs_ens_mean) get a small local
 * qualitative palette here -- unlike the rain-intensity bins, this
 * project never defined per-source display colours in Python (the
 * Streamlit view let Plotly's own default categorical palette pick
 * them), so there is no dashboard.colors constant to reuse for this one.
 *
 * SVG helpers below are named with a `weightMap`/`WEIGHT_MAP` prefix
 * rather than generic names (`el`, `SVG_NS`) -- this file and
 * skillTrends.js both load as plain <script> tags sharing one global
 * scope, and generic names collided across the two files (step 4's real
 * bug, caught driving the page in a browser), breaking the whole page's
 * script load. Each chart module stays self-contained instead of
 * depending on another module's load order.
 */

const WEIGHT_SOURCE_COLORS = {
  graphcast: "#4c78a8",
  hres: "#f58518",
  ifs_ens_mean: "#54a24b",
};
const WEIGHT_SOURCE_FALLBACK_COLOR = "#999999";
const FALLBACK_PATTERN_ID = "weight-map-fallback-hatch";
const WEIGHT_MAP_SVG_NS = "http://www.w3.org/2000/svg";

function weightMapEl(tag, attrs) {
  const node = document.createElementNS(WEIGHT_MAP_SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    node.setAttribute(key, value);
  }
  return node;
}

function buildFallbackHatchDefs() {
  const defs = weightMapEl("defs", {});
  const pattern = weightMapEl("pattern", {
    id: FALLBACK_PATTERN_ID,
    width: 6,
    height: 6,
    patternTransform: "rotate(45)",
    patternUnits: "userSpaceOnUse",
  });
  pattern.appendChild(weightMapEl("rect", { width: 3, height: 6, fill: "rgba(0,0,0,0.35)" }));
  defs.appendChild(pattern);
  return defs;
}

/**
 * Renders the grouped bar chart into `container` for one lead's real rows
 * (from GET /api/weight-map?lead=...). Returns the sorted list of real
 * fallback regions at this lead, for the caller to show a warning banner
 * from (matching dashboard/views/weight_map.py's own st.warning() logic).
 */
function renderWeightMapChart(container, rows) {
  container.innerHTML = "";

  const regions = [...new Set(rows.map((r) => r.region))].sort();
  const sources = [...new Set(rows.map((r) => r.source))].sort();

  const width = 760;
  const height = 380;
  const margin = { top: 16, right: 16, bottom: 56, left: 48 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const maxWeight = Math.max(1, ...rows.map((r) => r.weight));
  const yFor = (weight) => plotHeight - (weight / maxWeight) * plotHeight;

  const regionBandWidth = plotWidth / regions.length;
  const groupPadding = regionBandWidth * 0.15;
  const barWidth = (regionBandWidth - groupPadding * 2) / sources.length;

  const svg = weightMapEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Regional per-source weight bar chart",
  });
  svg.style.width = "100%";
  svg.style.height = "auto";
  svg.appendChild(buildFallbackHatchDefs());

  const plot = weightMapEl("g", { transform: `translate(${margin.left},${margin.top})` });
  svg.appendChild(plot);

  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const value = (maxWeight / yTicks) * i;
    const y = yFor(value);
    plot.appendChild(
      weightMapEl("line", { x1: 0, x2: plotWidth, y1: y, y2: y, stroke: "#e5e8ec" })
    );
    const label = weightMapEl("text", {
      x: -8,
      y: y + 4,
      "text-anchor": "end",
      "font-size": 11,
      fill: "#5b6472",
    });
    label.textContent = value.toFixed(2);
    plot.appendChild(label);
  }

  const fallbackRegions = new Set();

  regions.forEach((region, regionIndex) => {
    const regionX = regionIndex * regionBandWidth;
    const regionRows = rows.filter((r) => r.region === region);
    const isFallbackRegion = regionRows.some((r) => r.is_fallback);
    if (isFallbackRegion) fallbackRegions.add(region);

    sources.forEach((source, sourceIndex) => {
      const row = regionRows.find((r) => r.source === source);
      const weight = row ? row.weight : 0;
      const barY = yFor(weight);
      const barHeight = Math.max(plotHeight - barY, 0);
      const barX = regionX + groupPadding + sourceIndex * barWidth;

      const rect = weightMapEl("rect", {
        x: barX,
        y: barY,
        width: Math.max(barWidth - 2, 1),
        height: barHeight,
        fill: WEIGHT_SOURCE_COLORS[source] || WEIGHT_SOURCE_FALLBACK_COLOR,
      });
      const title = document.createElementNS(WEIGHT_MAP_SVG_NS, "title");
      const fallbackNote =
        row && row.is_fallback
          ? ` (fallback: ${row.reason}, n_train_points=${row.n_train_points})`
          : "";
      title.textContent = `${region} / ${source}: ${weight.toFixed(3)}${fallbackNote}`;
      rect.appendChild(title);
      plot.appendChild(rect);

      if (row && row.is_fallback) {
        plot.appendChild(
          weightMapEl("rect", {
            x: barX,
            y: barY,
            width: Math.max(barWidth - 2, 1),
            height: barHeight,
            fill: `url(#${FALLBACK_PATTERN_ID})`,
          })
        );
      }
    });

    const regionLabel = weightMapEl("text", {
      x: regionX + regionBandWidth / 2,
      y: plotHeight + 20,
      "text-anchor": "middle",
      "font-size": 12,
      fill: "#1a1f27",
    });
    regionLabel.textContent = region + (isFallbackRegion ? " *" : "");
    plot.appendChild(regionLabel);
  });

  container.appendChild(svg);

  const legend = document.createElement("div");
  legend.className = "legend";
  sources.forEach((source) => {
    const item = document.createElement("span");
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = WEIGHT_SOURCE_COLORS[source] || WEIGHT_SOURCE_FALLBACK_COLOR;
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(source));
    legend.appendChild(item);
  });
  if (fallbackRegions.size > 0) {
    const item = document.createElement("span");
    item.textContent = "* fallback region (equal split, not a real fit)";
    legend.appendChild(item);
  }
  container.appendChild(legend);

  return [...fallbackRegions].sort();
}

const WeightMapView = {
  /** Fetches real data for `lead` and renders chart + caption + warning into `container`. */
  async render(container, lead) {
    container.innerHTML = '<p class="loading-state">Loading weight map…</p>';

    let rows;
    try {
      rows = await fetchWeightMap(lead);
    } catch (err) {
      container.innerHTML = `<p class="error-state">Failed to load weight map: ${err.message}</p>`;
      return;
    }

    container.innerHTML = "";

    const subheader = document.createElement("h2");
    subheader.className = "view-subheader";
    subheader.textContent = "Weight map: per-region, per-lead source weights";
    container.appendChild(subheader);

    const caption = document.createElement("p");
    caption.className = "view-caption";
    caption.textContent =
      "This project's real weighting scheme is Phase 3's fit_region_weights " +
      "-- a per-region ordinary-least-squares fit over graphcast / hres / " +
      "ifs_ens_mean's own real historical errors, not a softmax or GBM gate. " +
      "Regions shown hatched fell back to an equal split (not enough real " +
      "training points to fit a weight) -- hover for the real reason.";
    container.appendChild(caption);

    const chartTitle = document.createElement("p");
    chartTitle.className = "chart-title";
    chartTitle.textContent = `Regional source weights at lead ${lead}h`;
    container.appendChild(chartTitle);

    const chartContainer = document.createElement("div");
    container.appendChild(chartContainer);
    const fallbackRegions = renderWeightMapChart(chartContainer, rows);

    if (fallbackRegions.length > 0) {
      const warning = document.createElement("div");
      warning.className = "warning-banner";
      warning.textContent =
        "Fallback regions at this lead (equal split, not a real fit): " +
        fallbackRegions.join(", ");
      container.appendChild(warning);
    }
  },
};
