/**
 * Extreme-probability-map view: reproduces
 * dashboard/views/extreme_probability.py's real behavior with a
 * hand-rolled <canvas> two-layer heatmap (per
 * docs/frontend-migration-scope.md's locked charting decision -- no JS
 * charting library).
 *
 * This view diverges from the blended-map's Tier 1 combiner (step 5):
 * Tier 1's deterministic regional blend has no predictive distribution to
 * compute an exceedance probability from at all. Instead this view uses
 * EMOS-CSG's `ifs_ens` combiner -- fit on the real 50-member IFS ensemble
 * (weavr.emos.fit_emos_csg) -- because the censored-shifted-gamma has a
 * real closed-form survival function
 * (weavr.emos.exceedance_probability_csgd), unlike BMA's mixture, which
 * has no closed form.
 *
 * 204.5mm is IMD's own real "extremely heavy rain" boundary
 * (weavr.verify.IMD_RAIN_THRESHOLDS_MM[-1]).
 * docs/phase4-data-and-combiner-scope.md found the extremely_heavy bin is
 * never fittable at any lead in this project's real 2020 JJAS data.
 * Cells whose own forecast fell in a bin EMOS-CSG could not fit for real
 * are drawn with a second, distinct semi-transparent grey pass -- a
 * fallback cell's ~0 probability must not look identical to a genuinely
 * low-risk, real fitted cell.
 *
 * The probability colourscale (GET /api/colors's probability_colorscale,
 * dashboard.colors.PROBABILITY_COLORSCALE server-side) is a continuous
 * 5-stop gradient, unlike the blended map's discrete rain bins -- this
 * view interpolates between stops per real probability value, matching
 * the Streamlit version's go.Heatmap continuous colourscale.
 */

const EXTREME_PROBABILITY_CELL_SIZE = 4;
const EXTREME_PROBABILITY_FALLBACK_RGBA = "rgba(120, 120, 120, 0.75)";

const EXTREME_PROBABILITY_CAPTION =
  "This map shows EMOS-CSG's real fitted ifs_ens combiner (the " +
  "real 50-member IFS ensemble) -- chosen over BMA because its " +
  "censored-shifted-gamma has a real closed-form exceedance " +
  "probability, unlike BMA's mixture. This diverges from the " +
  "blended map's Tier 1 combiner, which has no predictive " +
  "distribution to compute a probability from. 204.5mm is IMD's own " +
  "real 'extremely heavy rain' boundary. Grey-hatched cells have no " +
  "real fitted probability -- their own forecast fell in a " +
  "rain-intensity bin EMOS-CSG could not fit for real at this lead " +
  "(docs/phase4-data-and-combiner-scope.md found the extremely_heavy " +
  "bin is never fittable at any lead in this project's real data).";

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [
    parseInt(value.substring(0, 2), 16),
    parseInt(value.substring(2, 4), 16),
    parseInt(value.substring(4, 6), 16),
  ];
}

/**
 * Linearly interpolates `probability` (0..1) against `colorscale`'s real
 * [position, hexColor] stops (from /api/colors's probability_colorscale),
 * returning an "rgb(r,g,b)" string. Matches Plotly's own continuous
 * colourscale interpolation between the same stops the Streamlit view
 * uses.
 */
function interpolateProbabilityColor(colorscale, probability) {
  const p = Math.min(1, Math.max(0, probability));
  for (let i = 0; i < colorscale.length - 1; i++) {
    const [p0, hex0] = colorscale[i];
    const [p1, hex1] = colorscale[i + 1];
    if (p >= p0 && p <= p1) {
      const t = p1 === p0 ? 0 : (p - p0) / (p1 - p0);
      const [r0, g0, b0] = hexToRgb(hex0);
      const [r1, g1, b1] = hexToRgb(hex1);
      const r = Math.round(r0 + (r1 - r0) * t);
      const g = Math.round(g0 + (g1 - g0) * t);
      const b = Math.round(b0 + (b1 - b0) * t);
      return `rgb(${r},${g},${b})`;
    }
  }
  return hexToRgb(colorscale[colorscale.length - 1][1]).join(",");
}

/**
 * Draws `grid.probability` as a continuous heatmap (first pass), then
 * `grid.is_fallback` as a second, distinct semi-transparent grey pass on
 * top -- two real layers, not one colour choice, matching
 * dashboard/views/extreme_probability.py's two go.Heatmap traces.
 */
function renderExtremeProbabilityCanvas(container, grid, colorscale) {
  container.innerHTML = "";

  const latCount = grid.probability.length;
  const lonCount = grid.probability[0].length;

  const canvas = document.createElement("canvas");
  canvas.width = lonCount * EXTREME_PROBABILITY_CELL_SIZE;
  canvas.height = latCount * EXTREME_PROBABILITY_CELL_SIZE;
  canvas.style.width = "100%";
  canvas.style.height = "auto";
  canvas.style.imageRendering = "pixelated";
  canvas.setAttribute("role", "img");
  canvas.setAttribute(
    "aria-label",
    `P(rain > 204.5mm) map at lead ${grid.lead_hours}h`
  );

  const ctx = canvas.getContext("2d");

  // Layer 1: continuous probability heatmap.
  for (let latIndex = 0; latIndex < latCount; latIndex++) {
    const canvasRow = latCount - 1 - latIndex;
    for (let lonIndex = 0; lonIndex < lonCount; lonIndex++) {
      const probability = grid.probability[latIndex][lonIndex];
      ctx.fillStyle = interpolateProbabilityColor(colorscale, probability);
      ctx.fillRect(
        lonIndex * EXTREME_PROBABILITY_CELL_SIZE,
        canvasRow * EXTREME_PROBABILITY_CELL_SIZE,
        EXTREME_PROBABILITY_CELL_SIZE,
        EXTREME_PROBABILITY_CELL_SIZE
      );
    }
  }

  // Layer 2: a second, distinct pass -- semi-transparent grey only where
  // is_fallback is true, never blended into layer 1's colour choice.
  for (let latIndex = 0; latIndex < latCount; latIndex++) {
    const canvasRow = latCount - 1 - latIndex;
    for (let lonIndex = 0; lonIndex < lonCount; lonIndex++) {
      if (grid.is_fallback[latIndex][lonIndex]) {
        ctx.fillStyle = EXTREME_PROBABILITY_FALLBACK_RGBA;
        ctx.fillRect(
          lonIndex * EXTREME_PROBABILITY_CELL_SIZE,
          canvasRow * EXTREME_PROBABILITY_CELL_SIZE,
          EXTREME_PROBABILITY_CELL_SIZE,
          EXTREME_PROBABILITY_CELL_SIZE
        );
      }
    }
  }

  container.appendChild(canvas);
}

function renderExtremeProbabilityLegend(container, colorscale) {
  const wrapper = document.createElement("div");
  wrapper.style.marginTop = "12px";
  wrapper.style.maxWidth = "400px";

  const label = document.createElement("div");
  label.style.fontSize = "0.85rem";
  label.style.color = "var(--color-text-muted)";
  label.style.marginBottom = "4px";
  label.textContent = "P(rain > 204.5mm)";
  wrapper.appendChild(label);

  const gradientStops = colorscale.map(([pos, hex]) => `${hex} ${pos * 100}%`).join(", ");
  const bar = document.createElement("div");
  bar.style.height = "14px";
  bar.style.borderRadius = "3px";
  bar.style.border = "1px solid var(--color-border)";
  bar.style.background = `linear-gradient(to right, ${gradientStops})`;
  wrapper.appendChild(bar);

  const ticks = document.createElement("div");
  ticks.style.display = "flex";
  ticks.style.justifyContent = "space-between";
  ticks.style.fontSize = "0.75rem";
  ticks.style.color = "var(--color-text-muted)";
  ["0.0", "0.25", "0.5", "0.75", "1.0"].forEach((tick) => {
    const span = document.createElement("span");
    span.textContent = tick;
    ticks.appendChild(span);
  });
  wrapper.appendChild(ticks);

  const fallbackNote = document.createElement("div");
  fallbackNote.style.fontSize = "0.8rem";
  fallbackNote.style.color = "var(--color-text-muted)";
  fallbackNote.style.marginTop = "8px";
  fallbackNote.innerHTML =
    '<span class="legend-swatch" style="background:' +
    EXTREME_PROBABILITY_FALLBACK_RGBA +
    '"></span>fallback cell (not fittable at this lead)';
  wrapper.appendChild(fallbackNote);

  container.appendChild(wrapper);
}

const ExtremeProbabilityView = {
  /** Fetches real data for `lead` and renders canvas + captions + legend + warning into `container`. */
  async render(container, lead) {
    container.innerHTML = '<p class="loading-state">Loading extreme-probability map…</p>';

    let grid;
    let colors;
    try {
      [grid, colors] = await Promise.all([fetchExtremeProbability(lead), getColors()]);
    } catch (err) {
      container.innerHTML = `<p class="error-state">Failed to load extreme-probability map: ${err.message}</p>`;
      return;
    }

    container.innerHTML = "";

    const subheader = document.createElement("h2");
    subheader.className = "view-subheader";
    subheader.textContent = "Extreme-probability map: P(rain > 204.5mm)";
    container.appendChild(subheader);

    const caption = document.createElement("p");
    caption.className = "view-caption";
    caption.textContent = EXTREME_PROBABILITY_CAPTION;
    container.appendChild(caption);

    const title = document.createElement("p");
    title.className = "chart-title";
    title.textContent = `P(rain > 204.5mm) at lead ${grid.lead_hours}h (sample: ${grid.sample_time})`;
    container.appendChild(title);

    const canvasContainer = document.createElement("div");
    container.appendChild(canvasContainer);
    renderExtremeProbabilityCanvas(canvasContainer, grid, colors.probability_colorscale);

    renderExtremeProbabilityLegend(container, colors.probability_colorscale);

    // Computed from the real fetched is_fallback array's true count --
    // not hardcoded -- matching dashboard/views/extreme_probability.py's
    // own st.warning() logic (only shown when n_fallback > 0).
    const nFallback = grid.is_fallback.reduce(
      (sum, row) => sum + row.filter(Boolean).length,
      0
    );
    if (nFallback > 0) {
      const warning = document.createElement("div");
      warning.className = "warning-banner";
      warning.textContent =
        `${nFallback} gridpoint(s) at this lead are shown grey: their ` +
        "forecast fell in a rain-intensity bin with too few real " +
        "training days to fit EMOS-CSG (see caption above).";
      container.appendChild(warning);
    }
  },
};
