/**
 * Blended-map view: reproduces dashboard/views/blended_map.py's real
 * behavior with a hand-rolled <canvas> discrete heatmap (per
 * docs/frontend-migration-scope.md's locked charting decision -- no JS
 * charting library).
 *
 * "Blended" means this project's actual, real blend -- Tier 1's regional
 * weighting (weavr.weighting.fit_region_weights), the same combiner
 * docs/phase6-operational-scope.md decided the daily pipeline itself
 * should use, because Tier 2/3's fitted EMOS-CSG/BMA parameters don't
 * transfer to a live source substitution. This is not "the best" combiner
 * by some unstated metric -- Tier 2/3 have their own real, different
 * outcomes.
 *
 * The grid is a small, committed example export
 * (dashboard/data/example_blend_grid.npz), not a live re-blend of the
 * current local Zarr store -- the real gap frontendplan.md §1.4 states
 * plainly, reproduced in this view's own caption too.
 *
 * Colours come from GET /api/colors (rain_bin_colors / rain_bin_labels),
 * never hardcoded here -- the API's bin_index values (0..4, from
 * dashboard.colors.rain_bin_index server-side) are looked up against
 * rain_bin_labels' own order, so a cell's colour always matches the same
 * real IMD category the Streamlit view would show for it. Colour steps
 * are genuinely discrete (one solid fillRect per cell, no interpolation),
 * matching build_discrete_colorscale()'s real intent -- a continuous
 * heatmap would blend "dry" into "light" at cell boundaries, misrepresenting
 * IMD's real categorical warning scheme.
 */

const BLENDED_MAP_CELL_SIZE = 4;

const BLENDED_MAP_CAPTION =
  "This map shows Tier 1's real fitted regional blend " +
  "(weavr.weighting.fit_region_weights) -- the combiner " +
  "docs/phase6-operational-scope.md decided the daily pipeline itself " +
  "uses, not necessarily this project's single 'best' combiner by any " +
  "one metric (Tier 2/3 have their own different, real outcomes). " +
  "The grid is a small, committed example snapshot, not a live " +
  "re-blend of the current local data store -- see " +
  "docs/phase7-dashboard-scope.md.";

const BLENDED_MAP_COLOR_CAPTION =
  "Colour breakpoints are IMD's real published 24-hour " +
  "rainfall-warning thresholds (7.5 / 64.5 / 115.6 / 204.5mm) -- " +
  "see docs/phase7-dashboard-scope.md section 3.";

/**
 * Draws `grid.bin_index` (129x135, rows = latitude ascending, cols =
 * longitude ascending -- matching dashboard/views/blended_map.py's own
 * `origin="lower"`) as a discrete heatmap into `container`. `binColors`
 * is an array of hex strings indexed by bin index (0..4), built from
 * /api/colors's real rain_bin_colors/rain_bin_labels.
 */
function renderBlendedMapCanvas(container, grid, binColors) {
  container.innerHTML = "";

  const latCount = grid.bin_index.length;
  const lonCount = grid.bin_index[0].length;

  const canvas = document.createElement("canvas");
  canvas.width = lonCount * BLENDED_MAP_CELL_SIZE;
  canvas.height = latCount * BLENDED_MAP_CELL_SIZE;
  canvas.style.width = "100%";
  canvas.style.height = "auto";
  canvas.style.imageRendering = "pixelated";
  canvas.setAttribute("role", "img");
  canvas.setAttribute(
    "aria-label",
    `Blended precipitation map at lead ${grid.lead_hours}h`
  );

  const ctx = canvas.getContext("2d");

  for (let latIndex = 0; latIndex < latCount; latIndex++) {
    // grid.latitude is ascending (south to north); canvas y=0 is the top,
    // so the highest-latitude row must draw at the top -- flip vertically.
    const canvasRow = latCount - 1 - latIndex;
    for (let lonIndex = 0; lonIndex < lonCount; lonIndex++) {
      const binIndex = grid.bin_index[latIndex][lonIndex];
      ctx.fillStyle = binColors[binIndex] || "#000000";
      ctx.fillRect(
        lonIndex * BLENDED_MAP_CELL_SIZE,
        canvasRow * BLENDED_MAP_CELL_SIZE,
        BLENDED_MAP_CELL_SIZE,
        BLENDED_MAP_CELL_SIZE
      );
    }
  }

  container.appendChild(canvas);
}

function renderBlendedMapLegend(container, rainBinLabels, rainBinColors) {
  const legend = document.createElement("div");
  legend.className = "legend";
  rainBinLabels.forEach((label) => {
    const item = document.createElement("span");
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = rainBinColors[label] || "#000000";
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(label));
    legend.appendChild(item);
  });
  container.appendChild(legend);
}

const BlendedMapView = {
  /** Fetches real data for `lead` and renders canvas + captions + legend into `container`. */
  async render(container, lead) {
    container.innerHTML = '<p class="loading-state">Loading blended map…</p>';

    let grid;
    let colors;
    try {
      [grid, colors] = await Promise.all([fetchBlendedMap(lead), getColors()]);
    } catch (err) {
      container.innerHTML = `<p class="error-state">Failed to load blended map: ${err.message}</p>`;
      return;
    }

    container.innerHTML = "";

    const caption = document.createElement("p");
    caption.className = "view-caption";
    caption.textContent = BLENDED_MAP_CAPTION;
    container.appendChild(caption);

    const title = document.createElement("p");
    title.style.fontSize = "0.85rem";
    title.style.color = "var(--color-text-muted)";
    title.textContent = `Blended precipitation at lead ${grid.lead_hours}h (sample: ${grid.sample_time})`;
    container.appendChild(title);

    const binColors = colors.rain_bin_labels.map(
      (label) => colors.rain_bin_colors[label]
    );

    const canvasContainer = document.createElement("div");
    container.appendChild(canvasContainer);
    renderBlendedMapCanvas(canvasContainer, grid, binColors);

    renderBlendedMapLegend(container, colors.rain_bin_labels, colors.rain_bin_colors);

    const colorCaption = document.createElement("p");
    colorCaption.className = "view-caption";
    colorCaption.style.marginTop = "10px";
    colorCaption.textContent = BLENDED_MAP_COLOR_CAPTION;
    container.appendChild(colorCaption);
  },
};
