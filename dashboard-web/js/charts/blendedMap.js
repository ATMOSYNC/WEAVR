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

// Remembered across lead changes so the choice survives a re-render.
let blendedMapGeographyOn = false;

// One live map is kept and re-attached on every re-render (a lead change
// rebuilds the view's DOM). Creating a map costs about a second; reusing it
// makes a lead change only a data update.
let blendedMapGeographyCache = null; // { element, destroyed, handlePromise }

const BLENDED_MAP_GEOGRAPHY_NOTE =
  "Cells are drawn on an OpenStreetMap-derived basemap served from this " +
  "machine. Dry cells are transparent so the geography shows through; " +
  "the basemap draws no national boundaries.";
const BLENDED_MAP_GEOGRAPHY_UNAVAILABLE =
  "Basemap unavailable -- showing the plain grid.";

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

    const subheader = document.createElement("h2");
    subheader.className = "view-subheader";
    subheader.textContent = "Blended map: Tier 1's real regional blend over India";
    container.appendChild(subheader);

    const caption = document.createElement("p");
    caption.className = "view-caption";
    caption.textContent = BLENDED_MAP_CAPTION;
    container.appendChild(caption);

    const title = document.createElement("p");
    title.className = "chart-title";
    title.textContent = `Blended precipitation at lead ${grid.lead_hours}h (sample: ${grid.sample_time})`;
    container.appendChild(title);

    const binColors = colors.rain_bin_labels.map(
      (label) => colors.rain_bin_colors[label]
    );

    const toggleLabel = document.createElement("label");
    toggleLabel.className = "map-mode-toggle";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.id = "blended-map-geography";
    toggle.checked = blendedMapGeographyOn;
    toggleLabel.appendChild(toggle);
    toggleLabel.appendChild(document.createTextNode("Show geography"));
    container.appendChild(toggleLabel);

    const geographyNote = document.createElement("p");
    geographyNote.className = "view-caption";
    geographyNote.id = "blended-map-geography-note";
    container.appendChild(geographyNote);

    const canvasContainer = document.createElement("div");
    container.appendChild(canvasContainer);
    renderBlendedMapCanvas(canvasContainer, grid, binColors);

    const mapContainer = blendedMapGeographyCache
      ? blendedMapGeographyCache.element
      : document.createElement("div");
    mapContainer.id = "blended-map-geography-map";
    mapContainer.hidden = true;
    container.appendChild(mapContainer);

    // Bin 0 ("dry") is left transparent on the map: a solid white fill would
    // hide the very geography the toggle is for.
    const colorAt = (i, j) => {
      const bin = grid.bin_index[i][j];
      return bin === 0 ? null : binColors[bin] || "#000000";
    };

    const showCanvas = (message) => {
      mapContainer.hidden = true;
      canvasContainer.hidden = false;
      geographyNote.textContent = message || "";
    };

    const showGeography = async () => {
      if (!(await BasemapMap.isAvailable())) {
        throw new Error("basemap unavailable");
      }
      // The creation PROMISE is cached, not the finished handle: a lead
      // change while the first map is still loading must join that creation
      // rather than start a second map.
      if (blendedMapGeographyCache && blendedMapGeographyCache.destroyed) {
        blendedMapGeographyCache = null; // another view pruned it
      }
      if (!blendedMapGeographyCache) {
        const entry = { element: mapContainer, destroyed: false, handlePromise: null };
        entry.handlePromise = BasemapMap.create(mapContainer).then((created) => {
          const destroy = created.destroy;
          created.destroy = () => {
            entry.destroyed = true;
            destroy();
          };
          return created;
        });
        blendedMapGeographyCache = entry;
      }
      const entry = blendedMapGeographyCache;
      let handle;
      try {
        handle = await entry.handlePromise;
      } catch (err) {
        if (blendedMapGeographyCache === entry) blendedMapGeographyCache = null;
        throw err;
      }
      handle.setCells({ latitude: grid.latitude, longitude: grid.longitude, colorAt });
      canvasContainer.hidden = true;
      mapContainer.hidden = false;
      handle.resize();
      geographyNote.textContent = BLENDED_MAP_GEOGRAPHY_NOTE;
    };

    const applyMode = async () => {
      blendedMapGeographyOn = toggle.checked;
      if (!toggle.checked) {
        showCanvas("");
        return;
      }
      try {
        await showGeography();
      } catch (err) {
        // Any failure -- no tile file, no WebGL, bad tiles -- leaves the
        // plain grid in place, with the reason stated, never a blank map.
        toggle.checked = false;
        blendedMapGeographyOn = false;
        blendedMapGeographyCache = null;
        mapContainer.innerHTML = "";
        showCanvas(BLENDED_MAP_GEOGRAPHY_UNAVAILABLE);
      }
    };
    toggle.addEventListener("change", applyMode);
    if (blendedMapGeographyOn) applyMode();

    renderBlendedMapLegend(container, colors.rain_bin_labels, colors.rain_bin_colors);

    const colorCaption = document.createElement("p");
    colorCaption.className = "view-caption";
    colorCaption.style.marginTop = "10px";
    colorCaption.textContent = BLENDED_MAP_COLOR_CAPTION;
    container.appendChild(colorCaption);
  },
};
