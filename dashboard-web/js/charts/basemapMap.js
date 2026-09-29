/**
 * Reusable geographic map: draws a regular lat/lon grid (the project's
 * 0.25 degree common grid) over the local OpenStreetMap-derived basemap
 * (docs/basemap-scope.md). Used by the blended-map view and, later, the
 * extreme-probability view, via one shared API:
 *
 *   const available = await BasemapMap.isAvailable();
 *   const map = await BasemapMap.create(container);   // rejects on failure
 *   map.setCells({ latitude, longitude, colorAt });    // colorAt(i, j) -> css colour | null
 *   map.destroy();
 *
 * Cells are drawn as GeoJSON polygons, not a stretched picture: the map is
 * Web Mercator, which is not linear in latitude, so an image overlay over a
 * rectangular bounding box would misplace cells by kilometres (more so
 * toward the north). Grid coordinates are cell CENTRES, so each cell spans
 * half a grid step either side of its coordinate.
 *
 * Nothing here talks to any remote host: the style, tiles and libraries are
 * all served locally, and the basemap draws no boundaries (step 06 adds the
 * official one on top).
 */

const BASEMAP_MAP_TILES_PATH = "/basemap/india.pmtiles";
const BASEMAP_MAP_STYLE_PATH = "/basemap/style.json";
const BASEMAP_MAP_TILES_PLACEHOLDER = "__BASEMAP_TILES_URL__";
const BASEMAP_MAP_LOAD_TIMEOUT_MS = 10000;
const BASEMAP_MAP_CELLS_SOURCE = "weavr-cells";
const BASEMAP_MAP_CELLS_LAYER = "weavr-cells-fill";
const BASEMAP_MAP_CELL_OPACITY = 0.78;
const BASEMAP_MAP_OVERLAY_SOURCE = "weavr-overlay";
const BASEMAP_MAP_OVERLAY_LAYER = "weavr-overlay-fill";
const BASEMAP_MAP_OVERLAY_OPACITY = 0.75;

let basemapMapProtocolRegistered = false;
const basemapMapInstances = new Set();

/**
 * Spacing between adjacent grid coordinates. Derived from the data rather
 * than hard-coded, and checked to be regular.
 */
function basemapMapGridStep(values) {
  if (values.length < 2) {
    throw new Error("grid needs at least two coordinates to infer its spacing");
  }
  const step = values[1] - values[0];
  const tolerance = Math.abs(step) * 1e-6;
  for (let i = 2; i < values.length; i++) {
    if (Math.abs(values[i] - values[i - 1] - step) > tolerance) {
      throw new Error("grid coordinates are not regularly spaced");
    }
  }
  return step;
}

/**
 * One polygon per cell, spanning +-half a step around its centre.
 * `colorAt(i, j)` gives the colour for latitude index i, longitude index j;
 * a null/undefined colour leaves the cell out (transparent). Optional
 * `opacityAt(i, j)` gives each cell its own 0..1 fill opacity (default: the
 * layer's standard opacity).
 * Returns a GeoJSON FeatureCollection in [lon, lat] order.
 */
function basemapMapCellPolygons(latitude, longitude, colorAt, opacityAt) {
  const halfLat = Math.abs(basemapMapGridStep(latitude)) / 2;
  const halfLon = Math.abs(basemapMapGridStep(longitude)) / 2;
  const features = [];
  for (let i = 0; i < latitude.length; i++) {
    for (let j = 0; j < longitude.length; j++) {
      const color = colorAt(i, j);
      if (!color) continue;
      const west = longitude[j] - halfLon;
      const east = longitude[j] + halfLon;
      const south = latitude[i] - halfLat;
      const north = latitude[i] + halfLat;
      features.push({
        type: "Feature",
        properties: {
          c: color,
          o: opacityAt ? opacityAt(i, j) : BASEMAP_MAP_CELL_OPACITY,
        },
        geometry: {
          type: "Polygon",
          coordinates: [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
      });
    }
  }
  return { type: "FeatureCollection", features };
}

/** [west, south, east, north] of the whole grid, edges included. */
function basemapMapGridBounds(latitude, longitude) {
  const halfLat = Math.abs(basemapMapGridStep(latitude)) / 2;
  const halfLon = Math.abs(basemapMapGridStep(longitude)) / 2;
  return [
    Math.min(longitude[0], longitude[longitude.length - 1]) - halfLon,
    Math.min(latitude[0], latitude[latitude.length - 1]) - halfLat,
    Math.max(longitude[0], longitude[longitude.length - 1]) + halfLon,
    Math.max(latitude[0], latitude[latitude.length - 1]) + halfLat,
  ];
}

function basemapMapHasWebGl() {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch (err) {
    return false;
  }
}

/** True only if the libraries loaded, WebGL works and the tile file exists. */
async function basemapMapIsAvailable() {
  if (typeof maplibregl === "undefined" || typeof pmtiles === "undefined") return false;
  if (!basemapMapHasWebGl()) return false;
  try {
    const response = await fetch("/api/basemap/status");
    if (!response.ok) return false;
    const status = await response.json();
    return status.available === true;
  } catch (err) {
    return false;
  }
}

function basemapMapPruneDetached() {
  basemapMapInstances.forEach((instance) => {
    if (!instance.container.isConnected) {
      instance.destroy();
    }
  });
}

/**
 * Creates the map inside `container` and resolves once it has loaded.
 * Rejects (so the caller can fall back to its canvas) if the style or tiles
 * fail, or if loading takes longer than the timeout.
 */
async function basemapMapCreate(container, options = {}) {
  basemapMapPruneDetached();

  if (!basemapMapProtocolRegistered) {
    maplibregl.addProtocol("pmtiles", new pmtiles.Protocol().tile);
    basemapMapProtocolRegistered = true;
  }

  const styleResponse = await fetch(BASEMAP_MAP_STYLE_PATH);
  if (!styleResponse.ok) throw new Error(`basemap style -> ${styleResponse.status}`);
  const style = await styleResponse.json();
  const tilesUrl = `pmtiles://${window.location.origin}${BASEMAP_MAP_TILES_PATH}`;
  style.sources.basemap.url = style.sources.basemap.url.replace(
    `pmtiles://${BASEMAP_MAP_TILES_PLACEHOLDER}`,
    tilesUrl
  );

  container.classList.add("basemap-map");
  if (options.height) container.style.height = options.height;

  const map = new maplibregl.Map({
    container,
    style,
    center: [83, 22],
    zoom: 3.6,
    minZoom: 3,
    maxZoom: 10,
    attributionControl: { compact: false },
    dragRotate: false,
    pitchWithRotate: false,
  });
  map.touchZoomRotate.disableRotation();
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

  await new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error("basemap did not load in time")),
      BASEMAP_MAP_LOAD_TIMEOUT_MS
    );
    map.once("load", () => {
      clearTimeout(timer);
      resolve();
    });
    map.once("error", (event) => {
      clearTimeout(timer);
      reject(event.error || new Error("basemap failed to load"));
    });
  }).catch((err) => {
    map.remove();
    container.classList.remove("basemap-map");
    throw err;
  });

  let fitted = false;
  const instance = {
    container,
    map,
    destroyed: false,
    /**
     * grid = {latitude[], longitude[], colorAt(i, j), opacityAt?(i, j),
     *         overlayColorAt?(i, j)}
     * The optional overlay is a second, distinct layer drawn on top (used
     * for the extreme-probability "fallback cell" grey pass).
     */
    setCells(grid) {
      const upsert = (sourceId, layerId, data, opacityExpression) => {
        const source = map.getSource(sourceId);
        if (source) {
          source.setData(data);
          return;
        }
        map.addSource(sourceId, { type: "geojson", data });
        map.addLayer({
          id: layerId,
          type: "fill",
          source: sourceId,
          paint: {
            "fill-color": ["get", "c"],
            "fill-opacity": opacityExpression,
            "fill-antialias": false,
          },
        });
      };
      upsert(
        BASEMAP_MAP_CELLS_SOURCE,
        BASEMAP_MAP_CELLS_LAYER,
        basemapMapCellPolygons(grid.latitude, grid.longitude, grid.colorAt, grid.opacityAt),
        ["get", "o"]
      );
      const overlayAt = grid.overlayColorAt || (() => null);
      upsert(
        BASEMAP_MAP_OVERLAY_SOURCE,
        BASEMAP_MAP_OVERLAY_LAYER,
        basemapMapCellPolygons(grid.latitude, grid.longitude, overlayAt, () => BASEMAP_MAP_OVERLAY_OPACITY),
        ["get", "o"]
      );
      if (!fitted) {
        const [w, s, e, n] = basemapMapGridBounds(grid.latitude, grid.longitude);
        map.fitBounds([[w, s], [e, n]], { padding: 12, animate: false });
        fitted = true;
      }
    },
    /** Call when the container was hidden while the map existed. */
    resize() {
      map.resize();
    },
    destroy() {
      if (instance.destroyed) return;
      instance.destroyed = true;
      basemapMapInstances.delete(instance);
      map.remove();
      container.classList.remove("basemap-map");
    },
  };
  basemapMapInstances.add(instance);
  return instance;
}

const basemapMapSharedEntries = new Map();

/**
 * One long-lived map per view, re-attached each time the view re-renders (a
 * lead change rebuilds the view's DOM, and creating a map costs about a
 * second). The creation PROMISE is cached, not the finished handle, so a
 * re-render while the map is still loading joins that creation instead of
 * starting a second map.
 *
 *   const entry = BasemapMap.shared("blended");
 *   container.appendChild(entry.element);   // before calling handle()
 *   const handle = await entry.handle();     // rejects on failure
 */
function basemapMapShared(key) {
  let entry = basemapMapSharedEntries.get(key);
  if (entry && entry.destroyed) {
    basemapMapSharedEntries.delete(key); // another view pruned it
    entry = undefined;
  }
  if (!entry) {
    entry = {
      element: document.createElement("div"),
      destroyed: false,
      promise: null,
      handle() {
        if (!entry.promise) {
          entry.promise = basemapMapCreate(entry.element).then(
            (created) => {
              const destroy = created.destroy;
              created.destroy = () => {
                entry.destroyed = true;
                destroy();
              };
              return created;
            },
            (err) => {
              if (basemapMapSharedEntries.get(key) === entry) basemapMapSharedEntries.delete(key);
              throw err;
            }
          );
        }
        return entry.promise;
      },
    };
    basemapMapSharedEntries.set(key, entry);
  }
  return entry;
}

const BasemapMap = {
  isAvailable: basemapMapIsAvailable,
  create: basemapMapCreate,
  shared: basemapMapShared,
  // Exposed so the half-cell offset and the grid extent can be checked
  // directly in the browser (see docs/basemap-scope.md, step 04).
  cellPolygons: basemapMapCellPolygons,
  gridBounds: basemapMapGridBounds,
};
