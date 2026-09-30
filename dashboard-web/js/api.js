/**
 * Thin fetch() wrappers over dashboard/api.py's 6 real endpoints
 * (docs/frontend-migration-scope.md's locked decisions). Every function
 * here just fetches and JSON-decodes -- no data transformation happens in
 * this file, matching the plan's own separation between api.js (fetch)
 * and the chart modules (render).
 */

const API_BASE = "";

async function fetchJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = body.detail || response.statusText;
    throw new Error(`${path} -> ${response.status}: ${detail}`);
  }
  return response.json();
}

/** GET /api/weight-map?lead=<int> -> real per-region, per-source weight rows. */
async function fetchWeightMap(lead) {
  return fetchJson(`/api/weight-map?lead=${encodeURIComponent(lead)}`);
}

/** GET /api/skill-trends?metric=<rmse_mm|crps_mm> -> real per-method, per-lead rows. */
async function fetchSkillTrends(metric) {
  return fetchJson(`/api/skill-trends?metric=${encodeURIComponent(metric)}`);
}

/** GET /api/blended-map?lead=<int> -> the real 129x135 blended-forecast grid. */
async function fetchBlendedMap(lead) {
  return fetchJson(`/api/blended-map?lead=${encodeURIComponent(lead)}`);
}

/**
 * GET /api/extreme-probability?lead=<int>&threshold=<mm>
 *   -> the real 129x135 P(rain > threshold) grid, with the real per-cell
 *   `method` flags (csgd / csgd+gpd_tail / fallback).
 *
 * `threshold` is always sent explicitly, because the server's default is
 * 204.5 and this view lets the reader pick between the two real IMD
 * thresholds; omitting it would silently show 204.5 under a 115.6mm label.
 */
async function fetchExtremeProbability(lead, threshold) {
  return fetchJson(
    `/api/extreme-probability?lead=${encodeURIComponent(lead)}` +
      `&threshold=${encodeURIComponent(threshold)}`
  );
}

/** GET /api/colors -> the real IMD colour identities (dashboard/colors.py). */
async function fetchColors() {
  return fetchJson("/api/colors");
}

/** GET /api/meta/leads -> the 5 real lead times both example grids share. */
async function fetchLeads() {
  return fetchJson("/api/meta/leads");
}
