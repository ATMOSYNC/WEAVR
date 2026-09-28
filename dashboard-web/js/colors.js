/**
 * Turns /api/colors's response into a reusable lookup table for chart
 * modules. No hex values or IMD thresholds are hardcoded here or in any
 * chart module -- they all come from dashboard/colors.py via the API, so
 * this can't silently fork from that single source of truth
 * (frontendplan.md §2.4's own requirement).
 */

let colorsPromise = null;

/** Fetches /api/colors once and caches the result for the page's lifetime. */
function getColors() {
  if (!colorsPromise) {
    colorsPromise = fetchColors();
  }
  return colorsPromise;
}
