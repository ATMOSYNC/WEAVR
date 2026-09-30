/**
 * Canvas-level checks for the extreme-probability view.
 *
 * There is no browser in CI, so these load the real, unmodified
 * `extremeProbability.js` into a vm context with a recording stand-in for
 * `document`, and assert on the exact `fillRect` calls the real render path
 * makes. That is pixel sampling of the canvas that would be painted, without
 * a headless-browser download: it catches the failure that matters here --
 * a cell marked with the wrong estimator's glyph, or a marking that silently
 * stops being drawn -- rather than a CSS or layout regression.
 *
 * Run with:  node --test dashboard-web/tests/
 */

// A minimal DOM: enough for the real render functions, recording what they
// build so assertions can read the resulting text back out.
function makeElement(tagName) {
  const el = {
    tagName,
    style: {},
    children: [],
    attributes: {},
    className: "",
    id: "",
    textContent: "",
    innerHTML: "",
    hidden: false,
    listeners: {},
    appendChild(child) {
      el.children.push(child);
      return child;
    },
    addEventListener(name, handler) {
      el.listeners[name] = handler;
    },
    setAttribute(name, value) {
      el.attributes[name] = value;
    },
  };
  if (tagName === "canvas") {
    el.context = {
      fillStyle: null,
      fills: [],
      fillRect(x, y, w, h) {
        el.context.fills.push({ x, y, w, h, style: el.context.fillStyle });
      },
    };
    el.getContext = () => el.context;
  }
  return el;
}

/**
 * Loads `filename` in a fresh vm context and returns its globals.
 *
 * Top-level `function` declarations land on the vm global, but `const` and
 * `let` stay in the script's lexical scope -- so `exportNames` are read
 * through an epilogue evaluated in that same scope, which is how the
 * caption and the tail constants are reached.
 */
function loadView(filename, extraGlobals = {}, exportNames = []) {
  const fs = require("node:fs");
  const path = require("node:path");
  const vm = require("node:vm");

  let source = fs.readFileSync(path.join(__dirname, "..", filename), "utf8");
  if (exportNames.length > 0) {
    source += `\n;globalThis.__consts = { ${exportNames.join(", ")} };`;
  }

  const sandbox = {
    document: {
      createElement: makeElement,
      createTextNode: (text) => ({ nodeType: "text", textContent: String(text) }),
    },
    console,
    ...extraGlobals,
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename });
  if (exportNames.length > 0) {
    Object.assign(sandbox, sandbox.__consts);
  }
  return sandbox;
}

/** A 2x3 grid with one cell of each method, so every layer is exercised. */
function mixedMethodGrid() {
  return {
    threshold: 115.6,
    lead_hours: 24,
    sample_time: "2020-09-28T00:00:00",
    latitude: [10, 9],
    longitude: [70, 71, 72],
    probability: [
      [0.0, 0.1, 0.2],
      [0.3, 0.4, 0.5],
    ],
    is_fallback: [
      [false, false, false],
      [false, false, true], // (1,2) fallback
    ],
    method: [
      ["csgd", "csgd+gpd_tail", "csgd"], // (0,1) pooled-tail
      ["csgd", "csgd", "fallback"],
    ],
  };
}

const CELL = 4; // EXTREME_PROBABILITY_CELL_SIZE
const TAIL_STYLE = "rgba(20, 20, 20, 0.85)";
const FALLBACK_STYLE = "rgba(120, 120, 120, 0.75)";

const test = require("node:test");
const assert = require("node:assert/strict");

test("canvas paints three layers, each only on its own method's cells", () => {
  const view = loadView("js/charts/extremeProbability.js");
  const grid = mixedMethodGrid();
  const host = makeElement("div");

  view.renderExtremeProbabilityCanvas(
    host,
    grid,
    [
      [0.0, "#ffffff"],
      [1.0, "#0000ff"],
    ]
  );

  const canvas = host.children.find((c) => c.tagName === "canvas");
  assert.ok(canvas, "a canvas was appended");
  const fills = canvas.context.fills;

  // Layer 1: one probability fill per cell, 6 cells.
  const probabilityFills = fills.filter((f) => f.style !== FALLBACK_STYLE && f.style !== TAIL_STYLE);
  assert.equal(probabilityFills.length, 6, "every cell gets a probability colour");

  // Layer 2: grey on the single fallback cell only. The canvas row is
  // flipped, so latIndex 1 lands on canvasRow 0.
  const greyFills = fills.filter((f) => f.style === FALLBACK_STYLE);
  assert.equal(greyFills.length, 1, "only the fallback cell is greyed");
  assert.deepEqual(
    { x: greyFills[0].x, y: greyFills[0].y, w: greyFills[0].w, h: greyFills[0].h },
    { x: 2 * CELL, y: 0, w: CELL, h: CELL },
    "the fallback cell is at (lon 2, lat 1)"
  );

  // Layer 3: the tail cell's centred dot only -- the (0,1) cell, which is
  // canvasRow 1.
  const tailFills = fills.filter((f) => f.style === TAIL_STYLE);
  assert.equal(tailFills.length, 1, "only the csgd+gpd_tail cell is dotted");
  assert.deepEqual(
    { x: tailFills[0].x, y: tailFills[0].y, w: tailFills[0].w, h: tailFills[0].h },
    { x: 1 * CELL + 1, y: 1 * CELL + 1, w: 2, h: 2 },
    "the dot is centred in its cell"
  );

  // A fallback cell must never also be dotted: it has no real probability at
  // all, so marking it as merely uncertain would misrepresent it.
  assert.equal(
    fills.filter((f) => f.style === FALLBACK_STYLE && f.style === TAIL_STYLE).length,
    0
  );
});

test("a grid with no tail cells draws no dots at all", () => {
  const view = loadView("js/charts/extremeProbability.js");
  const grid = mixedMethodGrid();
  // The committed grid predates step 12: the server reports every cell as
  // plain csgd.
  grid.method = [
    ["csgd", "csgd", "csgd"],
    ["csgd", "csgd", "fallback"],
  ];
  const host = makeElement("div");
  view.renderExtremeProbabilityCanvas(host, grid, [[0.0, "#fff"], [1.0, "#00f"]]);

  const canvas = host.children.find((c) => c.tagName === "canvas");
  assert.equal(
    canvas.context.fills.filter((f) => f.style === TAIL_STYLE).length,
    0,
    "no dots when no cell uses the tail"
  );
});

test("method counts come from the real per-cell flags", () => {
  const view = loadView("js/charts/extremeProbability.js");
  // The returned object is built inside the vm realm, so its prototype
  // differs from this one; compare the counts rather than the identity.
  const assertCounts = (actual, expected) => {
    assert.equal(actual.csgd, expected.csgd, "csgd");
    assert.equal(actual.tail, expected.tail, "tail");
    assert.equal(actual.fallback, expected.fallback, "fallback");
  };

  assertCounts(view.extremeProbabilityMethodCounts(mixedMethodGrid()), {
    csgd: 4,
    tail: 1,
    fallback: 1,
  });

  // No `method` array at all (older server response shape): every cell is
  // counted from is_fallback rather than crashing.
  const legacy = mixedMethodGrid();
  delete legacy.method;
  assertCounts(view.extremeProbabilityMethodCounts(legacy), {
    csgd: 5,
    tail: 0,
    fallback: 1,
  });
});

test("the legend labels the threshold actually fetched", () => {
  const view = loadView("js/charts/extremeProbability.js");
  const grid = mixedMethodGrid(); // threshold 115.6
  const host = makeElement("div");

  view.renderExtremeProbabilityLegend(host, [[0.0, "#fff"], [1.0, "#00f"]], grid, {
    csgd: 4,
    tail: 1,
    fallback: 1,
  });

  const label = host.children[0].children[0];
  assert.equal(label.textContent, "P(rain > 115.6mm)", "label follows the grid, not a hardcoded 204.5");
  assert.ok(!label.textContent.includes("204.5"));

  // The tail entry states the real count...
  const tailNote = host.children[0].children[3];
  assert.match(tailNote.innerHTML, /pooled GPD tail/);
  assert.match(tailNote.innerHTML, /1 cell at this lead/);

  // ...and says so plainly when a grid uses none, instead of implying cells exist.
  const emptyHost = makeElement("div");
  view.renderExtremeProbabilityLegend(emptyHost, [[0.0, "#fff"], [1.0, "#00f"]], grid, {
    csgd: 5,
    tail: 0,
    fallback: 1,
  });
  const emptyNote = emptyHost.children[0].children[3];
  assert.match(emptyNote.innerHTML, /no cells at this threshold use it/);
});

test("the caption states the tail method and its limit", () => {
  const view = loadView(
    "js/charts/extremeProbability.js",
    {},
    ["EXTREME_PROBABILITY_CAPTION"]
  );
  const caption = view.EXTREME_PROBABILITY_CAPTION;
  assert.match(caption, /pooled extreme-value tail/);
  assert.match(caption, /generalised-Pareto tail/);
  // The limit has to be stated, not just the method.
  assert.match(caption, /least certain/);
  assert.match(caption, /extrapolat/);
  // Both real IMD thresholds are named, not just the one that used to be hardcoded.
  assert.match(caption, /115\.6mm/);
  assert.match(caption, /204\.5mm/);
});

test("the canvas aria-label follows the fetched threshold", () => {
  const view = loadView("js/charts/extremeProbability.js");
  const grid = mixedMethodGrid();
  const host = makeElement("div");
  view.renderExtremeProbabilityCanvas(host, grid, [[0.0, "#fff"], [1.0, "#00f"]]);
  const canvas = host.children.find((c) => c.tagName === "canvas");
  assert.equal(canvas.attributes["aria-label"], "P(rain > 115.6mm) map at lead 24h");
});

test("thresholds render without a trailing .0", () => {
  const view = loadView("js/charts/extremeProbability.js");
  assert.equal(view.formatThreshold(115.6), "115.6mm");
  assert.equal(view.formatThreshold(204.5), "204.5mm");
  assert.equal(view.formatThreshold(204), "204mm");
});

test("the threshold control offers the server's real values and reports changes", () => {
  const view = loadView("js/charts/extremeProbability.js");
  const host = makeElement("div");
  const picked = [];
  view.buildExtremeProbabilityThresholdControl(host, [115.6, 204.5], 204.5, (v) =>
    picked.push(v)
  );

  const group = host.children[0];
  assert.ok(group, "the control was appended to the container");
  assert.equal(group.attributes["aria-label"], "Rainfall threshold");
  assert.equal(group.attributes["role"], "radiogroup");
  const options = group.children.slice(1); // after the "Threshold" legend label
  assert.equal(options.length, 2, "one control per real threshold");
  const inputs = options.map((o) => o.children[0]);
  assert.deepEqual(
    inputs.map((i) => i.value),
    ["115.6", "204.5"]
  );
  // Each option is labelled with IMD's real name for that boundary, so the
  // reader is not left decoding a bare number.
  assert.match(options[0].children[1].textContent, /very heavy/);
  assert.match(options[1].children[1].textContent, /extremely heavy/);
  assert.equal(inputs[0].checked, false);
  assert.equal(inputs[1].checked, true, "the current threshold is pre-selected");

  inputs[0].checked = true;
  inputs[0].listeners.change();
  assert.deepEqual(picked, [115.6], "picking a threshold reports the new value");
});

test("api.js always sends the threshold explicitly", async () => {
  // The server's default is 204.5; omitting the parameter would render a
  // 115.6mm label over 204.5mm data.
  const requested = [];
  const sandbox = loadView("js/api.js", {
    fetch: (url) => {
      requested.push(url);
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    },
  });

  await sandbox.fetchExtremeProbability(24, 115.6);
  await sandbox.fetchExtremeProbability(24, 204.5);

  assert.equal(requested.length, 2);
  requested.forEach((url) => assert.match(url, /threshold=/, `${url} carries a threshold`));
  assert.match(requested[0], /lead=24/);
  assert.match(requested[0], /threshold=115\.6/);
  assert.match(requested[1], /threshold=204\.5/);
});
