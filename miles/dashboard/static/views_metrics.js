import { api } from "./api.js";
import { el } from "./app.js";
import { drawChart } from "./charts.js";

// wandb-shaped L0: metric names come verbatim from the tracking fan-out, so
// categories are simply the key prefixes. One category at a time (wandb
// sidebar idiom); every key of the active category renders as a line chart.
// "dump" only ever appears for dump-only dirs — the server does not
// advertise it while a telemetry stream exists.
const CATEGORY_ORDER = ["rollout", "perf", "train", "eval", "dump"];

const categoryOf = (key) => key.split("/")[0];

function axisOf(key) {
  if (key.startsWith("dump/")) return "dump";
  if (key.startsWith("train/")) return "train/step";
  if (key.startsWith("eval/")) return "eval/step";
  return "rollout/step";
}

function orderedCategories(meta) {
  const present = [...new Set(meta.metric_keys.map(categoryOf))];
  return [
    ...CATEGORY_ORDER.filter((c) => present.includes(c)),
    ...present.filter((c) => !CATEGORY_ORDER.includes(c)).sort(),
  ];
}

export async function renderMetrics(view, meta) {
  const cats = orderedCategories(meta);
  let active = sessionStorage.getItem("metricsCategory");
  if (!cats.includes(active)) active = cats[0] ?? null;

  const chartsPanel = el("div", {
    style: "flex: 3; min-width: 500px; display: grid; grid-template-columns: repeat(2, minmax(0, 500px)); gap: 0 14px; align-content: start",
  });
  const filterInput = el("input", { type: "text", placeholder: "filter…" });
  const catList = el("div", { class: "keylist" });

  const renderCategories = () => {
    catList.replaceChildren(
      ...cats.map((cat) =>
        el(
          "button",
          {
            class: cat === active ? "active" : "",
            style: "text-align: left",
            onclick: () => {
              active = cat;
              sessionStorage.setItem("metricsCategory", cat);
              renderCategories();
              buildPanels();
            },
          },
          [cat],
        ),
      ),
    );
  };

  const activeKeys = () => {
    const needle = filterInput.value.toLowerCase();
    return meta.metric_keys.filter((k) => categoryOf(k) === active && k.toLowerCase().includes(needle));
  };

  // slots persist across refreshes; a chart repaints ONLY when its data
  // changed (signature check), so a live page sits visually still between
  // real updates instead of flickering on every poll
  const slots = new Map(); // key -> {canvas, status, signature}
  let epoch = 0;

  function buildPanels() {
    slots.clear();
    const keys = activeKeys();
    if (!keys.length) {
      chartsPanel.replaceChildren(el("p", { class: "muted" }, [active ? "no metrics here" : "no metrics logged"]));
      return;
    }
    chartsPanel.replaceChildren(
      ...keys.map((key) => {
        const canvas = el("canvas", { class: "chart" });
        const status = el("p", { class: "muted" }, ["loading…"]);
        slots.set(key, { canvas, status, signature: null });
        return el("div", { class: "panel" }, [el("p", { class: "chart-title" }, [key]), status, canvas]);
      }),
    );
    refresh();
  }

  function refresh() {
    const current = ++epoch;
    const byAxis = new Map();
    for (const key of slots.keys()) {
      const axis = axisOf(key);
      if (!byAxis.has(axis)) byAxis.set(axis, []);
      byAxis.get(axis).push(key);
    }
    for (const [axis, keys] of byAxis.entries()) {
      api("/api/metrics", { keys: keys.join(","), x: axis === "dump" ? "rollout/step" : axis })
        .then((series) => {
          if (current !== epoch) return; // category/filter changed meanwhile
          for (const key of keys) {
            const slot = slots.get(key);
            if (!slot) continue;
            const s = series[key] ?? { x: [], y: [] };
            const signature = `${s.x.length}:${s.x.at(-1)}:${s.y.at(-1)}`;
            if (signature === slot.signature) continue; // unchanged: no repaint
            slot.signature = signature;
            slot.status.remove();
            drawChart(
              slot.canvas,
              s.x.map((x, i) => ({ x, y: s.y[i], label: `step ${x}\n${key} = ${s.y[i]}` })),
            );
          }
        })
        .catch((err) => {
          if (current !== epoch) return;
          for (const key of keys) {
            const slot = slots.get(key);
            if (slot && slot.signature === null) slot.status.textContent = String(err);
          }
        });
    }
  }

  filterInput.oninput = buildPanels;

  view.replaceChildren(
    el("div", { class: "row" }, [
      el("div", { class: "panel", style: "flex: 0 0 200px; min-width: 200px" }, [
        el("h3", {}, ["metrics"]),
        el("div", { class: "controls" }, [filterInput]),
        catList,
      ]),
      chartsPanel,
    ]),
  );
  renderCategories();
  buildPanels();
}
