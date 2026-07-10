// Minimal dependency-free canvas charts (line + scatter) with point picking.

const MARGIN = { left: 52, right: 14, top: 10, bottom: 24 };

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, width: rect.width, height: rect.height };
}

function niceTicks(min, max, target = 5) {
  if (!(max > min)) {
    max = min + (Math.abs(min) || 1);
  }
  const span = max - min;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / target)));
  const step = [1, 2, 5, 10].map((m) => m * step0).find((s) => span / s <= target) || step0 * 10;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-9 * span; t += step) ticks.push(t);
  return ticks;
}

const fmt = (v) => {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return String(Math.round(v * 100) / 100);
  return String(Math.round(v * 1000) / 1000);
};

// points: [{x, y, label?, flag?}]; opts: {line: bool, onClick(point), color}
// canvas._zoom = {x: [x0,x1]|null, y: [y0,y1]|null} — drag-selected domains;
// a null axis autoscales (y re-fits the points inside the x window)
const MAX_RAW_POINTS = 700;

export function drawChart(canvas, points, opts = {}) {
  const zoom = canvas._zoom ?? null;
  let pts = zoom?.x ? points.filter((p) => p.x >= zoom.x[0] && p.x <= zoom.x[1]) : points;
  let bucketed = false;
  if (opts.line !== false && pts.length > MAX_RAW_POINTS) {
    bucketed = true;
    const size = Math.ceil(pts.length / MAX_RAW_POINTS);
    const buckets = [];
    for (let i = 0; i < pts.length; i += size) {
      const bucket = pts.slice(i, i + size);
      const y = bucket.reduce((sum, p) => sum + p.y, 0) / bucket.length;
      buckets.push({
        x: bucket[0].x,
        y,
        label: `${fmt(bucket[0].x)}–${fmt(bucket.at(-1).x)}\nmean = ${fmt(y)} (${bucket.length} pts)`,
      });
    }
    pts = buckets;
  }
  const { ctx, width, height } = setupCanvas(canvas);
  const css = getComputedStyle(document.documentElement);
  const colText = css.getPropertyValue("--muted").trim();
  const colBorder = css.getPropertyValue("--border").trim();
  const colMain = opts.color || css.getPropertyValue("--accent").trim();
  const colFlag = css.getPropertyValue("--bad").trim();
  ctx.clearRect(0, 0, width, height);

  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  ctx.font = "11px ui-monospace, monospace";
  if (!points.length) {
    ctx.fillStyle = colText;
    ctx.fillText("no data", MARGIN.left + plotW / 2 - 20, MARGIN.top + plotH / 2);
    return;
  }

  let xMin, xMax;
  if (zoom?.x) {
    [xMin, xMax] = zoom.x;
  } else {
    const xs = pts.map((p) => p.x);
    [xMin, xMax] = [Math.min(...xs), Math.max(...xs)];
  }
  if (xMin === xMax) [xMin, xMax] = [xMin - 0.5, xMax + 0.5];
  let yMin, yMax;
  if (zoom?.y) {
    [yMin, yMax] = zoom.y;
  } else {
    const ys = pts.map((p) => p.y);
    [yMin, yMax] = ys.length ? [Math.min(...ys), Math.max(...ys)] : [0, 1];
    if (yMin === yMax) [yMin, yMax] = [yMin - 0.5, yMax + 0.5];
    const yPad = (yMax - yMin) * 0.08;
    yMin -= yPad;
    yMax += yPad;
  }
  const X = (v) => MARGIN.left + ((v - xMin) / (xMax - xMin)) * plotW;
  const Y = (v) => MARGIN.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  ctx.strokeStyle = colBorder;
  ctx.fillStyle = colText;
  ctx.lineWidth = 1;
  for (const t of niceTicks(yMin, yMax)) {
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, Y(t));
    ctx.lineTo(width - MARGIN.right, Y(t));
    ctx.stroke();
    ctx.fillText(fmt(t), 4, Y(t) + 3);
  }
  for (const t of niceTicks(xMin, xMax, 8)) {
    ctx.fillText(fmt(t), X(t) - 8, height - 6);
  }

  // marks are clipped to the plot area so zoomed-out data cannot paint over
  // the margins; the full point list is drawn so lines run to the box edges
  ctx.save();
  ctx.beginPath();
  ctx.rect(MARGIN.left, MARGIN.top, plotW, plotH);
  ctx.clip();
  if (opts.line !== false) {
    const linePts = bucketed ? pts : points;
    ctx.strokeStyle = colMain;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    linePts.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y))));
    ctx.stroke();
  }
  if (!bucketed) {
    for (const p of points) {
      ctx.fillStyle = p.flag ? colFlag : colMain;
      ctx.beginPath();
      ctx.arc(X(p.x), Y(p.y), opts.line !== false ? 3 : 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();

  const nearest = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    let best = null;
    let bestDist = 20;
    for (const p of pts) {
      const d = Math.hypot(X(p.x) - mx, Y(p.y) - my);
      if (d < bestDist) {
        bestDist = d;
        best = p;
      }
    }
    return best;
  };
  const hover = (ev) => {
    const p = nearest(ev);
    canvas.style.cursor = p && opts.onClick ? "pointer" : "crosshair";
    if (p) {
      showTooltip(ev.clientX, ev.clientY, p.label ?? `${fmt(p.x)}, ${fmt(p.y)}`);
    } else {
      hideTooltip();
    }
  };
  installDragZoom(canvas, {
    plotLeft: MARGIN.left,
    plotRight: width - MARGIN.right,
    plotTop: MARGIN.top,
    plotBottom: height - MARGIN.bottom,
    invX: (px) => xMin + ((px - MARGIN.left) / plotW) * (xMax - xMin),
    invY: (py) => yMax - ((py - MARGIN.top) / plotH) * (yMax - yMin),
    redraw: () => drawChart(canvas, points, opts),
    hover,
    onClick:
      opts.onClick &&
      ((ev) => {
        const p = nearest(ev);
        if (p) opts.onClick(p);
      }),
  });
}

// an axis participates in a zoom only when dragged at least this many px, so
// a horizontal-ish sweep zooms x and re-autoscales y instead of pinning y to
// the few pixels the hand wobbled
const AXIS_MIN_DRAG = 10;

// drag a box on the plot = zoom into the selected x/y ranges; double-click
// resets. Handlers are re-installed on every render so invX/invY match the
// pixels on screen; drag state and the zoom live on the canvas element
// because redraw() re-renders (and re-installs handlers) mid-drag. move/up
// listen on window: a real drag routes events to whatever element is under
// the pointer, so canvas-local handlers would lose any drag that strays
// outside the canvas band and never complete the zoom.
function installDragZoom(canvas, { plotLeft, plotRight, plotTop, plotBottom, invX, invY, redraw, hover, onClick }) {
  const pos = (ev) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.max(plotLeft, Math.min(plotRight, ev.clientX - rect.left)),
      y: Math.max(plotTop, Math.min(plotBottom, ev.clientY - rect.top)),
    };
  };
  canvas.onmousemove = (ev) => {
    if (canvas._dragAnchor == null) hover(ev);
  };
  canvas.onmousedown = (ev) => {
    ev.preventDefault(); // no native text selection mid-drag
    canvas._dragAnchor = pos(ev);
    hideTooltip();
    const onMove = (mv) => {
      const cur = pos(mv);
      redraw();
      const a = canvas._dragAnchor;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "rgba(47, 111, 235, 0.15)";
      // full-height band while the drag is x-only; a box once it has y extent
      const boxY = Math.abs(cur.y - a.y) >= AXIS_MIN_DRAG;
      ctx.fillRect(
        Math.min(a.x, cur.x),
        boxY ? Math.min(a.y, cur.y) : plotTop,
        Math.abs(cur.x - a.x),
        boxY ? Math.abs(cur.y - a.y) : plotBottom - plotTop,
      );
    };
    const onUp = (up) => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      const a = canvas._dragAnchor;
      canvas._dragAnchor = null;
      const cur = pos(up);
      const dx = Math.abs(cur.x - a.x);
      const dy = Math.abs(cur.y - a.y);
      if (dx < 4 && dy < 4) {
        redraw();
        if (onClick) onClick(up);
        return;
      }
      const prev = canvas._zoom ?? { x: null, y: null };
      canvas._zoom = {
        x: dx >= AXIS_MIN_DRAG ? [invX(Math.min(a.x, cur.x)), invX(Math.max(a.x, cur.x))] : prev.x,
        y:
          dy >= AXIS_MIN_DRAG
            ? [invY(Math.max(a.y, cur.y)), invY(Math.min(a.y, cur.y))]
            : dx >= AXIS_MIN_DRAG
              ? null // x-only sweep: let y re-fit the new window
              : prev.y,
      };
      redraw();
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };
  canvas.onmouseleave = hideTooltip;
  canvas.ondblclick = () => {
    canvas._zoom = null;
    redraw();
  };
}

// ------------------------------ tooltip -------------------------------------

let tooltipEl = null;

export function showTooltip(clientX, clientY, text) {
  if (!tooltipEl) {
    tooltipEl = document.createElement("div");
    tooltipEl.id = "tooltip";
    document.body.appendChild(tooltipEl);
  }
  tooltipEl.textContent = text;
  tooltipEl.style.display = "block";
  const pad = 12;
  const w = tooltipEl.offsetWidth;
  const x = clientX + pad + w > window.innerWidth ? clientX - w - pad : clientX + pad;
  tooltipEl.style.left = `${x}px`;
  tooltipEl.style.top = `${Math.min(clientY + pad, window.innerHeight - tooltipEl.offsetHeight - pad)}px`;
}

export function hideTooltip() {
  if (tooltipEl) tooltipEl.style.display = "none";
}

// diverging color for values around a center (e.g. imp_ratio around 1)
export function divergingColor(t) {
  // t in [-1, 1]: blue -> transparent -> red
  const a = Math.min(1, Math.abs(t)) * 0.85;
  return t >= 0 ? `rgba(224, 96, 96, ${a})` : `rgba(78, 161, 255, ${a})`;
}

// sequential color for magnitudes in [0, 1]
export function sequentialColor(t) {
  return `rgba(70, 194, 142, ${Math.min(1, Math.max(0, t)) * 0.85})`;
}
