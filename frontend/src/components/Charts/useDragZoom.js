import { useEffect, useState } from 'react';

// Drags smaller than this (in axis units) are treated as clicks, so click
// handlers on chart elements (e.g. head-dot selection) survive the zoom
// handlers wired to the same surface.
const MIN_DRAG = 0.4;

/**
 * Drag-to-zoom state for a recharts chart with two numeric axes. Spread
 * `mouseHandlers` onto the chart: recharts mouse events carry `xValue`/
 * `yValue` (the axis scales inverted at the cursor) on ScatterChart, which
 * feed the in-progress selection rectangle; releasing the mouse commits it
 * as the new axis domains.
 *
 * Kept out of the chart component (like useRRGFilters) so the chart stays a
 * pure visualization and the click-vs-drag contract is testable in isolation.
 *
 * @param {{x: number[], y: number[]}} defaultDomain - domains when not zoomed
 * @param {string} resetKey - dataset identity; zoom/drag reset when it changes
 */
export function useDragZoom(defaultDomain, resetKey) {
  const [drag, setDrag] = useState(null);
  const [zoom, setZoom] = useState(null);

  // A zoom into another dataset would be meaningless, so both the committed
  // zoom and any in-progress drag reset when the dataset identity changes.
  useEffect(() => {
    setZoom(null);
    setDrag(null);
  }, [resetKey]);

  const onMouseDown = (e) => {
    if (e?.xValue == null || e?.yValue == null) return;
    setDrag({ x1: e.xValue, y1: e.yValue, x2: e.xValue, y2: e.yValue });
  };

  const onMouseMove = (e) => {
    if (!drag || e?.xValue == null || e?.yValue == null) return;
    setDrag((d) => (d ? { ...d, x2: e.xValue, y2: e.yValue } : d));
  };

  const onMouseUp = () => {
    if (!drag) return;
    const { x1, y1, x2, y2 } = drag;
    setDrag(null);
    if (Math.abs(x2 - x1) < MIN_DRAG || Math.abs(y2 - y1) < MIN_DRAG) return;
    setZoom({
      x: [Math.min(x1, x2), Math.max(x1, x2)],
      y: [Math.min(y1, y2), Math.max(y1, y2)],
    });
  };

  const onMouseLeave = () => setDrag(null);

  return {
    xDomain: zoom?.x ?? defaultDomain.x,
    yDomain: zoom?.y ?? defaultDomain.y,
    drag,
    isZoomed: zoom != null,
    reset: () => setZoom(null),
    mouseHandlers: { onMouseDown, onMouseMove, onMouseUp, onMouseLeave },
  };
}
