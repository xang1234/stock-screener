import { describe, it, expect } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useDragZoom } from './useDragZoom';

const DEFAULT = { x: [88, 112], y: [88, 112] };

const renderZoom = (resetKey = 'groups|US') =>
  renderHook(({ key }) => useDragZoom(DEFAULT, key), { initialProps: { key: resetKey } });

const dragRect = (result, { x1, y1, x2, y2 }) => {
  act(() => result.current.mouseHandlers.onMouseDown({ xValue: x1, yValue: y1 }));
  act(() => result.current.mouseHandlers.onMouseMove({ xValue: x2, yValue: y2 }));
  act(() => result.current.mouseHandlers.onMouseUp());
};

describe('useDragZoom', () => {
  it('passes the default domains through when not zoomed', () => {
    const { result } = renderZoom();
    expect(result.current.xDomain).toEqual([88, 112]);
    expect(result.current.yDomain).toEqual([88, 112]);
    expect(result.current.isZoomed).toBe(false);
    expect(result.current.drag).toBeNull();
  });

  it('tracks the in-progress selection rectangle during a drag', () => {
    const { result } = renderZoom();
    act(() => result.current.mouseHandlers.onMouseDown({ xValue: 100, yValue: 100 }));
    act(() => result.current.mouseHandlers.onMouseMove({ xValue: 104, yValue: 96 }));
    expect(result.current.drag).toEqual({ x1: 100, y1: 100, x2: 104, y2: 96 });
  });

  it('commits a drag as ordered min/max domains', () => {
    const { result } = renderZoom();
    dragRect(result, { x1: 104, y1: 96, x2: 100, y2: 103 }); // dragged "backwards"
    expect(result.current.xDomain).toEqual([100, 104]);
    expect(result.current.yDomain).toEqual([96, 103]);
    expect(result.current.isZoomed).toBe(true);
    expect(result.current.drag).toBeNull();
  });

  it('treats a drag below the threshold as a click (no zoom)', () => {
    const { result } = renderZoom();
    dragRect(result, { x1: 100, y1: 100, x2: 100.2, y2: 100.2 });
    expect(result.current.isZoomed).toBe(false);
    expect(result.current.xDomain).toEqual([88, 112]);
  });

  it('ignores mouse events without axis values (outside the plot)', () => {
    const { result } = renderZoom();
    act(() => result.current.mouseHandlers.onMouseDown({ xValue: null, yValue: null }));
    expect(result.current.drag).toBeNull();
    act(() => result.current.mouseHandlers.onMouseUp());
    expect(result.current.isZoomed).toBe(false);
  });

  it('cancels an in-progress drag when the mouse leaves the chart', () => {
    const { result } = renderZoom();
    act(() => result.current.mouseHandlers.onMouseDown({ xValue: 100, yValue: 100 }));
    act(() => result.current.mouseHandlers.onMouseLeave());
    expect(result.current.drag).toBeNull();
    act(() => result.current.mouseHandlers.onMouseUp());
    expect(result.current.isZoomed).toBe(false);
  });

  it('reset() restores the default domains', () => {
    const { result } = renderZoom();
    dragRect(result, { x1: 100, y1: 96, x2: 104, y2: 103 });
    expect(result.current.isZoomed).toBe(true);
    act(() => result.current.reset());
    expect(result.current.isZoomed).toBe(false);
    expect(result.current.xDomain).toEqual([88, 112]);
  });

  it('resets the zoom when the dataset identity (resetKey) changes', () => {
    const { result, rerender } = renderZoom('groups|US');
    dragRect(result, { x1: 100, y1: 96, x2: 104, y2: 103 });
    expect(result.current.isZoomed).toBe(true);
    rerender({ key: 'sectors|US' });
    expect(result.current.isZoomed).toBe(false);
    expect(result.current.xDomain).toEqual([88, 112]);
  });
});
