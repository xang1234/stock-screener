/**
 * TradingView Advanced Chart widget wrapper.
 * Embeds TradingView charts using the widget library.
 */
import { useEffect, useRef, memo, useContext } from 'react';
import { Box } from '@mui/material';
import { ColorModeContext } from '../../contexts/ColorModeContext';

function TradingViewChart({ symbol, interval = 'D', range = '9M', hideSidebar = false }) {
  const containerRef = useRef(null);
  const { mode } = useContext(ColorModeContext);

  useEffect(() => {
    if (!containerRef.current) return;

    // Clear previous widget
    containerRef.current.innerHTML = '';

    // Create container div for TradingView
    const containerId = `tradingview_${symbol.replace(/[^a-zA-Z0-9]/g, '_')}_${Date.now()}`;
    const widgetContainer = document.createElement('div');
    widgetContainer.id = containerId;
    widgetContainer.style.height = '100%';
    widgetContainer.style.width = '100%';
    containerRef.current.appendChild(widgetContainer);

    // Load TradingView script
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/tv.js';
    script.async = true;
    script.onload = () => {
      if (typeof window.TradingView !== 'undefined' && document.getElementById(containerId)) {
        new window.TradingView.widget({
          autosize: true,
          symbol: symbol,
          interval: interval,
          timezone: 'America/New_York',
          theme: mode === 'dark' ? 'dark' : 'light',
          style: '1', // Candlestick
          locale: 'en',
          toolbar_bg: mode === 'dark' ? '#1e1e1e' : '#f1f3f6',
          enable_publishing: false,
          allow_symbol_change: false,
          container_id: containerId,
          range: range,
          hide_side_toolbar: hideSidebar,
          studies: [
            { id: 'MAExp@tv-basicstudies', inputs: { length: 10 }, styles: { plot: { color: '#88E17C' } } },
            { id: 'MAExp@tv-basicstudies', inputs: { length: 20 }, styles: { plot: { color: '#9AF1EC' } } },
            { id: 'MAExp@tv-basicstudies', inputs: { length: 50 }, styles: { plot: { color: '#5EBB4B' } } },
          ],
          overrides: {
            'mainSeriesProperties.candleStyle.upColor': '#2196f3',
            'mainSeriesProperties.candleStyle.downColor': '#E619CD',
            'mainSeriesProperties.candleStyle.borderUpColor': '#2196f3',
            'mainSeriesProperties.candleStyle.borderDownColor': '#E619CD',
            'mainSeriesProperties.candleStyle.wickUpColor': '#2196f3',
            'mainSeriesProperties.candleStyle.wickDownColor': '#E619CD',
            'mainSeriesProperties.priceAxisProperties.isLog': true,
          },
        });
      }
    };

    document.head.appendChild(script);

    return () => {
      // Cleanup
      if (script.parentNode) {
        script.parentNode.removeChild(script);
      }
    };
  }, [symbol, interval, range, mode]);

  return (
    <Box
      ref={containerRef}
      sx={{
        height: '100%',
        width: '100%',
        '& iframe': {
          border: 'none',
        },
      }}
    />
  );
}

// Memoize to prevent unnecessary re-renders
export default memo(TradingViewChart);
