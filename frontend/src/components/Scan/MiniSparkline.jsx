import { memo, useMemo } from 'react';
import { Box, Tooltip, Typography } from '@mui/material';

const PRICE_VIEWBOX_WIDTH = 100;
const RS_VIEWBOX_WIDTH = 60;

const normalizeValues = (data) => (
  Array.isArray(data)
    ? data.map(Number).filter((value) => Number.isFinite(value))
    : []
);

const sparklineDomain = (values, paddingRatio = 0.05) => {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 0.01;
  const padding = range * paddingRatio;
  return {
    min: min - padding,
    max: max + padding,
  };
};

const formatChangeText = (value) => {
  if (value == null) return null;
  return `${value >= 0 ? '+' : ''}${Number(value).toFixed(1)}%`;
};

function SparklinePlaceholder({ width, height }) {
  return (
    <Box sx={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'text.disabled', fontSize: 10 }}>
      -
    </Box>
  );
}

const sameSparklineProps = (prevProps, nextProps) => {
  const stableScalarProps = (
    prevProps.trend === nextProps.trend
    && prevProps.change1d === nextProps.change1d
    && prevProps.width === nextProps.width
    && prevProps.height === nextProps.height
    && prevProps.sparklineWidth === nextProps.sparklineWidth
    && prevProps.industry === nextProps.industry
    && prevProps.showChange === nextProps.showChange
  );

  if (!stableScalarProps) return false;
  if (prevProps.data === nextProps.data) return true;
  if (!prevProps.data || !nextProps.data) return false;
  if (prevProps.data.length !== nextProps.data.length) return false;

  return prevProps.data.every((value, index) => value === nextProps.data[index]);
};

function MiniPriceSparkline({
  data,
  trend,
  change1d,
  industry,
  width = 100,
  height = 28,
  showChange = true,
  sparklineWidth = 60,
}) {
  const { areaPath, color, fillColor, linePath, tooltipText } = useMemo(() => {
    const values = normalizeValues(data);
    if (values.length === 0) {
      return { areaPath: '', color: '#9e9e9e', fillColor: '#9e9e9e', linePath: '', tooltipText: 'No data' };
    }

    const { min, max } = sparklineDomain(values);
    const plotTop = 2;
    const plotBottom = height - 2;
    const plotHeight = Math.max(1, plotBottom - plotTop);
    const scaleY = (value) => plotTop + (1 - ((value - min) / (max - min))) * plotHeight;
    const scaleX = (index) => (
      values.length === 1 ? PRICE_VIEWBOX_WIDTH / 2 : (index / (values.length - 1)) * PRICE_VIEWBOX_WIDTH
    );
    const points = values.map((value, index) => `${scaleX(index).toFixed(2)},${scaleY(value).toFixed(2)}`);
    const pointPath = points.join(' L ');
    const isUp = trend === 1;
    const overallChange = values[0] !== 0 ? ((values[values.length - 1] - values[0]) / values[0]) * 100 : null;
    const trendText = trend === 1 ? 'Up' : trend === -1 ? 'Down' : 'Flat';
    const tooltipParts = [];
    if (industry) tooltipParts.push(industry);
    if (overallChange != null) {
      tooltipParts.push(`30d: ${overallChange >= 0 ? '+' : ''}${overallChange.toFixed(1)}% (${trendText})`);
    }

    return {
      areaPath: `M ${pointPath} L ${PRICE_VIEWBOX_WIDTH},${plotBottom} L 0,${plotBottom} Z`,
      color: isUp ? '#4caf50' : '#f44336',
      fillColor: isUp ? 'rgba(76, 175, 80, 0.3)' : 'rgba(244, 67, 54, 0.3)',
      linePath: `M ${pointPath}`,
      tooltipText: tooltipParts.join(' | ') || 'No data',
    };
  }, [data, height, industry, trend]);

  const changeText = useMemo(() => formatChangeText(change1d), [change1d]);
  const changeColor = change1d == null ? 'text.secondary' : change1d >= 0 ? 'success.main' : 'error.main';

  if (!linePath) {
    return <SparklinePlaceholder width={width} height={height} />;
  }

  return (
    <Tooltip title={tooltipText} arrow placement="top">
      <Box sx={{ width, height, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 0.5 }}>
        <Box sx={{ width: showChange ? sparklineWidth : '100%', height: '100%', flex: showChange ? '1 1 auto' : 1, minWidth: 0 }}>
          <svg
            aria-hidden="true"
            focusable="false"
            preserveAspectRatio="none"
            viewBox={`0 0 ${PRICE_VIEWBOX_WIDTH} ${height}`}
            width="100%"
            height="100%"
          >
            <path d={areaPath} fill={fillColor} />
            <path d={linePath} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
          </svg>
        </Box>
        {showChange && changeText && (
          <Typography sx={{ fontSize: 10, fontWeight: 600, fontFamily: 'monospace', color: changeColor, whiteSpace: 'nowrap', flexShrink: 0 }}>
            {changeText}
          </Typography>
        )}
      </Box>
    </Tooltip>
  );
}

function MiniRSSparkline({ data, trend, width = 60, height = 20 }) {
  const { bars, tooltipText } = useMemo(() => {
    const values = normalizeValues(data);
    if (values.length === 0) {
      return { bars: [], tooltipText: 'No RS data' };
    }

    const { min, max } = sparklineDomain(values, 0.1);
    const maxValue = Math.max(...values);
    const minValue = Math.min(...values);
    const maxIndex = values.indexOf(maxValue);
    const minIndex = values.indexOf(minValue);
    const gap = 1;
    const barWidth = Math.max(0.8, Math.min(3, (RS_VIEWBOX_WIDTH - gap * (values.length - 1)) / values.length));
    const plotTop = 1;
    const plotBottom = height - 1;
    const plotHeight = Math.max(1, plotBottom - plotTop);
    const scaleY = (value) => plotTop + (1 - ((value - min) / (max - min))) * plotHeight;
    const firstVal = values[0] || 1;
    const lastVal = values[values.length - 1] || 1;
    const change = ((lastVal - firstVal) / firstVal) * 100;
    const trendText = trend === 1 ? 'Improving' : trend === -1 ? 'Declining' : 'Flat';

    return {
      bars: values.map((value, index) => {
        const y = scaleY(value);
        return {
          x: values.length === 1 ? (RS_VIEWBOX_WIDTH - barWidth) / 2 : index * (barWidth + gap),
          y,
          width: barWidth,
          height: Math.max(1, plotBottom - y),
          fill: index === maxIndex ? '#1F97F4' : index === minIndex ? '#f44336' : '#90EE90',
        };
      }),
      tooltipText: `RS ${trendText} (${change >= 0 ? '+' : ''}${change.toFixed(1)}% over 30d)`,
    };
  }, [data, height, trend]);

  if (bars.length === 0) {
    return <SparklinePlaceholder width={width} height={height} />;
  }

  return (
    <Tooltip title={tooltipText} arrow placement="top">
      <Box sx={{ width, height, cursor: 'pointer' }}>
        <svg
          aria-hidden="true"
          focusable="false"
          preserveAspectRatio="none"
          viewBox={`0 0 ${RS_VIEWBOX_WIDTH} ${height}`}
          width="100%"
          height="100%"
        >
          {bars.map((bar, index) => (
            <rect
              key={index}
              x={bar.x.toFixed(2)}
              y={bar.y.toFixed(2)}
              width={bar.width.toFixed(2)}
              height={bar.height.toFixed(2)}
              rx="1"
              fill={bar.fill}
            />
          ))}
        </svg>
      </Box>
    </Tooltip>
  );
}

const MemoizedMiniPriceSparkline = memo(MiniPriceSparkline, sameSparklineProps);
const MemoizedMiniRSSparkline = memo(MiniRSSparkline, sameSparklineProps);

export {
  MemoizedMiniPriceSparkline as MiniPriceSparkline,
  MemoizedMiniRSSparkline as MiniRSSparkline,
};
