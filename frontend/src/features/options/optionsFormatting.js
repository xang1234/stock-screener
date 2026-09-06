export const formatOptionsMetric = (metric, { percent = false } = {}) => {
  if (!metric?.available || metric.value == null) return null;
  if (percent) return `${(metric.value * 100).toFixed(1)}%`;
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(metric.value);
};
