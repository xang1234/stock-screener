export const GROUP_RS_FIELDS = Object.freeze([
  Object.freeze({ field: 'avg_rs_rating', label: 'RS', staticLabel: 'Avg RS' }),
  Object.freeze({ field: 'avg_rs_rating_1m', label: '1M RS', staticLabel: '1M RS' }),
  Object.freeze({ field: 'avg_rs_rating_3m', label: '3M RS', staticLabel: '3M RS' }),
]);

export const formatGroupRs = (value) => (
  Number.isFinite(value) ? value.toFixed(1) : '-'
);
