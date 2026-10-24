// FastAPI sends `detail` as a string for HTTPException and as a list of
// objects for request validation errors; React cannot render the list.
export const formatErrorDetail = (detail) => {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map(formatErrorDetail).filter(Boolean).join('; ');
  }
  if (detail && typeof detail === 'object') {
    return typeof detail.msg === 'string' ? detail.msg : JSON.stringify(detail);
  }
  return detail == null ? '' : String(detail);
};

export const getErrorMessage = (error, fallback) => (
  formatErrorDetail(error?.response?.data?.detail)
  || error?.message
  || fallback
);
