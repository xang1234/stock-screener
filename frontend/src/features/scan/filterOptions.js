const normalizeOptionArray = (value) => (Array.isArray(value) ? value : []);

export const normalizeScanFilterOptions = (filterOptions = {}) => ({
  ibdIndustries: normalizeOptionArray(
    filterOptions.ibdIndustries ?? filterOptions.ibd_industries
  ),
  gicsSectors: normalizeOptionArray(
    filterOptions.gicsSectors ?? filterOptions.gics_sectors
  ),
  ratings: normalizeOptionArray(filterOptions.ratings),
});
