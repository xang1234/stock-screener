const normalizeOptionArray = (value) => (Array.isArray(value) ? value : []);

export const normalizeScanFilterOptions = (filterOptions = {}) => {
  const ibdIndustries = normalizeOptionArray(
    filterOptions.ibdIndustries ?? filterOptions.ibd_industries
  );
  const gicsSectors = normalizeOptionArray(
    filterOptions.gicsSectors ?? filterOptions.gics_sectors
  );
  const ratings = normalizeOptionArray(filterOptions.ratings);

  return {
    ibdIndustries,
    gicsSectors,
    ratings,
    optionValues: {
      ibd_industries: ibdIndustries,
      gics_sectors: gicsSectors,
      ratings,
    },
  };
};
