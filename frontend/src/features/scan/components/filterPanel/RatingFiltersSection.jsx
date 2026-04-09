import { Grid } from '@mui/material';
import {
  CompactRangeInput,
  CompactCheckbox,
  FilterSection,
} from '../../../../components/Scan/filters';

function RatingFiltersSection({
  filters,
  updateFilter,
  updateRangeFilter,
  activeCount,
  defaultExpanded,
}) {
  return (
    <FilterSection
      title="Rating / Score"
      category="rating"
      activeCount={activeCount}
      defaultExpanded={defaultExpanded}
    >
      <Grid container spacing={1.5}>
        <Grid item xs={6} sm={4} md={1.2}>
          <CompactRangeInput
            label="Composite"
            minValue={filters.compositeScore?.min}
            maxValue={filters.compositeScore?.max}
            onChange={(range) => updateRangeFilter('compositeScore', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.2}>
          <CompactRangeInput
            label="Minervini"
            minValue={filters.minerviniScore?.min}
            maxValue={filters.minerviniScore?.max}
            onChange={(range) => updateRangeFilter('minerviniScore', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.2}>
          <CompactRangeInput
            label="CANSLIM"
            minValue={filters.canslimScore?.min}
            maxValue={filters.canslimScore?.max}
            onChange={(range) => updateRangeFilter('canslimScore', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.2}>
          <CompactRangeInput
            label="IPO"
            minValue={filters.ipoScore?.min}
            maxValue={filters.ipoScore?.max}
            onChange={(range) => updateRangeFilter('ipoScore', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.2}>
          <CompactRangeInput
            label="Custom"
            minValue={filters.customScore?.min}
            maxValue={filters.customScore?.max}
            onChange={(range) => updateRangeFilter('customScore', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.2}>
          <CompactRangeInput
            label="Vol BT"
            minValue={filters.volBreakthroughScore?.min}
            maxValue={filters.volBreakthroughScore?.max}
            onChange={(range) => updateRangeFilter('volBreakthroughScore', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.2}>
          <CompactRangeInput
            label="SE Score"
            minValue={filters.seSetupScore?.min}
            maxValue={filters.seSetupScore?.max}
            onChange={(range) => updateRangeFilter('seSetupScore', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.2}>
          <CompactRangeInput
            label="Pvt Dist"
            minValue={filters.seDistanceToPivot?.min}
            maxValue={filters.seDistanceToPivot?.max}
            onChange={(range) => updateRangeFilter('seDistanceToPivot', range)}
            step={1}
            suffix="%"
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.2}>
          <CompactRangeInput
            label="Squeeze"
            minValue={filters.seBbSqueeze?.min}
            maxValue={filters.seBbSqueeze?.max}
            onChange={(range) => updateRangeFilter('seBbSqueeze', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.2}>
          <CompactRangeInput
            label="Vol/50d"
            minValue={filters.seVolumeVs50d?.min}
            maxValue={filters.seVolumeVs50d?.max}
            onChange={(range) => updateRangeFilter('seVolumeVs50d', range)}
            step={0.5}
            minLimit={0}
            suffix="x"
          />
        </Grid>
        <Grid item xs={6} sm={3} md={1}>
          <CompactCheckbox
            label="SE Ready"
            value={filters.seSetupReady}
            onChange={(value) => updateFilter('seSetupReady', value)}
          />
        </Grid>
        <Grid item xs={6} sm={3} md={1}>
          <CompactCheckbox
            label="RS Hi"
            value={filters.seRsLineNewHigh}
            onChange={(value) => updateFilter('seRsLineNewHigh', value)}
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.2}>
          <CompactRangeInput
            label="VCP Score"
            minValue={filters.vcpScore?.min}
            maxValue={filters.vcpScore?.max}
            onChange={(range) => updateRangeFilter('vcpScore', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={3} md={1}>
          <CompactCheckbox
            label="VCP"
            value={filters.vcpDetected}
            onChange={(value) => updateFilter('vcpDetected', value)}
          />
        </Grid>
        <Grid item xs={6} sm={3} md={1}>
          <CompactCheckbox
            label="VCP Ready"
            value={filters.vcpReady}
            onChange={(value) => updateFilter('vcpReady', value)}
          />
        </Grid>
        <Grid item xs={6} sm={3} md={1}>
          <CompactCheckbox
            label="Passes"
            value={filters.passesTemplate}
            onChange={(value) => updateFilter('passesTemplate', value)}
          />
        </Grid>
      </Grid>
    </FilterSection>
  );
}

export default RatingFiltersSection;
