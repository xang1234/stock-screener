import { Grid } from '@mui/material';
import {
  CompactRangeInput,
  CompactSelect,
  CompactCheckbox,
  FilterSection,
} from '../../../../components/Scan/filters';
import { STAGE_OPTIONS } from './constants';

function TechnicalFiltersSection({
  filters,
  updateFilter,
  updateRangeFilter,
  activeCount,
  defaultExpanded,
}) {
  return (
    <FilterSection
      title="Technical"
      category="technical"
      activeCount={activeCount}
      defaultExpanded={defaultExpanded}
    >
      <Grid container spacing={1.5}>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="ADR %"
            minValue={filters.adrPercent?.min}
            maxValue={filters.adrPercent?.max}
            onChange={(range) => updateRangeFilter('adrPercent', range)}
            step={0.5}
            minLimit={0}
            suffix="%"
          />
        </Grid>
        <Grid item xs={6} sm={3} md={1.5}>
          <CompactSelect
            label="Stage"
            value={filters.stage}
            options={STAGE_OPTIONS}
            onChange={(value) => updateFilter('stage', value)}
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="RS Rating"
            minValue={filters.rsRating?.min}
            maxValue={filters.rsRating?.max}
            onChange={(range) => updateRangeFilter('rsRating', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="RS 1M"
            minValue={filters.rs1m?.min}
            maxValue={filters.rs1m?.max}
            onChange={(range) => updateRangeFilter('rs1m', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="RS 3M"
            minValue={filters.rs3m?.min}
            maxValue={filters.rs3m?.max}
            onChange={(range) => updateRangeFilter('rs3m', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="RS 12M"
            minValue={filters.rs12m?.min}
            maxValue={filters.rs12m?.max}
            onChange={(range) => updateRangeFilter('rs12m', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="Beta"
            minValue={filters.beta?.min}
            maxValue={filters.beta?.max}
            onChange={(range) => updateRangeFilter('beta', range)}
            step={0.1}
            minLimit={0}
            maxLimit={5}
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="β-adj RS"
            minValue={filters.betaAdjRs?.min}
            maxValue={filters.betaAdjRs?.max}
            onChange={(range) => updateRangeFilter('betaAdjRs', range)}
            step={5}
            minLimit={0}
            maxLimit={100}
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={3} md={1}>
          <CompactCheckbox
            label="MA Align"
            value={filters.maAlignment}
            onChange={(value) => updateFilter('maAlignment', value)}
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="1D Chg %"
            minValue={filters.perfDay?.min}
            maxValue={filters.perfDay?.max}
            onChange={(range) => updateRangeFilter('perfDay', range)}
            step={1}
            suffix="%"
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="1W Chg %"
            minValue={filters.perfWeek?.min}
            maxValue={filters.perfWeek?.max}
            onChange={(range) => updateRangeFilter('perfWeek', range)}
            step={1}
            suffix="%"
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="1M Chg %"
            minValue={filters.perfMonth?.min}
            maxValue={filters.perfMonth?.max}
            onChange={(range) => updateRangeFilter('perfMonth', range)}
            step={1}
            suffix="%"
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="3M Chg %"
            minValue={filters.perf3m?.min}
            maxValue={filters.perf3m?.max}
            onChange={(range) => updateRangeFilter('perf3m', range)}
            step={5}
            suffix="%"
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="6M Chg %"
            minValue={filters.perf6m?.min}
            maxValue={filters.perf6m?.max}
            onChange={(range) => updateRangeFilter('perf6m', range)}
            step={10}
            suffix="%"
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="Gap %"
            minValue={filters.gapPercent?.min}
            maxValue={filters.gapPercent?.max}
            onChange={(range) => updateRangeFilter('gapPercent', range)}
            step={1}
            suffix="%"
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="Vol Surge"
            minValue={filters.volumeSurge?.min}
            maxValue={filters.volumeSurge?.max}
            onChange={(range) => updateRangeFilter('volumeSurge', range)}
            step={0.5}
            minLimit={0}
            suffix="x"
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="% vs EMA10"
            minValue={filters.ema10Distance?.min}
            maxValue={filters.ema10Distance?.max}
            onChange={(range) => updateRangeFilter('ema10Distance', range)}
            step={1}
            suffix="%"
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="% vs EMA20"
            minValue={filters.ema20Distance?.min}
            maxValue={filters.ema20Distance?.max}
            onChange={(range) => updateRangeFilter('ema20Distance', range)}
            step={1}
            suffix="%"
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="% vs EMA50"
            minValue={filters.ema50Distance?.min}
            maxValue={filters.ema50Distance?.max}
            onChange={(range) => updateRangeFilter('ema50Distance', range)}
            step={1}
            suffix="%"
            minOnly
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="% from 52W Hi"
            minValue={filters.week52HighDistance?.min}
            maxValue={filters.week52HighDistance?.max}
            onChange={(range) => updateRangeFilter('week52HighDistance', range)}
            step={1}
            suffix="%"
          />
        </Grid>
        <Grid item xs={6} sm={4} md={1.5}>
          <CompactRangeInput
            label="% from 52W Lo"
            minValue={filters.week52LowDistance?.min}
            maxValue={filters.week52LowDistance?.max}
            onChange={(range) => updateRangeFilter('week52LowDistance', range)}
            step={1}
            suffix="%"
          />
        </Grid>
      </Grid>
    </FilterSection>
  );
}

export default TechnicalFiltersSection;
