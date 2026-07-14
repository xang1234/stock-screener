import { useMemo, useState } from 'react';
import { applyScanFilterDefaults } from '../../features/scan/defaultFilters';
import { filterStaticScanRows } from '../scanClient';
import { annotateExpressionMatches } from '../../features/scan/filterExpressionEvaluator';

export function buildFiltersFromPreset(screen) {
  return applyScanFilterDefaults(screen?.filters ?? {});
}

export function usePresetScreens({
  screens,
  allRows,
  hydrationComplete,
}) {
  const [activeScreenId, setActiveScreenId] = useState(null);

  const matchCounts = useMemo(() => {
    if (!hydrationComplete || !screens?.length) return {};
    return Object.fromEntries(
      screens.map((s) => [
        s.id,
        s.filter_expression
          ? annotateExpressionMatches(allRows, s.filter_expression).length
          : filterStaticScanRows(allRows, buildFiltersFromPreset(s)).length,
      ]),
    );
  }, [allRows, hydrationComplete, screens]);

  return { activeScreenId, setActiveScreenId, matchCounts };
}
