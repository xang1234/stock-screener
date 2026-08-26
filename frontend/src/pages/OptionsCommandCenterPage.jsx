// Tailwind is scoped to this page only -- see src/styles/commandCenter.css
// for why (preflight is intentionally excluded so it can't affect the
// MUI-based pages elsewhere in the app).
import '../styles/commandCenter.css';

import { useQuery } from '@tanstack/react-query';

import MacroHealthTopBar from '../components/CommandCenter/MacroHealthTopBar';
import CommandGrid from '../components/CommandCenter/CommandGrid';
import ExecutiveAlertTicker from '../components/CommandCenter/ExecutiveAlertTicker';
import { getOptionsCommandCenterSnapshot } from '../api/optionsCommandCenter';

/**
 * Institutional Options Analytics Command Center -- the trading-floor
 * landing page. Wired to real data via GET /v1/options-command-center/,
 * which reads persisted options snapshots (see the backend endpoint's
 * docstring for coverage caveats -- ranking lists can legitimately be
 * short or empty while the snapshot table is still building up history).
 */
export default function OptionsCommandCenterPage() {
  const { data, isLoading, isError, error, dataUpdatedAt } = useQuery({
    queryKey: ['optionsCommandCenter'],
    queryFn: getOptionsCommandCenterSnapshot,
    staleTime: 30_000,
    refetchInterval: 60_000,
  });

  return (
    <div className="min-h-screen bg-slate-950 text-slate-300">
      <MacroHealthTopBar macro={data?.macro} />
      <ExecutiveAlertTicker alerts={data?.alerts ?? []} />

      {isLoading && (
        <div className="flex justify-center py-16 text-sm text-slate-500">Loading live options data&hellip;</div>
      )}

      {isError && (
        <div className="mx-auto max-w-[1800px] px-4 py-6">
          <div className="rounded-md border border-rose-900 bg-rose-950/40 px-4 py-3 text-sm text-rose-300">
            Failed to load Command Center data{error?.message ? `: ${error.message}` : '.'}
          </div>
        </div>
      )}

      {data && (
        <>
          <div className="mx-auto max-w-[1800px] px-4 pt-3 text-[11px] text-slate-600">
            {data.coverage?.activeUniverseSymbolsWithData ?? 0} active symbol(s) with persisted options data
            {dataUpdatedAt ? ` · refreshed ${new Date(dataUpdatedAt).toLocaleTimeString()}` : ''}
          </div>
          <CommandGrid data={data} />
        </>
      )}
    </div>
  );
}
