import { useState } from 'react';
import ScannerTable from '../ScannerTable';
import { SymbolCell } from '../Cells';
import { formatIv, formatPct } from '../format';

const columns = [
  { key: 'symbol', label: 'Symbol', render: (r) => <SymbolCell symbol={r.symbol} /> },
  { key: 'iv', label: 'ATM IV', align: 'right', render: (r) => formatIv(r.iv) },
  { key: 'hv', label: '20D HV', align: 'right', render: (r) => formatIv(r.hv) },
  {
    key: 'vrpPct',
    label: 'VRP',
    align: 'right',
    render: (r) => (
      <span className={r.vrpPct >= 0 ? 'text-amber-400' : 'text-cyan-400'}>{formatPct(r.vrpPct, 0)}</span>
    ),
  },
];

/** Toggles between "Top Rich VRP" (IV > HV, premium-selling candidates) and
 * "Top Cheap VRP" (IV < HV, premium-buying candidates) -- backed by two
 * strictly sign-separated rankings from the API (richVrp / cheapVrp), never
 * a client-side split of one list. */
export default function RichVrpTable({ rows = [], cheapRows = [] }) {
  const [mode, setMode] = useState('rich');
  const isRich = mode === 'rich';

  return (
    <ScannerTable
      title={
        <span className="flex items-center gap-2">
          <span>{isRich ? 'Top Rich VRP' : 'Top Cheap VRP'}</span>
          <span className="flex overflow-hidden rounded border border-slate-700 text-[9px] font-semibold uppercase">
            <button
              type="button"
              onClick={() => setMode('rich')}
              className={`px-1.5 py-0.5 ${isRich ? 'bg-amber-500/20 text-amber-300' : 'text-slate-500 hover:text-slate-300'}`}
            >
              Rich
            </button>
            <button
              type="button"
              onClick={() => setMode('cheap')}
              className={`px-1.5 py-0.5 ${!isRich ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-500 hover:text-slate-300'}`}
            >
              Cheap
            </button>
          </span>
        </span>
      }
      subtitle={
        isRich
          ? 'IV running richest vs. 20D realized -- premium selling candidates'
          : 'IV running cheapest vs. 20D realized -- premium buying candidates'
      }
      columns={columns}
      rows={isRich ? rows : cheapRows}
    />
  );
}
