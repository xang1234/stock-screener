import ScannerTable from '../ScannerTable';
import { SymbolCell, TypeBadge } from '../Cells';
import { formatPrice } from '../format';

const columns = [
  { key: 'symbol', label: 'Symbol', render: (r) => <SymbolCell symbol={r.symbol} /> },
  { key: 'strike', label: 'Strike', align: 'right', render: (r) => formatPrice(r.strike) },
  { key: 'type', label: 'Type', render: (r) => <TypeBadge type={r.type} /> },
  { key: 'volume', label: 'Volume', align: 'right', render: (r) => (r.volume ?? 0).toLocaleString() },
  { key: 'openInterest', label: 'OI', align: 'right', render: (r) => (r.openInterest ?? 0).toLocaleString() },
  {
    key: 'ratio',
    label: 'Vol/OI',
    align: 'right',
    render: (r) => <span className="font-bold text-amber-400">{r.ratio.toFixed(1)}x</span>,
  },
];

export default function UnusualVolumeOiTable({ rows = [] }) {
  return (
    <ScannerTable
      title="Unusual Volume / OI (OTM)"
      subtitle="Contracts trading well above existing open interest, latest snapshot"
      columns={columns}
      rows={rows}
    />
  );
}
