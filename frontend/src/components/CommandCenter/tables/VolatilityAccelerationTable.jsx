import ScannerTable from '../ScannerTable';
import { SymbolCell, SignedCell } from '../Cells';
import { formatPrice, formatUsdCompact, formatPct } from '../format';

const columns = [
  { key: 'symbol', label: 'Symbol', render: (r) => <SymbolCell symbol={r.symbol} /> },
  { key: 'price', label: 'Price', align: 'right', render: (r) => formatPrice(r.price) },
  {
    key: 'totalGex',
    label: 'Total GEX',
    align: 'right',
    render: (r) => <SignedCell value={r.totalGex}>{formatUsdCompact(r.totalGex)}</SignedCell>,
  },
  {
    key: 'distanceToFlipPct',
    label: 'vs. Flip',
    align: 'right',
    render: (r) => <SignedCell value={r.distanceToFlipPct}>{formatPct(r.distanceToFlipPct)}</SignedCell>,
  },
];

export default function VolatilityAccelerationTable({ rows = [] }) {
  return (
    <ScannerTable
      title="Top Volatility Acceleration"
      subtitle="Most negative Total GEX -- deepest short-gamma regimes"
      columns={columns}
      rows={rows}
    />
  );
}
