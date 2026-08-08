import ScannerTable from '../ScannerTable';
import { SymbolCell, SignedCell } from '../Cells';
import { formatPrice, formatPct } from '../format';

const columns = [
  { key: 'symbol', label: 'Symbol', render: (r) => <SymbolCell symbol={r.symbol} /> },
  { key: 'spot', label: 'Spot', align: 'right', render: (r) => formatPrice(r.spot) },
  { key: 'flipLevel', label: 'Flip Level', align: 'right', render: (r) => formatPrice(r.flipLevel) },
  {
    key: 'distancePct',
    label: 'Distance',
    align: 'right',
    render: (r) => <SignedCell value={r.distancePct}>{formatPct(r.distancePct)}</SignedCell>,
  },
];

export default function GammaFlipProximityTable({ rows = [], widened = false }) {
  return (
    <ScannerTable
      title="Gamma Flip Proximity"
      subtitle={
        widened
          ? 'Nothing within 1.5% right now -- showing the 3 closest tickers instead'
          : 'Within 1.5% of flip -- one print from a regime change'
      }
      columns={columns}
      rows={rows}
    />
  );
}
