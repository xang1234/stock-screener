import ScannerTable from '../ScannerTable';
import { SymbolCell, SignedCell } from '../Cells';

const columns = [
  { key: 'symbol', label: 'Symbol', render: (r) => <SymbolCell symbol={r.symbol} /> },
  {
    key: 'skew',
    label: '25Δ Skew',
    align: 'right',
    render: (r) => <SignedCell value={-r.skew}>{r.skew.toFixed(3)}</SignedCell>,
  },
];

export default function ExtremeSkewTable({ rows = [] }) {
  return (
    <ScannerTable
      title="Extreme Skew"
      subtitle="Most negative = strongest call skew (call IV over put IV)"
      columns={columns}
      rows={rows}
    />
  );
}
