import ScannerTable from '../ScannerTable';
import { SymbolCell, SignedCell } from '../Cells';
import { formatUsdCompact } from '../format';

const columns = [
  { key: 'symbol', label: 'Symbol', render: (r) => <SymbolCell symbol={r.symbol} /> },
  {
    key: 'callPremium',
    label: 'Call $',
    align: 'right',
    render: (r) => <span className="text-emerald-400">{formatUsdCompact(r.callPremium)}</span>,
  },
  {
    key: 'putPremium',
    label: 'Put $',
    align: 'right',
    render: (r) => <span className="text-rose-400">{formatUsdCompact(r.putPremium)}</span>,
  },
  {
    key: 'netPremium',
    label: 'Net $',
    align: 'right',
    render: (r) => <SignedCell value={r.netPremium}>{formatUsdCompact(r.netPremium)}</SignedCell>,
  },
];

export default function NetPremiumInflowTable({ rows = [] }) {
  return (
    <ScannerTable
      title="Net Dollar Premium Inflows"
      subtitle="Call premium traded vs. put premium traded, latest snapshot"
      columns={columns}
      rows={rows}
    />
  );
}
