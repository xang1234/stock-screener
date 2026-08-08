import { useNavigate } from 'react-router-dom';

/**
 * Generic dense data table used by every scanner in the Command Grid.
 *
 * `columns`: [{ key, label, align?: 'left'|'right', render(row) => node }]
 * `rows`: array of plain objects, each must have a `symbol` field -- rows
 * are clickable and route to that symbol's Options Analytics dashboard
 * (/options-analytics?ticker=SYMBOL), which reads the ticker query param on
 * mount to pre-select it.
 */
export default function ScannerTable({ title, subtitle, columns, rows }) {
  const navigate = useNavigate();

  return (
    <div className="overflow-hidden rounded-md border border-slate-800 bg-slate-900/60">
      <div className="border-b border-slate-800 px-3 py-2">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-300">{title}</h3>
        {subtitle && <p className="mt-0.5 text-[10px] text-slate-500">{subtitle}</p>}
      </div>
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr className="border-b border-slate-800 text-[10px] uppercase tracking-wider text-slate-500">
            {columns.map((col) => (
              <th
                key={col.key}
                className={`px-3 py-1.5 font-medium ${col.align === 'right' ? 'text-right' : 'text-left'}`}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr
              key={row.symbol + idx}
              onClick={() => navigate(`/options-analytics?ticker=${encodeURIComponent(row.symbol)}`)}
              className="cursor-pointer border-b border-slate-800/60 font-mono transition-colors last:border-0 hover:bg-slate-800/70 hover:ring-1 hover:ring-inset hover:ring-sky-500/40"
            >
              {columns.map((col) => (
                <td key={col.key} className={`px-3 py-1.5 ${col.align === 'right' ? 'text-right' : 'text-left'}`}>
                  {col.render ? col.render(row) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="px-3 py-4 text-center text-slate-600">
                No matches right now.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
