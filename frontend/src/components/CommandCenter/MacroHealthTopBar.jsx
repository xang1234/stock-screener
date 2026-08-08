import { formatPrice } from './format';

/**
 * Sticky macro-health header: broad-market gamma regime and SPY/QQQ
 * structural levels. This is the "is the tape calm or dangerous" glance --
 * everything below it is single-name detail.
 *
 * VIX Term Structure is intentionally not shown here: this app has no VIX
 * futures data source (only a single daily spot value elsewhere in the
 * codebase), so there is nothing real to report. Sourcing that is a
 * separate follow-up task, not stubbed with fake data in the meantime.
 *
 * There is also no genuine $SPX gamma reading -- SPX index options aren't
 * tracked, so the backend reports SPY's own regime as a labeled proxy
 * (macro.spxProxy.label already says "(SPY proxy)") rather than a
 * fabricated aggregate percentage.
 */
export default function MacroHealthTopBar({ macro }) {
  const spxProxy = macro?.spxProxy;
  const indices = macro?.indices ?? [];
  const spxLong = spxProxy?.regime === 'long_gamma';
  const spxKnown = spxProxy?.regime != null;

  return (
    <div className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950/95 backdrop-blur supports-[backdrop-filter]:bg-slate-950/80">
      <div className="mx-auto flex max-w-[1800px] flex-wrap items-center gap-x-8 gap-y-2 px-4 py-2.5 text-xs">
        {/* SPX (SPY proxy) gamma regime */}
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] uppercase tracking-wider text-slate-500">
            {spxProxy?.label ?? '$SPX (SPY proxy)'}
          </span>
          {spxKnown ? (
            <span
              className={`inline-flex items-center gap-1.5 rounded px-2 py-0.5 font-mono font-semibold ${
                spxLong ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'
              }`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${spxLong ? 'bg-emerald-400' : 'bg-rose-400'}`} />
              {spxLong ? 'Long Gamma' : 'Short Gamma'}
            </span>
          ) : (
            <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-slate-500">No data</span>
          )}
          <span className="font-mono text-slate-500">
            Flip {formatPrice(spxProxy?.flipLevel)} &middot; Spot {formatPrice(spxProxy?.spot)}
          </span>
        </div>

        <div className="h-4 w-px bg-slate-800" />

        {/* SPY / QQQ structural levels */}
        <div className="flex flex-wrap items-center gap-x-6 gap-y-1.5">
          {indices.map((idx) => (
            <div key={idx.symbol} className="flex items-center gap-2 font-mono">
              <span className="font-bold text-slate-200">{idx.symbol}</span>
              <span className="text-slate-400">{formatPrice(idx.spot)}</span>
              <LevelChip label="Flip" value={idx.flipLevel} tone="neutral" />
              <LevelChip label="Call Wall" value={idx.callWall} tone="bullish" />
              <LevelChip label="Put Wall" value={idx.putWall} tone="bearish" />
            </div>
          ))}
          {indices.length === 0 && <span className="text-slate-600">No index data yet</span>}
        </div>
      </div>
    </div>
  );
}

function LevelChip({ label, value, tone }) {
  const toneClass =
    tone === 'bullish'
      ? 'text-emerald-400'
      : tone === 'bearish'
        ? 'text-rose-400'
        : 'text-slate-300';
  return (
    <span className="text-[11px]">
      <span className="text-slate-600">{label} </span>
      <span className={toneClass}>{formatPrice(value)}</span>
    </span>
  );
}
