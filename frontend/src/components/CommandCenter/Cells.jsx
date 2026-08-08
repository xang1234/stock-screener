/** Small shared cell renderers so every table in the grid reads consistently. */

export function SymbolCell({ symbol }) {
  return <span className="font-bold text-slate-100">{symbol}</span>;
}

/** Colors green when positive, red when negative -- the semantic convention
 * used throughout the Command Center (green = bullish flow / long gamma,
 * red = bearish flow / short gamma). */
export function SignedCell({ children, value }) {
  const tone = value > 0 ? 'text-emerald-400' : value < 0 ? 'text-rose-400' : 'text-slate-400';
  return <span className={tone}>{children}</span>;
}

export function RegimeDot({ regime }) {
  const isLong = regime === 'long_gamma';
  return (
    <span
      className={`inline-block h-1.5 w-1.5 rounded-full ${isLong ? 'bg-emerald-400' : 'bg-rose-400'}`}
      title={isLong ? 'Long Gamma' : 'Short Gamma'}
    />
  );
}

export function TypeBadge({ type }) {
  const isCall = type === 'call';
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${
        isCall ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'
      }`}
    >
      {type}
    </span>
  );
}
