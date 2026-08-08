/** One of the three pillars (Structural & Dealer Risk / Volatility
 * Mispricing / Smart Money Flow) -- a titled column stacking its two
 * scanner tables. */
export default function PillarColumn({ title, icon, children }) {
  return (
    <div className="flex flex-col gap-4">
      <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-200">
        <span>{icon}</span>
        {title}
      </h2>
      <div className="flex flex-col gap-4">{children}</div>
    </div>
  );
}
