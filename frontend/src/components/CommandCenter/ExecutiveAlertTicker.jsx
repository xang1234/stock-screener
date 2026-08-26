const SEVERITY_STYLE = {
  critical: { icon: '🚨', text: 'text-rose-300' },
  warning: { icon: '⚠️', text: 'text-amber-300' },
  info: { icon: 'ℹ️', text: 'text-sky-300' },
};

/**
 * Continuously scrolling alert strip. The alert list is rendered twice
 * back-to-back and the wrapper animates exactly -50% on a loop, so the
 * second copy seamlessly takes over where the first left off -- standard
 * CSS marquee trick, no JS interval needed.
 */
export default function ExecutiveAlertTicker({ alerts }) {
  return (
    <div className="overflow-hidden border-y border-slate-800 bg-black py-2">
      <div className="flex w-max animate-[cc-marquee_45s_linear_infinite] gap-10 hover:[animation-play-state:paused]">
        {[alerts, alerts].map((pass, passIdx) => (
          <div key={passIdx} className="flex shrink-0 gap-10">
            {pass.map((alert, idx) => {
              const style = SEVERITY_STYLE[alert.severity] ?? SEVERITY_STYLE.info;
              return (
                <span
                  key={`${passIdx}-${alert.id}-${idx}`}
                  className={`flex items-center gap-2 whitespace-nowrap font-mono text-xs ${style.text}`}
                >
                  <span>{style.icon}</span>
                  {alert.text}
                </span>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
