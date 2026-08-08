import PillarColumn from './PillarColumn';
import VolatilityAccelerationTable from './tables/VolatilityAccelerationTable';
import GammaFlipProximityTable from './tables/GammaFlipProximityTable';
import RichVrpTable from './tables/RichVrpTable';
import ExtremeSkewTable from './tables/ExtremeSkewTable';
import NetPremiumInflowTable from './tables/NetPremiumInflowTable';
import UnusualVolumeOiTable from './tables/UnusualVolumeOiTable';

/** The "Top 10" Command Grid: three pillars, two scanner tables each.
 * `data` is the GET /v1/options-command-center/ response body -- each
 * ranking list may legitimately be shorter than 10 rows, or empty, when
 * the persisted-snapshot universe doesn't have enough coverage yet (see
 * the backend endpoint's docstring). ScannerTable already renders a
 * "No matches right now" empty state, so no extra handling needed here. */
export default function CommandGrid({ data }) {
  return (
    <div className="mx-auto grid max-w-[1800px] grid-cols-1 gap-6 px-4 py-6 lg:grid-cols-3">
      <PillarColumn title="Structural & Dealer Risk" icon="🎯">
        <VolatilityAccelerationTable rows={data?.volatilityAcceleration} />
        <GammaFlipProximityTable rows={data?.gammaFlipProximity?.rows} widened={data?.gammaFlipProximity?.widened} />
      </PillarColumn>

      <PillarColumn title="Volatility Mispricing" icon="📈">
        <RichVrpTable rows={data?.richVrp} cheapRows={data?.cheapVrp} />
        <ExtremeSkewTable rows={data?.extremeSkew} />
      </PillarColumn>

      <PillarColumn title="Smart Money Flow" icon="💰">
        <NetPremiumInflowTable rows={data?.netPremiumInflows} />
        <UnusualVolumeOiTable rows={data?.unusualVolumeOi} />
      </PillarColumn>
    </div>
  );
}
