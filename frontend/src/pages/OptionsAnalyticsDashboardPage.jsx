import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import {
  Container,
  Typography,
  Box,
  TextField,
  CircularProgress,
  Paper,
  Grid,
  Card,
  CardContent,
  Alert,
  Chip,
  Autocomplete,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import PrintIcon from '@mui/icons-material/Print';
import apiClient from '../api/client';
import SummaryCards from '../components/OptionsMetrics/SummaryCards';
import StrikeExposureChart from '../components/OptionsMetrics/StrikeExposureChart';
import IVSmileChart from '../components/OptionsMetrics/IVSmileChart';
import IVTermStructureChart from '../components/OptionsMetrics/IVTermStructureChart';
import UnusualVolumeTable from '../components/OptionsMetrics/UnusualVolumeTable';
import MetricHistoryChart from '../components/OptionsMetrics/MetricHistoryChart';
import ExpirationSelector from '../components/OptionsMetrics/ExpirationSelector';
import LastUpdated from '../components/OptionsMetrics/LastUpdated';

// Human-readable label + one-line rationale per evaluateMarketFactors() key,
// used only by the factor-breakdown table -- keeps that table's copy in one
// place instead of scattered across the sentence strings in evaluateMarketFactors.
const MARKET_FACTOR_LABELS = {
  gex: { label: 'Total GEX', why: 'Dominant dealer-hedging-flow signal' },
  skew: { label: '25Δ Volatility Skew', why: 'Put vs call IV demand' },
  maxPain: { label: 'Max Pain Pull', why: 'Weak, contested on its own' },
  premiumPcr: { label: 'Premium Put/Call Ratio', why: 'Dollar-weighted flow' },
  openInterest: { label: 'Open Interest Skew', why: 'Call OI vs put OI' },
  callWallBreak: { label: 'Call Wall Break', why: 'Resistance no longer holding' },
  putWallBreak: { label: 'Put Wall Break', why: 'Support no longer holding' },
};

/**
 * Divider + title + one-line "Goal" strapline, reused for each of the
 * dashboard's five metric groupings so a reader always knows what question
 * the cards below it are trying to answer, not just what the numbers are.
 */
function SectionHeader({ icon, title, goal }) {
  return (
    <Box sx={{ mt: 5, mb: 2, pt: 2, borderTop: '2px solid', borderColor: 'divider' }}>
      <Typography variant="h5" sx={{ fontWeight: 600 }}>
        {icon} {title}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
        Goal: {goal}
      </Typography>
    </Box>
  );
}

export default function OptionsAnalyticsDashboardPage() {
  const queryClient = useQueryClient();
  const [searchParams] = useSearchParams();

  const [tickerList, setTickerList] = useState([]);
  const [selectedTicker, setSelectedTicker] = useState(null);
  const [tickerInputValue, setTickerInputValue] = useState('');
  const [tickerQuery, setTickerQuery] = useState('');
  const [openTickerList, setOpenTickerList] = useState(false);
  const [defaultTickersLoaded, setDefaultTickersLoaded] = useState(false);
  const [loadingTickers, setLoadingTickers] = useState(false);
  const [loadingData, setLoadingData] = useState(false);
  const [selectedExpiration, setSelectedExpiration] = useState(null);

  const [gexData, setGexData] = useState(null);
  const [maxPainData, setMaxPainData] = useState(null);
  const [optionsMetrics, setOptionsMetrics] = useState(null);

  // Live, per-expiration term structure (Max Pain + GEX + options metrics
  // for the specific expiration picked in ExpirationSelector, computed from
  // today's chain rather than the nightly batch's nearest-expiration
  // snapshot). Drives the Gamma Exposure / Max Pain Analysis / Structural
  // Levels / Options Metrics cards below whenever selectedExpiration is set
  // -- otherwise those cards fall back to the batch data fetched above.
  const [termStructureData, setTermStructureData] = useState(null);
  const [termStructureLoading, setTermStructureLoading] = useState(false);
  const [termStructureError, setTermStructureError] = useState(false);
  const [error, setError] = useState(null);

  // Pre-select a ticker when arriving via /options-analytics?ticker=SYMBOL
  // (e.g. a row click from the Options Command Center). Only the symbol is
  // known at this point -- the Autocomplete's `name`/`exchange` fields are
  // cosmetic (used for its dropdown label, not for data fetching, which
  // only reads .symbol) and get filled in for real once tickerList loads,
  // if the user reopens the dropdown.
  useEffect(() => {
    const tickerParam = searchParams.get('ticker');
    if (tickerParam) {
      const symbol = tickerParam.toUpperCase();
      setSelectedTicker({ symbol });
      setTickerInputValue(symbol);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const fetchSymbols = async () => {
      setLoadingTickers(true);
      try {
        const params = {
          limit: 200,
        };
        if (tickerQuery) {
          params.q = tickerQuery;
        }
        const resp = await apiClient.get('/v1/universe/symbols', {
          params,
          signal: controller.signal,
        });
        setTickerList(resp.data.symbols || []);
        if (!tickerQuery && !defaultTickersLoaded) {
          setDefaultTickersLoaded(true);
        }
      } catch (err) {
        if (err.name !== 'CanceledError' && err.name !== 'AbortError') {
          console.error('Failed to load ticker list:', err);
        }
      } finally {
        setLoadingTickers(false);
      }
    };

    if (tickerQuery || (openTickerList && !defaultTickersLoaded)) {
      const timer = window.setTimeout(fetchSymbols, 250);
      return () => {
        window.clearTimeout(timer);
        controller.abort();
      };
    }

    return undefined;
  }, [tickerQuery, openTickerList, defaultTickersLoaded]);

  // Fetch all three data sources when ticker changes
  useEffect(() => {
    setSelectedExpiration(null); // expiration selection doesn't carry over between tickers

    if (!selectedTicker) {
      setGexData(null);
      setMaxPainData(null);
      setOptionsMetrics(null);
      setError(null);
      return;
    }

    (async () => {
      setLoadingData(true);
      setError(null);
      setGexData(null);
      setMaxPainData(null);
      setOptionsMetrics(null);

      try {
        const [gexResp, maxPainResp, optionsResp] = await Promise.all([
          apiClient.get('/v1/gex/dashboard', { params: { symbol: selectedTicker.symbol } }).catch(() => null),
          apiClient.get('/v1/max-pain/dashboard', { params: { symbol: selectedTicker.symbol } }).catch(() => null),
          apiClient.post('/v1/options/metrics', { symbol: selectedTicker.symbol }).catch(() => null),
        ]);

        setGexData(gexResp?.data ?? null);
        setMaxPainData(maxPainResp?.data ?? null);
        setOptionsMetrics(optionsResp?.data ?? null);

        if (!gexResp && !maxPainResp && !optionsResp) {
          setError('No data available for this ticker');
        }
      } catch (err) {
        setError(String(err));
      } finally {
        setLoadingData(false);
      }
    })();
  }, [selectedTicker]);

  // Live per-expiration term structure, refetched whenever the user picks a
  // specific expiration in ExpirationSelector. Cleared back to null (falling
  // back to the batch data above) when the expiration is cleared or the
  // ticker changes -- the ticker-change effect above always resets
  // selectedExpiration to null first, which this effect picks up.
  useEffect(() => {
    if (!selectedTicker || !selectedExpiration) {
      setTermStructureData(null);
      setTermStructureError(false);
      setTermStructureLoading(false);
      return undefined;
    }

    const controller = new AbortController();
    (async () => {
      setTermStructureLoading(true);
      setTermStructureError(false);
      try {
        const resp = await apiClient.get(`/v1/options/term-structure/${selectedTicker.symbol}`, {
          params: { expiration: selectedExpiration },
          signal: controller.signal,
        });
        setTermStructureData(resp.data);
        // The term-structure call just persisted a fresh snapshot for this
        // expiration -- invalidate so the history charts pick it up.
        queryClient.invalidateQueries({ queryKey: ['metric-history'] });
      } catch (err) {
        if (err.name !== 'CanceledError' && err.name !== 'AbortError') {
          setTermStructureError(true);
          setTermStructureData(null);
        }
      } finally {
        setTermStructureLoading(false);
      }
    })();

    return () => controller.abort();
  }, [selectedTicker, selectedExpiration, queryClient]);

  const getGexStatus = (value) => {
    if (value == null || Number.isNaN(Number(value))) return null;
    const num = Number(value);
    if (num > 0) {
      return {
        label: 'Long Gamma',
        color: 'success',
        description: 'Positive gamma exposure; option sellers may reduce hedges on large moves.',
      };
    }
    if (num < 0) {
      return {
        label: 'Short Gamma',
        color: 'warning',
        description: 'Negative gamma exposure; market makers may need aggressive re-hedging.',
      };
    }
    return {
      label: 'Neutral Gamma',
      color: 'info',
      description: 'Gamma exposure is balanced and less likely to trigger large hedging flows.',
    };
  };

  const getMaxPainStatus = (distancePct) => {
    if (distancePct == null || Number.isNaN(Number(distancePct))) return null;
    const pct = Number(distancePct);
    if (pct < -0.5) {
      return {
        label: 'Below Max Pain',
        color: 'info',
        description: 'Current price is below max pain; puts may be relatively expensive.',
      };
    }
    if (pct > 0.5) {
      return {
        label: 'Above Max Pain',
        color: 'info',
        description: 'Current price is above max pain; calls may be relatively expensive.',
      };
    }
    return {
      label: 'At Max Pain',
      color: 'success',
      description: 'Price is close to the max pain level, where open interest pain is minimized.',
    };
  };

  const getOpenInterestStatus = (callOi, putOi, side) => {
    if (callOi == null || putOi == null) return null;
    const callValue = Number(callOi);
    const putValue = Number(putOi);
    if (side === 'call') {
      if (callValue > putValue * 1.15) {
        return {
          label: 'Bullish OI',
          color: 'success',
          description: 'Call open interest exceeds put open interest, suggesting stronger bullish or resistance positioning.',
        };
      }
      if (callValue < putValue * 0.85) {
        return {
          label: 'Weak Call OI',
          color: 'warning',
          description: 'Call open interest is lower than put open interest, which may indicate less bullish conviction.',
        };
      }
      return {
        label: 'Balanced OI',
        color: 'info',
        description: 'Call and put open interest are roughly balanced, indicating neutral positioning.',
      };
    }

    if (side === 'put') {
      if (putValue > callValue * 1.15) {
        return {
          label: 'Bearish OI',
          color: 'warning',
          description: 'Put open interest exceeds call open interest, suggesting protective or bearish positioning.',
        };
      }
      if (putValue < callValue * 0.85) {
        return {
          label: 'Weak Put OI',
          color: 'success',
          description: 'Put open interest is lower than call open interest, which may indicate less bearish conviction.',
        };
      }
      return {
        label: 'Balanced OI',
        color: 'info',
        description: 'Call and put open interest are roughly balanced, indicating neutral positioning.',
      };
    }
    return null;
  };

  // When an expiration is selected, the lower cards read from the live
  // per-expiration term-structure fetch instead of the batch/nearest-
  // expiration data -- mapped onto the same field names the batch endpoints
  // already use so the render logic below (and getOverallConclusion/
  // getMarketSignal, which close over these same variables) doesn't need to
  // branch on which source it came from.
  const gexRow = selectedExpiration
    ? (termStructureData && {
        call_gex: termStructureData.total_call_gex,
        put_gex: termStructureData.total_put_gex,
        total_gex: termStructureData.total_gex,
        flip_level: termStructureData.key_levels?.zero_gamma,
        fetched_at: termStructureData.computed_at,
      })
    : gexData?.rows?.[0];

  const maxPainRow = selectedExpiration
    ? (termStructureData && {
        max_pain: termStructureData.max_pain_strike,
        distance_pct: termStructureData.max_pain_distance_pct,
        call_oi: termStructureData.total_call_oi,
        put_oi: termStructureData.total_put_oi,
        fetched_at: termStructureData.computed_at,
      })
    : maxPainData?.rows?.[0];

  // The term-structure payload IS the same shape calculate_options_metrics()
  // returns for /v1/options/metrics -- no field mapping needed, it drops
  // straight into SummaryCards.
  const displayOptionsMetrics = selectedExpiration ? termStructureData : optionsMetrics;

  // Structural Levels always derives from this same live payload now, in
  // both the default and live-expiration cases -- it previously fell back
  // to a SEPARATE, independently-computed endpoint (/v1/options/analysis,
  // the nightly batch's own 3-expiration GEX sweep) for the default case,
  // which could legitimately disagree with (or be null when) this payload's
  // own key_levels.zero_gamma wasn't -- e.g. Flip Level showing "N/A" here
  // while the Gamma Exposure card's own (still batch-sourced, untouched by
  // this) flip_level showed a real number, from two unrelated
  // computations that had no reason to agree in the first place.
  const displayAnalysisData = displayOptionsMetrics
    ? {
        call_wall: { strike: displayOptionsMetrics.call_wall, gex: displayOptionsMetrics.call_wall_gex },
        put_wall: { strike: displayOptionsMetrics.put_wall, gex: displayOptionsMetrics.put_wall_gex },
        flip_level:
          displayOptionsMetrics.key_levels?.zero_gamma != null
            ? { strike: displayOptionsMetrics.key_levels.zero_gamma, cumulative_gex: null }
            : null,
        spot_price: displayOptionsMetrics.underlying_price,
        timestamp: displayOptionsMetrics.computed_at,
      }
    : null;

  const callGexStatus = getGexStatus(gexRow?.call_gex);
  const putGexStatus = getGexStatus(gexRow?.put_gex);
  const totalGexStatus = getGexStatus(gexRow?.total_gex);
  const maxPainStatus = getMaxPainStatus(maxPainRow?.distance_pct);
  const callOiStatus = getOpenInterestStatus(maxPainRow?.call_oi, maxPainRow?.put_oi, 'call');
  const putOiStatus = getOpenInterestStatus(maxPainRow?.call_oi, maxPainRow?.put_oi, 'put');

  // Every directional factor the Market Conclusion draws on, computed once so
  // the headline label (getMarketSignal) and the narrative sentences
  // (getOverallConclusion) can never drift apart -- previously maxPain fed
  // the narrative text but was silently absent from the label's own logic.
  //
  // Deliberately excluded as votes (shown elsewhere on the page as context,
  // not opinions): IV Rank, Historical Volatility, VRP, and Expected Move
  // are magnitude/pricing-context metrics, not price-direction signals --
  // forcing them into a bullish/bearish vote would be dishonest. Net DEX/
  // VEX/CEX are already presented neutrally elsewhere on this page
  // (getExposureSignStatus never assigns a bullish/bearish color), so they
  // stay contextual here too rather than getting a vote this page doesn't
  // give them anywhere else.
  //
  // Weights: GEX is the dominant dealer-hedging-flow risk and gets the
  // heaviest weight (2), matching the priority-over-skew rule already
  // established below. Max pain gets the lightest weight (0.5) since it's
  // a weak/contested predictor even by its own card's description ("not a
  // reliable predictor on its own"). Wall breaks only vote when they've
  // actually happened (not "near"), since "near" is inherently ambiguous.
  const evaluateMarketFactors = () => {
    const factors = [];

    const totalGex = gexRow?.total_gex != null ? Number(gexRow.total_gex) : null;
    if (totalGex != null) {
      factors.push({
        key: 'gex',
        weight: 2,
        vote: totalGex > 0 ? 1 : totalGex < 0 ? -1 : 0,
        sentence:
          totalGex > 0
            ? 'Total GEX is positive, indicating a long-gamma regime that may support upward pressure as option sellers hedge into rising prices.'
            : totalGex < 0
              ? 'Total GEX is negative, indicating a short-gamma regime that may amplify downside moves as option sellers hedge into falling prices.'
              : 'Total GEX is neutral, indicating balanced gamma exposure.',
      });
    }

    const skew = displayOptionsMetrics?.skew != null ? Number(displayOptionsMetrics.skew) : null;
    if (skew != null) {
      factors.push({
        key: 'skew',
        weight: 1,
        vote: skew < 0 ? 1 : skew > 0 ? -1 : 0,
        sentence:
          skew > 0
            ? 'Volatility skew is positive, showing put skew and suggesting demand for downside protection.'
            : skew < 0
              ? 'Volatility skew is negative, showing call skew and suggesting bullish interest in upside risk.'
              : 'Volatility skew is neutral, showing no strong call/put bias.',
      });
    }

    const maxPain = maxPainRow?.distance_pct != null ? Number(maxPainRow.distance_pct) : null;
    if (maxPain != null) {
      // Max pain theory: price tends to drift toward max pain into expiry,
      // so being meaningfully above/below it is a mild pull in the
      // OPPOSITE direction -- not "above max pain = bullish".
      factors.push({
        key: 'maxPain',
        weight: 0.5,
        vote: maxPain > 0.5 ? -1 : maxPain < -0.5 ? 1 : 0,
        sentence:
          maxPain > 0.5
            ? 'Price is above max pain, which may reflect heavier call exposure and a mild pull back toward max pain into expiry.'
            : maxPain < -0.5
              ? 'Price is below max pain, which may reflect heavier put exposure and a mild pull back toward max pain into expiry.'
              : 'Price is close to max pain, suggesting the options market is relatively balanced around current levels.',
      });
    }

    const callPremium = displayOptionsMetrics?.call_premium_notional;
    const putPremium = displayOptionsMetrics?.put_premium_notional;
    const premiumPcr = callPremium != null && putPremium != null ? putPremium / (callPremium || 1) : null;
    if (premiumPcr != null) {
      // Same call-biased/put-biased thresholds as the Premium Put/Call
      // Ratio card itself (SummaryCards.jsx getMetricStatus('premium_pcr')).
      factors.push({
        key: 'premiumPcr',
        weight: 1,
        vote: premiumPcr < 0.7 ? 1 : premiumPcr > 1.5 ? -1 : 0,
        sentence:
          premiumPcr < 0.7
            ? 'Premium put/call ratio is call-biased -- real dollars are flowing predominantly into calls today.'
            : premiumPcr > 1.5
              ? 'Premium put/call ratio is put-biased -- real dollars are flowing predominantly into puts today.'
              : 'Premium put/call ratio is roughly balanced between calls and puts.',
      });
    }

    const callOi = maxPainRow?.call_oi != null ? Number(maxPainRow.call_oi) : null;
    const putOi = maxPainRow?.put_oi != null ? Number(maxPainRow.put_oi) : null;
    if (callOi != null && putOi != null) {
      // Same 1.15x threshold as getOpenInterestStatus above.
      const vote = callOi > putOi * 1.15 ? 1 : putOi > callOi * 1.15 ? -1 : 0;
      factors.push({
        key: 'openInterest',
        weight: 1,
        vote,
        sentence:
          vote === 1
            ? 'Call open interest exceeds put open interest, suggesting bullish or resistance-testing positioning.'
            : vote === -1
              ? 'Put open interest exceeds call open interest, suggesting protective or bearish positioning.'
              : 'Call and put open interest are roughly balanced.',
      });
    }

    const spot = displayAnalysisData?.spot_price;
    const callWallStrike = displayAnalysisData?.call_wall?.strike;
    const putWallStrike = displayAnalysisData?.put_wall?.strike;
    if (spot != null && callWallStrike != null && spot >= callWallStrike) {
      factors.push({
        key: 'callWallBreak',
        weight: 1,
        vote: 1,
        sentence: 'Price has pushed above the call wall, suggesting that resistance is no longer holding.',
      });
    }
    if (spot != null && putWallStrike != null && spot <= putWallStrike) {
      factors.push({
        key: 'putWallBreak',
        weight: 1,
        vote: -1,
        sentence: 'Price has fallen below the put wall, suggesting that support is no longer holding.',
      });
    }

    return factors;
  };

  const marketFactors = evaluateMarketFactors();

  // Weighted sum of every factor's vote -- see evaluateMarketFactors above
  // for what's included/excluded and why. Thresholds are set relative to
  // the max possible score (2 + 1 + 0.5 + 1 + 1 + 1(each wall) = 7.5) so
  // "Buy"/"Sell" require several factors actually agreeing, not just GEX
  // alone -- GEX alone (weight 2) now lands as Bullish/Bearish, matching
  // the old behavior for the single-factor case.
  const getMarketSignal = (factors) => {
    if (!factors || factors.length === 0) return null;
    const score = factors.reduce((sum, f) => sum + f.weight * f.vote, 0);

    if (score >= 3) {
      return { label: 'Buy', chipColor: 'success', textColor: 'success.main', advice: 'Buy.' };
    }
    if (score >= 1) {
      return { label: 'Bullish', chipColor: 'success', textColor: 'success.main', advice: 'Keep with bullish bias.' };
    }
    if (score <= -3) {
      return { label: 'Sell', chipColor: 'error', textColor: 'error.main', advice: 'Sell.' };
    }
    if (score <= -1) {
      return { label: 'Bearish', chipColor: 'error', textColor: 'error.main', advice: 'Keep with bearish/cautious bias.' };
    }
    return {
      label: 'Neutral',
      chipColor: 'default',
      textColor: 'text.secondary',
      advice: 'Keep position; signals are mixed or too weak for a clear bias.',
    };
  };

  const getOverallConclusion = (factors) => {
    if (!factors || factors.length === 0) return null;
    const sentences = factors.map((f) => f.sentence);
    const { advice } = getMarketSignal(factors);
    return `${sentences.join(' ')} ${advice}`;
  };

  const marketSignal = getMarketSignal(marketFactors);

  // Strategy pick crossed on two independent axes: Total Score gives
  // direction (bullish/bearish/neutral), VRP gives pricing (is premium rich
  // or cheap right now) -- a bullish read with rich vol calls for selling
  // premium in a bullish structure, not just buying calls. Thresholds are
  // the user-specified ones, not the same +-1/+-3 bands as getMarketSignal
  // above (that one's tuned for the Buy/Sell headline label; this one's
  // tuned for strategy selection and intentionally coarser).
  const getRecommendedStrategy = (score, vrpPct) => {
    if (score == null || vrpPct == null || Number.isNaN(score) || Number.isNaN(vrpPct)) return null;
    const rich = vrpPct > 10;
    if (score >= 1.5) {
      return rich
        ? 'Bull Put Spread / Covered Call (Vol is rich, sell premium)'
        : 'Long Call / Call Debit Spread (Vol is cheap/neutral, buy premium)';
    }
    if (score <= -1.5) {
      return rich
        ? 'Bear Call Spread (Vol is rich, sell premium)'
        : 'Long Put / Put Debit Spread (Vol is cheap/neutral, buy premium)';
    }
    return rich
      ? 'Iron Condor / Short Straddle (Expect pinning, sell premium)'
      : 'Long Straddle / Calendar Spread (Expect breakout, buy premium)';
  };

  const totalScore = marketFactors.reduce((sum, f) => sum + f.weight * f.vote, 0);
  const vrpPct =
    displayOptionsMetrics?.volatility_risk_premium != null
      ? displayOptionsMetrics.volatility_risk_premium * 100
      : null;
  const recommendedStrategy =
    marketFactors.length > 0 ? getRecommendedStrategy(totalScore, vrpPct) : null;

  return (
    <Container maxWidth="xl" sx={{ py: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Options Analytics Dashboard
        </Typography>
        {selectedTicker && (
          <Button
            className="no-print"
            variant="outlined"
            startIcon={<PrintIcon />}
            onClick={() => window.print()}
          >
            Print / Save as PDF
          </Button>
        )}
      </Box>

      {/* Ticker Selector */}
      <Paper className="no-print" sx={{ p: 2, mb: 3 }}>
        <Autocomplete
          open={openTickerList}
          onOpen={() => {
            setOpenTickerList(true);
            if (!defaultTickersLoaded) {
              setTickerQuery('');
            }
          }}
          onClose={() => setOpenTickerList(false)}
          openOnFocus
          options={tickerList}
          getOptionLabel={(opt) => `${opt.symbol} - ${opt.name || ''} (${opt.exchange})`}
          value={selectedTicker}
          onChange={(e, val) => {
            setSelectedTicker(val);
            setTickerInputValue(val ? `${val.symbol} - ${val.name || ''}` : '');
          }}
          inputValue={tickerInputValue}
          onInputChange={(e, val, reason) => {
            setTickerInputValue(val);
            if (reason === 'input') {
              setTickerQuery(val);
            }
          }}
          loading={loadingTickers}
          renderInput={(params) => (
            <TextField
              {...params}
              label="Select Ticker"
              placeholder="Type symbol or company name..."
              InputProps={{
                ...params.InputProps,
                endAdornment: (
                  <>
                    {loadingTickers ? <CircularProgress color="inherit" size={20} /> : null}
                    {params.InputProps.endAdornment}
                  </>
                ),
              }}
            />
          )}
        />
      </Paper>

      {loadingData && (
        <Box className="no-print" sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
        </Box>
      )}

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {displayOptionsMetrics?.is_stale_fallback && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          ⚠️ Live options data currently shows zero open interest for {selectedTicker?.symbol}
          {' '}(a known yfinance data gap during off-hours) -- showing the last known-good snapshot
          {displayOptionsMetrics.snapshot_fetched_at
            ? ` from ${new Date(displayOptionsMetrics.snapshot_fetched_at).toLocaleString()}`
            : ''}
          {' '}instead. Prices shown are current; positioning data below is not.
        </Alert>
      )}
      {displayOptionsMetrics?.data_source === 'live_zero_oi' && (
        <Alert severity="info" sx={{ mb: 2 }}>
          ℹ️ Live options data currently shows zero open interest for {selectedTicker?.symbol}
          {' '}(a known yfinance data gap during off-hours) and no prior snapshot is available yet
          to fall back to. The numbers below are likely not meaningful right now -- try again during
          market hours.
        </Alert>
      )}

      {selectedTicker && !loadingData && (
        <>
          {/* Ticker Header */}
          <Typography variant="h5" sx={{ mb: 2 }}>
            {selectedTicker.symbol} {selectedTicker.name && `- ${selectedTicker.name}`}
          </Typography>

          {/* Term structure: compare positioning across expirations using today's
              data (a cross-section, not a forecast -- see ExpirationSelector's
              doc comment). Selecting one also filters the history charts below
              to that expiration's own series. */}
          <ExpirationSelector
            symbol={selectedTicker.symbol}
            expiration={selectedExpiration}
            onExpirationChange={setSelectedExpiration}
            termStructureLoading={termStructureLoading}
            termStructureError={termStructureError}
            termStructureData={termStructureData}
          />

          <SectionHeader
            icon="🎯"
            title="Structural & Dealer Positioning Overview"
            goal="Identify market maker hedging levels, price ceilings/floors, and key volatility acceleration zones."
          />

          {/* History trend chart (separate from the point-in-time cards below --
              see MetricHistoryChart's doc comment for why these are never averaged) */}
          <MetricHistoryChart symbol={selectedTicker.symbol} metric="gex" expiration={selectedExpiration} />

          {/* GEX Summary */}
          {gexRow && (
            <Paper sx={{ p: 2, mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Gamma Exposure (GEX) {selectedExpiration ? `(Live: ${selectedExpiration})` : ''}
                </Typography>
                <LastUpdated timestamp={gexRow.fetched_at} />
              </Box>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary">Call GEX</Typography>
                      <Typography variant="h6">
                        {gexRow.call_gex?.toFixed(2) || 'N/A'}
                      </Typography>
                      {gexRow.call_gex != null && (
                        <Box sx={{ mt: 1 }}>
                          <Chip label={callGexStatus?.label} color={callGexStatus?.color} size="small" />
                          {callGexStatus?.description && (
                            <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary' }}>
                              {callGexStatus.description}
                            </Typography>
                          )}
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary">Put GEX</Typography>
                      <Typography variant="h6">
                        {gexRow.put_gex?.toFixed(2) || 'N/A'}
                      </Typography>
                      {gexRow.put_gex != null && (
                        <Box sx={{ mt: 1 }}>
                          <Chip label={putGexStatus?.label} color={putGexStatus?.color} size="small" />
                          {putGexStatus?.description && (
                            <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary' }}>
                              {putGexStatus.description}
                            </Typography>
                          )}
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary">Total GEX</Typography>
                      <Typography variant="h6">
                        {gexRow.total_gex?.toFixed(2) || 'N/A'}
                      </Typography>
                      {gexRow.total_gex != null && (
                        <Box sx={{ mt: 1 }}>
                          <Chip label={totalGexStatus?.label} color={totalGexStatus?.color} size="small" />
                          {totalGexStatus?.description && (
                            <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary' }}>
                              {totalGexStatus.description}
                            </Typography>
                          )}
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary">Flip Level</Typography>
                      <Typography variant="h6">
                        ${gexRow.flip_level?.toFixed(2) || 'N/A'}
                      </Typography>
                      <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
                        The price where dealer hedging flips character. Above it, dealers tend to smooth out
                        price swings; below it, their hedging can amplify moves instead.
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Paper>
          )}

          {/* Structural Levels - Call Wall, Put Wall, Flip Level */}
          {displayAnalysisData && (
            <Paper sx={{ p: 2, mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Structural Levels {selectedExpiration ? `(Live: ${selectedExpiration})` : '(Live: nearest expiration)'}
                </Typography>
                <LastUpdated timestamp={displayAnalysisData.timestamp} />
              </Box>
              {(() => {
                const callWallStrike = displayAnalysisData.call_wall?.strike;
                const putWallStrike = displayAnalysisData.put_wall?.strike;
                const flipLevelStrike = displayAnalysisData.flip_level?.strike;
                const invalidStructuralLevels = Boolean(
                  callWallStrike != null &&
                  putWallStrike != null &&
                  flipLevelStrike != null &&
                  callWallStrike === putWallStrike &&
                  callWallStrike === flipLevelStrike &&
                  displayAnalysisData.call_wall?.gex === 0 &&
                  displayAnalysisData.put_wall?.gex === 0 &&
                  (displayAnalysisData.flip_level?.cumulative_gex || 0) === 0
                );

                const formatStrike = (strike) =>
                  invalidStructuralLevels || strike == null ? 'N/A' : `$${strike.toFixed(2)}`;

                const formatGex = (value) =>
                  invalidStructuralLevels || value == null ? 'N/A' : value.toLocaleString('en-US', { maximumFractionDigits: 0 });

                const formatCumGex = (value) =>
                  invalidStructuralLevels || value == null ? 'N/A' : value.toLocaleString('en-US', { maximumFractionDigits: 0 });

                // Where price sits relative to each structural level, right now --
                // same "Above Max Pain" style badge as the Max Pain card above.
                const spot = displayAnalysisData.spot_price;

                const callWallStatus = (() => {
                  if (invalidStructuralLevels || spot == null || callWallStrike == null) return null;
                  if (spot >= callWallStrike) {
                    return { label: 'Above Call Wall', color: 'warning', description: 'Price has pushed above the call wall -- this resistance may no longer be holding, or a bigger move is underway.' };
                  }
                  if (((spot - callWallStrike) / callWallStrike) * 100 > -2) {
                    return { label: 'Near Call Wall', color: 'warning', description: 'Price is close to the call wall; upward moves may face resistance here as dealers hedge.' };
                  }
                  return { label: 'Below Call Wall', color: 'info', description: 'Price has room to run before reaching the call wall.' };
                })();

                const putWallStatus = (() => {
                  if (invalidStructuralLevels || spot == null || putWallStrike == null) return null;
                  if (spot <= putWallStrike) {
                    return { label: 'Below Put Wall', color: 'warning', description: 'Price has fallen below the put wall -- this support may no longer be holding.' };
                  }
                  if (((spot - putWallStrike) / putWallStrike) * 100 < 2) {
                    return { label: 'Near Put Wall', color: 'warning', description: 'Price is close to the put wall; downward moves may find support here as dealers hedge.' };
                  }
                  return { label: 'Above Put Wall', color: 'success', description: 'Price has room before reaching the put wall.' };
                })();

                const flipStatus = (() => {
                  if (invalidStructuralLevels || spot == null || flipLevelStrike == null) return null;
                  if (spot > flipLevelStrike) {
                    return { label: 'Long Gamma Regime', color: 'success', description: 'Price is above the flip level -- dealer hedging tends to dampen swings here, favoring calmer trading.' };
                  }
                  return { label: 'Short Gamma Regime', color: 'warning', description: 'Price is below the flip level -- dealer hedging can amplify moves here, favoring choppier trading.' };
                })();

                return (
                  <>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6} md={3}>
                        <Card>
                          <CardContent>
                            <Typography color="textSecondary">Call Wall</Typography>
                            <Typography variant="h6">
                              {formatStrike(callWallStrike)}
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                              GEX: {formatGex(displayAnalysisData.call_wall?.gex)}
                            </Typography>
                            {callWallStatus && (
                              <Box sx={{ mt: 1 }}>
                                <Chip label={callWallStatus.label} color={callWallStatus.color} size="small" />
                                <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary' }}>
                                  {callWallStatus.description}
                                </Typography>
                              </Box>
                            )}
                            <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
                              Strike with the heaviest call-side gamma -- tends to act like a ceiling the price
                              struggles to push above.
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                        <Card>
                          <CardContent>
                            <Typography color="textSecondary">Put Wall</Typography>
                            <Typography variant="h6">
                              {formatStrike(putWallStrike)}
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                              GEX: {formatGex(displayAnalysisData.put_wall?.gex)}
                            </Typography>
                            {putWallStatus && (
                              <Box sx={{ mt: 1 }}>
                                <Chip label={putWallStatus.label} color={putWallStatus.color} size="small" />
                                <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary' }}>
                                  {putWallStatus.description}
                                </Typography>
                              </Box>
                            )}
                            <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
                              Strike with the heaviest put-side gamma -- tends to act like a floor that offers support.
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                        <Card>
                          <CardContent>
                            <Typography color="textSecondary">Flip Level</Typography>
                            <Typography variant="h6">
                              {formatStrike(flipLevelStrike)}
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                              CumGEX: {formatCumGex(displayAnalysisData.flip_level?.cumulative_gex)}
                            </Typography>
                            {flipStatus && (
                              <Box sx={{ mt: 1 }}>
                                <Chip label={flipStatus.label} color={flipStatus.color} size="small" />
                                <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary' }}>
                                  {flipStatus.description}
                                </Typography>
                              </Box>
                            )}
                            <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
                              Where dealer hedging flips character -- calmer above it, choppier below it.
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                        <Card>
                          <CardContent>
                            <Typography color="textSecondary">Spot Price</Typography>
                            <Typography variant="h6">
                              ${displayAnalysisData.spot_price?.toFixed(2) || 'N/A'}
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                              Reference
                            </Typography>
                            <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
                              Today's actual stock price -- the point every level above is measured against.
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    </Grid>
                    {invalidStructuralLevels && (
                      <Typography variant="caption" sx={{ mt: 2, display: 'block', color: 'warning.main' }}>
                        ℹ️ Analysis data is too sparse to derive reliable key strike levels.
                      </Typography>
                    )}
                  </>
                );
              })()}
              <Typography variant="caption" sx={{ mt: 2, display: 'block', color: 'text.secondary' }}>
                {selectedExpiration
                  ? `ℹ️ Live term structure computed just now for ${selectedExpiration}`
                  : 'ℹ️ Live yfinance option-chain compute for the nearest expiration -- see the timestamp above for exactly when (may be a recent cache hit, not necessarily this instant).'}
              </Typography>
            </Paper>
          )}

          {/* Strike-level dealer positioning -- shares the live payload with
              every other section below, so it's fetched once here. */}
          {displayOptionsMetrics && (
            <>
              <LastUpdated timestamp={displayOptionsMetrics.computed_at} sx={{ display: 'block', mb: 1 }} />
              <SummaryCards data={displayOptionsMetrics} sections={['walls']} />
              <StrikeExposureChart
                strikes={displayOptionsMetrics.strikes}
                spot={displayOptionsMetrics.underlying_price}
                callWall={displayOptionsMetrics.key_levels?.call_wall}
                putWall={displayOptionsMetrics.key_levels?.put_wall}
                zeroGamma={displayOptionsMetrics.key_levels?.zero_gamma}
                expectedMove={displayOptionsMetrics.expected_move}
              />
            </>
          )}

          <SectionHeader
            icon="📈"
            title="Volatility & Skew Overview"
            goal="Determine if options are overpriced or underpriced, and detect structural directional skew across the option chain."
          />

          {displayOptionsMetrics && (
            <>
              <LastUpdated timestamp={displayOptionsMetrics.computed_at} sx={{ display: 'block', mb: 1 }} />
              <SummaryCards data={displayOptionsMetrics} sections={['volatility']} />
              <IVSmileChart
                ivSmile={displayOptionsMetrics.iv_smile}
                spot={displayOptionsMetrics.underlying_price}
              />
              {displayOptionsMetrics.next_earnings_date &&
                new Date(displayOptionsMetrics.next_earnings_date) >= new Date(new Date().toDateString()) && (
                  <Alert severity="warning" sx={{ mb: 2 }}>
                    ⚠️ Earnings Date: {displayOptionsMetrics.next_earnings_date}
                  </Alert>
              )}
              <IVTermStructureChart symbol={selectedTicker.symbol} />
            </>
          )}

          <SectionHeader
            icon="🔥"
            title="Options Flow & Positioning Heatmap Overview"
            goal="Spot institutional positioning, contract accumulation, and unusual activity without needing live tick data."
          />

          <MetricHistoryChart symbol={selectedTicker.symbol} metric="maxPain" expiration={selectedExpiration} />

          {/* Max Pain Summary */}
          {maxPainRow && (
            <Paper sx={{ p: 2, mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Max Pain Analysis {selectedExpiration ? `(Live: ${selectedExpiration})` : ''}
                </Typography>
                <LastUpdated timestamp={maxPainRow.fetched_at} />
              </Box>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary">Max Pain Level</Typography>
                      <Typography variant="h6">
                        ${maxPainRow.max_pain?.toFixed(2) || 'N/A'}
                      </Typography>
                      <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
                        The strike where option buyers collectively lose the most at expiration. Some traders
                        watch it as a possible price magnet into expiry, but it's not a reliable predictor on its own.
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary">Distance %</Typography>
                      <Typography variant="h6">
                        {maxPainRow.distance_pct?.toFixed(2) || 'N/A'}%
                      </Typography>
                      {maxPainRow.distance_pct != null && (
                        <Box sx={{ mt: 1 }}>
                          <Chip label={maxPainStatus?.label} color={maxPainStatus?.color} size="small" />
                          {maxPainStatus?.description && (
                            <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary' }}>
                              {maxPainStatus.description}
                            </Typography>
                          )}
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary">Call OI</Typography>
                      <Typography variant="h6">
                        {maxPainRow.call_oi?.toLocaleString() || 'N/A'}
                      </Typography>
                      {callOiStatus && (
                        <Box sx={{ mt: 1 }}>
                          <Chip label={callOiStatus.label} size="small" color={callOiStatus.color} />
                          <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary' }}>
                            {callOiStatus.description}
                          </Typography>
                        </Box>
                      )}
                      <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
                        Call open interest shows the amount of bullish/options resistance flow at strikes above the market.
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="textSecondary">Put OI</Typography>
                      <Typography variant="h6">
                        {maxPainRow.put_oi?.toLocaleString() || 'N/A'}
                      </Typography>
                      {putOiStatus && (
                        <Box sx={{ mt: 1 }}>
                          <Chip label={putOiStatus.label} size="small" color={putOiStatus.color} />
                          <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary' }}>
                            {putOiStatus.description}
                          </Typography>
                        </Box>
                      )}
                      <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>
                        Put open interest shows the amount of downside protection/support flow at strikes below the market.
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Paper>
          )}

          {displayOptionsMetrics && (
            <>
              <LastUpdated timestamp={displayOptionsMetrics.computed_at} sx={{ display: 'block', mb: 1 }} />
              <SummaryCards data={displayOptionsMetrics} sections={['flow']} />
              <UnusualVolumeTable contracts={displayOptionsMetrics.unusual_volume} />
            </>
          )}

          <SectionHeader
            icon="⏱️"
            title="Time Drift & Second-Order Greeks Overview"
            goal="Understand how time decay and volatility shifts will change market maker positioning over time."
          />

          {displayOptionsMetrics && (
            <>
              <LastUpdated timestamp={displayOptionsMetrics.computed_at} sx={{ display: 'block', mb: 1 }} />
              <SummaryCards data={displayOptionsMetrics} sections={['greeks']} />
            </>
          )}

          <SectionHeader
            icon="🧭"
            title="Executive Signal & Strategy Overview"
            goal="Combine all underlying metrics into an automated, actionable market conclusion and strategy recommendation."
          />

          {displayOptionsMetrics && (
            <>
              <Paper sx={{ p: 2, mt: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography variant="h6" sx={{ mr: 1 }}>
                    Market Conclusion
                  </Typography>
                  {marketSignal && (
                    <Box
                      sx={{
                        width: 12,
                        height: 12,
                        borderRadius: '50%',
                        bgcolor: marketSignal.textColor,
                        border: '1px solid',
                        borderColor: marketSignal.textColor,
                      }}
                    />
                  )}
                  {marketSignal && (
                    <Typography variant="subtitle2" sx={{ ml: 1, color: marketSignal.textColor }}>
                      {marketSignal.label}
                    </Typography>
                  )}
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {getOverallConclusion(marketFactors) || 'No conclusion available due to missing metric data.'}
                </Typography>

                {marketFactors.length > 0 && (
                  <TableContainer sx={{ mt: 2 }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Factor</TableCell>
                          <TableCell>Signal</TableCell>
                          <TableCell align="right">Weight</TableCell>
                          <TableCell align="right">Contribution</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {marketFactors.map((factor) => {
                          const meta = MARKET_FACTOR_LABELS[factor.key] || { label: factor.key, why: '' };
                          const contribution = factor.weight * factor.vote;
                          const voteLabel = factor.vote > 0 ? 'Bullish' : factor.vote < 0 ? 'Bearish' : 'Neutral';
                          const voteColor = factor.vote > 0 ? 'success' : factor.vote < 0 ? 'error' : 'default';
                          return (
                            <TableRow key={factor.key}>
                              <TableCell>
                                <Typography variant="body2">{meta.label}</Typography>
                                {meta.why && (
                                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                                    {meta.why}
                                  </Typography>
                                )}
                              </TableCell>
                              <TableCell>
                                <Chip label={voteLabel} color={voteColor} size="small" />
                              </TableCell>
                              <TableCell align="right">{factor.weight}</TableCell>
                              <TableCell
                                align="right"
                                sx={{
                                  color:
                                    contribution > 0
                                      ? 'success.main'
                                      : contribution < 0
                                        ? 'error.main'
                                        : 'text.secondary',
                                  fontWeight: 600,
                                }}
                              >
                                {contribution > 0 ? '+' : ''}
                                {contribution.toFixed(1)}
                              </TableCell>
                            </TableRow>
                          );
                        })}
                        <TableRow>
                          <TableCell colSpan={3}>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              Total score
                            </Typography>
                          </TableCell>
                          <TableCell align="right" sx={{ fontWeight: 700, color: marketSignal?.textColor }}>
                            {marketFactors.reduce((sum, f) => sum + f.weight * f.vote, 0) > 0 ? '+' : ''}
                            {marketFactors.reduce((sum, f) => sum + f.weight * f.vote, 0).toFixed(1)}
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}

                {recommendedStrategy && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2">Recommended Strategy</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {recommendedStrategy}
                    </Typography>
                  </Box>
                )}

                {/* Repeats the same timestamp shown next to the "Options Metrics"
                    heading above -- this card is long enough that the top-of-
                    section timestamp scrolls out of view, leaving the reader
                    with no visible answer to "how current is this?" right where
                    the conclusion (and its Buy/Sell/Bearish label) is read. */}
                <LastUpdated timestamp={displayOptionsMetrics.computed_at} sx={{ display: 'block', mt: 1 }} />
              </Paper>
            </>
          )}
        </>
      )}
    </Container>
  );
}
