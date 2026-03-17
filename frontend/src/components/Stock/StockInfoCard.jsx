import {
  Alert,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Box,
  Divider,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import BusinessIcon from '@mui/icons-material/Business';

function StockInfoCard({ stockData }) {
  const { info, fundamentals, technicals } = stockData;

  const formatCurrency = (value) => {
    if (!value) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatNumber = (value, decimals = 2) => {
    if (value === null || value === undefined) return 'N/A';
    return value.toFixed(decimals);
  };

  const formatLargeNumber = (value) => {
    if (!value) return 'N/A';
    if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    return `$${value.toFixed(0)}`;
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="div" sx={{ flexGrow: 1 }}>
            {info.symbol}
            {info.name && (
              <Typography variant="subtitle1" color="text.secondary">
                {info.name}
              </Typography>
            )}
          </Typography>
          {info.sector && (
            <Chip
              icon={<BusinessIcon />}
              label={info.sector}
              color="primary"
              variant="outlined"
            />
          )}
        </Box>

        {info.current_price && (
          <Typography variant="h4" color="primary" gutterBottom>
            {formatCurrency(info.current_price)}
          </Typography>
        )}

        <Divider sx={{ my: 2 }} />

        <Typography variant="h6" gutterBottom>
          Company Information
        </Typography>
        <Grid container spacing={2}>
          {info.industry && (
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="text.secondary">
                Industry
              </Typography>
              <Typography variant="body1">{info.industry}</Typography>
            </Grid>
          )}
          {info.market_cap && (
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="text.secondary">
                Market Cap
              </Typography>
              <Typography variant="body1">{formatLargeNumber(info.market_cap)}</Typography>
            </Grid>
          )}
        </Grid>

        {fundamentals && (
          <>
            <Divider sx={{ my: 2 }} />
            <Typography variant="h6" gutterBottom>
              Fundamentals
            </Typography>
            <Grid container spacing={2}>
              {fundamentals.pe_ratio && (
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    P/E Ratio
                  </Typography>
                  <Typography variant="body1">
                    {formatNumber(fundamentals.pe_ratio)}
                  </Typography>
                </Grid>
              )}
              {fundamentals.eps_current && (
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    EPS (TTM)
                  </Typography>
                  <Typography variant="body1">
                    {formatCurrency(fundamentals.eps_current)}
                  </Typography>
                </Grid>
              )}
              {fundamentals.eps_growth_quarterly !== null && (
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    EPS Growth (QoQ)
                  </Typography>
                  <Typography
                    variant="body1"
                    color={fundamentals.eps_growth_quarterly > 0 ? 'success.main' : 'error.main'}
                  >
                    {formatNumber(fundamentals.eps_growth_quarterly)}%
                  </Typography>
                </Grid>
              )}
            </Grid>
          </>
        )}

        {technicals && (
          <>
            <Divider sx={{ my: 2 }} />
            <Typography variant="h6" gutterBottom>
              Technical Indicators
            </Typography>
            <Grid container spacing={2}>
              {technicals.ma_50 && (
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    50-day MA
                  </Typography>
                  <Typography variant="body1">
                    {formatCurrency(technicals.ma_50)}
                  </Typography>
                </Grid>
              )}
              {technicals.ma_200 && (
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    200-day MA
                  </Typography>
                  <Typography variant="body1">
                    {formatCurrency(technicals.ma_200)}
                  </Typography>
                </Grid>
              )}
              {technicals.high_52w && technicals.low_52w && (
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="text.secondary">
                    52-Week Range
                  </Typography>
                  <Typography variant="body1">
                    {formatCurrency(technicals.low_52w)} - {formatCurrency(technicals.high_52w)}
                  </Typography>
                </Grid>
              )}
            </Grid>
          </>
        )}

        {(!fundamentals && !technicals) && (
          <Alert severity="info" sx={{ mt: 2 }}>
            Limited data available. Try waiting a moment and searching again.
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}

export default StockInfoCard;
