import { Box, Typography } from '@mui/material';

function TickerCell({ symbol, companyName, align = 'left' }) {
  if (!symbol) return '-';
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: align === 'center' ? 'center' : 'flex-start',
        gap: 0.25,
        minWidth: 0,
      }}
    >
      <Typography
        component="span"
        variant="body2"
        sx={{ fontWeight: 600, lineHeight: 1.2 }}
      >
        {symbol}
      </Typography>
      {companyName ? (
        <Typography
          variant="caption"
          color="text.secondary"
          noWrap
          title={companyName}
          sx={{
            display: 'block',
            lineHeight: 1.2,
            minWidth: 0,
            maxWidth: '100%',
            textAlign: align === 'center' ? 'center' : 'left',
          }}
        >
          {companyName}
        </Typography>
      ) : null}
    </Box>
  );
}

export default TickerCell;
