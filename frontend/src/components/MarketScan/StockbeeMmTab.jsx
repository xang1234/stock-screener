/**
 * Tab component that embeds the Stockbee MM page in an iframe.
 */
import { Box, Link, Typography } from '@mui/material';

const STOCKBEE_MM_URL = 'https://stockbee.blogspot.com/p/mm.html';

function StockbeeMmTab() {
  return (
    <Box sx={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column', gap: 1 }}>
      <Typography variant="body2" color="text.secondary">
        If the embedded page is blocked by your browser,{' '}
        <Link href={STOCKBEE_MM_URL} target="_blank" rel="noreferrer">
          open Stockbee MM in a new tab
        </Link>
        .
      </Typography>
      <iframe
        src={STOCKBEE_MM_URL}
        style={{ width: '100%', height: '100%', border: 'none', flex: 1 }}
        title="Stockbee MM"
        loading="lazy"
        referrerPolicy="strict-origin-when-cross-origin"
      />
    </Box>
  );
}

export default StockbeeMmTab;
