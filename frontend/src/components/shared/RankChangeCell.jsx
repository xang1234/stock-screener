import { Box } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

const RankChangeCell = ({ value, justifyContent = 'flex-end' }) => {
  if (value === null || value === undefined) {
    return <Box sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>-</Box>;
  }
  const color = value > 0 ? 'success.main' : value < 0 ? 'error.main' : 'text.secondary';
  const prefix = value > 0 ? '+' : '';
  return (
    <Box display="flex" alignItems="center" justifyContent={justifyContent} sx={{ fontFamily: 'monospace' }}>
      {value > 0 && <TrendingUpIcon sx={{ fontSize: 12, mr: 0.25, color }} />}
      {value < 0 && <TrendingDownIcon sx={{ fontSize: 12, mr: 0.25, color }} />}
      <Box component="span" sx={{ color, fontWeight: value !== 0 ? 600 : 400, fontSize: '11px' }}>
        {prefix}{value}
      </Box>
    </Box>
  );
};

export default RankChangeCell;
