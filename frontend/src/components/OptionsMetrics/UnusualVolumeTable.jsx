/**
 * UnusualVolumeTable
 *
 * Contracts trading today at more than 1.5x their existing open interest
 * (find_unusual_volume() in options_metrics.py) -- a rough "someone is
 * doing something today, not just holding" flag. Pre-filtered and
 * pre-sorted (highest ratio first) server-side; this just renders it.
 */
import {
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Alert,
} from '@mui/material';

/**
 * @param {Object} props
 * @param {Array<{strike:number, type:string, volume:number, open_interest:number, ratio:number}>} [props.contracts]
 * @param {number} [props.limit=25]
 */
export default function UnusualVolumeTable({ contracts, limit = 25 }) {
  if (!contracts) return null;

  const rows = contracts.slice(0, limit);

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 0.5 }}>
          Unusual Volume
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Contracts trading at over 1.5x their existing open interest today -- volume alone does not distinguish a new position from someone closing out an old one, so this is not proof of directional conviction, just a flag worth a closer look.
        </Typography>

        {rows.length === 0 && (
          <Alert severity="info">No contracts trading above 1.5x open interest right now.</Alert>
        )}

        {rows.length > 0 && (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Strike</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell align="right">Volume</TableCell>
                  <TableCell align="right">Open Interest</TableCell>
                  <TableCell align="right">Vol / OI</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {rows.map((row) => (
                  <TableRow key={`${row.type}-${row.strike}`}>
                    <TableCell>${row.strike.toFixed(2)}</TableCell>
                    <TableCell>
                      <Chip
                        label={row.type === 'call' ? 'Call' : 'Put'}
                        size="small"
                        color={row.type === 'call' ? 'success' : 'error'}
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell align="right">{row.volume.toLocaleString()}</TableCell>
                    <TableCell align="right">{row.open_interest.toLocaleString()}</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 600 }}>
                      {row.ratio.toFixed(2)}x
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        {contracts.length > limit && (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
            Showing top {limit} of {contracts.length} by ratio.
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}
