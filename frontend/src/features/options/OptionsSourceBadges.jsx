import { Box, Chip, Typography } from '@mui/material';

export default function OptionsSourceBadges({ badges = [], candidateRank, leaderRank }) {
  const isCandidate = badges.includes('candidate');
  const isLeader = badges.includes('leader');
  const label = isCandidate && isLeader ? 'Both' : isCandidate ? 'Candidate' : 'Leader';
  const ranks = [
    candidateRank == null ? null : `C${candidateRank}`,
    leaderRank == null ? null : `L${leaderRank}`,
  ].filter(Boolean).join(' · ');

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, whiteSpace: 'nowrap' }}>
      <Chip size="small" color={isCandidate && isLeader ? 'secondary' : 'primary'} label={label} />
      <Typography variant="caption" color="text.secondary">{ranks}</Typography>
    </Box>
  );
}
