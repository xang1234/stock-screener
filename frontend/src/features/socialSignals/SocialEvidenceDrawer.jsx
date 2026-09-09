import {
  Box, Button, Chip, CircularProgress, Divider, Drawer, Link, Stack, Typography,
} from '@mui/material';

import AddToWatchlistMenu from '../../components/common/AddToWatchlistMenu';
import { explanationMap, formatSocialScore, socialStateLabel } from './socialSignalPresentation';

const words = (value) => String(value || '').replaceAll('_', ' ');
const stateReasons = (value) => Array.isArray(value)
  ? value
  : String(value || '').split(',').filter(Boolean);

export default function SocialEvidenceDrawer({
  open, onClose, selected, query, onOpenChart, onOpenTheme, onScan,
}) {
  const data = query?.data;
  const item = data?.item || selected;
  const explanation = explanationMap(item);
  return (
    <Drawer anchor="right" open={open} onClose={onClose}>
      <Box sx={{ width: { xs: '100vw', sm: 520 }, p: 2 }} role="dialog" aria-label="Social evidence">
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="h6">{item?.canonical_symbol || 'Evidence'}</Typography>
            <Typography variant="caption" color="text.secondary">
              {item?.market || 'Global'} · {socialStateLabel(item?.state)}
            </Typography>
          </Box>
          <Button onClick={onClose}>Close</Button>
        </Stack>
        <Stack direction="row" spacing={1} sx={{ my: 1.5 }}>
          <Chip label={`Queue ${formatSocialScore(item?.queue_score)}`} />
          <Chip label={`Social ${formatSocialScore(item?.social_score)}`} />
          <Chip label={`Confirmation ${formatSocialScore(item?.confirmation_score)}`} />
        </Stack>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          <Button variant="contained" onClick={onOpenChart}>Open chart & setup</Button>
          <Button variant="outlined" onClick={onScan}>Send visible to Scan</Button>
          {explanation.theme_id ? <Button onClick={() => onOpenTheme(explanation.theme_id)}>Open theme</Button> : null}
          {item?.canonical_symbol && item?.market ? <AddToWatchlistMenu symbols={item.canonical_symbol} /> : null}
        </Stack>
        <Divider sx={{ my: 2 }} />
        <Typography variant="subtitle2">Why it surfaced</Typography>
        <Stack direction="row" gap={0.75} flexWrap="wrap" sx={{ mt: 1 }}>
          {stateReasons(explanation.state_reasons).map((reason) => (
            <Chip key={reason} size="small" label={words(reason)} />
          ))}
        </Stack>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Acceleration {formatSocialScore(explanation.acceleration)}
        </Typography>
        <ComponentMath title="Social components" components={explanation.social_components} />
        <ComponentMath title="Confirmation components" components={explanation.confirmation_components} />
        {query?.isLoading ? <CircularProgress aria-label="Loading evidence" sx={{ mt: 3 }} /> : null}
        {query?.isError ? <Typography color="error" sx={{ mt: 2 }}>Evidence could not be loaded.</Typography> : null}
        {data?.related_listings?.length ? (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2">Related listings</Typography>
            <Typography variant="body2">
              {data.related_listings.map((row) => `${row.market}:${row.canonical_symbol}`).join(' · ')}
            </Typography>
          </Box>
        ) : null}
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2">Source posts</Typography>
          {(data?.posts || []).slice(0, 3).map((post) => (
            <PaperPost key={post.post_id} post={post} />
          ))}
          {data?.available && !data.posts?.length ? (
            <Typography variant="body2" color="text.secondary">No post excerpts in this window.</Typography>
          ) : null}
        </Box>
      </Box>
    </Drawer>
  );
}

function ComponentMath({ title, components }) {
  const entries = Object.entries(components || {});
  if (!entries.length) return null;
  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="subtitle2">{title}</Typography>
      {entries.map(([name, component]) => (
        <Stack key={name} direction="row" justifyContent="space-between" gap={2}>
          <Typography variant="body2">{words(name)}</Typography>
          <Typography variant="body2" color="text.secondary">
            {formatSocialScore(component?.value)} · {component?.available_weight ?? 0}/{component?.total_weight ?? 0} weight
          </Typography>
        </Stack>
      ))}
    </Box>
  );
}

function PaperPost({ post }) {
  return (
    <Box sx={{ py: 1.5, borderBottom: 1, borderColor: 'divider' }}>
      <Typography variant="caption" color="text.secondary">
        @{post.author_handle} · {new Date(post.created_at).toLocaleString()}
      </Typography>
      <Typography variant="body2" sx={{ my: 0.75, whiteSpace: 'pre-wrap' }}>{post.excerpt}</Typography>
      <Stack direction="row" gap={0.5} alignItems="center" flexWrap="wrap">
        {(post.source_names || []).map((name) => <Chip key={name} size="small" label={name} />)}
        <Typography variant="caption">{post.engagement?.likes ?? 0} likes</Typography>
        <Link href={post.url} target="_blank" rel="noreferrer">Open on X</Link>
      </Stack>
    </Box>
  );
}
