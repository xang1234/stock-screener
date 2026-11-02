import {
  Box, Button, Chip, CircularProgress, Divider, Drawer, Link, Stack, Typography,
} from '@mui/material';

import AddToWatchlistMenu from '../../components/common/AddToWatchlistMenu';
import {
  explanationMap, featureLagLabel, formatSocialScore, socialStateLabel, stateReasonText,
} from './socialSignalPresentation';

const words = (value) => String(value || '').replaceAll('_', ' ');
const capitalize = (value) => value.charAt(0).toUpperCase() + value.slice(1);
const stateReasons = (value) => Array.isArray(value)
  ? value
  : String(value || '').split(',').filter(Boolean);

export default function SocialEvidenceDrawer({
  open, onClose, selected, query, onOpenChart, onScan,
}) {
  const data = query?.data;
  const item = data?.item || selected;
  const explanation = explanationMap(item);
  const featureLag = featureLagLabel(explanation);
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
          {item?.canonical_symbol && item?.market ? <AddToWatchlistMenu symbols={item.canonical_symbol} /> : null}
        </Stack>
        <Divider sx={{ my: 2 }} />
        <Typography variant="subtitle2">Why it surfaced</Typography>
        <Stack direction="row" gap={0.75} flexWrap="wrap" sx={{ mt: 1 }}>
          {stateReasons(explanation.state_reasons).map((reason) => (
            <Chip key={reason} size="small" label={stateReasonText(reason)} />
          ))}
          {featureLag ? <Chip size="small" color="warning" variant="outlined" label={featureLag} /> : null}
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
        {data?.available && Array.isArray(data.claims) ? <ClaimList claims={data.claims} /> : null}
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

const supportChip = {
  supported: { label: 'Supported', color: 'success' },
  uncertain: { label: 'Uncertain', color: 'warning' },
  unsupported: { label: 'Unsupported', color: 'default' },
};

// Attachment-backed excerpts quote the attachment, not the poster.
const attachmentLabel = { article: 'linked article', image: 'shared image' };

function ClaimList({ claims }) {
  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="subtitle2">What X is claiming</Typography>
      {claims.length ? (
        <Box component="ul" aria-label="Extracted claims"
          sx={{ listStyle: 'none', p: 0, m: 0 }}>
          {claims.map((claim) => {
            const chip = supportChip[claim.support] || supportChip.unsupported;
            return (
              <Box component="li" key={`${claim.post_id}:${claim.theme}:${claim.excerpt}`}
                sx={{ py: 1, borderBottom: 1, borderColor: 'divider' }}>
                <Stack direction="row" gap={0.75} alignItems="center" flexWrap="wrap">
                  <Typography variant="body2" fontWeight={600}>{claim.theme} · {claim.relationship}</Typography>
                  <Chip size="small" variant="outlined" color={chip.color} label={chip.label} />
                </Stack>
                <Typography variant="body2" sx={{ my: 0.5 }}>“{claim.excerpt}”</Typography>
                <Typography variant="caption" color="text.secondary">
                  {claim.source_url ? (
                    <>
                      From a{' '}
                      <Link href={claim.source_url} target="_blank" rel="noreferrer">
                        {attachmentLabel[claim.source] || 'attachment'}
                      </Link>
                      {' shared by '}
                    </>
                  ) : null}
                  <Link href={claim.url} target="_blank" rel="noreferrer">@{claim.author_handle}</Link>
                  {' · '}{new Date(claim.created_at).toLocaleString()}
                </Typography>
              </Box>
            );
          })}
        </Box>
      ) : (
        <Typography variant="body2" color="text.secondary">No company claims extracted in this window.</Typography>
      )}
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
        {post.post_kind && post.stance ? (
          <Chip size="small" variant="outlined" title="Model hint, not verified"
            aria-label={`Model hint, not verified: ${post.post_kind}, ${post.stance}`}
            label={`${capitalize(post.post_kind)} · ${capitalize(post.stance)}`} />
        ) : null}
        {(post.source_names || []).map((name) => <Chip key={name} size="small" label={name} />)}
        <Typography variant="caption">{post.engagement?.likes ?? 0} likes</Typography>
        <Link href={post.url} target="_blank" rel="noreferrer">Open on X</Link>
      </Stack>
    </Box>
  );
}
