# Stock Scanner v1.0.0

Stock Scanner v1.0.0 is the first tagged public release of the platform: a multi-market stock screening and research stack with Docker-first deployment and GHCR release images.

## Highlights

- Multi-market screening across the US, Hong Kong, Japan, and Taiwan with exchange-aware refresh pipelines and per-market scan badges.
- First-run bootstrap that stages universe, price, fundamentals, breadth, rankings, and autoscan hydration so a fresh install lands on usable data.
- Multi-strategy screening with saved filters and composite scoring across Minervini, CANSLIM, IPO, Volume Breakthrough, Setup Engine, and custom scans.
- AI-assisted research with chatbot workflows, theme discovery from social/news feeds, and market breadth analysis for higher-level context.
- Industry group rankings and watchlists with sparklines, movers, and stock-level drilldowns.
- Docker and GHCR deployment support with immutable `v1.0.0` image tags for server rollout and rollback.

## Deployment

Release images are published to GHCR under the `v1.0.0` tag:

- `ghcr.io/<owner>/stockscreenclaude-backend:v1.0.0`
- `ghcr.io/<owner>/stockscreenclaude-frontend:v1.0.0`

Update this file before future semver tags so each GitHub release carries a maintained capability summary alongside the tagged image version.
