# Social Signal worker deployment

Social Signals run in one optional Celery worker on the isolated `social_ingestion` queue. The feature starts fail-closed (`off` + `disabled`) and must be enabled from Operations after the worker and provider are ready. The scheduler requests a refresh every six hours.

There are two supported provider deployments:

- **Official X API:** uses the public backend image. Set `TWITTER_BEARER_TOKEN`; no private package or browser profile is needed.
- **Private xui-reader:** uses the `social-xui` Docker target. That image contains the installed private package and Playwright Chromium, so the image and every build cache containing it must remain private. Anyone who can pull it should be treated as having access to the package code.

GHCR is optional for a local Mac deployment. Build the xui image locally and point Compose at its local tag. GHCR is useful only when another trusted machine needs to pull the private image.

## Common setup

Copy `.env.docker.example` to the environment file used by Compose. Set a strong `SERVER_AUTH_PASSWORD`, `ADMIN_API_KEY`, and `OPENCODE_GO_API_KEY`. Leave these values in their environment file; do not put credentials or X session data in an image, Compose file, build argument, CI artifact, or Git commit. Social extraction uses `opencode-go/deepseek-v4-flash` through `https://opencode.ai/zen/go/v1`; the key is available only to backend services and is never sent to the browser.

Start the official/public worker with:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.social.yml \
  --profile social up -d celery-social
```

Then use **Operations → Social Signals administration** to choose `official`, test the configured lists, select `validation`, and inspect the preview before changing to `live`. Switching back to `disabled` is the immediate provider rollback.

## Private xui-reader: prepare the host profile

Use a dedicated X account/profile if possible. The integration is read-only from this application, but automated access can still trigger X rate limits, challenges, session invalidation, or account restrictions. A six-hour cadence reduces activity; it does not eliminate this risk.

Create a host directory that is not inside the repository and make it writable by uid 1000, the non-root container user. Initialize and authenticate locally:

```bash
mkdir -p "$XUI_PROFILE_DIR"
chmod 700 "$XUI_PROFILE_DIR"
xui config init --path "$XUI_PROFILE_DIR/config.toml"
xui profiles create automation --path "$XUI_PROFILE_DIR/config.toml"
xui auth login --path "$XUI_PROFILE_DIR/config.toml" --profile automation
xui auth status --path "$XUI_PROFILE_DIR/config.toml" --profile automation --json
```

If the container cannot update the session, change ownership of this dedicated directory to uid/gid 1000. Never broaden permissions on a home directory or unrelated data. When authentication expires, stop the Social worker, repeat `xui auth login` on the host, confirm `auth status`, and restart only that worker.

The profile is mounted read/write only at `/app/data/xui-reader` in `celery-social`. It is not mounted into the backend, scheduler, or general/data workers.

## Convenience wrapper

The wrapper combines prerequisite checks, the normal enabled-market stack, and
the isolated Social worker. It reads `.env`, then `.env.docker` when `.env` is
absent. To select another file, set `SOCIAL_STACK_ENV_FILE`; relative paths are
resolved from the repository root.

Build the private xui image locally and start the complete stack:

```bash
scripts/social-signal-stack.sh local up
```

Force-rebuild the normal backend, frontend, and worker images as well:

```bash
scripts/social-signal-stack.sh local up --build
```

The private xui image is rebuilt in both local forms. `--build` controls the
additional Compose rebuild of the normal application stack.

Or pull an already-published private GHCR image and start the complete stack:

```bash
scripts/social-signal-stack.sh ghcr up
```

GHCR mode requires `SOCIAL_WORKER_IMAGE` to use an immutable digest, a full
40-character `sha-*` tag, or a `v*` release tag. The wrapper rejects `latest`
and never performs `docker login`; authenticate separately before running it.

Operational commands target only the Social worker:

```bash
scripts/social-signal-stack.sh status
scripts/social-signal-stack.sh logs
scripts/social-signal-stack.sh stop
```

The wrapper never changes the database runtime to `validation` or `live`.
Activation remains an explicit administrator action in Operations. The manual
commands below remain useful for troubleshooting individual build and Compose
steps.

## GHCR-free local private build

Set `XUI_READER_REF` to the exact 40-character commit SHA you reviewed in the private repository. Create a `known_hosts` file using keys whose fingerprints you independently verified against [GitHub's published SSH host keys](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints). Do not populate this file from an unverified `ssh-keyscan` result.

BuildKit forwards the current SSH agent only for the private install step and mounts the verified host-key file as an ephemeral secret. Neither is copied into an image layer:

```bash
DOCKER_BUILDKIT=1 docker build \
  --ssh default \
  --secret id=github_known_hosts,src="$GITHUB_KNOWN_HOSTS_FILE" \
  --target social-xui \
  --build-arg XUI_READER_REF="$XUI_READER_REF" \
  -t stock-screener-social-xui:dev \
  -f backend/Dockerfile .
```

Start the locally built image:

```bash
SOCIAL_WORKER_IMAGE=stock-screener-social-xui:dev \
docker compose \
  -f docker-compose.yml \
  -f docker-compose.social.yml \
  -f docker-compose.social-xui.yml \
  --profile social up -d celery-social
```

The same Compose stack can build the private target itself when `XUI_READER_REF`, `GITHUB_KNOWN_HOSTS_FILE`, the SSH agent, and `XUI_PROFILE_DIR` are available. The explicit local build above is easier to audit and avoids a registry.

## Private GHCR image

Publish only to a private GHCR package in an access-controlled repository or organization. Do not export private build caches or verbose install logs to public CI artifacts. The private workflow deliberately refuses to push unless the package already exists and GitHub reports its visibility as `private`.

One-time trusted setup:

1. Protect `main` and the `v*` release-tag namespace with GitHub rulesets.
2. Create the `private-social-worker-release` GitHub environment and require approval from trusted maintainers before its secrets can be used.
3. Add `XUI_READER_DEPLOY_KEY` as an environment secret. It must be a read-only deploy key installed only on `xang1234/xui`.
4. Add `GITHUB_KNOWN_HOSTS` as an environment secret using host keys independently verified against [GitHub's published SSH fingerprints](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints).
5. Add `XUI_READER_REF` as an environment variable containing the exact 40-character private-repository commit SHA.
6. Bootstrap `ghcr.io/xang1234/stock-screener-social-xui` once from a trusted local build, then confirm **Package settings → Change visibility → Private** and disable repository-inherited access if it would expose the image through this public repository. GitHub documents that a newly published package starts private, but do not rely on that default without checking it. The workflow's preflight then prevents accidental publication to a missing or public package.

The workflow never runs for pull requests. It smoke-tests amd64 and arm64 against local HTML before publishing immutable `sha-*` and optional release tags; it never publishes a sole `latest` tag. GitHub-hosted provenance and SBOM records remain attached to the private package and are not uploaded as public artifacts.

For the bootstrap and later pulls, authenticate with a token that has only the package permissions needed by the host:

```bash
docker login ghcr.io
```

Set `SOCIAL_WORKER_IMAGE` to an immutable private image digest or release tag, then pull and start without building:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.social.yml \
  -f docker-compose.social-xui.yml \
  --profile social pull celery-social

docker compose \
  -f docker-compose.yml \
  -f docker-compose.social.yml \
  -f docker-compose.social-xui.yml \
  --profile social up -d --no-build celery-social
```

Access to a private container registry is not a substitute for package secrecy: a person who can pull the image can inspect the installed package.

## Validation and activation

Keep the runtime in `off` until the worker reports healthy. In Operations:

1. Add each X List using its readable name and immutable list ID/URL.
2. Test the list (at most five posts are sampled).
3. Enable the source.
4. Change runtime to `validation`; review source coverage, unresolved companies, candidate associations, and estimated LLM spend.
5. Change runtime to `live` only after the preview is acceptable.

The shared LLM allowance defaults to **$2 per Asia/Singapore day**. OpenCode Go input/output is budgeted at the documented peak DeepSeek V4 Flash rates. Work that cannot reserve budget waits for the next reset and is processed later; it is not discarded. Unknown or blocked model pricing also pauses dispatch rather than spending without a price contract.

Requests are additionally constrained to batches of at most 20 posts, one provider request at a time across the deployment, at least five seconds between request starts, 20 calls per refresh, and 80 calls per day. Provider failures are not automatically retried because a timed-out request may already have been billed. Remaining posts stay in the durable backlog for a later six-hour run. The four limits can be made stricter with `SOCIAL_LLM_BATCH_SIZE`, `SOCIAL_LLM_MIN_INTERVAL_SECONDS`, `SOCIAL_LLM_MAX_CALLS_PER_RUN`, and `SOCIAL_LLM_MAX_CALLS_PER_DAY`.

To stop collection immediately, set the provider to `disabled` in Operations. For a container-level rollback:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.social.yml \
  -f docker-compose.social-xui.yml \
  --profile social stop celery-social
```

Existing published Social results remain immutable and may become stale; the legacy Themes page and static site remain independent.

## Browser and image smoke checks

Before deploying a private image to each target architecture (arm64 and amd64), run it as uid 1000 and launch its bundled Chromium against local HTML only. Confirm that:

- `id -u` is not zero;
- `xui` and the Playwright-matched Chromium executable are readable/executable;
- Chromium starts without missing shared libraries and can read a local page;
- the runtime filesystem contains no forwarded SSH socket, private key, Git credential, or `known_hosts` build secret;
- the public image cannot import `xui_reader` and has no private browser/profile mount.

These smoke checks need no X account and must not make an X request. Run both the local-tag Compose path and the private-registry `--no-build` path before relying on a new image release.
