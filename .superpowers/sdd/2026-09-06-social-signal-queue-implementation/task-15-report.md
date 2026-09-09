# Task 15 report: local Docker and private worker image

Status: implementation complete inline. Private-package acquisition and private-image runtime smoke remain operator-gated because this session intentionally did not access an SSH credential, verified host-key file, xui profile, or private repository commit.

## Result

- Added one opt-in, non-root `celery-social` service consuming only `social_ingestion`; the ordinary backend/general/data workers never mount the xui profile.
- Added public and `social-xui` Docker targets while keeping `public` as the default final target. The public target does not install xui-reader or Playwright.
- The private builder requires a 40-character hexadecimal commit SHA, BuildKit SSH forwarding, and a separately mounted verified `known_hosts` secret. No SSH key, token, session path, pip configuration, or profile is accepted as an ARG/ENV/COPY layer.
- The private runtime installs Playwright's matching Chromium and OS libraries at `/opt/stockscanner/playwright`, restores `USER stockscanner`, and receives the host profile read/write only through the private Compose overlay.
- Added fail-closed Social settings to the shared backend/scheduler environment (`off`, `disabled`, six-hour cadence) and production resource/health/logging policy for the profiled worker.
- Added GHCR-free local-build, private GHCR pull, official API, xui login/reauthentication, validation/live activation, $2 budget deferral, six-hour account-risk, and rollback instructions.
- Expanded `.dockerignore` for SSH directories, private checkout names, and xui profile/session material.

## Verification

- RED: new security/topology suite failed four tests before the overlays, stages, and ignore rules existed; exact-SHA validation also failed before it was implemented.
- GREEN: `4 passed` in `test_social_worker_compose_contract.py`.
- Ordinary base Compose config: valid.
- Exact three-file private Compose config command: valid with fail-closed defaults.
- Docker build checks: public and private target graphs both report no warnings.
- Public arm64 image: built locally; runtime smoke confirmed Linux arm64, configured user `stockscanner`, uid 1000, and no `xui_reader` or `playwright` module.
- Public amd64 image: built locally under Docker Desktop emulation; the same non-root/private-dependency smoke passed.
- `git diff --check`: clean.

## Operator-gated verification

The private image was not built because a truthful build requires the operator's exact reviewed xui commit SHA, verified GitHub host-key file, and forwarded SSH agent. After those inputs exist, run the documented no-X browser smoke on arm64 and amd64. Do not substitute dummy credentials or an unverified `ssh-keyscan` result.
