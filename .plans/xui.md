# Plan: Replace in-tree xui-reader with private xui package + X API fallback



## Context



The repo currently vendors a full `xui-reader/` source tree (40+ files) and uses `sys.path` hacking to import it at runtime. The `news-tracker` repo demonstrates a cleaner pattern: install `xui-reader` as a pip package from a private GitHub repo (`git+https://github.com/xang1234/xui.git`), invoke it as a subprocess CLI, and fall back to the X API v2 when xui is unavailable.



This plan migrates StockScreenClaude to the same pattern: install the private `xui` package when available (via Docker build secret or local pip), invoke xui as a subprocess (not a Python import), and fall back to the X API v2 bearer token endpoint when xui is not installed or its auth fails.



## Architecture



```

┌─────────────────────────────────────┐

│       ContentIngestionService       │

│  fetchers["twitter"] = TwitterFetcher│

└──────────────┬──────────────────────┘

               │ .fetch(source, since)

               ▼

┌─────────────────────────────────────┐

│         TwitterFetcher              │

│  (content_ingestion_service.py)     │

│                                     │

│  delegates to:                      │

│   ┌───────────┐   ┌──────────────┐  │

│   │ XUI path  │──▶│ X API path   │  │

│   │ (primary) │   │ (fallback)   │  │

│   └───────────┘   └──────────────┘  │

└─────────────────────────────────────┘

        │                    │

        ▼                    ▼

  xui CLI subprocess    httpx GET to

  (xui read --json)     api.x.com/2/

                         tweets/search/recent

```



**Decision flow inside `TwitterFetcher.fetch()`:**

1. If `settings.xui_enabled` and xui CLI is on PATH → run xui subprocess

2. If xui succeeds → return results, done

3. If xui fails/unavailable AND `settings.twitter_bearer_token` is set → call X API v2

4. If neither available → raise RuntimeError with actionable message



## Changes



### 1. New file: `backend/app/services/twitter_xapi_fetcher.py`



X API v2 fallback fetcher. Responsible for calling `GET https://api.x.com/2/tweets/search/recent` (or user timeline endpoint) with a bearer token.



- Uses `httpx` (already a dependency) for HTTP calls

- Accepts a `ContentSource` and `since` datetime, returns `list[dict]` in the same format as the existing xui fetcher output (`external_id`, `title`, `content`, `url`, `author`, `published_at`)

- Builds query from source handle/list (e.g. `from:markminervini`)

- Handles rate limit (429) and auth errors (401/403) gracefully with logging

- Respects `settings.twitter_request_delay` between paginated calls

- X API v2 recent search endpoint: `https://api.x.com/2/tweets/search/recent`

  - Params: `query`, `max_results` (10-100), `tweet.fields=created_at,public_metrics,author_id`, `expansions=author_id`, `user.fields=name,username`

  - Auth: `Authorization: Bearer {token}`



### 2. Rewrite: `backend/app/services/xui_twitter_fetcher.py`



Replace the current in-process Python import approach with subprocess invocation (matching news-tracker pattern).



**Key changes:**

- Remove all `_load_xui_bindings`, `_ensure_xui_module_path`, `_import_xui_bindings` functions and the `_XUIBindings` dataclass

- Replace with subprocess invocation via `subprocess.run()` calling `xui read --json --sources user:<handle> --profile <profile> --path <config_path> --limit <N> --new --checkpoint-mode auto`

- Parse JSON output from stdout

- Add `_is_xui_available()` helper that checks if the `xui` CLI is on PATH (`shutil.which("xui")`)

- Keep the same public interface: `XUITwitterFetcher.fetch(source, since) -> list[dict]`

- Add fallback delegation: if xui unavailable or fails, delegate to `TwitterXAPIFetcher` if bearer token is configured



### 3. Update: `backend/app/services/xui_session_bridge_service.py`



The session bridge service also uses `_load_xui_auth_bindings` with the same `sys.path` hack. Update it to:

- Try importing `xui_reader.auth` normally (works when package is pip-installed)

- Remove `_repo_root_from_here()` and `_ensure_xui_module_path()` path hacking

- If import fails, raise `XUISessionBridgeError(503, "xui-reader package not installed...")`



### 4. Update: `backend/app/services/content_ingestion_service.py`



Modify `TwitterFetcher` class to handle the dual-path (xui primary, X API fallback) explicitly:



```python

class TwitterFetcher(BaseContentFetcher):

    def __init__(self):

        from .xui_twitter_fetcher import XUITwitterFetcher

        self._xui_fetcher = XUITwitterFetcher()



    def fetch(self, source, since=None):

        return self._xui_fetcher.fetch(source, since)  # fallback is internal

```



This stays the same — the fallback logic lives inside `XUITwitterFetcher.fetch()`.



### 5. Update: `backend/Dockerfile`



Replace the current xui-reader install approach with the news-tracker pattern:



**Current** (lines 29-36):

```dockerfile

COPY --chown=stockscanner:stockscanner xui-reader ./xui-reader

RUN pip install --no-cache-dir -e ./xui-reader && \

    python -m playwright install --with-deps chromium

```



**New:**

```dockerfile

ARG XUI_INSTALL=true

ARG XUI_PIP_SPEC=git+https://github.com/xang1234/xui.git@main



RUN --mount=type=secret,id=xui_github_token \

    if [ "$XUI_INSTALL" = "true" ]; then \

      spec="$XUI_PIP_SPEC"; \

      token_file="/run/secrets/xui_github_token"; \

      if [ -f "$token_file" ]; then \

        token="$(tr -d '\r\n' < "$token_file")"; \

        if [ -n "$token" ] && echo "$spec" | grep -q '^git+https://github.com/'; then \

          spec="$(echo "$spec" | sed "s#^git+https://github.com/#git+https://${token}@github.com/#")"; \

        fi; \

      fi; \

      pip install --no-cache-dir "xui-reader[cli] @ ${spec}" && \

      python -m playwright install --with-deps chromium; \

    fi

```



Remove the `COPY xui-reader ./xui-reader` line.



### 6. Update: `docker-compose.yml`



Add build args and secrets to the backend build anchor:



```yaml

x-backend-build: &backend-build

  context: .

  dockerfile: backend/Dockerfile

  args:

    XUI_INSTALL: ${XUI_INSTALL:-true}

    XUI_PIP_SPEC: ${XUI_PIP_SPEC:-git+https://github.com/xang1234/xui.git@main}

  secrets:

    - xui_github_token



# At bottom:

secrets:

  xui_github_token:

    file: ${XUI_GITHUB_TOKEN_FILE:-./.secrets/xui_github_token}

```



Add `TWITTER_BEARER_TOKEN` to the `x-app-env` anchor (it's already in settings.py but not plumbed through compose):

```yaml

TWITTER_BEARER_TOKEN: ${TWITTER_BEARER_TOKEN:-}

```



### 7. Update: `backend/app/config/settings.py`



Add new setting for xui command path (for subprocess invocation):

```python

xui_command: str = "xui"  # CLI command name or full path

```



Rename the `twitter_bearer_token` comment from "Legacy" to active:

```python

twitter_bearer_token: str = ""  # X API v2 bearer token (fallback when xui unavailable)

```



### 8. Update: `.env.docker.example` and `backend/.env.example`



- Add `XUI_COMMAND=xui` 

- Update `TWITTER_BEARER_TOKEN` comment to indicate it's the X API v2 fallback

- Add `.secrets/xui_github_token` setup instructions as comments

- Add `XUI_INSTALL`, `XUI_PIP_SPEC`, `XUI_GITHUB_TOKEN_FILE` Docker build vars



### 9. Update: `.gitignore`



Add `.secrets/` directory to gitignore.



### 10. Delete: `xui-reader/` directory



After the migration is complete, the entire in-tree `xui-reader/` directory (40+ files) can be removed. The functionality is now provided by the pip-installed private package.



## File summary



| File | Action |

|------|--------|

| `backend/app/services/twitter_xapi_fetcher.py` | **Create** — X API v2 fallback fetcher |

| `backend/app/services/xui_twitter_fetcher.py` | **Rewrite** — subprocess CLI invocation + fallback |

| `backend/app/services/xui_session_bridge_service.py` | **Update** — remove sys.path hack, use pip import |

| `backend/app/config/settings.py` | **Update** — add `xui_command`, update comment |

| `backend/Dockerfile` | **Update** — private package install via build secret |

| `docker-compose.yml` | **Update** — build args, secrets, bearer token env |

| `.env.docker.example` | **Update** — new env vars and docs |

| `backend/.env.example` | **Update** — new env vars and docs |

| `.gitignore` | **Update** — add `.secrets/` |

| `xui-reader/` | **Delete** — entire directory |



## Verification



1. **Unit tests** — Run existing tests to confirm no regressions:

   ```bash

   cd backend && source venv/bin/activate && pytest tests/unit/test_xui_twitter_fetcher.py tests/unit/test_xui_session_bridge_service.py -v

   ```

   Update these tests to mock `subprocess.run` instead of the old import-based bindings.



2. **xui CLI availability check** — Verify `shutil.which("xui")` returns correctly when package is/isn't installed.



3. **X API fallback** — Set `TWITTER_BEARER_TOKEN` in `.env`, disable xui (`XUI_ENABLED=false`), and verify twitter sources fetch via API.



4. **Docker build** — Build with and without the secret file:

   ```bash

   # With secret (private repo access)

   mkdir -p .secrets && echo 'ghp_xxx' > .secrets/xui_github_token

   docker-compose build backend

   

   # Without xui (API-only mode)  

   XUI_INSTALL=false docker-compose build backend

   ```



5. **Frontend** — The `xuiBridge.js` and browser extension are unaffected (they interact with the session bridge API, not the xui Python import directly). Verify the session status endpoint still works.

